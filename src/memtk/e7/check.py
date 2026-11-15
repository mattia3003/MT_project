"""E7 stage `check` (`docs/e6-e7-plan.md`, step 7): consistency checks of all E7 outputs.

Decisions X, Y, Z set; the run logs of `planted` and `reports` made with the current settings, in
that order, with S read by `reports` only; E5's lock unchanged and as recorded by `reports`; the
planted correctness criteria passed; every `planted` table reproduced; on S, the report selection,
verdicts and taxonomy of every report reproduced and every report figure present, the deletion test,
the taxonomy table and the patient profiles reproduced; the unit tests. Writes
`results/<pool>/e7/check_report.txt`; exit code 1 on any failure.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from memtk import checks, config, lesionmatch
from memtk.calibration import load_frozen
from memtk.e5.common import audit_config, lock_unchanged
from memtk.e6.common import exposure_config, load_heval, load_query, load_reference
from memtk.e7.common import chance_rates, class_rate, patient_name, reports_config
from memtk.e7.planted import correctness_tables, planted_reports, run_deletion
from memtk.e7.reports import (deletion_tables, group_of, profile_data, profile_table, select_reports, subset,
                              taxonomy_table)
from memtk.io import LabelStack, read_table
from memtk.reports import FLAGGED, deletion_summary, report_data, taxonomy, verdict
from memtk.sets import load_planted, load_set

LOG_KEYS = ["seed", "reports", "lesions.attach_gap"]
REPORT_KEYS = ["seed", "reports", "exposure", "calibration.bootstrap_draws"]
TOL = 1e-9


# --- stage ------------------------------------------------------------------------------------

def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None) -> None:
    """Consistency checks of all E7 outputs. Writes `results/<pool>/e7/check_report.txt`; exit code
    1 on any failure."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    planted_dir = config.results_dir(cfg, name, "e7") / "planted"
    out = config.results_dir(cfg, pool_name, "e7")
    reports_dir = out / "reports"
    report = checks.CheckReport()

    try:
        rep = reports_config(cfg)
        report.add("all E7 decisions (X, Y, Z) are set and implemented (reports:)", True, "")
    except SystemExit as e:
        report.add("all E7 decisions (X, Y, Z) are set and implemented (reports:)", False, str(e))
        rep = cfg["reports"]
    logs = {"planted": (planted_dir / "run_info.json", LOG_KEYS), "reports": (reports_dir / "run_info.json", REPORT_KEYS)}
    checks.check_run_logs(report, cfg, logs)
    info = {s: json.loads(p.read_text(encoding="utf-8")) if p.exists() else {} for s, (p, _) in logs.items()}
    report.add("'planted' never read S", not any(i.startswith("S") for i in info["planted"].get("inputs", [])),
               str(info["planted"].get("inputs")))
    try:
        now = lock_unchanged(cfg, name, pool_name)
        report.add("E5's lock unchanged since S was scored, and as recorded by 'reports'",
                   all(info["reports"].get(k) == v for k, v in now.items()), "")
    except SystemExit as e:
        report.add("E5's lock unchanged since S was scored, and as recorded by 'reports'", False, str(e))

    # planted copies, reproduced
    e3, _ = load_frozen(cfg, name)
    cut, bound = float(e3["detectors"]["lesion"]["candidate_min_s"]), int(e3["informative_bound_px"])
    chance = chance_rates(cfg, name)
    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    e4_dir = config.output_dir(cfg, name) / "e4"
    planted, planted_labels = load_planted(cfg, name)
    outcomes = read_table(config.results_dir(cfg, name, "e4") / "validate" / "planted_outcomes.csv")
    e2p = lesionmatch.load_tables(e4_dir, "planted", cut)
    e1p = read_table(e4_dir / "e1_planted.csv").set_index("query")
    les = read_table(e4_dir / "planted_lesions.csv")
    informative = les[les["area"] >= bound]
    sources = dict(zip(informative["lesion_id"].astype(str), informative["source_lesion"].fillna("").astype(str)))
    reports, table = planted_reports(outcomes, e1p, e2p, e3, sources, patient_name, chance,
                                     int(rep["correctness"]["slice_window"]))
    correct, criteria = correctness_tables(table, rep["correctness"])
    report.add("the planted correctness criteria pass (decision Z)", bool(criteria["pass"].all()),
               f"{int(criteria['pass'].sum())} of {len(criteria)}")
    flagged_p = table[table["gated"] & table["flagged"]]
    confusion = pd.crosstab(flagged_p["variant"], flagged_p["taxonomy"]).reset_index()
    controls = flagged_p[(flagged_p["stratum"] == "multi") & flagged_p["variant"].isin(rep["deletion"]["planted_controls"])]
    min_paired = int(rep["deletion"]["min_paired"])
    d_planted = run_deletion([reports[k] for k in controls["key"]], dict(zip(planted["key"], planted_labels)), min_paired,
                             t, e3)
    hcls = read_table(e4_dir / "heval_classes.csv")
    e2h = lesionmatch.load_tables(e4_dir, "heval", cut)
    e1h = read_table(e4_dir / "e1_heval.csv").set_index("query")
    hflag = hcls[hcls["e2_class"].isin(FLAGGED) & (hcls["stratum_e2"] == "multi")]
    h_reports = [report_data(r, e1h.loc[r["key"]] if r["key"] in e1h.index else None, e2h, e3) for _, r in hflag.iterrows()]
    h_labels = dict(zip(heval.keys, heval.labels))
    d_heval = run_deletion(h_reports, h_labels, min_paired, t, e3)
    deletion = pd.concat([d_planted.assign(set="planted controls"), d_heval.assign(set="H-eval chance flags")],
                         ignore_index=True)
    summary = pd.DataFrame([{"set": s, **deletion_summary(g)} for s, g in deletion.groupby("set", sort=False)])
    differ = checks.differing_tables({"reports": table, "correctness": correct, "criteria": criteria,
                                      "taxonomy_confusion": confusion, "deletion": deletion,
                                      "deletion_summary": summary}, planted_dir, TOL)
    report.add("every planted table reproduced", not differ, ", ".join(differ) if differ else "6 tables")

    # S: selection, verdicts, deletion, taxonomy, profiles
    audit, exp = audit_config(cfg), exposure_config(cfg)
    q = load_query("S", config.output_dir(cfg, pool_name) / "e5", audit["sets"]["main"], cut)
    h = load_heval(cfg, name, cut)
    flagged = read_table(config.results_dir(cfg, pool_name, "e5") / "analyze" / "flagged.csv")
    flagged = flagged[flagged["set"] == "main"].reset_index(drop=True)
    flagged = flagged.assign(group=flagged.apply(group_of, axis=1))
    e6_dir = config.results_dir(cfg, pool_name, "e6") / "exposure"
    e6 = {k: read_table(e6_dir / f"{k}.csv") for k in ("top_patients", "flag_details")}
    selection = select_reports(flagged, h.table, e6["top_patients"], e6["flag_details"], rep["selection"], exp,
                               np.random.default_rng(cfg["seed"]))
    index = read_table(reports_dir / "report_index.csv")
    cols = ["set", "key", "reason"]
    report.add("the report selection reproduced (decision X, same seed)",
               selection[cols].astype(str).equals(index[cols].astype(str)), f"{len(index)} reports")
    s_lm, h_lm = subset(q.e2, set(flagged["key"])), subset(h.e2, set(h.table["key"]))
    rows = {"S": flagged.set_index("key", drop=False), "H-eval": h.table.set_index("key", drop=False)}
    e1s = {"S": q.e1.set_index("query"), "H-eval": h.e1.set_index("query")}
    mismatch = []
    for r in index.to_dict("records"):
        d = report_data(rows[r["set"]].loc[r["key"]], e1s[r["set"]].loc[r["key"]], s_lm if r["set"] == "S" else h_lm, e3)
        line = verdict(d, patient_name, class_rate(chance, d.e2_class, d.stratum))
        if line != r["verdict"] or taxonomy(d) != r["taxonomy"]:
            mismatch.append(str(r["report"]))
    report.add("every report's verdict and taxonomy reproduced", not mismatch, ", ".join(mismatch[:5]))
    missing = [f for f in index["report"] if not (reports_dir / "reports" / f).exists()]
    report.add("every report figure exists", not missing, ", ".join(missing[:5]) or f"{len(index)} figures")
    pool_labels = LabelStack(config.output_dir(cfg, pool_name) / "labels.npz")
    deletion_s, summary_s = deletion_tables(flagged, h.table, s_lm, h_lm, pool_labels, h_labels, t, e3, min_paired)
    taxa = taxonomy_table(flagged, h.table, s_lm, h_lm, e3)
    profiles = profile_table(profile_data(flagged, q, e6["top_patients"], load_reference(cfg, name), exp, e3))
    differ = checks.differing_tables({"deletion": deletion_s, "deletion_summary": summary_s, "taxonomy": taxa,
                                      "patient_profiles": profiles}, reports_dir, TOL)
    report.add("deletion test, taxonomy and patient profiles on S reproduced", not differ,
               ", ".join(differ) if differ else "4 tables")
    missing = [p for p in profiles["patient_id"].unique() if not (reports_dir / "patients" / f"{p}.png").exists()]
    report.add("every patient profile figure exists", not missing, ", ".join(missing) or
               f"{profiles['patient_id'].nunique()} patients")

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = f"E7 check: pool '{pool_name}' against '{name}', seed {cfg['seed']}\n\n{report.to_text()}\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check"}, filename="check.run_info.json")
    print(f"\n{text}\nWrote {out / 'check_report.txt'}")
    if not report.all_passed:
        raise SystemExit(1)
