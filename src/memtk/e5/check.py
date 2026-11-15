"""E5 stage `check` (`docs/e5-plan.md`, step 6): consistency checks of all E5 outputs.

Decisions L-Q set; the run logs of `prepare`, `rehearse`, `score` and `analyze` made with the current
settings, in that order, with S read by `score` only; the lock (the `audit:` section, the frozen E3
file and the E5 file unchanged since `score`, and S scored once and completely); the `prepare`
values reproduced (sensitivity thresholds, tau_D*, held-out maxima, drop-only identity); every
`analyze` table and the outcome reading reproduced from the raw tables with the same seed; the unit
tests. Writes `results/<pool>/e5/check_report.txt`; exit code 1 on any failure. (`differing_tables` moved
to `memtk.checks` on 2026-11-15.)
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from memtk import calibration, checks, config, lesionmatch
from memtk.calibration import load_frozen
from memtk.audit import heldout, load_scored
from memtk.e5.analyze import analyze_scored, informative_areas
from memtk.e5.common import audit_config, audit_hash, score_marker
from memtk.e5.prepare import heldout_maxima, tau_d_star
from memtk.io import read_table
from memtk.sets import load_set

LOG_KEYS = ["seed", "audit", "calibration.bootstrap_draws", "lesions.attach_gap"]
INPUTS = {"prepare": ["T", "H-cal", "H-eval"], "rehearse": ["T", "H-cal", "H-eval", "planted"],
          "score": ["T", "S"], "analyze": ["T", "H-eval", "S tables"]}
TOL = 1e-9


# --- stage ------------------------------------------------------------------------------------

def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None) -> None:
    """Consistency checks of all E5 outputs. Writes `results/<pool>/e5/check_report.txt`; exit
    code 1 on any failure."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    real_dir, pool_dir = config.output_dir(cfg, name) / "e5", config.output_dir(cfg, pool_name) / "e5"
    out = config.results_dir(cfg, pool_name, "e5")
    analyze_dir = out / "analyze"
    report = checks.CheckReport()

    # decisions, run logs and inputs
    try:
        audit = audit_config(cfg)
        report.add("all E5 decisions (L-Q) are set and implemented (audit:)", True, "")
    except SystemExit as e:
        report.add("all E5 decisions (L-Q) are set and implemented (audit:)", False, str(e))
        audit = cfg["audit"]
    logs = {"prepare": (real_dir / "prepare.run_info.json", LOG_KEYS),
            "rehearse": (real_dir / "rehearse" / "positive" / "rehearse.run_info.json", LOG_KEYS),
            "score": (pool_dir / "score.run_info.json", LOG_KEYS),
            "analyze": (analyze_dir / "run_info.json", LOG_KEYS)}
    checks.check_run_logs(report, cfg, logs)
    info = {stage: json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            for stage, (path, _) in logs.items()}
    for stage, expected in INPUTS.items():
        inputs = info[stage].get("inputs")
        report.add(f"'{stage}' read only {', '.join(expected)}" + (" (S never read)" if "S" not in expected else ""),
                   inputs == expected, str(inputs))

    # the lock
    e3, e3_hash = load_frozen(cfg, name)
    e5, e5_hash = load_frozen(cfg, name, "e5")
    marker_path = score_marker(cfg, pool_name)
    marker = json.loads(marker_path.read_text(encoding="utf-8")) if marker_path.exists() else {}
    now = {"audit_sha256": audit_hash(cfg), "frozen_sha256": e3_hash, "e5_sha256": e5_hash}
    report.add("S scored to completion (score.complete.json)", bool(marker), str(marker_path))
    for key, value in now.items():
        recorded = {s: info[s].get(key) for s in ("score", "analyze")}
        report.add(f"{key.removesuffix('_sha256')} unchanged since S was scored (marker, score and analyze logs)",
                   marker.get(key) == value and all(v == value for v in recorded.values()),
                   f"now {value[:12]}, marker {str(marker.get(key))[:12]}")
    report.add("E5 file made with the current audit: section and frozen E3 file",
               e5.get("audit_sha256") == now["audit_sha256"] and e5.get("e3_frozen_sha256") == e3_hash, "")
    log = read_table(pool_dir / "score_log.csv") if (pool_dir / "score_log.csv").exists() else pd.DataFrame()
    variants = ["main", *audit["sensitivity"]]
    queries = read_table(pool_dir / "queries.csv")
    same_rows = all(read_table(pool_dir / f"classes_{v}.csv")["key"].tolist() == queries["key"].tolist()
                    for v in variants)
    report.add("one score run with every variant; each class table has one row per scored mask, in order",
               log.get("variant", pd.Series(dtype=str)).tolist() == variants and same_rows
               and len(queries) == marker.get("queries"), f"{len(queries)} masks, variants {log.get('variant', pd.Series()).tolist()}")

    # prepare, reproduced
    thresholds = read_table(config.results_dir(cfg, name, "e5") / "prepare" / "thresholds.csv")
    in_file = [e5["variants"][r.variant]["thresholds"][r.detector][r.threshold][r.stratum] for r in thresholds.itertuples()]
    report.add("sensitivity thresholds equal E3 (sensitivity.csv; drop-only: frozen) and the E5 file",
               bool(thresholds["equal"].astype(bool).all()) and np.allclose(in_file, thresholds["value"], atol=TOL, rtol=0),
               f"{len(thresholds)} values")
    e3_dir = config.output_dir(cfg, name) / "e3"
    e1_hcal = read_table(e3_dir / "e1_hcal.csv")
    star = tau_d_star(e1_hcal, e3)
    report.add("tau_D* (decision P) reproduced from the H-cal E1 table",
               all(abs(star["tau_D_star"][s] - e5["h2"]["tau_D_star"][s]) <= TOL for s in calibration.STRATA),
               str({s: round(v, 4) for s, v in star["tau_D_star"].items()}))
    e2_hcal = lesionmatch.load_tables(e3_dir, "hcal", float(e3["detectors"]["lesion"]["candidate_min_s"]))
    hcal = load_set(cfg, name, "H-cal")
    e4_dir = config.output_dir(cfg, name) / "e4"
    e1_max, e2_max, _ = heldout_maxima(cfg, name, e1_hcal, read_table(e4_dir / "e1_heval.csv"), e2_hcal,
                                       dict(zip(hcal.keys, hcal.patients)), read_table(e4_dir / "heval_classes.csv"),
                                       e5["near_exact"]["size_bins_px"])
    report.add("held-out maxima (decision Q) reproduced", e1_max == e5["near_exact"]["e1_maxima"]
               and e2_max == e5["near_exact"]["e2_C_maxima"], f"{len(e1_max)} E1 cells, E2 {e2_max}")
    drop = e5["drop_only_labels"]
    report.add("drop-only: held-out label maps unchanged at gap 0 (frozen thresholds apply)",
               drop["H-cal"]["changed_at_gap_0"] == 0 and drop["H-eval"]["changed_at_gap_0"] == 0,
               ", ".join(f"{s} {v['changed_at_gap_0']}" for s, v in drop.items()))

    # analyze, reproduced from the raw tables with the same seed
    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    s = load_scored("S", pool_dir, list(audit["sensitivity"]))
    tables, reading = analyze_scored(s, heldout(cfg, name, heval, list(audit["sensitivity"])),
                                     lesionmatch.load_tables(pool_dir, "main", float(e3["detectors"]["lesion"]["candidate_min_s"])),
                                     informative_areas(t, int(e3["informative_bound_px"])), e3, e5, audit,
                                     int(cfg["calibration"]["bootstrap_draws"]), np.random.default_rng(cfg["seed"]))
    differ = checks.differing_tables(tables, analyze_dir, TOL)
    report.add("every analyze table reproduced from the raw tables (same seed)", not differ,
               ", ".join(differ) if differ else f"{len(tables)} tables")
    report.add("outcome reading reproduced", reading == info["analyze"].get("reading"), reading["reading"])
    near = tables["near_exact"]
    heval_strong = near.loc[near["set"] == "H-eval", ["near_exact", "beyond_heldout_e2"]]
    report.add("no H-eval slice is near-exact or beyond the held-out maximum (by construction)",
               len(heval_strong) == 1 and bool((heval_strong == 0).all().all()), "")

    # the unit tests
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = f"E5 check: pool '{pool_name}' against '{name}', seed {cfg['seed']}\n\n{report.to_text()}\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check"}, filename="check.run_info.json")
    print(f"\n{text}\nWrote {out / 'check_report.txt'}")
    if not report.all_passed:
        raise SystemExit(1)
