"""E4 stage `check`: consistency checks of all E4 outputs.

Decisions set and outputs made with them (run logs, in order, with only T, H-eval and the planted
set as inputs); the frozen file unchanged since `plant` and `validate` ran and matching the current
settings; the planted set reproduced from the seed, P1 equal to its bases, bases from T and added
lesions from H-eval only; the gate table, the per-mask outcomes and the verdict reproduced from the
stored detector tables; the unit tests. Writes `results/<dataset>/e4/check_report.txt`; exit code
1 on any failure.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from memtk import calibration, checks, config
from memtk.e4.common import gate_status, load_frozen, load_planted
from memtk.e4.plant import planted_lesion_table, planted_table, plant_set
from memtk.e4.validate import fp_table, gate_table, gate_verdict, heval_classes, planted_outcomes
from memtk.ids import parse_lesion_id
from memtk.io import read_table
from memtk.lesionmatch import LesionMatch
from memtk.sets import load_set

DECISION_KEYS = ["validation.bases", "validation.variants", "validation.gate", "validation.caught",
                 "validation.criteria"]
PLANT_KEYS = ["seed", "lesions.attach_gap", "validation.bases", "validation.variants"]
VALIDATE_KEYS = [*PLANT_KEYS, "validation.gate", "validation.caught", "validation.criteria",
                 "validation.accepted_deviation", "calibration.bootstrap_draws"]
CAUGHT = ["e1_caught", "e2_caught", "any_caught", "both_caught"]


# --- stage ------------------------------------------------------------------------------------

def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None) -> None:
    """Consistency checks of all E4 outputs. Writes `results/<dataset>/e4/check_report.txt`;
    exit code 1 on any failure."""
    name, _ = config.real_dataset(cfg, dataset)
    out = config.results_dir(cfg, name, "e4")
    data_dir = config.output_dir(cfg, name) / "e4"
    report = checks.CheckReport()

    # decisions, run logs and inputs
    unset = [k for k in DECISION_KEYS if config.value(cfg, k) is None]
    report.add("all E4 decisions (I, J, K) are set in the config", not unset, ", ".join(unset))
    logs = {"plant": (data_dir / "plant.run_info.json", PLANT_KEYS),
            "validate": (data_dir / "validate.run_info.json", VALIDATE_KEYS)}
    checks.check_run_logs(report, cfg, logs)
    info = {stage: json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            for stage, (path, _) in logs.items()}
    for stage, expected in (("plant", ["T", "H-eval"]), ("validate", ["T", "H-eval", "planted"])):
        inputs = info[stage].get("inputs")
        report.add(f"'{stage}' read only {', '.join(expected)} (S never read)", inputs == expected, str(inputs))

    # the frozen file: unchanged since the stages ran, and matching the current settings
    frozen, frozen_hash = load_frozen(cfg, name)
    used = {stage: i.get("frozen_sha256") for stage, i in info.items()}
    report.add("frozen file unchanged since 'plant' and 'validate' ran (SHA-256)",
               all(h == frozen_hash for h in used.values()), ", ".join(f"{s}: {str(h)[:12]}" for s, h in used.items()))
    same = {"bound": frozen["informative_bound_px"] == cfg["calibration"]["informative_bound_px"],
            "minimum lesion area": frozen["min_lesion_area"] == cfg["lesions"]["min_lesion_area"],
            "whole_mask settings": frozen["detectors"]["whole_mask"] == dict(cfg["detectors"]["whole_mask"]),
            "lesion settings": all(frozen["detectors"]["lesion"][k] == v for k, v in cfg["detectors"]["lesion"].items())}
    report.add("frozen file settings == current config", all(same.values()),
               ", ".join(k for k, ok in same.items() if not ok))

    # the planted set
    planted, labels = load_planted(cfg, name)
    ps = plant_set(cfg, name, frozen)
    again = planted_table(ps)
    same_maps = (again["key"].tolist() == planted["key"].tolist()
                 and all(np.array_equal(p.labels, m) for p, m in zip(ps.planted, labels)))
    report.add("planted set reproduced from the seed (keys and label maps)", same_maps,
               f"{len(planted)} planted masks")
    stored_lesions = read_table(data_dir / "planted_lesions.csv").fillna({"source_lesion": ""})
    recomputed = planted_lesion_table(ps)
    same_sources = (recomputed["lesion_id"].tolist() == stored_lesions["lesion_id"].tolist()
                    and recomputed["source_lesion"].tolist() == stored_lesions["source_lesion"].tolist())
    report.add("provenance reproduced; every planted lesion has a source",
               same_sources and bool((stored_lesions["source_lesion"] != "").all()), f"{len(stored_lesions)} lesions")
    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    t_labels = dict(zip(t.keys, t.labels))
    p1 = (planted["variant"] == "P1").to_numpy()
    p1_bad = [k for k, b, m in zip(planted.loc[p1, "key"], planted.loc[p1, "base_key"], labels[p1])
              if not np.array_equal(m, t_labels[b])]
    report.add("every P1 label map equals its base map", not p1_bad, f"{int(p1.sum())} masks" + (f"; {p1_bad[:3]}" if p1_bad else ""))
    t_patients, h_keys = set(t.patients), set(heval.keys)
    added = planted["added"].dropna()
    donors_ok = all(parse_lesion_id(a)[0] in h_keys for a in added)
    report.add("bases are T slices; added lesions (P6-add) come from H-eval only",
               set(planted["patient_id"]) <= t_patients and planted["base_key"].isin(t.keys).all() and donors_ok,
               f"{planted['patient_id'].nunique()} base patients, {len(added)} added lesions")

    # the gate, reproduced from the stored detector tables
    e2_p, e2_h = stored_match(data_dir, "planted", frozen), stored_match(data_dir, "heval", frozen)
    l1_area = {k: int((t_labels[k] == 1).sum()) for k in planted["base_key"].unique()}
    outcomes = planted_outcomes(planted, read_table(data_dir / "e1_planted.csv"),
                                calibration.e2_classes(e2_p, frozen), l1_area)
    stored_out = read_table(out / "validate" / "planted_outcomes.csv")
    report.add("per-mask outcomes (caught by decision J) reproduced from the stored tables",
               outcomes[CAUGHT].equals(stored_out[CAUGHT].astype(bool)), f"{len(outcomes)} planted masks")
    hc = heval_classes(heval, read_table(data_dir / "e1_heval.csv"), e2_h, calibration.e2_classes(e2_h, frozen))
    criteria = cfg["validation"]["criteria"]
    fp = fp_table(hc, frozen, criteria, 1, np.random.default_rng(cfg["seed"]))
    verdict, gate = gate_verdict(gate_table(outcomes, fp, criteria), config.value(cfg, "validation.accepted_deviation"))
    stored_gate = read_table(out / "validate" / "gate.csv")
    report.add("gate.csv reproduced (values, pass, accepted)", same_gate(gate, stored_gate),
               f"{len(stored_gate)} criteria")
    logged = info["validate"].get("verdict")
    report.add("verdict reproduced and equal to the logged and stored one",
               verdict == logged == gate_status(cfg, name), f"{verdict} (logged {logged}, gate.csv {gate_status(cfg, name)})")

    # the unit tests
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = f"E4 check: dataset '{name}', seed {cfg['seed']}\n\n{report.to_text()}\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check"}, filename="check.run_info.json")
    print(f"\n{text}\nWrote {out / 'check_report.txt'}")
    if not report.all_passed:
        raise SystemExit(1)


# --- checks -----------------------------------------------------------------------------------

def stored_match(data_dir, tag: str, frozen: dict) -> LesionMatch:
    """The E2 results of one set (`planted` or `heval`) rebuilt from the four tables `validate` stored."""
    return LesionMatch(read_table(data_dir / f"e2_{tag}_queries.csv"), read_table(data_dir / f"e2_{tag}_lesions.csv"),
                       pd.read_parquet(data_dir / f"e2_{tag}_slice_pairs.parquet"),
                       pd.read_parquet(data_dir / f"e2_{tag}_lesion_pairs.parquet"),
                       float(frozen["detectors"]["lesion"]["candidate_min_s"]))


def same_gate(gate: pd.DataFrame, stored: pd.DataFrame, tol: float = 1e-9) -> bool:
    """Whether a recomputed gate table equals the stored one: the same criteria, values (within
    `tol`), pass and accepted."""
    if gate["criterion"].tolist() != stored["criterion"].tolist():
        return False
    return bool(np.allclose(gate["value"].astype(float), stored["value"].astype(float), atol=tol, rtol=0)
                and (gate["pass"].astype(bool) == stored["pass"].astype(bool)).all()
                and (gate["accepted"].astype(bool) == stored["accepted"].astype(bool)).all())
