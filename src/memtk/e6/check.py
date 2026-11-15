"""E6 stage `check` (`docs/e6-e7-plan.md`, step 7): consistency checks of all E6 outputs.

Decisions R, U, V, W set; the run logs of `reference`, `rehearse` and `exposure` made with the
current settings, in that order, with S read by `exposure` only; E5's lock unchanged since S was
scored and recorded by `exposure`; every reference table reproduced from T and the stored H-eval
tables; the rehearsal readings as expected (null and placement mocks not "patient copying", the
positive mock "patient copying"); every `exposure` table and the reading reproduced from E5's stored
tables with the same seed; the unit tests. Writes `results/<pool>/e6/check_report.txt`; exit code 1
on any failure.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from memtk import checks, config
from memtk.calibration import frozen_settings, load_frozen
from memtk.e5.common import audit_config, lock_unchanged
from memtk.e6.common import exposure_config, load_heval, load_query, load_reference, reference_dir
from memtk.e6.exposure import analyze_query
from memtk.e6.reference import lesions3d_tables, nn_counts
from memtk.exposure import knn_rarity, patient_features, slice_rarity, slice_shares
from memtk.io import read_table
from memtk.sets import load_set

LOG_KEYS = ["seed", "exposure", "calibration.bootstrap_draws", "lesions.attach_gap"]
# `reference` depends only on these keys of `exposure:` (the placement mock, added later, does not touch it)
REFERENCE_KEYS = ["seed", "exposure.lesions3d", "exposure.rarity", "lesions.attach_gap"]
TOL = 1e-9
# What each rehearsal case must not / must read (decision U)
EXPECTED_READING = {"null": "not patient copying", "positive": "patient copying",
                    "placement": "not patient copying", "placement_compact": "not patient copying"}


# --- stage ------------------------------------------------------------------------------------

def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None) -> None:
    """Consistency checks of all E6 outputs. Writes `results/<pool>/e6/check_report.txt`; exit code
    1 on any failure."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    data_dir = reference_dir(cfg, name)
    rehearse_dir = config.results_dir(cfg, name, "e6") / "rehearse"
    out = config.results_dir(cfg, pool_name, "e6")
    report = checks.CheckReport()

    try:
        exp = exposure_config(cfg)
        report.add("all E6 decisions (R, U, V, W) are set and implemented (exposure:)", True, "")
    except SystemExit as e:
        report.add("all E6 decisions (R, U, V, W) are set and implemented (exposure:)", False, str(e))
        exp = cfg["exposure"]
    logs = {"reference": (data_dir / "reference.run_info.json", REFERENCE_KEYS),
            "rehearse": (rehearse_dir / "positive" / "run_info.json", LOG_KEYS),
            "exposure": (out / "exposure" / "run_info.json", LOG_KEYS)}
    checks.check_run_logs(report, cfg, logs)
    info = {s: json.loads(p.read_text(encoding="utf-8")) if p.exists() else {} for s, (p, _) in logs.items()}
    for stage in ("reference", "rehearse"):
        inputs = info[stage].get("inputs", [])
        report.add(f"'{stage}' never read S", not any(i.startswith("S") for i in inputs), str(inputs))
    report.add("'exposure' read S only from E5's stored tables", "S tables (E5)" in info["exposure"].get("inputs", []),
               str(info["exposure"].get("inputs")))

    try:
        now = lock_unchanged(cfg, name, pool_name)
        report.add("E5's lock unchanged since S was scored, and as recorded by 'exposure'",
                   all(info["exposure"].get(k) == v for k, v in now.items()), "")
    except SystemExit as e:
        report.add("E5's lock unchanged since S was scored, and as recorded by 'exposure'", False, str(e))

    # reference, reproduced
    e3, _ = load_frozen(cfg, name)
    t = load_set(cfg, name, "T")
    lesions = read_table(config.output_dir(cfg, name) / "lesions.csv")
    lesions = lesions[(lesions["set"] == "T") & lesions["key"].isin(t.keys)]
    methods = [exp["lesions3d"]["main"], exp["lesions3d"]["sensitivity"]]
    table3d, lesion_map = lesions3d_tables(cfg, name, t, lesions, methods, int(e3["informative_bound_px"]))
    wm, les = frozen_settings(e3)
    rarity = slice_rarity(t, wm, les)
    main = exp["lesions3d"]["main"]
    feats = patient_features(lesion_map.rename(columns={f"lesion3d_id_{main}": "lesion3d_id"}))
    r = exp["rarity"]
    feats["rarity"] = knn_rarity(feats, r["features"], r["log1p"], int(r["patient_k"]))
    patients = feats.join(slice_shares(t.patients).rename("t_slice_share")).join(
        pd.Series(t.patients).value_counts().rename("t_slices")).rename_axis("patient_id").reset_index()
    heldout_nn = nn_counts(read_table(config.output_dir(cfg, name) / "e4" / "heval_classes.csv"))
    differ = checks.differing_tables({"lesions3d": table3d, "lesion_map": lesion_map, "rarity_slices": rarity,
                                      "patients": patients, "heldout_nn": heldout_nn}, data_dir, TOL)
    report.add("every reference table reproduced (3D lesions, 2D -> 3D map, rarity, patients, H-eval NN)", not differ,
               ", ".join(differ) if differ else "5 tables")

    # rehearsal readings
    for case, expected in EXPECTED_READING.items():
        path = rehearse_dir / case / "run_info.json"
        reading = json.loads(path.read_text(encoding="utf-8")).get("reading", {}) if path.exists() else {}
        cells = {c: v["reading"] for c, v in reading.get("cells", {}).items()}
        ok = (bool(cells) and (all(v == "patient copying" for v in cells.values()) if expected == "patient copying"
                               else all(v != "patient copying" for v in cells.values())))
        report.add(f"rehearsal '{case}' reads {expected} (decision U)", ok, str(cells))

    # exposure, reproduced from E5's stored tables with the same seed
    audit = audit_config(cfg)
    cut = float(e3["detectors"]["lesion"]["candidate_min_s"])
    q = load_query("S", config.output_dir(cfg, pool_name) / "e5", audit["sets"]["main"], cut)
    tables, reading = analyze_query(q, load_heval(cfg, name, cut), load_reference(cfg, name), e3, exp,
                                    int(cfg["calibration"]["bootstrap_draws"]), np.random.default_rng(cfg["seed"]))
    differ = checks.differing_tables(tables, out / "exposure", TOL)
    report.add("every exposure table reproduced from E5's stored tables (same seed)", not differ,
               ", ".join(differ) if differ else f"{len(tables)} tables")
    report.add("the reading reproduced", reading == info["exposure"].get("reading"), reading["reading"])

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = f"E6 check: pool '{pool_name}' against '{name}', seed {cfg['seed']}\n\n{report.to_text()}\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check"}, filename="check.run_info.json")
    print(f"\n{text}\nWrote {out / 'check_report.txt'}")
    if not report.all_passed:
        raise SystemExit(1)
