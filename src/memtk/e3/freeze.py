"""E3 stage `freeze` (needs decision H): writes the frozen threshold file that E4 and E5 read.

The point values are recomputed from the H-cal tables stored by `calibrate` and must equal its
`thresholds.csv` (otherwise the inputs changed and `calibrate` must run again); the intervals come
from that table. Before writing, the conditions that keep the E2 shortcuts exact are confirmed.
Writes `configs/frozen/e3_<dataset>.yaml` (in git) and its run log next to it.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from memtk import config
from memtk.calibration import THRESHOLDS, chance_rates_at, frozen_thresholds, shortcut_problems, thresholds
from memtk.e3.common import lesion_settings, load_hcal_tables
from memtk.io import read_table
from memtk.stages import decided, refuse_overwrite, require


# --- stages -----------------------------------------------------------------------------------

def run_freeze(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Freeze tau_D, tau_L and tau_C (per stratum or pooled, by decision H) with their intervals,
    the class rule, the bound and the detector settings. Needs `calibrate`. Writes
    `configs/frozen/e3_<dataset>.yaml`."""
    name, _ = config.real_dataset(cfg, dataset)
    path = config.frozen_file(cfg, name)
    refuse_overwrite([path], overwrite)
    strata = {d: str(decided(cfg, f"calibration.strata.{d}", "e3_calibrate.py calibrate")) for d in THRESHOLDS}
    table_path = config.results_dir(cfg, name, "e3") / "calibrate" / "thresholds.csv"
    require(table_path, "calibrate")
    percentile = float(cfg["calibration"]["percentile"])
    settings = lesion_settings(cfg)

    e1, e2, patients = load_hcal_tables(cfg, name)
    recomputed = thresholds(e1, e2, patients, percentile, 0, np.random.default_rng(cfg["seed"]))
    stored = read_table(table_path)
    differs = compare_tables(recomputed, stored)
    if differs:
        raise SystemExit(f"Thresholds recomputed from the stored H-cal tables differ from {table_path} "
                         f"({'; '.join(differs[:3])}); run 'e3_calibrate.py calibrate --overwrite' first.")
    frozen_values, intervals = frozen_thresholds(stored, strata)
    problems = shortcut_problems(frozen_values, settings)
    if problems:
        raise SystemExit("Not frozen: " + "; ".join(problems))

    frozen = {
        "dataset": name,
        "frozen": datetime.now().isoformat(timespec="seconds"),
        "calibration_set": {"set": "H-cal", "patients": len(set(patients.values())),
                            "slices_e1": len(e1), "slices_e2": int((e2.queries["n_informative"] > 0).sum())},
        "percentile": percentile,
        "strata": strata,
        "stratum_rule": {"whole_mask": "lesions >= lesions.min_lesion_area: 1 = single, >= 2 = multi",
                         "lesion": "informative lesions (>= informative_bound_px): 1 = single, >= 2 = multi"},
        "thresholds": frozen_values,
        "intervals_95": intervals,
        "class_rule": {"whole_mask": "copy if score >= tau_D",
                       "lesion": "full if C(q, NN) >= tau_C; partial if not full but the strongest lesion pair "
                                 ">= tau_L; novel otherwise"},
        "informative_bound_px": int(decided(cfg, "calibration.informative_bound_px", "e3_calibrate.py bound-review")),
        "min_lesion_area": int(cfg["lesions"]["min_lesion_area"]),
        "detectors": {"whole_mask": dict(cfg["detectors"]["whole_mask"]),
                      "lesion": {**cfg["detectors"]["lesion"], "sigma_px": settings["sigma_px"],
                                 "prefilter_px": settings["prefilter_px"]}},
        "hcal_chance_rates": chance_rates_at(e1, e2, patients, frozen_values),
        "source": {"thresholds": config.repo_relative(table_path),
                   "tables": config.repo_relative(config.output_dir(cfg, name) / "e3")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# Frozen E3 thresholds for '{name}', written by 'e3_calibrate.py freeze'. Do not edit by hand:\n"
              f"# E4 and E5 read this file; changing it means re-running E3 on H-cal (never tuning on S).\n")
    path.write_text(header + yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True), encoding="utf-8")
    config.write_run_info(path.parent, cfg, script=__file__, filename=f"e3_{name}.run_info.json",
                          extra={"stage": "freeze", "inputs": ["T", "H-cal"], "frozen_file": path.name})
    print(f"\n{yaml.safe_dump({'thresholds': frozen_values, 'hcal_chance_rates': frozen['hcal_chance_rates']}, sort_keys=False)}"
          f"Wrote {path}")


# --- frozen values ----------------------------------------------------------------------------

def compare_tables(recomputed: pd.DataFrame, stored: pd.DataFrame, tol: float = 1e-9) -> list[str]:
    """Threshold values that differ between a recomputation and the stored `thresholds.csv`."""
    merged = recomputed.merge(stored, on=["detector", "threshold", "stratum"], suffixes=("_new", "_stored"),
                              how="outer")
    bad = merged[~np.isclose(merged["value_new"], merged["value_stored"], atol=tol, rtol=0)]
    return [f"{r.detector} {r.threshold} {r.stratum}: {r.value_new} vs {r.value_stored}" for r in bad.itertuples()]

# `frozen_thresholds`, `shortcut_problems` and `chance_rates_at` moved to `memtk.calibration` on
# 2026-11-15 (E5 `prepare` uses them for the sensitivity thresholds).
