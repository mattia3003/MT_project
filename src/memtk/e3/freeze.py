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
from memtk.calibration import frozen_threshold
from memtk.e3.calibrate import STRATA, lesion_slices, thresholds
from memtk.e3.common import lesion_settings, load_hcal_tables
from memtk.io import read_table
from memtk.stages import decided, refuse_overwrite, require

DETECTORS = {"whole_mask": ("tau_D",), "lesion": ("tau_L", "tau_C")}


# --- stages -----------------------------------------------------------------------------------

def run_freeze(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Freeze tau_D, tau_L and tau_C (per stratum or pooled, by decision H) with their intervals,
    the class rule, the bound and the detector settings. Needs `calibrate`. Writes
    `configs/frozen/e3_<dataset>.yaml`."""
    name, _ = config.real_dataset(cfg, dataset)
    path = config.frozen_file(cfg, name)
    refuse_overwrite([path], overwrite)
    strata = {d: str(decided(cfg, f"calibration.strata.{d}", "e3_calibrate.py calibrate")) for d in DETECTORS}
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
        "source": {"thresholds": repo_relative(table_path), "tables": repo_relative(config.output_dir(cfg, name) / "e3")},
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

def repo_relative(path) -> str:
    """A path relative to the repository root when it lies inside it, else as given."""
    try:
        return path.relative_to(config.REPO_ROOT).as_posix()
    except ValueError:
        return str(path)

def compare_tables(recomputed: pd.DataFrame, stored: pd.DataFrame, tol: float = 1e-9) -> list[str]:
    """Threshold values that differ between a recomputation and the stored `thresholds.csv`."""
    merged = recomputed.merge(stored, on=["detector", "threshold", "stratum"], suffixes=("_new", "_stored"),
                              how="outer")
    bad = merged[~np.isclose(merged["value_new"], merged["value_stored"], atol=tol, rtol=0)]
    return [f"{r.detector} {r.threshold} {r.stratum}: {r.value_new} vs {r.value_stored}" for r in bad.itertuples()]


def frozen_thresholds(table: pd.DataFrame, strata: dict) -> tuple[dict, dict]:
    """Per detector and threshold, the single and multi values (both the pooled value when the
    detector is pooled, decision H) and their bootstrap intervals."""
    values, intervals = {}, {}
    for detector, names in DETECTORS.items():
        values[detector], intervals[detector] = {}, {}
        for name in names:
            rows = table[(table["detector"] == detector) & (table["threshold"] == name)].set_index("stratum")
            source = {s: (s if strata[detector] == "stratified" else "all") for s in STRATA}
            values[detector][name] = {s: float(rows.loc[src, "value"]) for s, src in source.items()}
            intervals[detector][name] = {s: [float(rows.loc[src, "ci_low"]), float(rows.loc[src, "ci_high"])]
                                         for s, src in source.items()}
    return values, intervals


def shortcut_problems(values: dict, settings: dict) -> list[str]:
    """Conditions under which the E2 shortcuts stay exact and the classes are meaningful: every
    tau_L >= the candidate cut and above the largest score possible at the prefilter distance
    (s <= position factor there), every tau_C above 0."""
    problems = []
    tau_L, tau_C = min(values["lesion"]["tau_L"].values()), min(values["lesion"]["tau_C"].values())
    at_prefilter = float(np.exp(-settings["prefilter_px"] ** 2 / (2 * settings["sigma_px"] ** 2)))
    if tau_L < settings["candidate_min_s"]:
        problems.append(f"tau_L {tau_L:.3f} is below the candidate cut {settings['candidate_min_s']}")
    if tau_L <= at_prefilter:
        problems.append(f"tau_L {tau_L:.3f} does not exceed the largest score at the prefilter distance {at_prefilter:.4f}")
    if tau_C <= 0:
        problems.append(f"tau_C {tau_C} is not above 0")
    return problems


def chance_rates_at(e1: pd.DataFrame, e2, patients: dict, values: dict) -> dict:
    """The share of H-cal slices (all chance) that the frozen thresholds flag, per stratum: E1
    copies, E2 full and partial copies and either."""
    rates = {}
    e1_flag = e1["score"] >= frozen_threshold({"thresholds": values}, "whole_mask", "tau_D", e1["n_lesions"].to_numpy())
    les = lesion_slices(e2, patients)
    tau_L = pd.Series(frozen_threshold({"thresholds": values}, "lesion", "tau_L", les["n_informative"].to_numpy()),
                      index=les.index)
    tau_C = pd.Series(frozen_threshold({"thresholds": values}, "lesion", "tau_C", les["n_informative"].to_numpy()),
                      index=les.index)
    classes = e2.summary(tau_L, tau_C).set_index("query")["class"].reindex(les.index)
    for s in STRATA:
        c = classes[les["stratum"] == s]
        rates[s] = {"e1_copy": round(float(e1_flag[e1["stratum"] == s].mean()), 4),
                    "e2_full": round(float((c == "full").mean()), 4),
                    "e2_partial": round(float((c == "partial").mean()), 4),
                    "e2_flagged": round(float((c != "novel").mean()), 4)}
    return rates
