"""E3 building blocks: chance-match rates of held-out lesions by size and the informative-lesion
bound (`docs/experiment-plan.md`, E3 step 1; `docs/e1-e3-plan.md`, decision G), and percentile
thresholds with bootstrap-by-patient intervals (E3 step 2; decision H). For E4 and E5: the frozen
threshold file with its hash, its detector settings, both detectors run with them, and the classes
the frozen thresholds give the detectors' results.

A held-out (H-cal) lesion was never trained on, so its best match against the training lesions
is chance. The chance-match rate of an area bin is the share of its lesions whose best lesion
score reaches tau_L. The bound is the smallest bin edge from which the rate stays below a cut
(5% or 10%) for every larger bin. Thresholds are percentiles of H-cal slice scores; slices of
one patient are near-duplicates, so their intervals resample patients, not slices.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from memtk import config, lesionmatch, wholemask
from memtk.ids import parse_slice_key
from memtk.lesionmatch import LesionMatch
from memtk.sets import SliceSet
from memtk.stages import require

# The stage that writes each experiment's frozen file
FROZEN_BY = {"e3": "e3_calibrate.py freeze", "e5": "e5_audit.py prepare"}
STRATA = ("single", "multi")
# The thresholds of each detector
THRESHOLDS = {"whole_mask": ("tau_D",), "lesion": ("tau_L", "tau_C")}

def area_bins(area: np.ndarray, edges: Sequence[int]) -> pd.Series:
    """The lower edge of the area bin of each lesion (bins [edges[i], edges[i + 1]), the last one
    open); lesions below the first edge get NaN."""
    edges_arr = np.asarray(edges)
    pos = np.searchsorted(edges_arr, np.asarray(area), side="right") - 1
    return pd.Series(np.where(pos >= 0, edges_arr[np.clip(pos, 0, None)], np.nan))


def bin_label(lower: int, edges: Sequence[int]) -> str:
    """A readable label for the bin starting at `lower`, e.g. "10-11 px" or ">= 40 px"."""
    i = list(edges).index(lower)
    return f">= {lower} px" if i == len(edges) - 1 else f"{lower}-{edges[i + 1] - 1} px"


def chance_rates(best_s: np.ndarray, area: np.ndarray, patients: np.ndarray | Sequence[str],
                 edges: Sequence[int], taus: Sequence[float]) -> pd.DataFrame:
    """Per area bin and tau_L: lesions, patients, lesions whose best score reaches tau_L
    (`matched`; NaN scores never do) and their share (`rate`)."""
    table = pd.DataFrame({"lower": area_bins(area, edges), "best_s": np.asarray(best_s, dtype=float),
                          "patient": np.asarray(patients)})
    rows = []
    for lower, d in table.dropna(subset=["lower"]).groupby("lower"):
        for tau in taus:
            matched = int((d["best_s"] >= tau).sum())
            rows.append({"lower": int(lower), "bin": bin_label(int(lower), edges), "tau_L": tau, "lesions": len(d),  # pyright: ignore[reportArgumentType]  # pandas stubs type the group key as any scalar, complex included
                         "patients": d["patient"].nunique(), "matched": matched, "rate": matched / len(d)})
    return pd.DataFrame(rows)


def percentile_threshold(scores: np.ndarray, percentile: float) -> float:
    """The `percentile` of scores, a missing score (no match) counting as 0."""
    return float(np.percentile(np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0), percentile))


def stratum(n_lesions: np.ndarray) -> np.ndarray:
    """"single" for slices with one lesion, "multi" for slices with two or more."""
    return np.where(np.asarray(n_lesions) >= 2, "multi", "single")


def bootstrap_interval(scores: np.ndarray, patients: np.ndarray, percentile: float, draws: int,
                       rng: np.random.Generator, level: float = 0.95) -> tuple[float, float]:
    """Interval of the `percentile` of slice scores when patients (all slices of each) are
    resampled with replacement `draws` times; a missing score counts as 0."""
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    patients = np.asarray(patients)
    groups = [scores[patients == p] for p in np.unique(patients)]
    stats = np.empty(draws)
    for i in range(draws):
        chosen = rng.integers(0, len(groups), size=len(groups))
        stats[i] = np.percentile(np.concatenate([groups[j] for j in chosen]), percentile)
    low, high = np.percentile(stats, [50 * (1 - level), 50 * (1 + level)])
    return float(low), float(high)


def bootstrap_share(flags: np.ndarray, patients: np.ndarray, draws: int, rng: np.random.Generator,
                    level: float = 0.95) -> tuple[float, float]:
    """Interval of the share of flagged slices when patients (all slices of each) are resampled
    with replacement `draws` times; (NaN, NaN) without slices."""
    flags, patients = np.asarray(flags, dtype=bool), np.asarray(patients)
    if not len(flags):
        return np.nan, np.nan
    _, codes = np.unique(patients, return_inverse=True)
    k, n = np.bincount(codes, weights=flags), np.bincount(codes)
    chosen = rng.integers(0, len(n), size=(draws, len(n)))
    shares = k[chosen].sum(axis=1) / n[chosen].sum(axis=1)
    low, high = np.percentile(shares, [50 * (1 - level), 50 * (1 + level)])
    return float(low), float(high)


def read_frozen(path: str | Path) -> dict:
    """The frozen E3 threshold file (written by `e3_calibrate.py freeze`), as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_frozen(cfg: dict, dataset: str, experiment: str = "e3") -> tuple[dict, str]:
    """A frozen file of `dataset` (`config.frozen_file`; E3 thresholds by default) and its SHA-256,
    recorded in the run logs so that a `check` can confirm it did not change between stages.
    Moved from `memtk.e4.common` on 2026-11-15 (E5 needs it)."""
    path = config.frozen_file(cfg, dataset, experiment)
    require(path, FROZEN_BY[experiment])
    return read_frozen(path), hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_threshold(frozen: dict, detector: str, name: str, n_lesions: np.ndarray) -> np.ndarray:
    """Each slice's frozen threshold `name` of `detector` (e.g. "lesion", "tau_C"), chosen by its
    stratum from its lesion count (the count the detector uses)."""
    values = frozen["thresholds"][detector][name]
    return np.where(stratum(n_lesions) == "multi", float(values["multi"]), float(values["single"]))


def frozen_settings(frozen: dict) -> tuple[dict, dict]:
    """Keywords for `wholemask.match` (top_k_dice, top_k_centroid, max_shift) and for
    `lesionmatch.match` (min_area = the informative-lesion bound, sigma_px, prefilter_px,
    candidate_min_s), read from the frozen file: E4 and E5 take their settings only from there."""
    wm, les = frozen["detectors"]["whole_mask"], frozen["detectors"]["lesion"]
    if les["position_frame"] != "pixel":
        raise ValueError(f"Frozen position_frame '{les['position_frame']}' is not implemented (only 'pixel')")
    return ({"top_k_dice": int(wm["top_k_dice"]), "top_k_centroid": int(wm["top_k_centroid"]),
             "max_shift": int(wm["max_shift_px"])},
            {"min_area": int(frozen["informative_bound_px"]), "sigma_px": float(les["sigma_px"]),
             "prefilter_px": float(les["prefilter_px"]), "candidate_min_s": float(les["candidate_min_s"])})


def run_frozen(labels: np.ndarray, keys: list[str], t: SliceSet,
               frozen: dict) -> tuple[pd.DataFrame, LesionMatch, pd.DataFrame]:
    """E1 (masks with a lesion only) and E2 of a stack of label maps against T with the frozen
    settings; the E1 table with its frozen classes, the E2 results, and the E2 classes. Moved
    from `memtk.e4.validate.run_detectors` on 2026-11-15 (E5 needs it)."""
    wm, les = frozen_settings(frozen)
    n_lesions = labels.reshape(len(labels), -1).max(axis=1)
    scorable = n_lesions > 0
    e1 = wholemask.match(labels[scorable], t.labels, t.patients, [k for k, s in zip(keys, scorable) if s],
                         t.keys, **wm)
    e2 = lesionmatch.match(labels, keys, t.labels, t.keys, **les)
    return e1_classes(e1, n_lesions[scorable], frozen), e2, e2_classes(e2, frozen)


def e1_classes(e1: pd.DataFrame, n_lesions: np.ndarray, frozen: dict) -> pd.DataFrame:
    """The E1 table (`wholemask.match`) with each query's `n_lesions` (lesions >= the minimum
    area, in the table's row order), `stratum`, frozen `tau_D` and `copy` (score >= tau_D)."""
    out = e1.copy()
    out["n_lesions"] = np.asarray(n_lesions)
    out["stratum"] = stratum(out["n_lesions"].to_numpy())
    out["tau_D"] = frozen_threshold(frozen, "whole_mask", "tau_D", out["n_lesions"].to_numpy())
    out["copy"] = out["score"] >= out["tau_D"]
    return out


def e2_classes(e2: LesionMatch, frozen: dict) -> pd.DataFrame:
    """`LesionMatch.summary` under the frozen tau_L and tau_C of each query's stratum: one row per
    query slice with an informative lesion, with its `stratum` (by informative lesions), `tau_L`,
    `tau_C`, and the patients of the nearest neighbour and of the strongest-pair slice
    (`nn_patient`, `strongest_patient`; "" without a candidate)."""
    n = e2.queries[e2.queries["n_informative"] > 0].set_index("query")["n_informative"]
    tau_L = pd.Series(frozen_threshold(frozen, "lesion", "tau_L", n.to_numpy()), index=n.index)
    tau_C = pd.Series(frozen_threshold(frozen, "lesion", "tau_C", n.to_numpy()), index=n.index)
    out = e2.summary(tau_L, tau_C)
    out["stratum"] = stratum(out["n_informative"].to_numpy())
    out["tau_L"], out["tau_C"] = out["query"].map(tau_L), out["query"].map(tau_C)
    for col, key in (("nn_patient", "nn"), ("strongest_patient", "strongest_train")):
        out[col] = [parse_slice_key(k)[0] if k else "" for k in out[key]]
    return out


# --- thresholds from H-cal --------------------------------------------------------------------
# Moved from `memtk.e3.calibrate` and `memtk.e3.freeze` on 2026-11-15: E3 sets the thresholds with
# them and E5 `prepare` re-derives the sensitivity thresholds the same way.

def lesion_slices(e2: LesionMatch, patients: dict) -> pd.DataFrame:
    """One row per H-cal slice with an informative lesion: patient, stratum (by informative
    lesions), the strongest single lesion pair over all candidate slices and the configuration
    score C of the nearest neighbour (both 0 without a candidate; C does not depend on tau_L)."""
    q = e2.queries[e2.queries["n_informative"] > 0].copy()
    strongest = e2.slice_pairs.groupby("query")["best_pair_s"].max()
    q["strongest_s"] = q["query"].map(strongest).fillna(0.0)
    q["C"] = q["query"].map(e2.slice_scores(e2.candidate_min_s).groupby("query")["C"].max()).fillna(0.0)
    q["patient_id"] = q["query"].map(patients)
    q["stratum"] = stratum(q["n_informative"].to_numpy())
    return q.set_index("query")


def threshold_rows(detector: str, name: str, scores: pd.DataFrame, pooled_col: str, stratum_col: str,
                   percentile: float, draws: int, rng: np.random.Generator) -> list[dict]:
    """Rows for one threshold: pooled (from `pooled_col`) and per stratum (from `stratum_col`),
    each with its bootstrap interval (skipped when draws is 0), slices, patients and the share of
    the slices that the pooled threshold flags (score >= threshold)."""
    pooled = percentile_threshold(scores[pooled_col].to_numpy(), percentile)
    rows = []
    for st, d, col in [("all", scores, pooled_col), *((s, scores[scores["stratum"] == s], stratum_col)
                                                      for s in STRATA)]:
        value = percentile_threshold(d[col].to_numpy(), percentile) if len(d) else np.nan
        low, high = (bootstrap_interval(d[col].to_numpy(), d["patient_id"].to_numpy(), percentile, draws, rng)
                     if draws and len(d) else (np.nan, np.nan))
        rows.append({"detector": detector, "threshold": name, "stratum": st, "value": value,
                     "ci_low": low, "ci_high": high, "slices": len(d), "patients": d["patient_id"].nunique(),
                     "flagged_at_pooled": float((d[pooled_col].fillna(0) >= pooled).mean()) if len(d) else np.nan})
    return rows


def thresholds(e1: pd.DataFrame, e2: LesionMatch, patients: dict, percentile: float, draws: int,
               rng: np.random.Generator) -> pd.DataFrame:
    """tau_D, tau_L and tau_C at `percentile`, pooled and per stratum, with intervals (skipped
    when draws is 0). `e1` needs `score`, `stratum` and `patient_id`."""
    les = lesion_slices(e2, patients)
    rows = threshold_rows("whole_mask", "tau_D", e1, "score", "score", percentile, draws, rng)
    rows += threshold_rows("lesion", "tau_L", les, "strongest_s", "strongest_s", percentile, draws, rng)
    rows += threshold_rows("lesion", "tau_C", les, "C", "C", percentile, draws, rng)
    return pd.DataFrame(rows)


def frozen_thresholds(table: pd.DataFrame, strata: dict) -> tuple[dict, dict]:
    """Per detector and threshold, the single and multi values (both the pooled value when the
    detector is pooled, decision H) and their bootstrap intervals, from a `thresholds` table."""
    values, intervals = {}, {}
    for detector, names in THRESHOLDS.items():
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
    (s <= position factor there), every tau_C above 0. `settings`: sigma_px, prefilter_px,
    candidate_min_s."""
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


def chance_rates_at(e1: pd.DataFrame, e2: LesionMatch, patients: dict, values: dict) -> dict:
    """The share of H-cal slices (all chance) that thresholds `values` flag, per stratum: E1
    copies, E2 full and partial copies and either. `e1` needs `score`, `n_lesions`, `stratum`."""
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


def propose_bound(rates: pd.DataFrame, cut: float) -> int | None:
    """The smallest bin edge from which the chance-match rate of every bin at or above it is
    below `cut` (`rates`: one row per bin with `lower` and `rate`); None if even the largest bin
    reaches the cut."""
    ordered = rates.sort_values("lower")
    bound = None
    for lower, rate in zip(ordered["lower"][::-1], ordered["rate"][::-1]):
        if rate >= cut:
            break
        bound = int(lower)
    return bound
