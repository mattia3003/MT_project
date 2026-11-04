"""E3 building blocks: chance-match rates of held-out lesions by size and the informative-lesion
bound (`docs/experiment-plan.md`, E3 step 1; `docs/e1-e3-plan.md`, decision G), and percentile
thresholds with bootstrap-by-patient intervals (E3 step 2; decision H). For E4 and E5: the frozen
threshold file, its detector settings, and the classes it gives the detectors' results.

A held-out (H-cal) lesion was never trained on, so its best match against the training lesions
is chance. The chance-match rate of an area bin is the share of its lesions whose best lesion
score reaches tau_L. The bound is the smallest bin edge from which the rate stays below a cut
(5% or 10%) for every larger bin. Thresholds are percentiles of H-cal slice scores; slices of
one patient are near-duplicates, so their intervals resample patients, not slices.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from memtk.ids import parse_slice_key
from memtk.lesionmatch import LesionMatch


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
