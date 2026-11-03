"""E3 building blocks: chance-match rates of held-out lesions by size and the informative-lesion
bound (`docs/experiment-plan.md`, E3 step 1; `docs/e1-e3-plan.md`, decision G).

A held-out (H-cal) lesion was never trained on, so its best match against the training lesions
is chance. The chance-match rate of an area bin is the share of its lesions whose best lesion
score reaches tau_L. The bound is the smallest bin edge from which the rate stays below a cut
(5% or 10%) for every larger bin.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


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
