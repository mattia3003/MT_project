"""E8 building blocks (`docs/e8-plan.md`, decisions AB, AC): image similarity between a query image
(synthetic MRI, held-out or mock BRAVO) and the BRAVO of a training comparison slice, and the matched
control draw. No knowledge of stages or file locations.

- **Alignment (AC):** `best_match` finds the best shifted Dice of a query mask against candidate
  masks (E1's shift search and tie rule, over every candidate given); `align` moves an image, mask or
  label map by that shift with zero fill (E1's convention: pixel (r, c) goes to (r + dr, c + dc);
  pixels moved off the image are lost, unlike `planted.shift`, which refuses that).
- **Normalisation (AC):** `zscore_in_brain` scales a whole image by the mean and SD of its own brain,
  so the background stays one constant value.
- **Regions (AC):** `lesion_region` (informative lesions grown by a few px), `regions` (lesion
  neighbourhood, whole brain and ring, each inside both brains).
- **SSIM (AC):** `ssim_map` (Gaussian-weighted SSIM map, fixed data range) and `region_means`.
- **Controls (AB):** `match_controls` draws k controls per case from the same cell, without
  replacement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

from memtk.lesions import grow
from memtk.wholemask import shift_order, shifted_dice

REGIONS = ("lesion", "brain", "ring")   # decision AC: primary, secondary, sensitivity


# --- alignment --------------------------------------------------------------------------------

def best_match(query: np.ndarray, candidates: np.ndarray, max_shift: int) -> tuple[int, float, int, int]:
    """The best candidate for a query mask by Dice over shifts within ±max_shift: (index, score,
    shift_row, shift_col). Every candidate is searched (E1 searches a shortlist); ties as in E1: the
    higher plain Dice, then the first candidate; among a candidate's equal shifts the smallest move.
    Masks are boolean; the query must hold a pixel."""
    scores = shifted_dice(np.asarray(query, dtype=bool), np.asarray(candidates, dtype=bool), max_shift)
    best_shift = scores.argmax(1)
    best = scores[np.arange(len(candidates)), best_shift]
    plain = scores[:, 0]                       # shift (0, 0) comes first in shift_order
    i = int(np.lexsort((np.arange(len(candidates)), -plain, -best))[0])
    dr, dc = shift_order(max_shift)[best_shift[i]]
    return i, float(best[i]), dr, dc


def align(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """The array moved by (dr, dc) pixels with zero fill: pixel (r, c) goes to (r + dr, c + dc);
    pixels moved off the image are lost (no wrap-around)."""
    out = np.zeros_like(array)
    h, w = array.shape
    src_r, dst_r = slice(max(0, -dr), min(h, h - dr)), slice(max(0, dr), min(h, h + dr))
    src_c, dst_c = slice(max(0, -dc), min(w, w - dc)), slice(max(0, dc), min(w, w + dc))
    out[dst_r, dst_c] = array[src_r, src_c]
    return out


# --- normalisation and regions ----------------------------------------------------------------

def zscore_in_brain(image: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """The whole image as (x - mean) / SD of its brain pixels (float64); the background stays one
    constant value. Refuses an empty or constant brain."""
    values = np.asarray(image, dtype=float)[brain]
    if values.size < 2 or values.std() == 0:
        raise ValueError("Brain is empty or constant; the image cannot be z-scored")
    return (np.asarray(image, dtype=float) - values.mean()) / values.std()


def lesion_region(labels: np.ndarray, bound: int, grow_px: int) -> np.ndarray:
    """The informative lesions of a label map (area >= bound px) grown outward by `grow_px` px
    (8-connected); empty if none is informative."""
    areas = np.bincount(labels.ravel())
    informative = np.flatnonzero(areas >= bound)
    informative = informative[informative > 0]
    return grow(np.isin(labels, informative), grow_px)


def regions(region: np.ndarray, lesions_q: np.ndarray, lesions_t: np.ndarray, brain_q: np.ndarray,
            brain_t: np.ndarray) -> dict[str, np.ndarray]:
    """Decision AC's three regions, all inside both brains (query arrays already aligned): `lesion`
    (the lesion neighbourhood), `brain` (whole brain), `ring` (the neighbourhood without the lesion
    pixels of either mask)."""
    both = brain_q & brain_t
    lesion = region & both
    return {"lesion": lesion, "brain": both, "ring": lesion & ~lesions_q & ~lesions_t}


# --- SSIM -------------------------------------------------------------------------------------

def ssim_map(x: np.ndarray, y: np.ndarray, data_range: float, sigma: float) -> np.ndarray:
    """The local SSIM map of two images (Gaussian weights with this sigma, population covariance:
    the settings of Wang et al. 2004), with a fixed data range."""
    # the stub's return type also covers gradient=True (3 values); full=True alone returns 2
    _, smap = structural_similarity(x, y, data_range=data_range, gaussian_weights=True, sigma=sigma,  # pyright: ignore[reportAssignmentType]
                                    use_sample_covariance=False, full=True)
    return np.asarray(smap)


def region_means(smap: np.ndarray, masks: dict[str, np.ndarray]) -> dict[str, float]:
    """Mean SSIM and pixel count of each region (`ssim_<name>`, `px_<name>`); NaN for an empty
    region."""
    out: dict[str, float] = {}
    for name, m in masks.items():
        n = int(m.sum())
        out[f"ssim_{name}"] = float(smap[m].mean()) if n else np.nan
        out[f"px_{name}"] = n
    return out


# --- matched controls -------------------------------------------------------------------------

def match_controls(cases: pd.DataFrame, pool: pd.DataFrame, k: int, rng: np.random.Generator) -> pd.DataFrame:
    """Decision AB: for each case (columns `key`, `cell`; in the given order) up to k controls from
    the pool rows of the same cell, drawn at random without replacement over all cases. Returns one
    row per pair (`case`, `control`, `cell`); a case short of controls gets what its cell has left."""
    left: dict[object, list[str]] = {c: list(g["key"]) for c, g in pool.groupby("cell", sort=True)}
    rows = []
    for key, cell in zip(cases["key"], cases["cell"]):
        avail = left.get(cell, [])
        take = rng.choice(len(avail), size=min(k, len(avail)), replace=False) if avail else np.array([], dtype=int)
        chosen = [avail[i] for i in sorted(take)]
        left[cell] = [a for i, a in enumerate(avail) if i not in set(take.tolist())]
        rows += [{"case": key, "control": c, "cell": cell} for c in chosen]
    return pd.DataFrame(rows, columns=["case", "control", "cell"])


def size_cell(values: Sequence[float] | np.ndarray | pd.Series, edges: Sequence[float]) -> np.ndarray:
    """Bin index of each value for bins starting at `edges` (lower edges, ascending): 0 below the
    first edge, i for edges[i-1] <= v < edges[i], len(edges) at or above the last."""
    return np.searchsorted(np.asarray(edges, dtype=float), np.asarray(values, dtype=float), side="right")
