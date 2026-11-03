"""E2: the lesion-level matching detector (`docs/experiment-plan.md`, E2; `docs/e1-e3-plan.md`).

Each query slice is compared with each training slice lesion by lesion. Only informative
lesions (area >= the bound) take part. Two lesions a, b score s = shape x position x size:
shape = IoU of their pixel sets with the rounded centroids aligned (centred crops), position =
exp(-d^2 / 2 sigma^2) with d the centroid distance in pixels (decision F), size = smaller area /
larger area. A slice pair (q, t) pairs its lesions one-to-one (Hungarian, maximising the summed
s) and scores the configuration C = sum of the paired s that reach candidate_min_s / max(lesions in
q, lesions in t): how alike the whole layout is, independent of tau_L (decided 2026-11-03; first
planned with only the pairs reaching tau_L, which made C 0 for most held-out slices). Classes:
full copy when C of the nearest neighbour reaches tau_C, partial when it does not but a lesion
pair over all candidate slices reaches tau_L, novel otherwise.

Two shortcuts give exactly the same result as comparing everything: lesion pairs farther apart
than the prefilter distance are skipped (their position factor is below exp(-8), so s stays
below any tau_L >= candidate_min_s), and C is computed only for candidate slices, those sharing
a lesion pair with s >= candidate_min_s (every other slice has C = 0 for tau_L >= that value).

All candidate pairs are stored; tau_L and tau_C are applied afterwards (`LesionMatch.slice_scores`,
`LesionMatch.summary`; one value each, or one per query slice for stratified thresholds), so
thresholds and sensitivity analyses need no recomputation. Used by E3
(H-cal), E4 (planted copies) and E5 (S); the stages pass the settings from `detectors.lesion`
and `calibration`, converted to pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree

from memtk.ids import lesion_id

# Query slices scored at once: bounds the number of lesion pairs held in memory
BLOCK = 1000
# Lesion pairs whose shape is computed at once: bounds the packed-crop arrays in memory
SHAPE_CHUNK = 200_000


# --- lesion records ---

@dataclass
class Lesions:
    """The informative lesions of a stack of label maps, one entry per lesion, in slice order."""
    keys: list[str]            # slice keys of the stack
    slice: np.ndarray          # per lesion: index into `keys`
    number: np.ndarray         # per lesion: its L-number in the slice (1 = largest)
    area: np.ndarray           # per lesion: pixels (attached strays included)
    centroid: np.ndarray       # (n, 2): row, col in pixels
    offsets: list[np.ndarray]  # per lesion: (area, 2) pixel positions relative to the rounded centroid
    n_informative: np.ndarray  # per slice: informative lesions

    def ids(self, idx) -> list[str]:
        """Lesion IDs (`<slice key>/L<n>`) of the lesions at positions `idx`."""
        return [lesion_id(self.keys[s], int(n)) for s, n in zip(self.slice[idx], self.number[idx])]

    def half_size(self) -> int:
        """The largest distance (rows or columns) of a lesion pixel from its rounded centroid."""
        return max((int(np.abs(o).max()) for o in self.offsets), default=0)


def lesion_records(label_maps: np.ndarray, keys: Sequence[str], min_area: int) -> Lesions:
    """The lesions of each label map (N, H, W) with an area of at least `min_area` (the
    informative-lesion bound); smaller lesions are left out, as if absent."""
    slice_idx, number, area, centroid, offsets = [], [], [], [], []
    n_informative = np.zeros(len(label_maps), dtype=int)
    for i, lab in enumerate(label_maps):
        rows, cols = np.nonzero(lab)
        values = lab[rows, cols]
        for n in np.unique(values):
            sel = values == n
            if sel.sum() < min_area:
                continue
            r, c = rows[sel], cols[sel]
            centre = np.array([r.mean(), c.mean()])
            slice_idx.append(i)
            number.append(int(n))
            area.append(int(sel.sum()))
            centroid.append(centre)
            offsets.append(np.stack([r, c], axis=1) - np.rint(centre).astype(int))
            n_informative[i] += 1
    return Lesions(list(keys), np.array(slice_idx, dtype=int), np.array(number, dtype=int),
                   np.array(area, dtype=int), np.array(centroid).reshape(-1, 2), offsets, n_informative)


def packed_crops(lesions: Lesions, half: int) -> np.ndarray:
    """Each lesion as a (2 * half + 1)^2 crop centred on its rounded centroid, bit-packed per
    lesion (one row of bytes per lesion), for fast intersections."""
    side = 2 * half + 1
    crops = np.zeros((len(lesions.area), side * side), dtype=bool)
    for k, off in enumerate(lesions.offsets):
        crops[k, (off[:, 0] + half) * side + off[:, 1] + half] = True
    return np.packbits(crops, axis=1)


# --- lesion scores ---

def position_factor(distance: np.ndarray, sigma_px: float) -> np.ndarray:
    """exp(-d^2 / 2 sigma^2) for centroid distances d in pixels."""
    return np.exp(-np.square(distance) / (2 * sigma_px ** 2))


def size_factor(area_a: np.ndarray, area_b: np.ndarray) -> np.ndarray:
    """Smaller area / larger area."""
    return np.minimum(area_a, area_b) / np.maximum(area_a, area_b)


def shape_factor(crops_a: np.ndarray, crops_b: np.ndarray, area_a: np.ndarray, area_b: np.ndarray) -> np.ndarray:
    """IoU of paired packed crops (rows of `crops_a` with the same rows of `crops_b`)."""
    out = np.empty(len(crops_a))
    for lo in range(0, len(crops_a), SHAPE_CHUNK):
        sl = slice(lo, lo + SHAPE_CHUNK)
        inter = np.bitwise_count(crops_a[sl] & crops_b[sl]).sum(axis=1)
        out[sl] = inter / (area_a[sl] + area_b[sl] - inter)
    return out


def assign_pairs(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairing of the rows (query lesions) and columns (training lesions) of a score
    matrix that maximises the summed score (Hungarian); pairs with score 0 are dropped."""
    rows, cols = linear_sum_assignment(s, maximize=True)
    keep = s[rows, cols] > 0
    return rows[keep], cols[keep]


# --- the detector ---

@dataclass
class LesionMatch:
    """E2 results for a set of query slices against the training slices, before thresholds.

    `queries`: one row per query slice (`query`, `n_informative`; slices without an informative
    lesion have 0 and take no further part). `lesions`: one row per informative query lesion
    with its best single match over all training lesions (`best_s` and factors; NaN when no
    pair reaches `candidate_min_s`). `slice_pairs`: one row per candidate slice pair (`query`,
    `train`, `train_idx` = position in the reference stack, `n_query`, `n_train` informative
    lesions, `n_matched` Hungarian pairs with s > 0, `best_pair_s` = strongest lesion pair of
    the two slices). `lesion_pairs`: the Hungarian pairs of every candidate slice pair (both
    lesion IDs, `distance`, `shape`, `position`, `size`, `s`)."""
    queries: pd.DataFrame
    lesions: pd.DataFrame
    slice_pairs: pd.DataFrame
    lesion_pairs: pd.DataFrame
    candidate_min_s: float

    def slice_scores(self, tau_L: float | pd.Series) -> pd.DataFrame:
        """`slice_pairs` with the configuration score C (paired s reaching candidate_min_s, summed,
        / max(n_query, n_train); independent of tau_L), `n_kept` (paired lesions with s >= tau_L)
        and `all_copied` (every informative lesion of both slices kept). `tau_L` is one value or a
        Series by query slice (stratified thresholds). Exact only for tau_L >= candidate_min_s,
        which is enforced."""
        lowest = float(tau_L.min()) if isinstance(tau_L, pd.Series) else float(tau_L)
        if lowest < self.candidate_min_s:
            raise ValueError(f"tau_L {lowest} is below candidate_min_s {self.candidate_min_s}: the "
                             f"candidate shortcut would no longer be exact")
        pairs = self.lesion_pairs
        threshold = pairs["query"].map(tau_L) if isinstance(tau_L, pd.Series) else tau_L
        paired = pairs[pairs["s"] >= self.candidate_min_s].groupby(["query", "train"])["s"].sum().rename("paired_s")
        n_kept = pairs[pairs["s"] >= threshold].groupby(["query", "train"]).size().rename("n_kept")
        out = self.slice_pairs.join(paired, on=["query", "train"]).join(n_kept, on=["query", "train"])
        out[["paired_s", "n_kept"]] = out[["paired_s", "n_kept"]].fillna(0)
        out["n_kept"] = out["n_kept"].astype(int)
        out["C"] = out["paired_s"] / np.maximum(out["n_query"], out["n_train"])
        out["all_copied"] = (out["n_kept"] == out["n_query"]) & (out["n_kept"] == out["n_train"])
        return out.drop(columns="paired_s")

    def summary(self, tau_L: float | pd.Series, tau_C: float | pd.Series) -> pd.DataFrame:
        """One row per query slice with an informative lesion: nearest neighbour `nn` (highest C;
        ties: higher `best_pair_s`, then the first training slice) with its `C`, `n_kept` and
        `all_copied`; the strongest single lesion pair over all candidate slices (`strongest_s`,
        `strongest_train`); and the class: `full` if C(q, NN) >= tau_C, `partial` if not but
        `strongest_s` >= tau_L, `novel` otherwise. Each threshold is one value or a Series by
        query slice; tau_C must be above 0 (every slice has C >= 0)."""
        lowest_c = float(tau_C.min()) if isinstance(tau_C, pd.Series) else float(tau_C)
        if lowest_c <= 0:
            raise ValueError(f"tau_C {lowest_c} must be above 0: C >= 0 would make every slice a full copy")
        scored = self.slice_scores(tau_L).sort_values(
            ["query", "C", "best_pair_s", "train_idx"], ascending=[True, False, False, True])
        nn = scored.drop_duplicates("query").set_index("query")
        out = self.queries[self.queries["n_informative"] > 0][["query", "n_informative"]].copy()
        out["nn"] = out["query"].map(nn["train"]).fillna("")
        out["C"] = out["query"].map(nn["C"]).fillna(0.0)
        out["n_kept"] = out["query"].map(nn["n_kept"]).fillna(0).astype(int)
        out["all_copied"] = out["query"].map(nn["all_copied"]).eq(True)  # no candidate maps to NaN: False
        out["strongest_s"] = out["query"].map(self.slice_pairs.groupby("query")["best_pair_s"].max())
        strongest = (self.slice_pairs.sort_values(["query", "best_pair_s", "train_idx"], ascending=[True, False, True])
                     .drop_duplicates("query").set_index("query"))
        out["strongest_train"] = out["query"].map(strongest["train"]).fillna("")
        c_cut = out["query"].map(tau_C) if isinstance(tau_C, pd.Series) else tau_C
        l_cut = out["query"].map(tau_L) if isinstance(tau_L, pd.Series) else tau_L
        out["class"] = np.where(out["C"] >= c_cut, "full",
                                np.where(out["strongest_s"].fillna(0.0) >= l_cut, "partial", "novel"))
        return out.reset_index(drop=True)


def match(query_labels: np.ndarray, query_keys: Sequence[str], reference_labels: np.ndarray,
          reference_keys: Sequence[str], min_area: int, sigma_px: float, prefilter_px: float,
          candidate_min_s: float) -> LesionMatch:
    """E2 for a stack of query label maps against the reference (training) label maps: every
    lesion pair within `prefilter_px`, candidate slices, Hungarian pairs. Thresholds are applied
    afterwards with `LesionMatch.slice_scores` / `summary`."""
    q, ref = lesion_records(query_labels, query_keys, min_area), lesion_records(reference_labels, reference_keys, min_area)
    half = max(q.half_size(), ref.half_size())
    q_crops, ref_crops = packed_crops(q, half), packed_crops(ref, half)
    n_ref_slices = len(reference_keys)
    lesion_rows, slice_rows, pair_rows = [], [], []
    tree = KDTree(ref.centroid) if len(ref.area) else None

    for start in range(0, len(query_keys), BLOCK):
        in_block = np.flatnonzero((q.slice >= start) & (q.slice < start + BLOCK))
        if tree is None or not len(in_block):
            continue
        neighbours = tree.query_ball_point(q.centroid[in_block], r=prefilter_px)
        qi = np.repeat(in_block, [len(n) for n in neighbours])
        ri = np.concatenate([np.asarray(n, dtype=int) for n in neighbours])
        distance = np.hypot(*(q.centroid[qi] - ref.centroid[ri]).T)
        position = position_factor(distance, sigma_px)
        size = size_factor(q.area[qi], ref.area[ri])
        shape = np.full(len(qi), np.nan)
        # s <= position x size, so the shape is only needed where that product reaches candidate_min_s ...
        need = position * size >= candidate_min_s
        shape[need] = shape_factor(q_crops[qi[need]], ref_crops[ri[need]], q.area[qi[need]], ref.area[ri[need]])
        s = position * size * shape
        code = q.slice[qi] * n_ref_slices + ref.slice[ri]
        candidate_codes = np.unique(code[s >= candidate_min_s])
        in_candidate = np.isin(code, candidate_codes)
        # ... and for every pair inside a candidate slice pair, where it decides the Hungarian pairing
        todo = in_candidate & ~need
        shape[todo] = shape_factor(q_crops[qi[todo]], ref_crops[ri[todo]], q.area[qi[todo]], ref.area[ri[todo]])
        s = position * size * shape

        lesion_rows.append(_best_per_lesion(q, ref, in_block, qi, ri, s, shape, position, size, candidate_min_s))
        sel = np.flatnonzero(in_candidate)
        slice_part, pair_part = _slice_pairs(q, ref, n_ref_slices, qi[sel], ri[sel], code[sel], s[sel],
                                             shape[sel], position[sel], size[sel], distance[sel])
        slice_rows.append(slice_part)
        pair_rows.append(pair_part)

    queries = pd.DataFrame({"query": list(query_keys), "n_informative": q.n_informative})
    return LesionMatch(queries, _concat(lesion_rows, LESION_COLUMNS), _concat(slice_rows, SLICE_COLUMNS),
                       _concat(pair_rows, PAIR_COLUMNS), candidate_min_s)


# --- tables ---

LESION_COLUMNS = ["query", "query_lesion", "area", "best_train_lesion", "best_s", "best_shape",
                  "best_position", "best_size"]
SLICE_COLUMNS = ["query", "train", "train_idx", "n_query", "n_train", "n_matched", "best_pair_s"]
PAIR_COLUMNS = ["query", "train", "query_lesion", "train_lesion", "distance", "shape", "position", "size", "s"]


def _concat(parts: list[pd.DataFrame], columns: list[str]) -> pd.DataFrame:
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=columns)


def _best_per_lesion(q: Lesions, ref: Lesions, in_block, qi, ri, s, shape, position, size,
                     candidate_min_s: float) -> pd.DataFrame:
    """Per query lesion of the block: its best-scoring training lesion among the pairs reaching
    `candidate_min_s` (ties: the first); NaN when there is none."""
    ok = s >= candidate_min_s  # NaN (shape not computed: s below candidate_min_s) compares False
    order = np.lexsort((ri[ok], -s[ok], qi[ok]))
    first = np.unique(qi[ok][order], return_index=True)[1]
    best = np.flatnonzero(ok)[order[first]]
    table = pd.DataFrame({"query": [q.keys[i] for i in q.slice[in_block]], "query_lesion": q.ids(in_block),
                          "area": q.area[in_block]}, index=in_block)
    found = pd.DataFrame({"best_train_lesion": ref.ids(ri[best]), "best_s": s[best], "best_shape": shape[best],
                          "best_position": position[best], "best_size": size[best]}, index=qi[best])
    return table.join(found).reset_index(drop=True)


def _slice_pairs(q: Lesions, ref: Lesions, n_ref_slices: int, qi, ri, code, s, shape, position, size,
                 distance) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hungarian pairs and slice-pair rows for the lesion pairs of the candidate slice pairs."""
    if not len(code):
        return pd.DataFrame(columns=SLICE_COLUMNS), pd.DataFrame(columns=PAIR_COLUMNS)
    pairs = pd.DataFrame({"code": code, "qi": qi, "ri": ri, "s": s})
    group = pairs.groupby("code", sort=True)
    # One lesion on a side (most slice pairs): the Hungarian pairing is simply the best pair
    simple = (group["qi"].transform("nunique") == 1) | (group["ri"].transform("nunique") == 1)
    best = (pairs[simple].sort_values(["code", "s", "qi", "ri"], ascending=[True, False, True, True])
            .drop_duplicates("code"))
    matched = best.index[best["s"] > 0].tolist()
    for _, g in pairs[~simple].groupby("code"):
        q_pos, r_pos = np.unique(g["qi"], return_inverse=True)[1], np.unique(g["ri"], return_inverse=True)[1]
        grid = np.zeros((q_pos.max() + 1, r_pos.max() + 1))
        grid[q_pos, r_pos] = g["s"]
        index = np.full(grid.shape, -1)
        index[q_pos, r_pos] = g.index
        rows, cols = assign_pairs(grid)
        matched.extend(index[rows, cols].tolist())
    m = np.sort(np.array(matched, dtype=int))

    summary = group.agg(best_pair_s=("s", "max"))
    summary["n_matched"] = pd.Series(code[m]).value_counts().reindex(summary.index, fill_value=0)
    codes = summary.index.to_numpy()
    q_slice, r_slice = codes // n_ref_slices, codes % n_ref_slices
    slices = pd.DataFrame({
        "query": [q.keys[i] for i in q_slice], "train": [ref.keys[i] for i in r_slice], "train_idx": r_slice,
        "n_query": q.n_informative[q_slice], "n_train": ref.n_informative[r_slice],
        "n_matched": summary["n_matched"].to_numpy(), "best_pair_s": summary["best_pair_s"].to_numpy()})
    pairs = pd.DataFrame({
        "query": [q.keys[i] for i in q.slice[qi[m]]], "train": [ref.keys[i] for i in ref.slice[ri[m]]],
        "query_lesion": q.ids(qi[m]), "train_lesion": ref.ids(ri[m]), "distance": distance[m],
        "shape": shape[m], "position": position[m], "size": size[m], "s": s[m]})
    return slices, pairs
