"""E1: the whole-mask nearest-neighbour detector (`docs/experiment-plan.md`, E1).

For every query mask, find the most similar training mask by Dice overlap, allowing small shifts:
plain Dice to every training mask, a short list of candidates (the best by plain Dice plus the
nearest by the centroid of the largest lesion, which catches shifted copies with low plain Dice),
then the best Dice over shifts of up to `max_shift` px for each candidate. Masks are the cleaned
masks (`labels > 0`); the centroid is that of lesion L1 (the largest). Used by E3 (H-cal), E4
(planted copies) and E5 (S); the stages pass the settings from `detectors.whole_mask`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

# Queries scored at once: bounds the dense Dice block (queries x training masks) in memory
BLOCK = 512


# --- building blocks ---

def largest_lesion_centroids(labels: np.ndarray) -> np.ndarray:
    """(N, 2) centroid (row, col) of lesion L1 (label 1, the largest) of each label map in a
    stack of shape (N, H, W); NaN for a map without lesions."""
    out = np.full((len(labels), 2), np.nan)
    for i, lab in enumerate(labels):
        rows, cols = np.nonzero(lab == 1)
        if len(rows):
            out[i] = rows.mean(), cols.mean()
    return out


def dice_matrix(queries: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Plain Dice (no shift) of every query mask against every reference mask: (Nq, Nr) float.
    Masks are boolean stacks (N, H, W); intersections are one sparse matrix product."""
    q = sparse.csr_matrix(queries.reshape(len(queries), -1).astype(np.float32))
    r = sparse.csr_matrix(reference.reshape(len(reference), -1).astype(np.float32))
    inter = np.asarray((q @ r.T).todense())
    sizes = queries.reshape(len(queries), -1).sum(1)[:, None] + reference.reshape(len(reference), -1).sum(1)[None, :]
    return 2 * inter / sizes


def shift_order(max_shift: int) -> list[tuple[int, int]]:
    """All shifts (row, col) within ±max_shift, smallest move first (then by row, col), so that
    among shifts with the same score the smallest one is reported."""
    shifts = [(dr, dc) for dr in range(-max_shift, max_shift + 1) for dc in range(-max_shift, max_shift + 1)]
    return sorted(shifts, key=lambda s: (s[0] ** 2 + s[1] ** 2, s[0], s[1]))


def shifted_dice(query: np.ndarray, candidates: np.ndarray, max_shift: int) -> np.ndarray:
    """Dice of the query moved by every shift in `shift_order(max_shift)` against each candidate:
    (n_candidates, n_shifts). Moving by (dr, dc) puts query pixel (r, c) at (r + dr, c + dc);
    pixels moved off the image are lost (zero padding, no wrap-around)."""
    s = max_shift
    rows, cols = np.nonzero(query)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    crop = query[r0:r1, c0:c1]
    padded = np.pad(candidates, ((0, 0), (s, s), (s, s)))
    inter = np.empty((len(candidates), (2 * s + 1) ** 2))
    for k, (dr, dc) in enumerate(shift_order(s)):
        window = padded[:, r0 + dr + s:r1 + dr + s, c0 + dc + s:c1 + dc + s]
        inter[:, k] = (window & crop).sum(axis=(1, 2))
    sizes = query.sum() + candidates.reshape(len(candidates), -1).sum(1)
    return 2 * inter / sizes[:, None]


def candidate_indices(plain: np.ndarray, query_centroid: np.ndarray, reference_centroids: np.ndarray,
                      top_k_dice: int, top_k_centroid: int) -> np.ndarray:
    """Reference indices to shift-search for one query: the `top_k_dice` best by plain Dice plus
    the `top_k_centroid` nearest by the centroid of the largest lesion (ties by index), sorted."""
    by_dice = np.argsort(-plain, kind="stable")[:top_k_dice]
    distance = np.hypot(*(reference_centroids - query_centroid).T)
    by_centroid = np.argsort(distance, kind="stable")[:top_k_centroid]
    return np.union1d(by_dice, by_centroid)


# --- the detector ---

def match(query_labels: np.ndarray, reference_labels: np.ndarray, reference_patients: Sequence[str],
          query_keys: Sequence[str], reference_keys: Sequence[str], top_k_dice: int,
          top_k_centroid: int, max_shift: int) -> pd.DataFrame:
    """E1 for a stack of query label maps against the reference (training) label maps. Masks are
    `labels > 0`; every mask must hold a lesion (callers leave out and report empty ones).

    Returns one row per query: `query`, the nearest neighbour `nn` (reference key) and its
    `nn_patient`, `score` (best Dice over the shifts), `shift_row`, `shift_col` (how far the query
    was moved), `plain_dice` (the NN at no shift), `runner_up_patient` and `runner_up_score` (the
    best other patient: shifted score for candidates, plain Dice otherwise), `margin` (score minus
    runner-up score) and `n_candidates`. Ties: the higher plain Dice, then the first reference."""
    queries, reference = query_labels > 0, reference_labels > 0
    for name, masks in (("query", queries), ("reference", reference)):
        empty = ~masks.reshape(len(masks), -1).any(1)
        if empty.any():
            raise ValueError(f"{int(empty.sum())} {name} mask(s) without a lesion; leave them out first")
    patients = np.asarray(reference_patients)
    patient_names, patient_codes = np.unique(patients, return_inverse=True)
    by_patient = np.argsort(patient_codes, kind="stable")
    patient_starts = np.flatnonzero(np.r_[True, np.diff(patient_codes[by_patient]) > 0])
    q_centroids, r_centroids = largest_lesion_centroids(query_labels), largest_lesion_centroids(reference_labels)
    shifts = shift_order(max_shift)

    rows = []
    for start in range(0, len(queries), BLOCK):
        plain_block = dice_matrix(queries[start:start + BLOCK], reference)
        best_plain = np.maximum.reduceat(plain_block[:, by_patient], patient_starts, axis=1)
        for i, plain in enumerate(plain_block):
            q = start + i
            cand = candidate_indices(plain, q_centroids[q], r_centroids, top_k_dice, top_k_centroid)
            scores = shifted_dice(queries[q], reference[cand], max_shift)
            best_shift = scores.argmax(1)  # first in shift_order: the smallest move among ties
            cand_score = scores[np.arange(len(cand)), best_shift]
            nn_pos = np.lexsort((cand, -plain[cand], -cand_score))[0]
            nn = int(cand[nn_pos])
            patient_best = best_plain[i].copy()
            np.maximum.at(patient_best, patient_codes[cand], cand_score)
            others = patient_best.copy()
            others[patient_codes[nn]] = -np.inf
            runner_up = int(others.argmax())
            runner_up_score = float(others[runner_up]) if len(others) > 1 else 0.0
            rows.append({
                "query": query_keys[q], "nn": reference_keys[nn], "nn_patient": patients[nn],
                "score": float(cand_score[nn_pos]), "shift_row": shifts[best_shift[nn_pos]][0],
                "shift_col": shifts[best_shift[nn_pos]][1], "plain_dice": float(plain[nn]),
                "runner_up_patient": patient_names[runner_up] if len(others) > 1 else "",
                "runner_up_score": runner_up_score, "margin": float(cand_score[nn_pos]) - runner_up_score,
                "n_candidates": len(cand)})
    return pd.DataFrame(rows)
