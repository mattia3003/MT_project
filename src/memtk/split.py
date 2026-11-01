"""Seeded patient split of a real dataset into T, H-cal and H-eval.

All patients of the source training split go to T. The held-out patients are divided between
two sets (H-cal, H-eval) by matched pairs: sort them by number of lesion slices, cut the sorted
list into consecutive pairs, and send one patient of each pair at random to each set. Both sets
then cover the same range of lesion burden. An odd patient out goes to the second set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from memtk.io import list_patients, load_image, patient_files


def count_lesion_slices(mask_path, axis: int) -> int:
    """Number of slices along `axis` with at least one non-zero mask voxel (native resolution)."""
    mask = np.asanyarray(load_image(mask_path).dataobj)
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    return int((mask != 0).any(axis=other_axes).sum())


def paired_split(patient_ids: list[str], counts: list[int], set_names: list[str],
                 rng: np.random.Generator) -> dict[str, str]:
    """Matched-pair random assignment of patients to two sets. Returns {patient_id: set_name}."""
    if len(set_names) != 2:
        raise ValueError(f"Expected two held-out set names, got {set_names}")
    order = sorted(zip(counts, patient_ids))  # ties broken by ID, so the order is deterministic
    assignment = {}
    for i in range(0, len(order) - 1, 2):
        first, second = order[i][1], order[i + 1][1]
        if rng.random() < 0.5:
            first, second = second, first
        assignment[first], assignment[second] = set_names
    if len(order) % 2:
        assignment[order[-1][1]] = set_names[1]
    return assignment


def make_split(dataset_cfg: dict, seed: int, slice_axis: int) -> pd.DataFrame:
    """Build the split table: patient_id, orig_split, set, n_lesion_slices (one row per patient)."""
    rows = []
    for split in (dataset_cfg["train_split"], dataset_cfg["heldout_split"]):
        for pid in list_patients(dataset_cfg, split):
            _, mask_path = patient_files(dataset_cfg, split, pid)
            rows.append({"patient_id": pid, "orig_split": split,
                         "n_lesion_slices": count_lesion_slices(mask_path, slice_axis)})
    df = pd.DataFrame(rows)
    if df["patient_id"].duplicated().any():
        raise ValueError(f"Patient IDs occur in both source splits: "
                         f"{df.loc[df['patient_id'].duplicated(), 'patient_id'].tolist()}")

    heldout = df[df["orig_split"] == dataset_cfg["heldout_split"]]
    assignment = paired_split(heldout["patient_id"].tolist(), heldout["n_lesion_slices"].tolist(),
                              dataset_cfg["heldout_sets"], np.random.default_rng(seed))
    df["set"] = df["patient_id"].map(assignment).fillna("T")
    return df[["patient_id", "orig_split", "set", "n_lesion_slices"]]


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    """Patients and lesion-slice statistics per set."""
    return df.groupby("set", sort=False).agg(
        patients=("n_lesion_slices", "size"),
        lesion_slices=("n_lesion_slices", "sum"),
        median=("n_lesion_slices", "median"),
        min=("n_lesion_slices", "min"),
        max=("n_lesion_slices", "max"),
    )
