"""The slice sets the detectors compare, as label-map stacks read from the E0b lesion tables: the
real sets (T, H-cal, H-eval; held-out sets go through decision E, `filters.real_sets`; T is never
filtered) and a synthetic pool by mask class (E5, decision L). Also label maps made again from
masks with other attach-or-drop settings (E5's drop-only variant). Used by E3 (T, H-cal), E4 (T,
H-eval, base masks) and E5. Moved from `memtk.e3.common` on 2026-11-04 when E4 needed it
(`docs/e4-plan.md`, step 1); `load_pool` and `relabel` added 2026-11-15 (`docs/e5-plan.md`, step 1).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from memtk import config
from memtk.io import LabelStack, read_table
from memtk.lesions import attach_or_drop
from memtk.stages import decided, require


@dataclass
class SliceSet:
    """The lesion slices of one real set that take part, with their label maps."""
    name: str
    keys: list[str]
    patients: np.ndarray
    labels: np.ndarray         # (N, H, W) label maps; cleaned mask = labels > 0
    left_out: dict[str, int]   # reason -> slices left out


def load_set(cfg: dict, dataset: str, name: str) -> SliceSet:
    """The slices of set `name` (T, H-cal or H-eval) from the E0b lesion tables. Held-out sets go
    through decision E (`filters.real_sets`); T is never filtered (it is the reference). Slices
    without a lesion of at least the minimum area are left out (neither detector can score them)."""
    out_dir = config.output_dir(cfg, dataset)
    slices = read_table(out_dir / "slices.csv")
    slices = slices[slices["set"] == name]
    left_out = {}
    if name != "T":
        passes = decision_e(slices, cfg)
        left_out["removed by decision E"] = int((~passes).sum())
        slices = slices[passes]
    empty = slices["n_lesions"] == 0
    left_out["no lesion of at least the minimum area"] = int(empty.sum())
    slices = slices[~empty]
    stack = LabelStack(out_dir / "labels.npz")
    labels = np.stack([stack[k] for k in slices["key"]])
    return SliceSet(name, slices["key"].tolist(), slices["patient_id"].to_numpy(), labels, left_out)


def load_pool(cfg: dict, pool: str, mask_classes: Sequence[str]) -> SliceSet:
    """The synthetic samples of a pool whose `mask_class` (`qc.csv`) is one of `mask_classes`, with
    their label maps (uint8: the detectors only use `labels > 0` and the lesion numbers, and a
    pool of ~36,000 maps then takes ~0.6 GB). Samples are independent, so each sample ID is its own
    "patient" (cluster). Samples without a lesion of at least the minimum area are left out and
    counted, as in `load_set`."""
    out_dir = config.output_dir(cfg, pool)
    require(out_dir / "qc.csv", "e0b_audit.py qc")
    qc = read_table(out_dir / "qc.csv")
    keep = qc["mask_class"].isin(mask_classes)
    left_out = {f"mask_class {c}": int(n) for c, n in qc.loc[~keep, "mask_class"].value_counts().sort_index().items()}
    qc = qc[keep]
    empty = qc["n_lesions"] == 0
    left_out["no lesion of at least the minimum area"] = int(empty.sum())
    keys = qc.loc[~empty, "key"].tolist()
    stack = LabelStack(out_dir / "labels.npz")
    labels = np.zeros((len(keys), *stack.shape), dtype=np.uint8)
    for i, k in enumerate(keys):
        m = stack[k]
        if m.max() > np.iinfo(np.uint8).max:
            raise ValueError(f"Sample {k} has more than 255 lesions")
        labels[i] = m
    return SliceSet(pool, keys, np.asarray(keys), labels, left_out)


def load_planted(cfg: dict, dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """The E4 planted set written by `e4_validate.py plant`: `planted.csv` and the label maps in its
    row order. Moved from `memtk.e4.common` on 2026-11-15 (E5 `rehearse` and E7 need it)."""
    d = config.output_dir(cfg, dataset) / "e4"
    require(d / "planted_labels.npz", "e4_validate.py plant")
    planted = read_table(d / "planted.csv")
    stack = LabelStack(d / "planted_labels.npz")
    return planted, np.stack([stack[k] for k in planted["key"]])


def relabel(masks: np.ndarray, min_area: int, gap: int) -> np.ndarray:
    """Label maps of a stack of masks as saved (N, H, W) through attach-or-drop with the given
    minimum area and gap (gap 0 = drop-only: every speck dropped, none attached); uint8 when every
    label fits, as `load_pool` gives, else int32."""
    labels = [attach_or_drop(np.asarray(m, dtype=bool), min_area, gap).labels for m in masks]
    out = np.stack(labels) if labels else np.zeros((0, *masks.shape[1:]), dtype=np.int32)
    return out.astype(np.uint8) if out.size and out.max() <= np.iinfo(np.uint8).max else out


def decision_e(slices: pd.DataFrame, cfg: dict) -> pd.Series:
    """Which real slices pass the sampling filters of decision E (`filters.real_sets`), on the
    masks as saved. `max_mask_px` comes straight from `slices.csv` (all mask pixels = lesion
    pixels with strays + dropped specks); other filters need the masks and are not used here yet."""
    passes = pd.Series(True, index=slices.index)
    for name in decided(cfg, "filters.real_sets", "e0b_audit.py filters-review"):
        if name == "max_mask_px":
            passes &= (slices["lesion_px"] + slices["speck_px"]) < cfg["filters"]["max_mask_px"]
        else:
            raise SystemExit(f"Decision E filter '{name}' is not implemented (only max_mask_px); "
                             f"add it to memtk.sets.decision_e.")
    return passes
