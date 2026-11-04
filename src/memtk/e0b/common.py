"""Helpers shared by the E0b stages: brain boxes, and access to any slice of a dataset or pool.
Helpers every stage package uses (decisions, overwrite and input guards, random picks) are in
`memtk.stages`; loading a patient's real volumes is `memtk.io.load_real` (moved 2026-11-04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from memtk import config, ids
from memtk.io import MaskStack, load_real, read_table
from memtk.synthetic import load_sample

# Brain-box columns of a brain.csv (inclusive rows and columns; -1 for a slice without brain)
BOX_COLUMNS = ["box_r0", "box_r1", "box_c0", "box_c1"]


def boxes_of(brain_csv: pd.DataFrame, key_col: str) -> dict[str, tuple[int, int, int, int] | None]:
    """Brain box per slice key from a brain.csv (None where the slice has no brain)."""
    return {str(k): (int(r0), int(r1), int(c0), int(c1)) if r0 >= 0 else None
            for k, r0, r1, c0, c1 in brain_csv[[key_col, *BOX_COLUMNS]].itertuples(index=False)}


class Slices:
    """Image, mask as saved, brain mask and brain box of any slice of one source, by slice key.
    Real data when `pool_cfg` is None (volumes in `out_dir`), else a synthetic pool. Needs the
    outputs of the `brain` stage."""

    def __init__(self, out_dir: Path, pool_cfg: dict | None = None):
        self.out_dir, self.pool_cfg = out_dir, pool_cfg
        self.brains = MaskStack(out_dir / "brain_masks.npz")
        self.boxes = boxes_of(read_table(out_dir / "brain.csv"), "key" if pool_cfg is None else "sample_id")
        if pool_cfg is not None:
            self.raw = MaskStack(out_dir / "masks_raw.npz")
            index = read_table(out_dir / "synthetic_index.csv")
            self.files = dict(zip(index["sample_id"], index["filename"]))
            self.pool_dir = config.resolve_path(pool_cfg["dir"])

    def __contains__(self, key: str) -> bool:
        return key in self.brains

    def __getitem__(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple | None]:
        if self.pool_cfg is None:
            pid, z = ids.parse_slice_key(key)
            bravo, seg = load_real(self.out_dir, pid)
            image, mask = bravo[:, :, z], seg[:, :, z] > 0
        else:
            image, mask = load_sample(self.pool_dir / self.files[key], self.pool_cfg)[0], self.raw[key]
        return image, mask, self.brains[key], self.boxes[key]
