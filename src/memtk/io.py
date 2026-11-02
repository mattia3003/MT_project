"""File access for real datasets and synthetic pools, driven by config entries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.dataobj_images import DataobjImage

from memtk.config import resolve_path

# Columns that hold IDs. They must stay text: a sample ID such as `1033` would otherwise be read as
# a number and no longer match its key.
ID_COLUMNS = ("source", "key", "lesion_id", "patient_id", "sample_id", "set", "filename")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a pipeline CSV with every ID column as text (see ID_COLUMNS)."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns
    return pd.read_csv(path, dtype={c: str for c in ID_COLUMNS if c in header})


def load_image(path: str | Path) -> DataobjImage:
    """Load a NIfTI (or other array-backed) image; fails clearly on anything else."""
    img = nib.load(path)
    if not isinstance(img, DataobjImage):
        raise TypeError(f"Not an array-backed image: {path}")
    return img


def save_nifti(array: np.ndarray, path: str | Path) -> None:
    """Save an array as NIfTI with its own dtype and an identity affine.

    Model-space arrays are resampled, so the native affine no longer applies; array index
    (i, j, z) is the only coordinate system used downstream.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)


def list_patients(dataset_cfg: dict, split: str) -> list[str]:
    """Sorted patient IDs (folder names) in one source split of a real dataset."""
    split_dir = resolve_path(dataset_cfg["root"]) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def patient_files(dataset_cfg: dict, split: str, patient_id: str) -> tuple[Path, Path]:
    """(image, mask) paths of one patient. Raises if either file is missing."""
    folder = resolve_path(dataset_cfg["root"]) / split / patient_id
    image = folder / dataset_cfg["image_file"]
    mask = folder / dataset_cfg["mask_file"]
    for path in (image, mask):
        if not path.is_file():
            raise FileNotFoundError(f"Missing file for patient {patient_id}: {path}")
    return image, mask


# --- mask stacks: many 2D boolean masks in one file, looked up by key ------------------------
# Stored bit-packed (1 bit per pixel) in a compressed .npz with arrays `keys`, `packed` and
# `shape`, so a whole synthetic pool fits in memory and each mask is unpacked only when used.

class MaskStackWriter:
    """Collects 2D boolean masks of one shape under unique keys; `save` writes the stack."""

    def __init__(self, shape: Sequence[int]):
        self.shape = (int(shape[0]), int(shape[1]))
        self.keys: list[str] = []
        self._packed: list[np.ndarray] = []

    def add(self, key: str, mask: np.ndarray) -> None:
        if mask.shape != self.shape:
            raise ValueError(f"Mask '{key}' has shape {mask.shape}, expected {self.shape}")
        self.keys.append(key)
        self._packed.append(np.packbits(np.asarray(mask, dtype=bool).ravel()))

    def save(self, path: str | Path) -> Path:
        if len(set(self.keys)) != len(self.keys):
            raise ValueError(f"Duplicate keys in mask stack {path}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        packed = np.stack(self._packed) if self._packed else np.zeros((0, 0), dtype=np.uint8)
        np.savez_compressed(path, keys=np.array(self.keys, dtype=str), packed=packed,
                            shape=np.array(self.shape))
        return path


class MaskStack:
    """A saved mask stack: `stack[key]` gives one mask as a boolean (H, W) array."""

    def __init__(self, path: str | Path):
        with np.load(path) as data:
            self.keys: list[str] = data["keys"].tolist()
            self._packed = data["packed"]
            self.shape = (int(data["shape"][0]), int(data["shape"][1]))
        self._row = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        return key in self._row

    def __getitem__(self, key: str) -> np.ndarray:
        bits = np.unpackbits(self._packed[self._row[key]], count=self.shape[0] * self.shape[1])
        return bits.reshape(self.shape).astype(bool)


# --- label stacks: many 2D integer label maps in one file, looked up by key -------------------
# Stored sparse: only the non-zero pixels (flat index and label) of each map, concatenated, with
# per-key offsets. Lesion label maps are almost empty, so a whole pool takes a few MB.

class LabelStackWriter:
    """Collects 2D integer label maps (0 = background) of one shape under unique keys."""

    def __init__(self, shape: Sequence[int]):
        self.shape = (int(shape[0]), int(shape[1]))
        self.keys: list[str] = []
        self._index: list[np.ndarray] = []
        self._values: list[np.ndarray] = []

    def add(self, key: str, labels: np.ndarray) -> None:
        if labels.shape != self.shape:
            raise ValueError(f"Label map '{key}' has shape {labels.shape}, expected {self.shape}")
        flat = np.asarray(labels).ravel()
        index = np.flatnonzero(flat)
        if index.size and (flat[index].min() < 0 or flat[index].max() > np.iinfo(np.uint16).max):
            raise ValueError(f"Label map '{key}' has labels outside 0..65535")
        self.keys.append(key)
        self._index.append(index.astype(np.uint32))
        self._values.append(flat[index].astype(np.uint16))

    def save(self, path: str | Path) -> Path:
        if len(set(self.keys)) != len(self.keys):
            raise ValueError(f"Duplicate keys in label stack {path}")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        offsets = np.concatenate([[0], np.cumsum([len(i) for i in self._index])]).astype(np.int64)
        np.savez_compressed(path, keys=np.array(self.keys, dtype=str), offsets=offsets,
                            index=np.concatenate(self._index) if self._index else np.zeros(0, np.uint32),
                            values=np.concatenate(self._values) if self._values else np.zeros(0, np.uint16),
                            shape=np.array(self.shape))
        return path


class LabelStack:
    """A saved label stack: `stack[key]` gives one label map as an int32 (H, W) array."""

    def __init__(self, path: str | Path):
        with np.load(path) as data:
            self.keys: list[str] = data["keys"].tolist()
            self._offsets, self._index, self._values = data["offsets"], data["index"], data["values"]
            self.shape = (int(data["shape"][0]), int(data["shape"][1]))
        self._row = {k: i for i, k in enumerate(self.keys)}

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key: str) -> bool:
        return key in self._row

    def __getitem__(self, key: str) -> np.ndarray:
        i = self._row[key]
        lo, hi = self._offsets[i], self._offsets[i + 1]
        flat = np.zeros(self.shape[0] * self.shape[1], dtype=np.int32)
        flat[self._index[lo:hi]] = self._values[lo:hi]
        return flat.reshape(self.shape)
