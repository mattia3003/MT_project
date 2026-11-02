"""File access for real datasets and synthetic pools, driven by config entries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.dataobj_images import DataobjImage

from memtk.config import resolve_path


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
