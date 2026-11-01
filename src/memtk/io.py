"""File access for real datasets and synthetic pools, driven by config entries."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
from nibabel.dataobj_images import DataobjImage

from memtk.config import resolve_path


def load_image(path: str | Path) -> DataobjImage:
    """Load a NIfTI (or other array-backed) image; fails clearly on anything else."""
    img = nib.load(path)
    if not isinstance(img, DataobjImage):
        raise TypeError(f"Not an array-backed image: {path}")
    return img


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
