"""Conversion of real volumes to the 128x128 "model space" the diffusion models were trained in.

Reproduces the training transform of the original code (amibekk/Thesis,
Generation/diffusion_training.py), applied per 3D volume to both the image and the mask:
    LoadImage -> EnsureChannelFirst -> ToTensor -> ScaleIntensity(0, 1) -> Resize((128, 128, -1))
followed by mask > threshold (strict, as `make_binary` there). ScaleIntensity uses the min and
max of the whole volume, so the conversion must run on volumes, never on single slices.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
# Imported from their defining modules: monai.transforms has no __all__, so type checkers
# treat its re-exports as private.
from monai.transforms.compose import Compose
from monai.transforms.intensity.array import ScaleIntensity
from monai.transforms.io.array import LoadImage
from monai.transforms.spatial.array import Resize
from monai.transforms.utility.array import EnsureChannelFirst, ToTensor


def model_space_transform(prep_cfg: dict) -> Compose:
    """The training transform; the slice axis keeps its length (-1 in Resize)."""
    spatial_size = list(prep_cfg["spatial_size"])
    spatial_size.insert(prep_cfg["slice_axis"], -1)
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ToTensor(),
        ScaleIntensity(minv=0.0, maxv=1.0),
        Resize(spatial_size=tuple(spatial_size), mode=prep_cfg["interpolation"]),
    ])


def apply_transform(path: str | Path, transform: Compose) -> np.ndarray:
    """Run `transform` on one file and return the single-channel result as a float32 array."""
    out = transform(str(path))
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Expected a tensor from the transform, got {type(out).__name__}: {path}")
    return out[0].detach().cpu().numpy().astype(np.float32)


def binarize_mask(mask_soft: np.ndarray, threshold: float) -> np.ndarray:
    """Binary uint8 mask: 1 where mask_soft > threshold (strict)."""
    return (mask_soft > threshold).astype(np.uint8)


def convert_volume_pair(image_path: str | Path, mask_path: str | Path,
                        prep_cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Model-space (image, soft mask, binary mask) of one patient."""
    transform = model_space_transform(prep_cfg)
    image = apply_transform(image_path, transform)
    mask_soft = apply_transform(mask_path, transform)
    return image, mask_soft, binarize_mask(mask_soft, prep_cfg["mask_threshold"])


def lesion_px_per_slice(mask: np.ndarray, axis: int) -> np.ndarray:
    """Number of non-zero mask pixels in each slice along `axis`."""
    other_axes = tuple(i for i in range(mask.ndim) if i != axis)
    return (mask != 0).sum(axis=other_axes)
