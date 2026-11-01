"""Lesion labelling and brain masks in model space (128x128 slices).

Masks are labelled into 8-connected components. Brain masks of the paired MRI give the brain
box used for brain-relative positions and the inside/outside-brain tests of quality control.
Attach-or-drop of specks and lesion features are added for the E0b `lesions` stage.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)

# Fixed foreground cut of the `low_threshold` brain method. Measured 2026-11-01 on 600 T and 600
# synthetic slices: background lies near 0 (99.9th percentile ~0.001), dim brain tissue near 0.09
# (2nd percentile of the Otsu brain), so 0.02 separates them in both real and synthetic images.
LOW_THRESHOLD = 0.02

# Smallest foreground region kept by the "all regions" rule. Low slices show the brain as
# separate pieces (lower lobes, cerebellum), which keeping only the largest region throws away;
# 30 px is well below such pieces and well above isolated bright pixels.
MIN_BRAIN_REGION = 30

# Candidate brain-mask methods: name -> (foreground rule, outward dilation in px, keep all regions)
BRAIN_METHODS = {
    "otsu": ("otsu", 0, False),
    "otsu_dilate1": ("otsu", 1, False),
    "otsu_dilate2": ("otsu", 2, False),
    "otsu_all": ("otsu", 0, True),
    "low_threshold": ("low", 0, False),
    "low_threshold_all": ("low", 0, True),
}


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected components of a 2D mask: (label image with 1..n, n)."""
    result = ndimage.label(mask > 0, structure=EIGHT_CONNECTED)
    if not isinstance(result, tuple):  # only an int when an `output` array is passed
        raise TypeError("ndimage.label returned no label array")
    labels, n = result
    return labels, int(n)


def count_components(mask: np.ndarray) -> int:
    """Number of 8-connected foreground components of a 2D mask."""
    return label_components(mask)[1]


def brain_mask(image: np.ndarray, method: str) -> np.ndarray:
    """Brain of one 2D MRI slice as a boolean mask; empty if the image has no foreground.

    Foreground (Otsu threshold, or the fixed LOW_THRESHOLD), then the largest 8-connected
    region, or every region of at least MIN_BRAIN_REGION px for the "all regions" methods,
    with holes filled (so dark ventricles count as brain), then grown outward by the method's
    dilation.
    """
    if method not in BRAIN_METHODS:
        raise ValueError(f"Unknown brain method '{method}'; choose from {sorted(BRAIN_METHODS)}")
    rule, dilation, keep_all = BRAIN_METHODS[method]
    if rule == "otsu":
        if image.max() <= image.min():  # constant image: Otsu is undefined
            return np.zeros(image.shape, dtype=bool)
        foreground = image > threshold_otsu(image)
    else:
        foreground = image > LOW_THRESHOLD

    labels, n = label_components(foreground)
    if n == 0:
        return np.zeros(image.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())[1:]
    if keep_all:
        kept = np.isin(labels, np.flatnonzero(sizes >= MIN_BRAIN_REGION) + 1)
    else:
        kept = labels == int(np.argmax(sizes)) + 1
    brain = np.asarray(ndimage.binary_fill_holes(kept), dtype=bool)
    if dilation:
        brain = np.asarray(ndimage.binary_dilation(brain, structure=EIGHT_CONNECTED, iterations=dilation),
                           dtype=bool)
    return brain


def tissue_px(image: np.ndarray) -> int:
    """Amount of brain tissue in a slice: pixels of the `low_threshold_all` brain.

    One fixed measure for every image, independent of the chosen brain method; a near-blank
    (failed) synthetic MRI has almost none.
    """
    return int(brain_mask(image, "low_threshold_all").sum())


def brain_box(brain: np.ndarray) -> tuple[int, int, int, int] | None:
    """Inclusive bounding box (row0, row1, col0, col1) of a brain mask; None if it is empty."""
    rows, cols = np.flatnonzero(brain.any(axis=1)), np.flatnonzero(brain.any(axis=0))
    if rows.size == 0:
        return None
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def fraction_outside(component: np.ndarray, brain: np.ndarray) -> float:
    """Share of a component's pixels that lie outside the brain mask."""
    size = int(component.sum())
    return float((component & ~brain).sum() / size) if size else 0.0
