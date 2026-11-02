"""Lesion labelling and brain masks in model space (128x128 slices).

Masks are labelled into 8-connected components, then attach-or-drop turns components below a
minimum area (specks) into stray pixels of a nearby lesion or drops them. Brain masks of the
paired MRI give the brain box used for brain-relative positions and the inside/outside-brain
tests of quality control. Lesion features are added for the E0b `lesions` stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)

# The foreground cut of the `low_threshold` methods (`low_threshold` argument) depends on the data,
# so it comes from the config (`brain.low_threshold`), checked per dataset by E0b brain-review.

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


def chessboard_distance(region: np.ndarray) -> np.ndarray:
    """Per pixel, the Chebyshev (chessboard) distance to the nearest pixel of `region` (0 on
    the region itself). The region must not be empty."""
    result = ndimage.distance_transform_cdt(~np.asarray(region, dtype=bool), metric="chessboard")
    if not isinstance(result, np.ndarray):  # None only when a `distances` output array is passed
        raise TypeError("distance_transform_cdt returned no distance array")
    return result


def grow(region: np.ndarray, px: int) -> np.ndarray:
    """A boolean region grown outward by `px` pixels in every direction (8-connected)."""
    return np.asarray(ndimage.binary_dilation(region, structure=EIGHT_CONNECTED, iterations=px), dtype=bool)


# --- attach-or-drop --------------------------------------------------------------------------
# Gap = background pixels between the nearest pixels of two components, in Chebyshev distance:
# touching components are already one component, so a gap of 1 means one empty pixel between.

@dataclass
class LesionLabels:
    """A mask after attach-or-drop. Nothing is moved or deleted: every mask pixel is either in
    `labels` (a lesion, attached strays included) or in `dropped`."""
    labels: np.ndarray    # int32; 0 = background or dropped speck, 1..n = lesions (L1 = largest)
    attached: np.ndarray  # bool; pixels of the specks attached to a lesion (stray pixels)
    dropped: np.ndarray   # bool; pixels of the isolated specks, left out of `labels`
    n_lesions: int
    n_strays: int         # speck components attached
    n_specks: int         # speck components dropped (isolated specks)


def _components(mask: np.ndarray, min_area: int):
    """Components (labels 1..n) and their sizes, the labels of the lesions (>= min_area px), and
    per pixel the Chebyshev distance to the nearest lesion pixel (None if there is no lesion)."""
    comp, n = label_components(mask)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    sizes[0] = 0                                     # background is not a component
    lesion_ids = np.flatnonzero(sizes >= min_area)
    in_lesion = np.isin(comp, lesion_ids)
    dist = chessboard_distance(in_lesion) if lesion_ids.size else None
    return comp, n, sizes, lesion_ids, in_lesion, dist


def speck_gaps(mask: np.ndarray, min_area: int) -> list[tuple[int, int, int, int]]:
    """Every speck (component < min_area px) of a mask as (size, gap, row, col): its gap to the
    nearest lesion (-1 if the mask has no lesion) and its first pixel."""
    comp, n, sizes, _, _, dist = _components(mask, min_area)
    out = []
    for i in range(1, n + 1):
        if sizes[i] >= min_area:
            continue
        rows, cols = np.nonzero(comp == i)
        gap = int(dist[rows, cols].min()) - 1 if dist is not None else -1
        out.append((int(sizes[i]), gap, int(rows[0]), int(cols[0])))
    return out


def attach_or_drop(mask: np.ndarray, min_area: int, gap: int) -> LesionLabels:
    """Label a mask into lesions. Components of at least `min_area` px are lesions; smaller ones
    are specks. A speck within `gap` background px of a lesion is a stray: it takes the label of
    the nearest lesion (ties: the larger lesion, then the lower component number). Other specks
    are dropped. gap = 0 means drop-only. Two lesions are never merged.

    Lesions are numbered by final area (strays included), largest first; ties go to the lesion
    whose first pixel comes first in row-major order (top-left first).
    """
    comp, n, sizes, lesion_ids, in_lesion, dist = _components(mask, min_area)
    owner = np.zeros(n + 1, dtype=np.int64)          # component -> lesion component it belongs to
    owner[lesion_ids] = lesion_ids
    attached = np.zeros(mask.shape, dtype=bool)
    dropped = np.zeros(mask.shape, dtype=bool)
    n_strays = n_specks = 0
    for i in range(1, n + 1):
        if sizes[i] >= min_area:
            continue
        pix = comp == i
        d = int(dist[pix].min()) if dist is not None else None   # Chebyshev distance = gap + 1
        if d is not None and d - 1 <= gap:
            # the lesions at that distance: those reached by growing the speck d px
            near = np.unique(comp[grow(pix, d) & in_lesion])
            owner[i] = min(near, key=lambda j: (-sizes[j], j))
            attached |= pix
            n_strays += 1
        else:
            dropped |= pix
            n_specks += 1

    grouped = owner[comp]                            # 0 for background and dropped specks
    areas = np.bincount(grouped.ravel(), minlength=n + 1)
    first = {j: int(np.flatnonzero(grouped == j)[0]) for j in lesion_ids}
    order = sorted(lesion_ids, key=lambda j: (-areas[j], first[j]))
    relabel = np.zeros(n + 1, dtype=np.int32)
    relabel[order] = np.arange(1, len(order) + 1)
    return LesionLabels(relabel[grouped], attached, dropped, len(order), n_strays, n_specks)


def brain_mask(image: np.ndarray, method: str, low_threshold: float) -> np.ndarray:
    """Brain of one 2D MRI slice as a boolean mask; empty if the image has no foreground.

    Foreground (Otsu threshold, or pixels above `low_threshold` for the low-threshold methods;
    the Otsu methods ignore it), then the largest 8-connected region, or every region of at
    least MIN_BRAIN_REGION px for the "all regions" methods, with holes filled (so dark
    ventricles count as brain), then grown outward by the method's dilation.
    """
    if method not in BRAIN_METHODS:
        raise ValueError(f"Unknown brain method '{method}'; choose from {sorted(BRAIN_METHODS)}")
    rule, dilation, keep_all = BRAIN_METHODS[method]
    if rule == "otsu":
        if image.max() <= image.min():  # constant image: Otsu is undefined
            return np.zeros(image.shape, dtype=bool)
        foreground = image > threshold_otsu(image)
    else:
        foreground = image > low_threshold

    labels, n = label_components(foreground)
    if n == 0:
        return np.zeros(image.shape, dtype=bool)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0                                     # background is not a region
    if keep_all:
        kept = np.isin(labels, np.flatnonzero(sizes >= MIN_BRAIN_REGION))
    else:
        kept = labels == int(np.argmax(sizes))
    brain = np.asarray(ndimage.binary_fill_holes(kept), dtype=bool)
    return grow(brain, dilation) if dilation else brain


def tissue_px(image: np.ndarray, low_threshold: float) -> int:
    """Amount of brain tissue in a slice: pixels of the `low_threshold_all` brain.

    One fixed measure for every image, independent of the chosen brain method; a near-blank
    (failed) synthetic MRI has almost none.
    """
    return int(brain_mask(image, "low_threshold_all", low_threshold).sum())


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


# --- lesion features -------------------------------------------------------------------------

def lesion_features(res: LesionLabels, brain: np.ndarray, box: tuple[int, int, int, int] | None) -> list[dict]:
    """One row per lesion of an attach-or-drop result, in label order (L1 first).

    The centroid is given in three frames: pixels (row, col) of the model-space grid; the same
    as a fraction of the image size; and brain-relative, as a fraction of the brain box (0 at
    its first row/column, 1 at its last; outside [0, 1] for a lesion beyond the box; NaN without
    a brain). Shape measures come from scikit-image regionprops.
    """
    height, width = res.labels.shape
    rows = []
    for p in regionprops(res.labels):
        lesion = res.labels == p.label
        r, c = p.centroid
        r0, c0, r1, c1 = p.bbox                      # regionprops: exclusive end
        if box is not None:
            b_r0, b_r1, b_c0, b_c1 = box
            row_brain = (r - b_r0) / (b_r1 - b_r0) if b_r1 > b_r0 else 0.5
            col_brain = (c - b_c0) / (b_c1 - b_c0) if b_c1 > b_c0 else 0.5
        else:
            row_brain = col_brain = float("nan")
        rows.append({
            "label": int(p.label), "area": int(p.area), "stray_px": int((lesion & res.attached).sum()),
            "centroid_row": float(r), "centroid_col": float(c),
            "centroid_row_img": float(r / height), "centroid_col_img": float(c / width),
            "centroid_row_brain": float(row_brain), "centroid_col_brain": float(col_brain),
            "bbox_r0": int(r0), "bbox_r1": int(r1 - 1), "bbox_c0": int(c0), "bbox_c1": int(c1 - 1),
            "equiv_diameter": float(p.equivalent_diameter_area), "eccentricity": float(p.eccentricity),
            "solidity": float(p.solidity), "frac_outside_brain": fraction_outside(lesion, brain),
        })
    return rows
