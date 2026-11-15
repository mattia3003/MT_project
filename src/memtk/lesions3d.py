"""3D lesions of real volumes in model space (`docs/e6-e7-plan.md`, decision W).

E1–E5 compare single slices; E6's duplication test (H1) and E7's Panel C need to know which 2D
lesions of neighbouring slices are the same metastasis. Two definitions, both on the model-space
mask **as saved** (`seg.nii.gz`, every component, so the thin first and last slices of a
metastasis count):

- **Slice-to-slice linking (main):** the 8-connected components of each slice are linked to those
  of the next slice one-to-one, by Hungarian assignment on pixel IoU (pairs with IoU 0 are never
  linked). A chain of linked components is one 3D lesion. Two metastases that touch in one slice
  are one component there; the one-to-one rule lets only one of them continue through it, so a
  merge or split never joins two chains.
- **26-connectivity (sensitivity):** 3D connected components. Neighbouring metastases that touch in
  any slice become one 3D lesion (`max_parts` > 1 shows where).

The E0b 2D lesion labels (cleaned, numbered by size per slice) stay unchanged; each maps to the 3D
lesion holding most of its pixels (`map_2d_to_3d`). Volumes are (H, W, D), slice index z on the last
axis, as written by E0a.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from memtk.ids import lesion3d_id
from memtk.lesions import label_components

TWENTY_SIX_CONNECTED = np.ones((3, 3, 3), dtype=bool)


# --- 3D labelling -----------------------------------------------------------------------------

def link_slices(seg: np.ndarray) -> np.ndarray:
    """3D lesion labels (H, W, D) by slice-to-slice linking: per slice the 8-connected components
    of `seg > 0`; a component continues the chain of the component of the previous slice it is
    paired with (Hungarian, maximising the summed IoU, IoU > 0); unpaired components start a new
    chain. Chains are numbered from 1 in order of first appearance (z, then component order)."""
    seg = np.asarray(seg) > 0
    out = np.zeros(seg.shape, dtype=np.int32)
    prev_labels, prev_chain = None, np.zeros(1, dtype=np.int32)
    n_chains = 0
    for z in range(seg.shape[2]):
        labels, n = label_components(seg[:, :, z])
        chain = np.zeros(n + 1, dtype=np.int32)          # component label -> chain number
        if n and prev_labels is not None and len(prev_chain) > 1:
            rows, cols = _paired(prev_labels, len(prev_chain) - 1, labels, n)
            chain[cols] = prev_chain[rows]
        for c in range(1, n + 1):
            if not chain[c]:
                n_chains += 1
                chain[c] = n_chains
        out[:, :, z] = chain[labels]
        prev_labels, prev_chain = labels, chain
    return out


def components_26(seg: np.ndarray) -> np.ndarray:
    """3D lesion labels (H, W, D) as 26-connected components of `seg > 0`, renumbered from 1 in
    order of first appearance (lowest z first), like `link_slices`."""
    result = ndimage.label(np.asarray(seg) > 0, structure=TWENTY_SIX_CONNECTED)
    if not isinstance(result, tuple):  # only an int when an `output` array is passed
        raise TypeError("ndimage.label returned no label array")
    labels = np.asarray(result[0], dtype=np.int32)
    values, first = np.unique(np.moveaxis(labels, 2, 0).ravel(), return_index=True)  # slice by slice, raster order
    order = values[np.argsort(first)]
    order = order[order > 0]
    renumber = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    renumber[order] = np.arange(1, len(order) + 1)
    return renumber[labels]


def _paired(a: np.ndarray, na: int, b: np.ndarray, nb: int) -> tuple[np.ndarray, np.ndarray]:
    """One-to-one pairs (labels in `a`, labels in `b`) of two label images that maximise the
    summed IoU, keeping only pairs that overlap."""
    both = (a > 0) & (b > 0)
    inter = np.bincount(a[both] * (nb + 1) + b[both], minlength=(na + 1) * (nb + 1)).reshape(na + 1, nb + 1)[1:, 1:]
    if not inter.any():
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    area_a = np.bincount(a.ravel(), minlength=na + 1)[1:]
    area_b = np.bincount(b.ravel(), minlength=nb + 1)[1:]
    iou = inter / (area_a[:, None] + area_b[None, :] - inter)
    rows, cols = linear_sum_assignment(iou, maximize=True)
    keep = iou[rows, cols] > 0
    return rows[keep] + 1, cols[keep] + 1


# --- tables -----------------------------------------------------------------------------------

def lesion3d_table(labels3d: np.ndarray, patient_id: str) -> pd.DataFrame:
    """One row per 3D lesion of a label volume (H, W, D): `lesion3d_id`, `patient_id`, `number`,
    `z_first`, `z_last`, `span` (slices that contain it), `volume_px`, `largest_area_px` (its
    largest area in one slice) and `max_parts` (most separate 8-connected pieces it has in one
    slice: above 1 only for 26-connected lesions that join neighbouring metastases)."""
    rows = []
    for z in range(labels3d.shape[2]):
        sl = labels3d[:, :, z]
        for n in np.unique(sl[sl > 0]):
            region = sl == n
            rows.append({"number": int(n), "z": z, "area": int(region.sum()), "parts": label_components(region)[1]})
    columns = ["lesion3d_id", "patient_id", "number", "z_first", "z_last", "span", "volume_px", "largest_area_px",
               "max_parts"]
    if not rows:
        return pd.DataFrame(columns=columns)
    per_slice = pd.DataFrame(rows)
    out = per_slice.groupby("number").agg(z_first=("z", "min"), z_last=("z", "max"), span=("z", "nunique"),
                                          volume_px=("area", "sum"), largest_area_px=("area", "max"),
                                          max_parts=("parts", "max")).reset_index()
    out.insert(0, "patient_id", patient_id)
    out.insert(0, "lesion3d_id", [lesion3d_id(patient_id, n) for n in out["number"]])
    return out[columns]


def map_2d_to_3d(labels2d: np.ndarray, labels3d_slice: np.ndarray) -> dict[int, int]:
    """For each 2D lesion number of a slice's label map, the 3D lesion number holding most of its
    pixels in the same slice (ties: the lower number); 0 if none of its pixels lies in a 3D lesion."""
    out = {}
    for n in np.unique(labels2d[labels2d > 0]):
        under = labels3d_slice[labels2d == n]
        under = under[under > 0]
        out[int(n)] = int(np.bincount(under).argmax()) if len(under) else 0
    return out
