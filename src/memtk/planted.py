"""E4: planted copies of training masks with known origin (`docs/experiment-plan.md`, E4;
`docs/e4-plan.md`, decision I).

Base masks are T lesion slices (`choose_bases`: one group of patients with many multi-lesion
slices, one of other patients; slices of a patient spaced in z). Each base gives the variants
P1 exact, P2 / P3 shift (one axis direction, exact distance, nothing pushed off the image), P4
rotation about the centroid of all lesion pixels, P5 erosion and dilation with the 3x3 cross,
P6-remove (one informative lesion) and P6-add (one informative held-out lesion at its own
position, inside the base's brain, apart from the base lesions). Every planted mask is then
labelled with attach-or-drop like a real mask, and each planted lesion is traced to the lesion it
comes from (`provenance`), which E7 uses to check whether its explanations name the right
lesions. Pure functions on arrays; the `plant` stage of `scripts/e4_validate.py` reads and writes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage

from memtk.ids import lesion_id, parse_slice_key
from memtk.lesions import attach_or_drop, grow

VARIANTS = ("P1", "P2", "P3", "P4", "P5-erode", "P5-dilate", "P6-remove", "P6-add")
AXIS_DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
# The smallest 1 px change: 4-connected neighbours only
CROSS = ndimage.generate_binary_structure(2, 1)


# --- base masks ---

def choose_bases(slices: pd.DataFrame, p: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, list[str]]:
    """The base slices (decision I) from a table of T slices (`key`, `patient_id`, `z`,
    `n_informative`). Eligible: patients with at least `min_informative_slices` slices holding an
    informative lesion. `patients_multi` patients are drawn from those with at least
    `min_multi_slices` multi-lesion slices and give `multi_per_patient` multi- and
    `single_per_patient` single-lesion slices; `patients_other` are drawn from the remaining
    eligible patients and give `slices_per_patient` random informative slices. Slices of one
    patient are at least `min_z_gap` apart. A drawn patient whose slices cannot meet the quota
    with that spacing is skipped and the next one in the drawn order is used (a patient skipped
    in the first group can still be drawn in the second).

    Returns the chosen rows with `group` ("multi" or "other"), in draw order, and the skipped
    patients."""
    slices = slices[slices["n_informative"] > 0]
    per_patient = slices.groupby("patient_id")["n_informative"].agg(n="size", multi=lambda n: int((n >= 2).sum()))
    eligible = per_patient[per_patient["n"] >= p["min_informative_slices"]]
    multi_pool = sorted(eligible.index[eligible["multi"] >= p["min_multi_slices"]])
    chosen, drawn, skipped = [], [], []

    def draw(pool: list[str], n_patients: int, quotas, group: str) -> None:
        taken = 0
        for patient in [pool[i] for i in rng.permutation(len(pool))]:
            if taken == n_patients:
                break
            rows = slices[slices["patient_id"] == patient]
            picked: list = []
            for want, n in quotas:
                found = _spaced(rows[want(rows["n_informative"])], n, rows.loc[picked, "z"].tolist(),
                                p["min_z_gap"], rng)
                if found is None:
                    break
                picked += found
            if len(picked) < sum(n for _, n in quotas):
                skipped.append(patient)
                continue
            chosen.append(rows.loc[picked].assign(group=group))
            drawn.append(patient)
            taken += 1
        if taken < n_patients:
            raise ValueError(f"Only {taken} of {n_patients} '{group}' patients meet the base quotas")

    draw(multi_pool, p["patients_multi"],
         [(lambda n: n >= 2, p["multi_per_patient"]), (lambda n: n == 1, p["single_per_patient"])], "multi")
    other_pool = sorted(set(eligible.index) - set(drawn))
    draw(other_pool, p["patients_other"], [(lambda n: n >= 1, p["slices_per_patient"])], "other")
    return pd.concat(chosen).reset_index(drop=True), skipped


def _spaced(rows: pd.DataFrame, n: int, taken_z: list[int], gap: int, rng: np.random.Generator) -> list[int] | None:
    """Index labels of `n` random rows whose `z` lie at least `gap` from each other and from
    `taken_z` (greedy in random order); None if the greedy pass cannot find `n`."""
    picked, zs = [], list(taken_z)
    for i in rng.permutation(len(rows)):
        z = int(rows["z"].iloc[i])
        if all(abs(z - t) >= gap for t in zs):
            picked.append(rows.index[i])
            zs.append(z)
            if len(picked) == n:
                return picked
    return None


def planted_key(variant: str, base_key: str) -> str:
    """Key of a planted mask: `<variant>__<patient_id>_z<z>`, e.g. `P2__Mets_005_z127`."""
    patient, z = parse_slice_key(base_key)
    return f"{variant}__{patient}_z{z}"


# --- variants ---

def shift(array: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """The array moved by (dr, dc) pixels (mask or label map); refuses to push a non-zero pixel off
    the image, so the shift is exact."""
    rows, cols = np.nonzero(array)
    r, c = rows + dr, cols + dc
    if len(r) and (r.min() < 0 or c.min() < 0 or r.max() >= array.shape[0] or c.max() >= array.shape[1]):
        raise ValueError(f"Shift ({dr}, {dc}) would move pixels off the image")
    out = np.zeros_like(array)
    out[r, c] = array[rows, cols]
    return out


def shift_direction(mask: np.ndarray, distance: int, rng: np.random.Generator) -> tuple[int, int] | None:
    """A random axis direction (unit row, col step) along which the mask can move `distance`
    pixels without leaving the image; None if there is none."""
    rows, cols = np.nonzero(mask)
    h, w = mask.shape
    fits = [(dr, dc) for dr, dc in AXIS_DIRECTIONS
            if 0 <= rows.min() + dr * distance and rows.max() + dr * distance < h
            and 0 <= cols.min() + dc * distance and cols.max() + dc * distance < w]
    return fits[rng.integers(len(fits))] if fits else None


def lesion_centroid(mask: np.ndarray) -> np.ndarray:
    """(row, col) centroid of all lesion pixels of a mask or label map."""
    return np.argwhere(mask).mean(axis=0)


def _rotated(array: np.ndarray, degrees: float, centre: np.ndarray, order: int) -> np.ndarray:
    """The array rotated by `degrees` about `centre` (row, col); each output pixel is read from the
    input position rotated back (interpolation `order`), zero outside."""
    t = np.deg2rad(degrees)
    back = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    return ndimage.affine_transform(array, back, offset=centre - back @ centre,  # pyright: ignore[reportArgumentType]  # scipy stubs type offset as float; a per-axis sequence is documented and accepted
                                    order=order, mode="constant", cval=0)


def rotate(mask: np.ndarray, degrees: float, threshold: float) -> np.ndarray:
    """P4: the mask rotated by `degrees` (the sign sets the direction) about the centroid of its
    lesion pixels, bilinear on the float mask and re-binarized at `threshold`."""
    return _rotated(mask.astype(float), degrees, lesion_centroid(mask), order=1) >= threshold


def rotate_labels(labels: np.ndarray, degrees: float) -> np.ndarray:
    """A label map rotated like `rotate` rotates its mask (same centre; nearest neighbour), for
    tracing the planted lesions back to the base lesions."""
    return _rotated(labels, degrees, lesion_centroid(labels > 0), order=0)


def morph(mask: np.ndarray, op: str, px: int) -> np.ndarray:
    """P5: the mask eroded or dilated (`op` "erode" / "dilate") by `px` pixels with the 3x3 cross."""
    if op not in ("erode", "dilate"):
        raise ValueError(f"Unknown morphology '{op}' (erode or dilate)")
    f = ndimage.binary_erosion if op == "erode" else ndimage.binary_dilation
    return np.asarray(f(mask, structure=CROSS, iterations=px), dtype=bool)


def add_lesion(mask: np.ndarray, lesion: np.ndarray, brain: np.ndarray, min_gap: int) -> np.ndarray | None:
    """P6-add: the mask with `lesion` (a boolean mask at its own position) added, or None when the
    lesion leaves the brain or lies within `min_gap` background pixels of a mask pixel (it must stay
    a separate lesion)."""
    if (lesion & ~brain).any() or (grow(lesion, min_gap) & mask).any():
        return None
    return mask | lesion


# --- provenance ---

def provenance(planted: np.ndarray, reference: np.ndarray) -> dict[int, int]:
    """For each lesion of a planted label map, the reference label it overlaps most (ties: the
    lower label; 0 if it overlaps none). `reference` is the base label map put through the same
    geometric change (shift, rotation) or unchanged (morphology, removal)."""
    out = {}
    for n in range(1, int(planted.max()) + 1):
        under = reference[planted == n]
        under = under[under > 0]
        out[n] = int(np.bincount(under).argmax()) if under.size else 0
    return out


@dataclass
class Planted:
    """One planted mask: its label map after attach-or-drop, how it was made, where each lesion
    comes from, and whether every informative base lesion is still a lesion of at least the bound
    (the rule that gates P5)."""
    variant: str
    key: str
    base_key: str
    labels: np.ndarray
    params: dict
    sources: dict[int, str]    # planted lesion number -> source lesion ID ("" if none)
    keeps_base_lesions: bool


def variants_of(base_key: str, base_labels: np.ndarray, brain: np.ndarray, donors: list[tuple[str, np.ndarray]],
                p: dict, bound: int, min_area: int, gap: int,
                rng: dict[str, np.random.Generator]) -> tuple[list[Planted], dict[str, str]]:
    """Every variant of one base label map (decision I; `p` = `validation.variants`). `donors`:
    the informative held-out lesions for P6-add as (lesion ID, boolean mask). `rng` holds one
    generator per purpose: "direction", "angle", "remove", "add". Planted masks are labelled with
    attach-or-drop (`min_area`, `gap`). Returns the planted masks and, per variant that could not
    be made, the reason."""
    mask = base_labels > 0
    areas = np.bincount(base_labels.ravel())
    informative = [n for n in range(1, len(areas)) if areas[n] >= bound]
    base_names = {n: lesion_id(base_key, n) for n in range(1, len(areas))}
    planted, skipped = [], {}

    def make(variant: str, planted_mask: np.ndarray, params: dict, reference: np.ndarray, names: dict[int, str]):
        labels = attach_or_drop(planted_mask, min_area, gap).labels
        sources = {n: names.get(r, "") for n, r in provenance(labels, reference).items()}
        sizes = np.bincount(labels.ravel(), minlength=len(sources) + 1)
        kept = {sources[n] for n in sources if sizes[n] >= bound}
        planted.append(Planted(variant, planted_key(variant, base_key), base_key, labels, params, sources,
                               all(base_names[n] in kept for n in informative)))

    make("P1", mask, {}, base_labels, base_names)

    direction = shift_direction(mask, p["shift_large_px"], rng["direction"])
    for variant, distance in (("P2", p["shift_small_px"]), ("P3", p["shift_large_px"])):
        if direction is None:
            skipped[variant] = f"no axis direction leaves room for {p['shift_large_px']} px"
            continue
        dr, dc = direction[0] * distance, direction[1] * distance
        make(variant, shift(mask, dr, dc), {"shift_row": dr, "shift_col": dc}, shift(base_labels, dr, dc), base_names)

    angle = float(rng["angle"].choice([-1, 1]) * p["rotate_deg"])
    make("P4", rotate(mask, angle, p["rotate_threshold"]), {"angle": angle}, rotate_labels(base_labels, angle),
         base_names)

    for op in ("erode", "dilate"):
        make(f"P5-{op}", morph(mask, op, p["morph_px"]), {"op": op, "px": p["morph_px"]}, base_labels, base_names)

    if len(informative) >= 2:
        n = int(rng["remove"].choice(informative))
        make("P6-remove", mask & (base_labels != n), {"removed": base_names[n]}, base_labels, base_names)
    else:
        skipped["P6-remove"] = "fewer than 2 informative lesions"

    tries = rng["add"].permutation(len(donors))[:p["add_max_tries"]]
    for k, i in enumerate(tries, start=1):
        donor_id, lesion = donors[i]
        added = add_lesion(mask, lesion, brain, p["add_min_gap_px"])
        if added is not None:
            donor_label = len(areas)
            reference = np.where(lesion, donor_label, base_labels)
            make("P6-add", added, {"added": donor_id, "tries": k}, reference, {**base_names, donor_label: donor_id})
            break
    else:
        skipped["P6-add"] = f"no held-out lesion fits after {len(tries)} tries"
    return planted, skipped
