"""Unit tests for memtk.planted (E4 planted copies) on small hand-built masks with known answers."""

import numpy as np
import pandas as pd
import pytest

from memtk.lesions import attach_or_drop
from memtk.planted import (add_lesion, choose_bases, lesion_centroid, morph, planted_key, provenance, rotate,
                           rotate_labels, shift, shift_direction, variants_of)

VARIANT_PARAMS = {"shift_small_px": 2, "shift_large_px": 5, "rotate_deg": 5, "rotate_threshold": 0.5,
                  "morph_px": 1, "add_min_gap_px": 3, "add_max_tries": 50}
BASE_PARAMS = {"min_informative_slices": 5, "patients_multi": 1, "min_multi_slices": 3, "multi_per_patient": 3,
               "single_per_patient": 2, "patients_other": 2, "slices_per_patient": 5, "min_z_gap": 2}


def rect(shape, r0, c0, h, w):
    m = np.zeros(shape, dtype=bool)
    m[r0:r0 + h, c0:c0 + w] = True
    return m


def rngs(seed=0):
    streams = np.random.SeedSequence(seed).spawn(4)
    return {k: np.random.default_rng(s) for k, s in zip(("direction", "angle", "remove", "add"), streams)}


@pytest.fixture
def base():
    """A 64x64 base label map with two informative lesions: L1 7x7 (49 px), L2 6x6 (36 px)."""
    mask = rect((64, 64), 20, 20, 7, 7) | rect((64, 64), 34, 36, 6, 6)
    return attach_or_drop(mask, 4, 2).labels


# --- variants ---------------------------------------------------------------------------------

def test_shift_moves_every_pixel_exactly_and_keeps_labels():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2, 3], labels[5, 5] = 1, 2
    moved = shift(labels, 2, -1)
    assert moved[4, 2] == 1 and moved[7, 4] == 2 and moved.sum() == 3


def test_shift_refuses_to_push_pixels_off_the_image():
    mask = rect((10, 10), 0, 4, 2, 2)
    with pytest.raises(ValueError):
        shift(mask, -1, 0)


def test_shift_direction_only_where_there_is_room():
    rng = np.random.default_rng(0)
    top = rect((20, 20), 0, 8, 3, 3)                             # touches the top edge
    assert all(shift_direction(top, 5, rng) != (-1, 0) for _ in range(50))
    assert shift_direction(rect((20, 20), 0, 0, 20, 20), 1, rng) is None


def test_rotation_by_zero_is_the_identity(base):
    assert (rotate(base > 0, 0, 0.5) == (base > 0)).all()
    assert (rotate_labels(base, 0) == base).all()


@pytest.mark.parametrize("angle", [-5, 5])
def test_small_rotation_keeps_area_and_centroid(angle):
    mask = rect((64, 64), 20, 18, 6, 14)
    out = rotate(mask, angle, 0.5)
    assert abs(int(out.sum()) - int(mask.sum())) <= 0.1 * mask.sum()
    assert np.abs(lesion_centroid(out) - lesion_centroid(mask)).max() < 0.5
    assert not (out == mask).all()


def test_morphology_with_the_cross():
    square = rect((20, 20), 5, 5, 5, 5)
    assert morph(square, "erode", 1).sum() == 9                 # 3x3 left
    assert morph(square, "dilate", 1).sum() == 25 + 4 * 5       # one row/column on each side, no corners
    with pytest.raises(ValueError):
        morph(square, "open", 1)


def test_added_lesion_must_stay_inside_the_brain_and_apart():
    mask, brain = rect((30, 30), 10, 10, 3, 3), rect((30, 30), 2, 2, 26, 26)
    at_gap_2 = rect((30, 30), 10, 15, 3, 3)                     # columns 13, 14 empty: 2 background px
    at_gap_3 = rect((30, 30), 10, 16, 3, 3)
    outside = rect((30, 30), 0, 20, 3, 3)
    assert add_lesion(mask, at_gap_2, brain, 3) is None
    assert add_lesion(mask, outside, brain, 3) is None
    added = add_lesion(mask, at_gap_3, brain, 3)
    assert added is not None and added.sum() == 18


def test_provenance_follows_the_largest_overlap():
    reference = np.zeros((10, 10), dtype=np.int32)
    reference[1:4, 1:4], reference[1:4, 4:7] = 2, 1
    planted = np.zeros((10, 10), dtype=np.int32)
    planted[1:4, 2:5] = 1                                       # 6 px over label 2, 3 px over label 1
    planted[7:9, 7:9] = 2                                      # over nothing
    assert provenance(planted, reference) == {1: 2, 2: 0}
    tie = np.zeros((10, 10), dtype=np.int32)
    tie[1:4, 3:5] = 1                                           # 3 px over each: the lower label
    assert provenance(tie, reference) == {1: 1}


def test_planted_key():
    assert planted_key("P2", "Mets_005/z127") == "P2__Mets_005_z127"


def test_every_variant_of_a_two_lesion_base(base):
    brain = rect((64, 64), 4, 4, 56, 56)
    donors = [("Heval/z1/L1", rect((64, 64), 0, 0, 4, 4)),     # outside the brain: never added
              ("Heval/z2/L1", rect((64, 64), 8, 44, 4, 4))]
    planted, skipped = variants_of("Pat/z9", base, brain, donors, VARIANT_PARAMS, 12, 4, 2, rngs())
    by = {p.variant: p for p in planted}
    assert skipped == {} and list(by) == ["P1", "P2", "P3", "P4", "P5-erode", "P5-dilate", "P6-remove", "P6-add"]
    assert (by["P1"].labels == base).all() and by["P1"].key == "P1__Pat_z9"
    assert by["P1"].sources == {1: "Pat/z9/L1", 2: "Pat/z9/L2"}
    small, large = by["P2"].params, by["P3"].params
    assert (abs(small["shift_row"]) + abs(small["shift_col"]), abs(large["shift_row"]) + abs(large["shift_col"])) == (2, 5)
    assert np.sign(small["shift_row"]) == np.sign(large["shift_row"]) and np.sign(small["shift_col"]) == np.sign(large["shift_col"])
    assert (by["P2"].labels == shift(base, small["shift_row"], small["shift_col"])).all()
    assert abs(by["P4"].params["angle"]) == 5 and by["P4"].sources == {1: "Pat/z9/L1", 2: "Pat/z9/L2"}
    assert all(by[v].keeps_base_lesions for v in ("P1", "P2", "P3", "P4", "P5-erode", "P5-dilate", "P6-add"))
    removed = by["P6-remove"]
    assert removed.labels.max() == 1 and not removed.keeps_base_lesions
    assert removed.sources[1] != removed.params["removed"]
    added = by["P6-add"]
    assert added.params["added"] == "Heval/z2/L1" and sorted(added.sources.values()) == ["Heval/z2/L1", "Pat/z9/L1", "Pat/z9/L2"]


def test_variants_repeat_with_the_same_seeds(base):
    brain = np.ones((64, 64), dtype=bool)
    donors = [(f"H/z{i}/L1", rect((64, 64), 4 + 6 * i, 54, 4, 4)) for i in range(5)]
    first, _ = variants_of("Pat/z9", base, brain, donors, VARIANT_PARAMS, 12, 4, 2, rngs(7))
    again, _ = variants_of("Pat/z9", base, brain, donors, VARIANT_PARAMS, 12, 4, 2, rngs(7))
    assert [p.params for p in first] == [p.params for p in again]
    assert all((a.labels == b.labels).all() for a, b in zip(first, again))


def test_erosion_below_the_bound_is_not_a_survivor(base):
    planted, _ = variants_of("Pat/z9", base, np.ones((64, 64), bool), [], VARIANT_PARAMS, 20, 4, 2, rngs())
    erode = next(p for p in planted if p.variant == "P5-erode")
    assert not erode.keeps_base_lesions                          # L2 6x6 -> 4x4 = 16 px < 20


def test_single_lesion_base_has_no_removal_and_no_donor_means_no_addition():
    single = attach_or_drop(rect((64, 64), 20, 20, 6, 6), 4, 2).labels
    planted, skipped = variants_of("Pat/z1", single, np.ones((64, 64), bool), [], VARIANT_PARAMS, 12, 4, 2, rngs())
    assert set(skipped) == {"P6-remove", "P6-add"}
    assert [p.variant for p in planted] == ["P1", "P2", "P3", "P4", "P5-erode", "P5-dilate"]


# --- base masks -------------------------------------------------------------------------------

def slice_table() -> pd.DataFrame:
    """T slices: M1 and M2 have spaced multi-lesion slices; Tight has 3 multi-lesion slices only at
    adjacent z (no 3 of them 2 apart); O1-O3 have single-lesion slices only; Few has 4 slices."""
    rows = []
    for patient, multi_z in (("M1", (0, 2, 4, 6)), ("M2", (1, 5, 9)), ("Tight", (10, 11, 12))):
        rows += [(patient, z, 2 if z in multi_z else 1) for z in range(14)]
    rows += [(p, z, 1) for p in ("O1", "O2", "O3") for z in range(12)]
    rows += [("Few", z, 2) for z in range(4)]
    rows += [("M1", 20, 0)]                                     # no informative lesion: never chosen
    t = pd.DataFrame(rows, columns=["patient_id", "z", "n_informative"])
    t["key"] = t["patient_id"] + "/z" + t["z"].astype(str)
    return t


def test_bases_meet_quotas_and_spacing():
    bases, _ = choose_bases(slice_table(), {**BASE_PARAMS, "patients_multi": 2}, np.random.default_rng(3))
    multi = bases[bases["group"] == "multi"]
    assert set(multi["patient_id"]) == {"M1", "M2"}                  # Tight can never give 3 spaced multi slices
    for _, rows in multi.groupby("patient_id"):
        assert (rows["n_informative"] >= 2).sum() == 3 and (rows["n_informative"] == 1).sum() == 2
    other = bases[bases["group"] == "other"]
    assert other["patient_id"].nunique() == 2 and other.groupby("patient_id").size().eq(5).all()
    assert "Few" not in set(bases["patient_id"]) and (bases["n_informative"] > 0).all()
    for _, rows in bases.groupby("patient_id"):
        z = np.sort(rows["z"].to_numpy())
        assert (np.diff(z) >= 2).all()


def test_bases_repeat_with_a_seed_and_fail_without_enough_patients():
    table = slice_table()
    first, _ = choose_bases(table, BASE_PARAMS, np.random.default_rng(11))
    assert first.equals(choose_bases(table, BASE_PARAMS, np.random.default_rng(11))[0])
    with pytest.raises(ValueError):
        choose_bases(table, {**BASE_PARAMS, "patients_multi": 3}, np.random.default_rng(0))
