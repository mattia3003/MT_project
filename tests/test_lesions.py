"""Unit tests for memtk.lesions on small hand-built images with known answers."""

import numpy as np
import pytest

from memtk.lesions import (BRAIN_METHODS, MIN_BRAIN_REGION, attach_or_drop, brain_box, brain_mask,
                           count_components, fraction_outside, label_components, lesion_features,
                           speck_gaps, tissue_px)

LOW = 0.02   # low threshold used in these tests: between background (0) and the dim rim (0.08)


def disk(shape, center, radius):
    rr, cc = np.ogrid[:shape[0], :shape[1]]
    return (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius ** 2


@pytest.fixture
def head():
    """A 64x64 'MRI': bright core (0.5), dim rim (0.08), dark hole in the core (0.0), a separate
    piece of tissue (49 px, like a lower lobe in a low slice) and a small bright blob (9 px)."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[disk(img.shape, (30, 30), 16)] = 0.08                  # dim rim
    img[disk(img.shape, (30, 30), 12)] = 0.5                   # bright core
    img[disk(img.shape, (30, 30), 3)] = 0.0                    # dark 'ventricle' inside
    img[2:9, 52:59] = 0.3                                      # separate tissue piece, 49 px
    img[58:61, 58:61] = 0.5                                    # separate blob, 9 px
    return img


# --- labelling -------------------------------------------------------------------------------

def test_diagonal_pixels_are_one_component():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = mask[2, 2] = mask[3, 3] = 1
    assert count_components(mask) == 1


def test_one_pixel_gap_separates_components():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 0] = mask[2, 2] = 1
    labels, n = label_components(mask)
    assert n == 2 and labels[2, 0] != labels[2, 2]


def test_empty_mask_has_no_components():
    assert count_components(np.zeros((8, 8))) == 0


# --- attach-or-drop --------------------------------------------------------------------------

def lesion_with_speck(speck_col):
    """A 2x2 lesion at rows 5-6, cols 5-6 (4 px) and a 1 px speck at (5, speck_col).
    Speck at col 8 = gap 1 (col 7 empty), col 9 = gap 2, col 10 = gap 3."""
    mask = np.zeros((12, 14), dtype=np.uint8)
    mask[5:7, 5:7] = 1
    mask[5, speck_col] = 1
    return mask


@pytest.mark.parametrize("speck_col, speck_gap", [(8, 1), (9, 2), (10, 3)])
def test_speck_is_attached_up_to_the_gap_and_dropped_beyond(speck_col, speck_gap):
    mask = lesion_with_speck(speck_col)
    assert speck_gaps(mask, 4) == [(1, speck_gap, 5, speck_col)]
    for gap in (0, 1, 2, 3):
        res = attach_or_drop(mask, min_area=4, gap=gap)
        attached = speck_gap <= gap
        assert res.n_lesions == 1
        assert res.n_strays == int(attached) and res.n_specks == int(not attached)
        assert res.labels[5, speck_col] == (1 if attached else 0)
        assert res.dropped[5, speck_col] == (not attached)


def test_gap_zero_is_drop_only():
    res = attach_or_drop(lesion_with_speck(8), min_area=4, gap=0)
    assert res.n_strays == 0 and res.n_specks == 1 and res.labels.sum() == 4


def test_nothing_is_moved_or_deleted():
    rng = np.random.default_rng(0)
    mask = (rng.random((40, 40)) > 0.85).astype(np.uint8)    # many small components
    mask[10:16, 10:16] = 1
    res = attach_or_drop(mask, min_area=4, gap=2)
    assert np.array_equal((res.labels > 0) | res.dropped, mask > 0)
    assert not ((res.labels > 0) & res.dropped).any()
    assert not (res.attached & ~(res.labels > 0)).any()       # strays carry a lesion label


def test_two_lesions_one_pixel_apart_stay_separate():
    mask = np.zeros((6, 10), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    mask[2:4, 5:7] = 1                                        # gap 1 (col 4 empty)
    res = attach_or_drop(mask, min_area=4, gap=2)
    assert res.n_lesions == 2 and set(np.unique(res.labels)) == {0, 1, 2}


def test_stray_joins_the_nearest_lesion_and_ties_go_to_the_larger():
    mask = np.zeros((10, 20), dtype=np.uint8)
    mask[4:6, 2:4] = 1                       # small lesion, 4 px, cols 2-3
    mask[3:7, 12:16] = 1                     # large lesion, 16 px, cols 12-15
    mask[4, 6] = 1                           # speck: gap 2 to small, gap 5 to large
    mask[4, 8] = 1                           # speck: gap 4 to small, gap 3 to large
    res = attach_or_drop(mask, min_area=4, gap=3)
    big, small = res.labels[4, 13], res.labels[4, 2]
    assert res.labels[4, 6] == small and res.labels[4, 8] == big
    tie = np.zeros((10, 20), dtype=np.uint8)
    tie[4:6, 2:4] = 1                        # 4 px, cols 2-3
    tie[3:7, 7:11] = 1                       # 16 px, cols 7-10
    tie[4, 5] = 1                            # gap 1 to both
    res = attach_or_drop(tie, min_area=4, gap=1)
    assert res.labels[4, 5] == res.labels[4, 8]


def test_lesions_are_numbered_by_area_with_strays_then_top_left():
    mask = np.zeros((12, 20), dtype=np.uint8)
    mask[1:3, 12:15] = 1                     # 6 px, top right
    mask[8:10, 1:4] = 1                      # 6 px, bottom left
    mask[5:7, 1:3] = 1                       # 4 px
    mask[5, 4] = 1                           # stray (gap 1): the 4 px lesion becomes 5 px
    res = attach_or_drop(mask, min_area=4, gap=1)
    assert res.labels[1, 12] == 1            # equal areas: the top-left first pixel wins
    assert res.labels[8, 1] == 2
    assert res.labels[5, 1] == 3 and res.labels[5, 4] == 3


def test_mask_without_lesion():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = mask[5, 5] = 1
    res = attach_or_drop(mask, min_area=4, gap=2)
    assert res.n_lesions == 0 and res.n_specks == 2 and not res.labels.any()
    assert speck_gaps(mask, 4) == [(1, -1, 1, 1), (1, -1, 5, 5)]
    empty = attach_or_drop(np.zeros((8, 8)), min_area=4, gap=2)
    assert empty.n_lesions == empty.n_specks == empty.n_strays == 0


# --- lesion features -------------------------------------------------------------------------

def test_square_lesion_features_in_all_three_frames():
    mask = np.zeros((20, 40), dtype=np.uint8)
    mask[4:8, 10:14] = 1                     # 4x4 square: centroid (5.5, 11.5)
    mask[4, 15] = 1                          # stray at gap 1: centroid moves, area 17
    brain = np.zeros((20, 40), dtype=bool)
    brain[2:12, 10:30] = True                # box rows 2-11, cols 10-29; the stray is inside
    res = attach_or_drop(mask, min_area=4, gap=2)
    (row,) = lesion_features(res, brain, brain_box(brain))
    assert row["label"] == 1 and row["area"] == 17 and row["stray_px"] == 1
    r = (16 * 5.5 + 4) / 17
    c = (16 * 11.5 + 15) / 17
    assert row["centroid_row"] == pytest.approx(r) and row["centroid_col"] == pytest.approx(c)
    assert row["centroid_row_img"] == pytest.approx(r / 20) and row["centroid_col_img"] == pytest.approx(c / 40)
    assert row["centroid_row_brain"] == pytest.approx((r - 2) / 9)
    assert row["centroid_col_brain"] == pytest.approx((c - 10) / 19)
    assert (row["bbox_r0"], row["bbox_r1"], row["bbox_c0"], row["bbox_c1"]) == (4, 7, 10, 15)
    assert row["frac_outside_brain"] == 0.0


def test_lesion_features_without_brain_and_partly_outside():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:4, 2:6] = 1                       # 8 px
    res = attach_or_drop(mask, min_area=4, gap=2)
    (row,) = lesion_features(res, np.zeros((10, 10), dtype=bool), None)
    assert np.isnan(row["centroid_row_brain"]) and np.isnan(row["centroid_col_brain"])
    assert row["frac_outside_brain"] == 1.0
    brain = np.zeros((10, 10), dtype=bool)
    brain[:, :4] = True                      # left half of the lesion inside
    (row,) = lesion_features(res, brain, brain_box(brain))
    assert row["frac_outside_brain"] == 0.5


def test_lesion_features_follow_label_order():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[1:3, 1:3] = 1                       # 4 px -> L2
    mask[6:9, 6:9] = 1                       # 9 px -> L1
    res = attach_or_drop(mask, min_area=4, gap=2)
    rows = lesion_features(res, np.ones((12, 12), dtype=bool), (0, 11, 0, 11))
    assert [r["label"] for r in rows] == [1, 2] and [r["area"] for r in rows] == [9, 4]


# --- brain masks -----------------------------------------------------------------------------

def test_low_threshold_keeps_dim_rim_and_contains_otsu(head):
    # Whether Otsu keeps the dim rim depends on the histogram, so only properties that hold
    # either way are tested: the low threshold keeps all tissue, Otsu never reaches beyond it.
    core = disk(head.shape, (30, 30), 12)
    rim = disk(head.shape, (30, 30), 16)
    otsu = brain_mask(head, "otsu", LOW)
    low = brain_mask(head, "low_threshold", LOW)
    assert np.array_equal(low, rim)          # rim included, hole filled, blob left out
    assert otsu[core].all()                  # the bright core is always brain
    assert not (otsu & ~low).any()           # Otsu brain lies within the low-threshold brain


def test_low_threshold_value_is_used(head):
    # a threshold above the dim rim (0.08) leaves only the bright core, hole filled
    assert np.array_equal(brain_mask(head, "low_threshold", 0.1), disk(head.shape, (30, 30), 12))
    assert np.array_equal(brain_mask(head, "otsu", 0.1), brain_mask(head, "otsu", LOW))   # Otsu ignores it


def test_dilation_grows_the_otsu_brain(head):
    sizes = [brain_mask(head, m, LOW).sum() for m in ("otsu", "otsu_dilate1", "otsu_dilate2")]
    assert sizes[0] < sizes[1] < sizes[2]
    assert brain_mask(head, "otsu_dilate1", LOW)[brain_mask(head, "otsu", LOW)].all()   # only grows


def test_holes_are_filled(head):
    assert brain_mask(head, "otsu", LOW)[30, 30]


def test_small_blobs_are_never_brain(head):
    assert 9 < MIN_BRAIN_REGION
    for method in BRAIN_METHODS:
        assert not brain_mask(head, method, LOW)[59, 59]


def test_only_all_regions_method_keeps_separate_tissue_piece(head):
    for method, (_, _, keep_all) in BRAIN_METHODS.items():
        assert brain_mask(head, method, LOW)[5, 55] == keep_all


def test_tissue_px_counts_all_tissue(head):
    rim_area = int(disk(head.shape, (30, 30), 16).sum())
    assert tissue_px(head, LOW) == rim_area + 49
    assert tissue_px(np.zeros((16, 16)), LOW) == 0


@pytest.mark.parametrize("method", sorted(BRAIN_METHODS))
def test_empty_and_constant_images_give_empty_brain(method):
    assert not brain_mask(np.zeros((16, 16)), method, LOW).any()
    assert not brain_mask(np.full((16, 16), 0.01), method, LOW).any()


def test_unknown_method_raises(head):
    with pytest.raises(ValueError):
        brain_mask(head, "magic", LOW)


def test_brain_box_is_inclusive():
    brain = np.zeros((10, 10), dtype=bool)
    brain[2:5, 3:8] = True
    assert brain_box(brain) == (2, 4, 3, 7)
    assert brain_box(np.zeros((10, 10), dtype=bool)) is None


def test_fraction_outside():
    brain = np.zeros((4, 4), dtype=bool)
    brain[:, :2] = True
    component = np.zeros((4, 4), dtype=bool)
    component[0, :4] = True                  # 2 of 4 pixels inside
    assert fraction_outside(component, brain) == 0.5
    assert fraction_outside(np.zeros((4, 4), dtype=bool), brain) == 0.0
