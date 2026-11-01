"""Unit tests for memtk.lesions on small hand-built images with known answers."""

import numpy as np
import pytest

from memtk.lesions import (BRAIN_METHODS, MIN_BRAIN_REGION, brain_box, brain_mask, count_components,
                           fraction_outside, label_components, tissue_px)


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


# --- brain masks -----------------------------------------------------------------------------

def test_low_threshold_keeps_dim_rim_and_contains_otsu(head):
    # Whether Otsu keeps the dim rim depends on the histogram, so only properties that hold
    # either way are tested: the low threshold keeps all tissue, Otsu never reaches beyond it.
    core = disk(head.shape, (30, 30), 12)
    rim = disk(head.shape, (30, 30), 16)
    otsu = brain_mask(head, "otsu")
    low = brain_mask(head, "low_threshold")
    assert np.array_equal(low, rim)          # rim included, hole filled, blob left out
    assert otsu[core].all()                  # the bright core is always brain
    assert not (otsu & ~low).any()           # Otsu brain lies within the low-threshold brain


def test_dilation_grows_the_otsu_brain(head):
    sizes = [brain_mask(head, m).sum() for m in ("otsu", "otsu_dilate1", "otsu_dilate2")]
    assert sizes[0] < sizes[1] < sizes[2]
    assert brain_mask(head, "otsu_dilate1")[brain_mask(head, "otsu")].all()   # only grows


def test_holes_are_filled(head):
    assert brain_mask(head, "otsu")[30, 30]


def test_small_blobs_are_never_brain(head):
    assert 9 < MIN_BRAIN_REGION
    for method in BRAIN_METHODS:
        assert not brain_mask(head, method)[59, 59]


def test_only_all_regions_method_keeps_separate_tissue_piece(head):
    for method, (_, _, keep_all) in BRAIN_METHODS.items():
        assert brain_mask(head, method)[5, 55] == keep_all


def test_tissue_px_counts_all_tissue(head):
    rim_area = int(disk(head.shape, (30, 30), 16).sum())
    assert tissue_px(head) == rim_area + 49
    assert tissue_px(np.zeros((16, 16))) == 0


@pytest.mark.parametrize("method", sorted(BRAIN_METHODS))
def test_empty_and_constant_images_give_empty_brain(method):
    assert not brain_mask(np.zeros((16, 16)), method).any()
    assert not brain_mask(np.full((16, 16), 0.01), method).any()


def test_unknown_method_raises(head):
    with pytest.raises(ValueError):
        brain_mask(head, "magic")


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
