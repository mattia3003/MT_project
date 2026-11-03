"""The sampling script's filters, as checked by memtk.e0b.filters.filter_checks, on hand-built masks."""

import numpy as np

from memtk.e0b.filters import filter_checks
from memtk.lesions import attach_or_drop


def checks(mask, brain, max_px=2000, small_px=40):
    return filter_checks(mask, attach_or_drop(mask, min_area=4, gap=2).labels, brain, max_px, small_px)


def test_a_speck_counts_as_small_lesion_only_in_the_mask_as_saved():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:18, 10:18] = 1                   # 64 px lesion
    mask[40, 40] = 1                         # isolated 1 px speck
    c = checks(mask, np.ones((60, 60), dtype=bool))
    assert c["raw_small_lesion"] and not c["clean_small_lesion"]


def test_inside_brain_and_pixel_sum_in_both_versions():
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:16, 10:16] = 1                   # 36 px lesion, inside
    mask[50, 55] = 1                         # speck outside the brain
    brain = np.zeros((60, 60), dtype=bool)
    brain[:40, :40] = True
    c = checks(mask, brain, max_px=37)
    assert not c["raw_inside_brain"] and c["clean_inside_brain"]       # only the speck is outside
    assert not c["raw_max_mask_px"] and c["clean_max_mask_px"]         # 37 px raw, 36 px cleaned
    assert c["raw_small_lesion"] and c["clean_small_lesion"]           # 36 < 40


def test_mask_without_lesion_fails_small_lesion_after_cleaning():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5, 5] = 1
    c = checks(mask, np.ones((20, 20), dtype=bool))
    assert c["raw_small_lesion"] and not c["clean_small_lesion"] and c["clean_inside_brain"]
