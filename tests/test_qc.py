"""QC classes: precedence, the speck cut-off and the edge count, on hand-built tables."""

import pandas as pd

from memtk.qc import MASK_CLASSES, QC_CLASSES, edge_lesions, mask_class, qc_class


def test_mask_class_follows_lesions_then_speck_count():
    n_lesions = pd.Series([0, 0, 2, 1, 1, 3])
    n_specks = pd.Series([0, 50, 10, 9, 1, 0])
    assert mask_class(n_lesions, n_specks, max_specks=9).tolist() == \
        ["no_lesion", "no_lesion", "noisy", "minor", "minor", "clean"]


def test_cut_off_is_inclusive_for_minor():
    classes = mask_class(pd.Series([1, 1, 1]), pd.Series([3, 4, 0]), max_specks=3)
    assert classes.tolist() == ["minor", "noisy", "clean"]


def test_failed_mri_comes_first_but_keeps_the_mask_class():
    masks = mask_class(pd.Series([1, 0, 1, 1]), pd.Series([0, 0, 20, 2]), max_specks=9)
    failed = pd.Series([True, True, False, False])
    assert masks.tolist() == ["clean", "no_lesion", "noisy", "minor"]   # the mask alone
    assert qc_class(failed, masks).tolist() == ["mri_failed", "mri_failed", "noisy", "minor"]


def test_every_sample_gets_exactly_one_known_class():
    masks = mask_class(pd.Series([0, 1, 2, 1]), pd.Series([5, 0, 12, 4]), max_specks=9)
    classes = qc_class(pd.Series([False, False, False, True]), masks)
    assert classes.notna().all() and set(classes) <= set(QC_CLASSES) and set(masks) <= set(MASK_CLASSES)


def test_edge_lesions_counts_only_lesions_beyond_the_fraction():
    lesions = pd.DataFrame({"key": ["a", "a", "b", "c"], "frac_outside_brain": [0.6, 0.5, 1.0, 0.0]})
    counts = edge_lesions(lesions, 0.5)
    assert counts.to_dict() == {"a": 1, "b": 1}                          # 0.5 is not more than half
