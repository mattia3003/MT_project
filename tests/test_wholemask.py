"""Unit tests for memtk.wholemask (E1) on small hand-built label maps with known answers."""

import numpy as np
import pytest

from memtk.wholemask import (dice_matrix, largest_lesion_centroids, match, shift_order, shifted_dice)

SIZE = 32


def labels(*squares):
    """A label map with one square lesion per (row, col, side), numbered 1, 2, ... in order
    (list the largest first, as in real label maps where L1 is the largest)."""
    lab = np.zeros((SIZE, SIZE), dtype=np.int32)
    for n, (r, c, side) in enumerate(squares, start=1):
        lab[r:r + side, c:c + side] = n
    return lab


def run(query, refs, patients, top_k_dice=20, top_k_centroid=20, max_shift=3):
    return match(np.stack([query]), np.stack(refs), patients, ["q"], [f"t{i}" for i in range(len(refs))],
                 top_k_dice, top_k_centroid, max_shift).iloc[0]


def test_dice_matrix_known_values():
    a, b, c = labels((0, 0, 4)) > 0, labels((0, 2, 4)) > 0, labels((20, 20, 4)) > 0
    d = dice_matrix(np.stack([a]), np.stack([a, b, c]))
    assert d[0].tolist() == pytest.approx([1.0, 0.5, 0.0])   # b overlaps a in 8 of 16 px


def test_largest_lesion_centroid_is_that_of_label_1():
    lab = labels((2, 2, 4), (20, 20, 2))
    assert largest_lesion_centroids(np.stack([lab, np.zeros_like(lab)])).tolist()[0] == [3.5, 3.5]
    assert np.isnan(largest_lesion_centroids(np.stack([np.zeros_like(lab)]))).all()


def test_shift_order_starts_with_no_move_and_covers_the_square():
    order = shift_order(3)
    assert order[0] == (0, 0) and len(order) == 49 and len(set(order)) == 49


def test_shifted_exact_copy_is_found_with_its_shift():
    train = labels((10, 10, 3), (20, 5, 2))
    query = np.roll(train, (2, -1), axis=(0, 1))   # the training mask moved down 2, left 1
    row = run(query, [train], ["A"])
    assert row["score"] == pytest.approx(1.0)
    assert (row["shift_row"], row["shift_col"]) == (-2, 1)   # moving the query back
    assert row["plain_dice"] < 1.0


def test_no_wrap_around_at_the_border():
    query = labels((10, 0, 2)) > 0                     # touches the left edge
    train = labels((10, SIZE - 2, 2)) > 0              # touches the right edge
    assert shifted_dice(query, np.stack([train]), 3).max() == 0.0


def test_shift_beyond_the_range_is_not_found():
    train = labels((10, 10, 2))
    query = np.roll(train, 5, axis=1)
    assert run(query, [train], ["A"], max_shift=3)["score"] == 0.0


def test_centroid_candidates_catch_a_shifted_copy_with_no_plain_overlap():
    source = labels((10, 10, 2))
    query = np.roll(source, 3, axis=1)                 # no plain overlap with its source
    decoy = labels((25, 25, 6), (10, 13, 2))           # its L2 overlaps the query exactly; its L1 is far away
    row = run(query, [decoy, source], ["A", "B"], top_k_dice=1, top_k_centroid=1)
    assert row["n_candidates"] == 2
    assert row["nn"] == "t1" and row["score"] == pytest.approx(1.0)


def test_margin_is_measured_against_another_patient():
    query = labels((10, 10, 4))
    same = labels((10, 10, 4))
    same_patient_near = labels((10, 11, 4))
    other = labels((25, 25, 4), (10, 12, 4))           # another patient, partial match
    row = run(query, [same, same_patient_near, other], ["A", "A", "B"], max_shift=0)
    assert (row["nn"], row["nn_patient"], row["runner_up_patient"]) == ("t0", "A", "B")
    other_score = 2 * 8 / (16 + 32)
    assert row["runner_up_score"] == pytest.approx(other_score)
    assert row["margin"] == pytest.approx(1.0 - other_score)


def test_ties_report_the_smallest_shift():
    query = labels((15, 15, 1))                        # one pixel inside a large square:
    train = labels((10, 10, 12))                       # every shift within ±3 scores the same
    row = run(query, [train], ["A"])
    assert (row["shift_row"], row["shift_col"]) == (0, 0)


def test_masks_without_a_lesion_are_refused():
    lab = labels((10, 10, 2))
    with pytest.raises(ValueError, match="query"):
        run(np.zeros_like(lab), [lab], ["A"])
    with pytest.raises(ValueError, match="reference"):
        run(lab, [lab, np.zeros_like(lab)], ["A", "B"])
