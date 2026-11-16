"""Unit tests for memtk.paired (E8 decisions AB, AC) on small hand-built inputs: SSIM of identical
images, invariance under a*x + b after z-scoring, realignment of a shifted image and mask (shift
sign), zero-filled alignment, the lesion region and the three regions, and the matched control draw."""

import numpy as np
import pandas as pd
import pytest

from memtk.paired import (align, best_match, lesion_region, match_controls, region_means, regions, size_cell,
                          ssim_map, zscore_in_brain)

RNG = np.random.default_rng(0)


def brain_image(shape=(48, 48)) -> tuple[np.ndarray, np.ndarray]:
    """A textured disc on a black background, and its brain mask."""
    rr, cc = np.mgrid[:shape[0], :shape[1]]
    brain = (rr - 24) ** 2 + (cc - 24) ** 2 <= 18 ** 2
    image = np.where(brain, 0.4 + 0.2 * RNG.random(shape), 0.0)
    return image, brain


def test_identical_images_give_ssim_one_in_any_region():
    image, brain = brain_image()
    z = zscore_in_brain(image, brain)
    smap = ssim_map(z, z, data_range=8.0, sigma=1.5)
    region = np.zeros_like(brain)
    region[20:28, 20:28] = True
    means = region_means(smap, {"brain": brain, "small": region})
    assert means["ssim_brain"] == pytest.approx(1.0) and means["ssim_small"] == pytest.approx(1.0)
    assert means["px_small"] == 64


def test_zscore_makes_ssim_invariant_to_affine_intensity():
    image, brain = brain_image()
    other = np.where(brain, image + 0.05 * RNG.random(image.shape), 0.0)
    z = zscore_in_brain(image, brain)
    assert z[brain].mean() == pytest.approx(0.0) and z[brain].std() == pytest.approx(1.0)
    base = ssim_map(z, zscore_in_brain(other, brain), 8.0, 1.5)
    scaled = ssim_map(z, zscore_in_brain(3.0 * other + 0.7, brain), 8.0, 1.5)
    assert scaled == pytest.approx(base)
    with pytest.raises(ValueError):
        zscore_in_brain(image, np.zeros_like(brain))


def test_shifted_image_and_mask_are_realigned():
    image, brain = brain_image()
    mask = np.zeros_like(brain)
    mask[20:25, 26:30] = True
    dr, dc = 2, -3
    moved_img, moved_mask = align(image, dr, dc), align(mask, dr, dc)
    other = np.zeros_like(mask)
    other[5:8, 5:8] = True
    i, score, sr, sc = best_match(moved_mask, np.stack([other, mask]), max_shift=3)
    assert (i, score, sr, sc) == (1, 1.0, -dr, -dc)        # the query is moved back onto the target
    assert np.array_equal(align(moved_img, sr, sc)[brain], image[brain])
    assert ssim_map(align(moved_img, sr, sc), image, 1.0, 1.5)[10:38, 10:38] == pytest.approx(1.0)


def test_best_match_prefers_higher_plain_dice_then_first_candidate():
    q = np.zeros((16, 16), dtype=bool)
    q[5:9, 5:9] = True
    same, moved = q.copy(), align(q, 1, 0)
    assert best_match(q, np.stack([moved, same]), 2)[0] == 1       # both reach 1.0 by shift; plain Dice decides
    assert best_match(q, np.stack([same, same]), 2)[0] == 0


def test_align_fills_with_zeros_and_loses_pixels_moved_off():
    a = np.arange(1, 17).reshape(4, 4)
    out = align(a, 1, -2)
    assert out[0].tolist() == [0, 0, 0, 0]
    assert out[1].tolist() == [3, 4, 0, 0]
    assert out.sum() == a[:3, 2:].sum()
    assert np.array_equal(align(a, 0, 0), a)


def test_lesion_region_grows_informative_lesions_only():
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[10:14, 10:14] = 1              # 16 px: informative at bound 12
    labels[25, 25] = 2                    # 1 px: not informative
    region = lesion_region(labels, bound=12, grow_px=8)
    rows, cols = np.nonzero(region)
    assert (rows.min(), rows.max(), cols.min(), cols.max()) == (2, 21, 2, 21)
    assert not region[25, 25]             # the 1 px lesion does not define a region
    assert not lesion_region(labels == 2, bound=12, grow_px=8).any()


def test_regions_stay_inside_both_brains_and_ring_drops_lesion_pixels():
    brain_q = np.zeros((20, 20), dtype=bool)
    brain_q[2:18, 2:18] = True
    brain_t = np.zeros_like(brain_q)
    brain_t[2:18, 6:18] = True
    les_q, les_t = np.zeros_like(brain_q), np.zeros_like(brain_q)
    les_q[8:10, 8:10] = True
    les_t[9:11, 9:11] = True
    region = np.ones_like(brain_q)
    r = regions(region, les_q, les_t, brain_q, brain_t)
    assert np.array_equal(r["brain"], brain_q & brain_t)
    assert not (r["lesion"] & ~(brain_q & brain_t)).any()
    assert not (r["ring"] & (les_q | les_t)).any()
    assert r["ring"].sum() == r["lesion"].sum() - (les_q | les_t).sum()


def test_match_controls_respects_cells_k_and_seed_without_reuse():
    cases = pd.DataFrame({"key": ["a1", "a2", "a3"], "cell": ["x", "x", "y"]})
    pool = pd.DataFrame({"key": [f"b{i}" for i in range(12)], "cell": ["x"] * 8 + ["y"] * 2 + ["z"] * 2})
    pairs = match_controls(cases, pool, k=5, rng=np.random.default_rng(3))
    assert not pairs["control"].duplicated().any()
    cell_of = dict(zip(pool["key"], pool["cell"]))
    assert all(cell_of[c] == cell for c, cell in zip(pairs["control"], pairs["cell"]))
    assert pairs.groupby("case").size().to_dict() == {"a1": 5, "a2": 3, "a3": 2}   # x has 8, y has 2
    again = match_controls(cases, pool, k=5, rng=np.random.default_rng(3))
    assert again.equals(pairs)


def test_size_cell_uses_lower_edges():
    assert size_cell([5, 12, 19, 20, 79, 80, 500], [12, 20, 40, 80]).tolist() == [0, 1, 1, 2, 3, 4, 4]
