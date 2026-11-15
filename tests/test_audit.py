"""Unit tests for memtk.audit (E5) on small constructed inputs."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from memtk.audit import (auc, auc_bootstrap, beyond_max, cell_maxima, cell_rates, chain_groups, copied_area_share,
                         excess_test, holm, interval, one_sided_p, resample_sums, standardized_rate)

EDGES = (4, 12, 20)


def test_chain_groups_join_runs_within_the_gap_and_keep_patients_apart():
    patients = ["A", "A", "A", "A", "A", "B", "B"]
    z = [7, 1, 4, 2, 8, 2, 3]
    groups = chain_groups(patients, z, max_gap=2)
    assert groups.tolist() == [1, 0, 0, 0, 1, 2, 2]          # A: {1, 2, 4}, {7, 8}; B never joins A
    assert chain_groups(["A", "A"], [5, 5], 2).tolist() == [0, 0]   # the same slice twice: one group
    assert chain_groups(["A", "A"], [1, 4], 2).tolist() == [0, 1]   # gap 3 > 2


def test_holm_against_known_values():
    adjusted = holm([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 = 0.02, 0.03, 0.06, 0.04 -> monotone 0.02, 0.03, 0.06, 0.06
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.5, 0.9]).max() <= 1.0


def test_auc_equals_mann_whitney_u_with_ties():
    s, h = np.array([0.2, 0.5, 0.5, 0.9]), np.array([0.1, 0.5, 0.7])
    u = mannwhitneyu(s, h, alternative="greater").statistic
    assert auc(s, h) == pytest.approx(u / (len(s) * len(h)))
    assert auc([1.0], [0.0]) == 1.0 and auc([0.5], [0.5]) == 0.5


def test_resample_sums_draws_whole_clusters():
    flags = np.array([1, 1, 0, 0, 0])
    values = np.c_[flags, np.ones(5)]
    boot = resample_sums(values, ["A", "A", "B", "B", "B"], 500, np.random.default_rng(0))
    pairs = {tuple(r) for r in boot}
    assert pairs <= {(4.0, 4.0), (2.0, 5.0), (0.0, 6.0)}      # AA, AB, BB
    assert len(pairs) == 3
    again = resample_sums(values, ["A", "A", "B", "B", "B"], 500, np.random.default_rng(0))
    assert np.array_equal(boot, again)


def test_resample_sums_with_one_cluster_per_row_is_an_ordinary_bootstrap():
    flags = np.r_[np.ones(300), np.zeros(700)]
    boot = resample_sums(np.c_[flags, np.ones(1000)], np.arange(1000), 2000, np.random.default_rng(1))
    shares = boot[:, 0] / boot[:, 1]
    assert (boot[:, 1] == 1000).all()
    assert shares.mean() == pytest.approx(0.3, abs=0.005)
    assert shares.std() == pytest.approx(np.sqrt(0.3 * 0.7 / 1000), rel=0.1)


def test_excess_test_finds_no_excess_for_equal_samples_and_one_for_a_clear_shift():
    rng = np.random.default_rng(2)
    h = np.tile([True] + [False] * 9, 26)                    # 10% in each of 26 patients
    patients = np.repeat(np.arange(26), 10)
    same = excess_test(np.tile([True] + [False] * 9, 200), np.arange(2000), h, patients, 1000, rng)
    assert same["excess"] == pytest.approx(0.0)
    assert same["excess_low"] <= 0 <= same["excess_high"] and same["p"] > 0.2
    shifted = excess_test(np.tile([True, False], 1000), np.arange(2000), h, patients, 1000, rng)
    assert shifted["excess"] == pytest.approx(0.4)
    assert shifted["excess_low"] > 0 and shifted["p"] < 0.01
    assert shifted["n_s"] == 2000 and shifted["flagged_h"] == 26


def test_interval_and_one_sided_p():
    boot = np.arange(-10, 90, dtype=float)                  # 10 of 100 draws below 0, one at 0
    assert one_sided_p(boot) == pytest.approx(12 / 101)
    low, high = interval(boot)
    assert low < 0 < high


def test_auc_bootstrap_centres_on_the_point_estimate():
    rng = np.random.default_rng(3)
    s = rng.normal(0.5, 1, 400)
    h = rng.normal(0.0, 1, 200)
    patients = np.repeat(np.arange(20), 10)
    boot = auc_bootstrap(s, np.arange(400), h, patients, 1000, np.random.default_rng(4))
    point = auc(s, h)
    low, high = interval(boot)
    assert low < point < high
    assert boot.mean() == pytest.approx(point, abs=0.02)


def test_standardized_rate_reweights_groups_and_skips_empty_ones():
    assert standardized_rate(np.array([1, 9]), np.array([10, 10]), np.array([0.5, 0.5])) == pytest.approx(0.5)
    assert standardized_rate(np.array([1, 9]), np.array([10, 10]), np.array([1.0, 0.0])) == pytest.approx(0.1)
    assert standardized_rate(np.array([1, 0]), np.array([10, 0]), np.array([0.5, 0.5])) == pytest.approx(0.1)
    boot = standardized_rate(np.array([[1, 9], [2, 8]]), np.array([[10, 10], [10, 10]]), np.array([0.5, 0.5]))
    assert boot == pytest.approx([0.5, 0.5])


def test_cell_rates_equal_the_crude_rate_with_own_weights_and_reweight_otherwise():
    flags = np.array([[1, 0], [0, 0], [1, 1], [1, 1], [0, 0], [0, 1]], dtype=bool)   # two flag columns
    cells = np.array(["a", "a", "b", "b", "b", "b"])
    clusters = np.arange(6)
    own = np.array([2, 4]) / 6
    point, boot = cell_rates(flags, cells, clusters, ["a", "b"], own, 300, np.random.default_rng(0))
    assert point == pytest.approx(flags.mean(axis=0))
    assert boot.shape == (300, 2) and boot.mean(axis=0) == pytest.approx(flags.mean(axis=0), abs=0.05)
    other, _ = cell_rates(flags, cells, clusters, ["a", "b"], np.array([1.0, 0.0]), 10, np.random.default_rng(0))
    assert other == pytest.approx([0.5, 0.0])                   # only cell a: 1 of 2, 0 of 2


def test_copied_area_share_counts_kept_pairs_on_the_source_slice_only():
    pairs = pd.DataFrame({
        "query": ["q", "q", "q", "r"],
        "train": ["P/z5", "P/z5", "P/z9", "P/z5"],
        "query_lesion": ["q/L1", "q/L2", "q/L1", "r/L1"],
        "train_lesion": ["P/z5/L1", "P/z5/L2", "P/z9/L1", "P/z5/L1"],
        "s": [0.9, 0.5, 0.99, 0.95]})
    query_area = pd.Series({"q/L1": 20, "q/L2": 10, "r/L1": 30})
    train_area = pd.Series({"P/z5/L1": 20, "P/z5/L2": 10, "P/z5/L3": 10, "P/z9/L1": 25})
    sources = pd.Series({"q": "P/z5", "r": "P/z5", "u": ""})
    share = copied_area_share(pairs, sources, pd.Series({"q": 0.8, "r": 0.8, "u": 0.8}), query_area, train_area)
    assert share["q"] == pytest.approx((20 + 20) / (30 + 40))   # L2 pair below tau_L; z9 is not the source
    assert share["r"] == pytest.approx((30 + 20) / (30 + 40))
    assert np.isnan(share["u"])                                   # no source


def test_cell_maxima_and_beyond_max_at_bin_edges():
    maxima = cell_maxima(np.array([0.95, 0.97, 0.80, 0.85]), np.array(["single", "single", "single", "multi"]),
                         np.array([5, 11, 12, 30]), EDGES)
    cells = maxima.set_index(["stratum", "lower"])["max"]
    assert cells[("single", 4)] == 0.97 and cells[("single", 12)] == 0.80 and cells[("single", -1)] == 0.97
    scores = np.array([0.97, 0.98, 0.81, 0.81, 0.90, 0.86])
    strata = np.array(["single", "single", "single", "single", "single", "multi"])
    sizes = np.array([11, 11, 12, 19, 20, 50])
    # equal to the maximum is not beyond it; 12 px opens the next bin; no held-out single slice >= 20 px: whole stratum
    assert beyond_max(scores, strata, sizes, maxima, EDGES).tolist() == [False, True, True, True, False, True]
    assert not beyond_max(np.array([1.0]), np.array(["none"]), np.array([30]), maxima, EDGES)[0]  # no held-out stratum
