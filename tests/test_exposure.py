"""Unit tests for memtk.exposure (E6 decisions R, U, V, W) on small hand-built inputs: smoothed
expected shares, exposure ratios equal to 1 when flags follow pi, the concentration test, E2 margins,
slice rarity against other patients only, patient rarity, Spearman, lesion credits and their
bootstrap, and the H1 table."""

import numpy as np
import pandas as pd
import pytest

from memtk.exposure import (concentration_test, credit_counts, credit_draws, e2_margins, expected_shares,
                            exposure_table, h1_table, knn_rarity, lesion_credits, rowwise_spearman, slice_rarity,
                            span_rates, spearman_test)
from memtk.lesionmatch import LesionMatch
from memtk.sets import SliceSet

WEIGHTS = pd.Series([0.4, 0.3, 0.2, 0.1], index=["A", "B", "C", "D"])


def test_expected_shares_are_smoothed_toward_the_weights():
    pi = expected_shares(pd.Series({"A": 6, "B": 4}), WEIGHTS, m=10.0)
    assert pi.sum() == pytest.approx(1.0) and (pi > 0).all()
    assert pi["D"] == pytest.approx(10 * 0.1 / 20)                  # no held-out NN: the prior share only
    assert expected_shares(pd.Series({"A": 6, "B": 4}), WEIGHTS, m=0.0)["A"] == pytest.approx(0.6)
    with pytest.raises(ValueError):
        expected_shares(pd.Series({"Z": 1}), WEIGHTS, m=1.0)


def test_exposure_ratio_is_one_when_flags_follow_pi():
    nn = ["A"] * 4 + ["B"] * 3 + ["C"] * 2 + ["D"] * 1
    sources = ["A"] * 8 + ["B"] * 6 + ["C"] * 4 + ["D"] * 2
    # held-out counts proportional to the weights: pi = w for any m
    table, ratio_draws, pi_draws = exposure_table(sources, [f"s{i}" for i in range(len(sources))], nn,
                                                  [f"h{i}" for i in range(len(nn))], WEIGHTS, m=5.0, draws=500,
                                                  rng=np.random.default_rng(0))
    assert table["ratio"].to_numpy() == pytest.approx(np.ones(4))
    assert (table["ratio_low"] <= 1).all() and (table["ratio_high"] >= 1).all()
    assert ratio_draws.shape == pi_draws.shape == (500, 4)
    assert pi_draws.sum(axis=1) == pytest.approx(np.ones(500))


def test_concentration_finds_flags_piled_on_one_patient():
    pi = WEIGHTS.to_numpy()
    pi_draws = np.broadcast_to(pi, (999, 4))
    even, _ = concentration_test(pi * 100, pi, pi_draws, top_k=1, rng=np.random.default_rng(1))
    assert even["dispersion"] == pytest.approx(0.0) and even["dispersion_p"] > 0.5
    piled, band = concentration_test(np.array([0, 0, 0, 100.0]), pi, pi_draws, top_k=1, rng=np.random.default_rng(1))
    assert piled["dispersion_p"] == pytest.approx(1 / 1000) and piled["top1_share"] == 1.0
    assert piled["above_null_band"] == 1 and piled["above_null_band_expected"] < 0.5
    assert (band["null_low"] <= 1).all() and (band["null_high"] >= 1).all()


def test_e2_margin_is_the_gap_to_the_best_other_patient():
    slice_pairs = pd.DataFrame({"query": ["q", "q", "q", "r"], "train": ["P/z1", "Q/z2", "P/z3", "P/z1"],
                                "train_idx": [0, 1, 2, 0], "n_query": 1, "n_train": 1, "n_matched": 1,
                                "best_pair_s": [0.9, 0.6, 0.8, 0.5]})
    lesion_pairs = slice_pairs[["query", "train"]].assign(
        query_lesion="x", train_lesion="y", distance=0.0, shape=1.0, position=1.0, size=1.0,
        s=slice_pairs["best_pair_s"])
    e2 = LesionMatch(pd.DataFrame({"query": ["q", "r"], "n_informative": [1, 1]}), pd.DataFrame(), slice_pairs,
                     lesion_pairs, 0.1)
    margins = e2_margins(e2)
    assert margins["q"] == pytest.approx(0.9 - 0.6)   # P's best slice against Q's, not P's second slice
    assert margins["r"] == pytest.approx(0.5)         # one patient only


def blob(row: int, col: int, size: int = 4) -> np.ndarray:
    lab = np.zeros((32, 32), dtype=np.int32)
    lab[row:row + size, col:col + size] = 1
    return lab


def test_slice_rarity_ignores_the_patients_own_slices():
    # A has the same lesion on two slices; B has it 2 px away; C has a distant lesion. (2 px, not 1:
    # E2 centres crops on the rounded centroid, and a 4 px lesion moved by 1 px rounds differently)
    labels = np.stack([blob(10, 10), blob(10, 10), blob(10, 12), blob(25, 25)])  # pyright: ignore[reportCallIssue]  # numpy stub: np.stack overloads mis-resolve here
    t = SliceSet("T", ["A/z1", "A/z2", "B/z1", "C/z1"], np.array(["A", "A", "B", "C"]), labels, {})
    wm = {"top_k_dice": 5, "top_k_centroid": 5, "max_shift": 2}
    les = {"min_area": 4, "sigma_px": 6.4, "prefilter_px": 25.6, "candidate_min_s": 0.1}
    r = slice_rarity(t, wm, les).set_index("key").to_dict("index")
    assert r["A/z1"]["rarity_e1"] == pytest.approx(0.0)          # B's lesion after a 2 px shift
    assert r["A/z1"]["e1_best_other_slice"] == "B/z1"
    assert 0 < r["A/z1"]["rarity_e2"] < 0.1                        # same shape, 2 px apart
    assert r["C/z1"]["rarity_e2"] == pytest.approx(1.0)            # no other slice is a candidate


def test_knn_rarity_ranks_the_outlier_highest():
    f = pd.DataFrame({"n": [1, 1, 2, 2, 30], "spread": [0.1, 0.12, 0.11, 0.1, 0.5]}, index=list("abcde"))
    r = knn_rarity(f, ["n", "spread"], ["n"], k=2)
    assert r.idxmax() == "e" and (r > 0).all()


def test_rowwise_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(3)
    x, y = rng.integers(0, 5, 40).astype(float), rng.integers(0, 5, 40).astype(float)
    assert rowwise_spearman(x[None], y[None])[0] == pytest.approx(pd.Series(x).corr(pd.Series(y), method="spearman"))
    assert np.isnan(rowwise_spearman(x[None], np.ones((1, 40)))[0])


def test_spearman_test_on_a_monotone_relation():
    x = np.arange(12, dtype=float)
    out = spearman_test(x, x ** 2, permutations=999, draws=200, rng=np.random.default_rng(4))
    assert out["rho"] == pytest.approx(1.0) and out["p_permutation"] == pytest.approx(1 / 1000)
    assert out["rho_low"] <= 1.0


def test_lesion_credits_count_a_mask_once_per_3d_lesion():
    pairs = pd.DataFrame({"query": ["q", "q", "q", "r"], "train_lesion": ["P/z1/L1", "P/z2/L1", "P/z2/L2", "P/z1/L1"],
                          "s": [0.9, 0.95, 0.85, 0.5]})
    of = pd.Series({"P/z1/L1": "P/D1", "P/z2/L1": "P/D1", "P/z2/L2": "P/D2"})
    credits = lesion_credits(pairs, pd.Series({"q": 0.8, "r": 0.8}), of)
    assert sorted(map(tuple, credits.to_numpy())) == [("q", "P/D1"), ("q", "P/D2")]   # r below tau_L
    lesions = pd.Index(["P/D1", "P/D2", "P/D3"])
    assert credit_counts(credits, lesions).tolist() == [1, 1, 0]


def test_credit_draws_centre_on_the_counts():
    credits = pd.DataFrame({"query": ["q1", "q2", "q2"], "lesion3d_id": ["D1", "D1", "D2"]})
    lesions = pd.Index(["D1", "D2"])
    iid = credit_draws(credits, lesions, 4000, np.random.default_rng(5), n_queries=10)
    assert iid.mean(axis=0) == pytest.approx([2, 1], abs=0.1)
    clusters = pd.Series({"q1": "A", "q2": "A", "q3": "B"})
    by_cluster = credit_draws(credits, lesions, 2000, np.random.default_rng(6), clusters=clusters)
    assert set(np.unique(by_cluster[:, 0])) <= {0, 2, 4}                # cluster A credits D1 twice, as a whole


def test_h1_table_difference_is_positive_when_only_the_query_set_rises_with_span():
    span = np.array([1, 2, 3, 4, 5, 1, 2, 3, 4, 5], dtype=float)
    bins = np.array(["small"] * 5 + ["large"] * 5)
    c_s = span * 2
    c_h = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
    table = h1_table(span, bins, c_s, c_h, np.broadcast_to(c_s, (50, 10)), np.broadcast_to(c_h, (50, 10)))
    rows = table.set_index("bin").to_dict("index")
    assert rows["weighted"]["difference"] > 0.5 and rows["weighted"]["difference_low"] > 0
    assert rows["small"]["rho_s"] == pytest.approx(1.0)


def test_span_rates_relative_to_the_bin_average():
    span = np.array([1, 1, 4, 4, 9, 9], dtype=float)
    bins = np.array(["a"] * 6)
    groups, order = ((1, 2), (3, 5), (6, None)), ["a"]
    flat = span_rates(span, bins, np.full(6, 2.0), np.full((20, 6), 2.0), 1000, groups, order)
    assert flat["relative"].to_numpy() == pytest.approx([1, 1, 1]) and flat["rate_per_1000"].iloc[0] == pytest.approx(2.0)
    rising = span_rates(span, bins, span, np.broadcast_to(span, (20, 6)), 1000, groups, order)
    assert rising["relative"].is_monotonic_increasing and rising["span"].tolist() == ["1-2", "3-5", ">= 6"]

