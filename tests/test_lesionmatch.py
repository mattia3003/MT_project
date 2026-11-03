"""Unit tests for memtk.lesionmatch (E2) on small hand-built label maps with known answers, and
the shortcuts against brute force on random ones."""

import numpy as np
import pandas as pd
import pytest

from memtk.lesionmatch import (LesionMatch, assign_pairs, lesion_records, match, packed_crops,
                               position_factor, shape_factor, size_factor)

SIZE = 64
SIGMA, PREFILTER, MIN_S = 6.4, 25.6, 0.1


def labels(*rects):
    """A label map with one rectangle lesion per (row, col, height, width), numbered in order."""
    lab = np.zeros((SIZE, SIZE), dtype=np.int32)
    for n, (r, c, h, w) in enumerate(rects, start=1):
        lab[r:r + h, c:c + w] = n
    return lab


def run(queries, refs, min_area=4, min_s=MIN_S):
    return match(np.stack(queries), [f"q{i}" for i in range(len(queries))], np.stack(refs),
                 [f"t{i}" for i in range(len(refs))], min_area, SIGMA, PREFILTER, min_s)


# --- lesion score ---

def test_identical_lesion_scores_1_and_is_a_full_copy():
    lab = labels((20, 20, 4, 5), (40, 30, 3, 3))
    res = run([lab], [lab])
    assert res.lesion_pairs["s"].tolist() == pytest.approx([1.0, 1.0])
    row = res.summary(tau_L=0.5, tau_C=0.5).iloc[0]
    assert (row["nn"], row["C"], row["class"], row["all_copied"]) == ("t0", pytest.approx(1.0), "full", True)


def test_factors_at_known_distance_and_size():
    assert position_factor(np.array([0.0, 6.4]), SIGMA) == pytest.approx([1.0, np.exp(-0.5)])
    assert size_factor(np.array([4]), np.array([16])) == pytest.approx([0.25])


def test_translated_lesion_keeps_its_shape_and_loses_position():
    query, train = labels((20, 20, 4, 5)), labels((22, 20, 4, 5))   # moved 2 px down
    pair = run([query], [train]).lesion_pairs.iloc[0]
    assert (pair["shape"], pair["size"]) == (pytest.approx(1.0), pytest.approx(1.0))
    assert pair["position"] == pytest.approx(np.exp(-4 / (2 * SIGMA ** 2)))


def test_shape_is_the_iou_of_centred_crops():
    small, large = labels((21, 21, 2, 2)), labels((20, 20, 4, 4))   # same rounded centroid
    pair = run([small], [large], min_s=0.0).lesion_pairs.iloc[0]
    assert pair["shape"] == pytest.approx(4 / 16) and pair["size"] == pytest.approx(4 / 16)


def test_score_is_symmetric():
    a, b = labels((20, 20, 4, 6)), labels((23, 22, 5, 3))
    s_ab = run([a], [b], min_s=0.0).lesion_pairs["s"].iloc[0]
    s_ba = run([b], [a], min_s=0.0).lesion_pairs["s"].iloc[0]
    assert s_ab == pytest.approx(s_ba)


# --- slice-pair score and classes ---

def test_hungarian_pairing_beats_greedy():
    s = np.array([[0.9, 0.8], [0.85, 0.1]])   # greedy takes 0.9 + 0.1; the best total is 0.8 + 0.85
    rows, cols = assign_pairs(s)
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 1), (1, 0)]


def hand_made(slice_rows, pair_rows, queries):
    return LesionMatch(pd.DataFrame(queries, columns=["query", "n_informative"]), pd.DataFrame(),
                       pd.DataFrame(slice_rows, columns=["query", "train", "train_idx", "n_query", "n_train",
                                                         "n_matched", "best_pair_s"]),
                       pd.DataFrame(pair_rows, columns=["query", "train", "s"]), MIN_S)


def test_worked_example_of_the_experiment_plan():
    # q has lesions A, B; t has X, Y, Z; best pairing A-X 0.9, B-Z 0.2; only A-X passes tau_L.
    # C counts every paired lesion reaching the candidate cut: (0.9 + 0.2) / 3
    res = hand_made([("q", "t", 0, 2, 3, 2, 0.9)], [("q", "t", 0.9), ("q", "t", 0.2)], [("q", 2)])
    scored = res.slice_scores(tau_L=0.5).iloc[0]
    assert (scored["C"], scored["n_kept"], scored["all_copied"]) == (pytest.approx(1.1 / 3), 1, False)
    assert res.summary(tau_L=0.5, tau_C=0.5).iloc[0]["class"] == "partial"


def test_C_counts_pairs_from_the_candidate_cut_and_ignores_tau_L():
    # two pairs: 0.9 and 0.05 (below the cut of 0.1, not counted); C is the same for any tau_L
    res = hand_made([("q", "t", 0, 2, 2, 2, 0.9)], [("q", "t", 0.9), ("q", "t", 0.05)], [("q", 2)])
    assert res.slice_scores(tau_L=0.2)["C"].iloc[0] == pytest.approx(0.45)
    assert res.slice_scores(tau_L=0.95)["C"].iloc[0] == pytest.approx(0.45)


def test_classes_and_nearest_neighbour():
    res = hand_made(
        [("q1", "t0", 0, 1, 1, 1, 0.6), ("q1", "t1", 1, 1, 1, 1, 0.8),   # q1: NN t1, C 0.8: full
         ("q2", "t0", 0, 2, 1, 1, 0.7),                                   # q2: C 0.35, a pair >= tau_L: partial
         ("q3", "t0", 0, 1, 1, 1, 0.2),                                   # q3: C 0.2, no pair >= tau_L: novel
         ("q6", "t0", 0, 3, 3, 3, 0.45)],                                 # q6: three pairs at 0.45: C 0.45
        [("q1", "t0", 0.6), ("q1", "t1", 0.8), ("q2", "t0", 0.7), ("q3", "t0", 0.2)] + [("q6", "t0", 0.45)] * 3,
        [("q1", 1), ("q2", 2), ("q3", 1), ("q4", 1), ("q5", 0), ("q6", 3)])  # q4: no candidate; q5: none informative
    out = res.summary(tau_L=0.5, tau_C=0.4).set_index("query")
    # q6: a configuration copy (C >= tau_C) although no single lesion reaches tau_L
    assert out["class"].to_dict() == {"q1": "full", "q2": "partial", "q3": "novel", "q4": "novel", "q6": "full"}
    assert (out.loc["q1", "nn"], out.loc["q1", "C"]) == ("t1", pytest.approx(0.8))
    assert (out.loc["q2", "C"], out.loc["q2", "n_kept"]) == (pytest.approx(0.35), 1)
    assert (out.loc["q6", "n_kept"], out.loc["q6", "strongest_s"]) == (0, pytest.approx(0.45))
    assert (out.loc["q3", "strongest_train"], out.loc["q4", "nn"]) == ("t0", "")


def test_thresholds_per_query_slice():
    res = hand_made([("q1", "t0", 0, 1, 1, 1, 0.7), ("q2", "t0", 0, 1, 1, 1, 0.7), ("q3", "t0", 0, 1, 1, 1, 0.7)],
                    [("q1", "t0", 0.7), ("q2", "t0", 0.7), ("q3", "t0", 0.7)], [("q1", 1), ("q2", 1), ("q3", 1)])
    out = res.summary(tau_L=pd.Series({"q1": 0.6, "q2": 0.6, "q3": 0.8}),
                      tau_C=pd.Series({"q1": 0.6, "q2": 0.8, "q3": 0.8})).set_index("query")
    assert out["class"].to_dict() == {"q1": "full", "q2": "partial", "q3": "novel"}


def test_thresholds_that_would_break_the_shortcut_or_the_classes_are_refused():
    with pytest.raises(ValueError, match="candidate_min_s"):
        hand_made([], [], [("q", 1)]).slice_scores(tau_L=0.05)
    with pytest.raises(ValueError, match="candidate_min_s"):
        hand_made([], [], [("q", 1)]).slice_scores(tau_L=pd.Series({"q": 0.05, "r": 0.5}))
    with pytest.raises(ValueError, match="tau_C"):
        hand_made([], [], [("q", 1)]).summary(tau_L=0.5, tau_C=0.0)


# --- informative lesions and the prefilter ---

def test_lesions_below_the_bound_take_no_part():
    small_only = labels((20, 20, 2, 2))                    # 4 px
    mixed = labels((20, 20, 5, 5), (40, 40, 2, 2))         # 25 px and 4 px
    res = run([small_only, mixed], [mixed], min_area=10)
    assert res.queries["n_informative"].tolist() == [0, 1]
    assert res.summary(0.5, 0.5)["query"].tolist() == ["q1"]
    assert res.slice_pairs.iloc[0][["n_query", "n_train"]].tolist() == [1, 1]


def test_pairs_beyond_the_prefilter_distance_are_never_compared():
    query = labels((10, 10, 3, 3))
    near, far = labels((10, 30, 3, 3)), labels((10, 41, 3, 3))   # 20 px and 31 px away
    res = run([query], [near, far], min_s=0.0)
    assert res.slice_pairs["train"].tolist() == ["t0"]


# --- the shortcuts give the same result as brute force ---

def random_maps(rng, n):
    maps = []
    for _ in range(n):
        lab = np.zeros((SIZE, SIZE), dtype=np.int32)
        for k in range(1, rng.integers(1, 4) + 1):
            for _ in range(20):                                  # place without touching another lesion
                h, w = rng.integers(1, 7, size=2)
                r, c = rng.integers(16, 44, size=2)
                if not lab[max(r - 1, 0):r + h + 1, max(c - 1, 0):c + w + 1].any():
                    lab[r:r + h, c:c + w] = k
                    break
        maps.append(lab)
    return np.stack(maps)


def brute_force(queries, refs, min_area):
    """C of every slice pair from all lesion pairs (Hungarian over all of them, paired s from
    the candidate cut counted), no prefilter and no candidate shortcut."""
    q, t = lesion_records(queries, [f"q{i}" for i in range(len(queries))], min_area), \
        lesion_records(refs, [f"t{i}" for i in range(len(refs))], min_area)
    half = max(q.half_size(), t.half_size())
    qc, tc = packed_crops(q, half), packed_crops(t, half)
    qi, ti = np.meshgrid(np.arange(len(q.area)), np.arange(len(t.area)), indexing="ij")
    qi, ti = qi.ravel(), ti.ravel()
    d = np.hypot(*(q.centroid[qi] - t.centroid[ti]).T)
    s = (position_factor(d, SIGMA) * size_factor(q.area[qi], t.area[ti])
         * shape_factor(qc[qi], tc[ti], q.area[qi], t.area[ti])).reshape(len(q.area), len(t.area))
    best, C = {}, {}
    for a in range(len(queries)):
        for b in range(len(refs)):
            sub = s[np.ix_(q.slice == a, t.slice == b)]
            if not sub.size:
                continue
            best[(f"q{a}", f"t{b}")] = sub.max()
            rows, cols = assign_pairs(sub)
            paired = sub[rows, cols][sub[rows, cols] >= MIN_S]
            C[(f"q{a}", f"t{b}")] = paired.sum() / max(sub.shape)
    return best, C


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shortcuts_equal_brute_force(seed):
    rng = np.random.default_rng(seed)
    queries, refs = random_maps(rng, 12), random_maps(rng, 30)
    res = run(list(queries), list(refs))
    best, C = brute_force(queries, refs, min_area=4)
    candidates = {k for k, v in best.items() if v >= MIN_S}
    found = set(zip(res.slice_pairs["query"], res.slice_pairs["train"]))
    assert found == candidates and len(found) > 0
    scored = res.slice_scores(tau_L=0.3).set_index(["query", "train"])["C"]
    for key, value in C.items():
        assert scored.get(key, 0.0) == pytest.approx(value, abs=1e-9)
    lesion_best = res.lesions.set_index("query_lesion")["best_s"].dropna()
    assert len(lesion_best) > 0 and (lesion_best >= MIN_S).all()
