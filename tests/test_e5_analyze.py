"""Unit tests for the E5 analysis rules in memtk.e5.analyze on small constructed class tables: the
analysis sets (decision L), the source slice of each flag (decision J), chained group labels
(decision N) and the mechanical outcome reading (decision M)."""

import pandas as pd

from memtk.e5.analyze import chained, outcome, sources, subsets

SETS = {"main": ["clean", "minor"], "noisy": ["noisy"]}


def table() -> pd.DataFrame:
    return pd.DataFrame({
        "key": ["a", "b", "c", "d"],
        "mask_class": ["clean", "minor", "noisy", "clean"],
        "stratum_e1": ["single", "multi", "single", None],
        "e1_copy": [True, False, True, None],
        "e1_nn": ["P/z5", "P/z6", "Q/z1", None],
        "stratum_e2": ["single", "multi", "none", "single"],
        "e2_class": ["full", "partial", "none", "novel"],
        "e2_nn": ["P/z5", "R/z9", "", "P/z1"],
        "e2_strongest_train": ["P/z5", "P/z7", "", "P/z1"]})


def test_subsets_by_mask_class_and_main_plus_noisy_only_with_both():
    s = subsets(table(), SETS)
    assert s["main"].tolist() == [True, True, False, True]
    assert s["noisy"].tolist() == [False, False, True, False]
    assert s["main+noisy"].all()
    only_main = subsets(table().assign(mask_class="clean"), SETS)
    assert set(only_main) == {"main"}


def test_sources_follow_decision_j():
    t = table()
    assert sources(t, "e1_copy").tolist() == ["P/z5", "Q/z1"]          # E1: the nearest neighbour
    assert sources(t, "e2_full").tolist() == ["P/z5"]
    assert sources(t, "e2_partial").tolist() == ["P/z7"]               # partial: the strongest-pair slice
    assert sources(t, "e2_flagged").tolist() == ["P/z5", "P/z7"]


def test_chained_labels_name_patient_and_first_slice():
    src = pd.Series(["P/z5", "P/z7", "P/z11", "Q/z1"], index=[10, 11, 12, 13])
    labels = chained(src, max_gap=2)
    assert labels.tolist() == ["P:z5", "P:z5", "P:z11", "Q:z1"]
    assert labels.index.tolist() == [10, 11, 12, 13]
    assert chained(pd.Series(dtype=str), 2).empty


def test_outcome_reading_is_mechanical():
    audit = {"e8_min_strong": 20}
    near = pd.DataFrame({"set": ["query main"], "near_exact": [0], "e8_strong_groups": [0], "e8_eligible": [False]})
    rates = pd.DataFrame({"class": ["e1_copy", "e2_full"], "stratum": ["single", "multi"],
                          "primary": [True, True], "beyond_chance": [False, False]})
    assert outcome(rates, near, audit)["reading"] == "validated negative"
    assert outcome(rates, near.assign(near_exact=3), audit)["reading"] == "rare but real memorization"
    beyond = outcome(rates.assign(beyond_chance=[False, True]), near, audit)
    assert beyond["reading"] == "memorization beyond chance" and beyond["primary_beyond"] == ["e2_full multi"]
