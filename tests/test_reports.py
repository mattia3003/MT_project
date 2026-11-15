"""Unit tests for memtk.reports (E7 decisions X, Y, Z) on hand-built report data: the verdict rules
(full copy, configuration match without a lesion pair above tau_L, the strongest pair naming another
patient), correctness on a toy P6-add and P6-remove, the top and smallest contributor, lesion
removal, the taxonomy, and the deletion statistics."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from memtk.reports import (ReportData, contributors, correctness_row, deletion_summary, profile_tables, taxonomy,
                           verdict, without)


def data(pairs: Sequence[tuple[str, str, float, float, float, float]], e2_class: str = "full", n_query: int = 2,
         n_train: int = 2, source: str = "Mets_001/z10", strongest: str = "Mets_001/z10") -> ReportData:
    """Report data for query `q` with pairs (query lesion, train lesion, shape, position, size, s)."""
    p = pd.DataFrame(pairs, columns=["query_lesion", "train_lesion", "shape", "position", "size", "s"])
    p["contribution"] = p["s"] / max(n_query, n_train)
    p["kept"] = p["s"] >= 0.8
    return ReportData("q", "multi", e2_class, float(p["s"].sum()) / max(n_query, n_train), source, source,
                      float(p["s"].max()) if len(p) else 0.0, strongest, 0.8, source, (0, 0), 0.8, 0.4, 0.9,
                      n_query, n_train, p)


def name(patient: str) -> str:
    return {"Mets_001": "P001", "Mets_002": "P002"}[patient]


def test_verdict_full_copy_names_source_and_chance_rate():
    d = data([("q/L1", "Mets_001/z10/L1", 1, 1, 1, 1.0), ("q/L2", "Mets_001/z10/L2", 1, 0.95, 1, 0.95)])
    text = verdict(d, name, 0.029)
    assert text.startswith("2 of 2 informative lesions match P001, slice 10") and "-> full copy" in text
    assert "2.9%" in text


def test_verdict_configuration_match_is_not_called_a_copy():
    d = data([("q/L1", "Mets_001/z10/L1", 0.7, 0.9, 0.9, 0.57), ("q/L2", "Mets_001/z10/L2", 0.6, 0.9, 0.8, 0.43)])
    text = verdict(d, name, None)
    assert text.startswith("Configuration match with P001, slice 10") and "copy" not in text


def test_verdict_adds_the_strongest_pair_patient_when_it_differs():
    d = data([("q/L1", "Mets_001/z10/L1", 1, 1, 1, 0.9)], e2_class="partial", n_query=2,
             strongest="Mets_002/z4")
    text = verdict(d, name, None)
    assert "1 of 2" in text and "1 without a partner" in text and "-> partial copy" in text
    assert "points to P002, slice 4" in text


def test_verdict_full_class_without_every_lesion_is_a_configuration_copy():
    d = data([("q/L1", "Mets_001/z10/L1", 1, 1, 1, 0.9), ("q/L2", "Mets_001/z10/L2", 0.6, 0.9, 0.9, 0.49)])
    text = verdict(d, name, None)
    assert "1 of 2" in text and "1 paired but not closer than chance" in text
    assert "configuration copy" in text and "full copy" not in text


def test_correctness_of_p6_add_and_p6_remove():
    pairs = [("q/L1", "Mets_001/z10/L1", 1, 1, 1, 1.0), ("q/L2", "Mets_001/z10/L2", 1, 1, 1, 1.0)]
    added = {"q/L1": "Mets_001/z10/L1", "q/L2": "Mets_001/z10/L2", "q/L3": "Heval/z2/L1"}
    ok = correctness_row(data(pairs, n_query=3), "Mets_001/z10", "Mets_001", added, 2)
    assert ok["patient_ok"] and ok["slice_ok"] and ok["lesions_ok"]
    wrong = pairs + [("q/L3", "Mets_001/z10/L3", 1, 1, 1, 0.9)]    # the added lesion paired above tau_L
    assert not correctness_row(data(wrong, n_query=3, n_train=3), "Mets_001/z10", "Mets_001", added, 2)["lesions_ok"]
    swapped = [("q/L1", "Mets_001/z10/L2", 1, 1, 1, 0.9)]           # a base lesion paired with the wrong lesion
    removed = {"q/L1": "Mets_001/z10/L1"}
    assert not correctness_row(data(swapped, n_query=1), "Mets_001/z10", "Mets_001", removed, 2)["lesions_ok"]
    far = correctness_row(data(pairs, source="Mets_001/z14"), "Mets_001/z10", "Mets_001", {}, 2)
    assert far["patient_ok"] and not far["slice_ok"]


def test_contributors_and_removal():
    d = data([("q/L1", "T/z1/L1", 1, 1, 1, 0.9), ("q/L3", "T/z1/L2", 1, 1, 1, 0.2), ("q/L2", "T/z1/L3", 1, 1, 1, 0.5)],
             n_query=3, n_train=3)
    assert contributors(d.pairs, 2) == (1, 3)
    assert contributors(d.pairs.head(1), 2) is None
    labels = np.array([[1, 0, 2], [3, 3, 0]])
    assert without(labels, 3).tolist() == [[1, 0, 2], [0, 0, 0]] and labels[1, 0] == 3


def test_taxonomy_categories():
    same = [("q/L1", "T/z1/L1", 1, 1, 1, 1.0), ("q/L2", "T/z1/L2", 1, 1, 1, 1.0)]
    assert taxonomy(data(same)) == "exact"
    assert taxonomy(data([(a, b, s, 0.95, z, 0.95) for a, b, s, _, z, _ in same])) == "shifted"
    assert taxonomy(data([(a, b, 0.8, p, z, 0.8) for a, b, _, p, z, _ in same])) == "deformed"
    assert taxonomy(data(same[:1], n_query=2)) == "partial"
    assert taxonomy(data(same, e2_class="novel")) == "not flagged"


def test_deletion_summary_favours_the_top_contributor():
    d = pd.DataFrame({"C_before": [0.9, 0.8, 0.7], "C_top": [0.3, 0.2, 0.3], "C_small": [0.8, 0.7, 0.65],
                      "e1_before": [0.9, 0.9, 0.9], "e1_top": [0.4, 0.5, 0.6], "e1_small": [0.85, 0.8, 0.88],
                      "flag_before": [True] * 3, "flag_top": [False, False, True], "flag_small": [True, True, True]})
    s = deletion_summary(d)
    assert s["C_drop_top"] > s["C_drop_small"] and s["e1_drop_top"] > s["e1_drop_small"]
    assert s["flips_top"] == 2 and s["flips_small"] == 0 and s["mcnemar_p"] == pytest.approx(0.25)


def test_verdict_names_a_whole_mask_match_before_the_lesion_level_reading():
    d = data([("q/L1", "Mets_001/z10/L1", 0.8, 0.9, 0.9, 0.65)], e2_class="novel", n_query=1, n_train=1)
    d.e1_score, d.e1_nn, d.tau_D = 0.91, "Mets_002/z4", 0.73
    text = verdict(d, name, None)
    assert text.startswith("Matches P002, slice 4 as a whole closer than chance (E1 0.91 >= 0.73).")
    assert text.endswith("(E2: novel)")


def test_profile_tables_count_flags_per_slice_and_matches_per_lesion():
    lesion_map = pd.DataFrame({"lesion_id": ["P/z1/L1", "P/z2/L1", "P/z2/L2", "Q/z1/L1"],
                               "patient_id": ["P", "P", "P", "Q"], "z": [1, 2, 2, 1], "area": [20, 25, 12, 30],
                               "d3": ["P/D1", "P/D1", "P/D2", "Q/D1"]})
    credits = pd.DataFrame({"query": ["a", "b", "b", "c"], "train_lesion": ["P/z1/L1", "P/z1/L1", "P/z2/L2", "Q/z1/L1"]})
    slices, lesions = profile_tables("P", lesion_map, "d3", credits, pd.Series(["P/z2", "P/z2", "Q/z1"]),
                                     pd.Series(["P/z1"]))
    assert slices.to_dict("list") == {"z": [1, 2], "e1_flags": [0, 2], "e2_flags": [1, 0]}
    assert lesions.set_index("lesion_id")["credits"].to_dict() == {"P/z1/L1": 2, "P/z2/L1": 0, "P/z2/L2": 1}

