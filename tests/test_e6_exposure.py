"""Unit tests for the E6 analysis rules in memtk.e6.exposure: the reading of decision U per primary
cell and over cells, and the flags of one cell (class x stratum) with their source patient."""

import pandas as pd

from memtk.e6.exposure import cell_reading, flags, overall_reading

CONC_BEYOND = {"dispersion": 500.0, "dispersion_null_pct": 250.0}
CONC_WITHIN = {"dispersion": 200.0, "dispersion_null_pct": 250.0}


def margins(m_low: float, r_high: float, m_high: float = 0.9) -> dict:
    return {"margin_auc_low": m_low, "margin_auc_high": m_high, "rarity_auc_high": r_high}


def test_cell_reading_branches():
    assert cell_reading(CONC_BEYOND, margins(0.6, 0.9), 0.5)["reading"] == "patient copying"
    assert cell_reading(CONC_BEYOND, margins(0.4, 0.45), 0.5)["reading"] == "typical placement"
    assert cell_reading(CONC_WITHIN, margins(0.6, 0.9), 0.5)["reading"] == "undetermined"     # margin alone
    assert cell_reading(CONC_WITHIN, margins(0.4, 0.6), 0.5)["reading"] == "undetermined"
    assert cell_reading(CONC_BEYOND, margins(0.5, 0.45), 0.5)["reading"] == "typical placement"  # 0.5 is not above
    # margins wholly below chance: typical placement whatever the rarity
    assert cell_reading(CONC_WITHIN, margins(0.2, 0.6, m_high=0.48), 0.5)["reading"] == "typical placement"


def test_overall_reading_agrees_or_is_mixed():
    same = overall_reading({"a": {"reading": "undetermined"}, "b": {"reading": "undetermined"}})
    assert same["reading"] == "undetermined"
    assert overall_reading({"a": {"reading": "patient copying"}, "b": {"reading": "undetermined"}})["reading"] == "mixed"


def test_flags_of_one_cell_follow_decision_j():
    t = pd.DataFrame({"key": ["a", "b", "c", "d"], "cluster": ["a", "b", "c", "d"],
                      "stratum_e1": ["multi", "single", "multi", "multi"], "e1_copy": [True, True, False, True],
                      "e1_nn": ["P/z5", "Q/z1", "P/z6", "R/z2"],
                      "stratum_e2": ["multi", "single", "multi", "none"], "e2_class": ["partial", "full", "full", "none"],
                      "e2_nn": ["S/z1", "Q/z1", "P/z6", ""], "e2_strongest_train": ["P/z9", "Q/z1", "P/z6", ""]})
    e1 = flags(t, "e1_copy", "multi")
    assert e1["key"].tolist() == ["a", "d"] and e1["source_patient"].tolist() == ["P", "R"]
    e2 = flags(t, "e2_flagged", "multi")
    assert e2["source"].tolist() == ["P/z9", "P/z6"]           # partial: the strongest-pair slice
