"""Unit tests for memtk.calibration (E3) on small constructed score tables."""

import numpy as np
import pandas as pd
import pytest

from memtk.calibration import area_bins, bin_label, chance_rates, percentile_threshold, propose_bound

EDGES = (4, 10, 20)


def test_area_bins_and_labels():
    assert area_bins(np.array([3, 4, 9, 10, 19, 20, 500]), EDGES).tolist()[1:] == [4, 4, 10, 10, 20, 20]
    assert np.isnan(area_bins(np.array([3]), EDGES)[0])
    assert [bin_label(e, EDGES) for e in EDGES] == ["4-9 px", "10-19 px", ">= 20 px"]


def test_chance_rates_count_lesions_patients_and_missing_scores():
    best = np.array([0.9, 0.2, np.nan, 0.6, 0.7])
    area = np.array([5, 6, 7, 12, 25])
    rates = chance_rates(best, area, ["A", "A", "B", "C", "C"], EDGES, [0.5]).set_index("lower")
    assert list(rates.loc[4, ["lesions", "patients", "matched"]]) == [3, 2, 1]
    assert rates.loc[4, "rate"] == pytest.approx(1 / 3)                 # NaN (no match) never counts
    assert rates.loc[10, "rate"] == 1.0 and rates.loc[20, "rate"] == 1.0


def test_percentile_counts_missing_scores_as_zero():
    assert percentile_threshold(np.array([np.nan, np.nan, 1.0, 1.0]), 50) == pytest.approx(0.5)


@pytest.mark.parametrize("rates, cut, expected", [
    ([0.30, 0.08, 0.04, 0.02], 0.05, 20),   # falls below 5% from the 20 px bin on
    ([0.30, 0.08, 0.04, 0.02], 0.10, 10),
    ([0.30, 0.02, 0.06, 0.02], 0.05, 30),   # a later bin above the cut moves the bound up
    ([0.30, 0.20, 0.10, 0.06], 0.05, None), # even the largest bin reaches the cut
])
def test_bound_is_where_every_larger_bin_stays_below_the_cut(rates, cut, expected):
    table = pd.DataFrame({"lower": [4, 10, 20, 30], "rate": rates})
    assert propose_bound(table, cut) == expected
