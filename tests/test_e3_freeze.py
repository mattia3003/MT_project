"""The E3 freeze helpers: they catch changed thresholds, apply decision H, and refuse thresholds
under which the E2 shortcuts would not be exact."""

import pandas as pd
import pytest

from memtk.e3.freeze import compare_tables, frozen_thresholds, shortcut_problems

SETTINGS = {"sigma_px": 6.4, "prefilter_px": 25.6, "candidate_min_s": 0.1}


def table(tau_L_multi=0.8):
    rows = []
    for det, name, values in (("whole_mask", "tau_D", (0.9, 0.91, 0.73)), ("lesion", "tau_L", (0.77, 0.74, tau_L_multi)),
                              ("lesion", "tau_C", (0.71, 0.72, 0.41))):
        for stratum, v in zip(("all", "single", "multi"), values):
            rows.append({"detector": det, "threshold": name, "stratum": stratum, "value": v,
                         "ci_low": v - 0.01, "ci_high": v + 0.01})
    return pd.DataFrame(rows)


def test_compare_tables_finds_a_changed_value():
    assert compare_tables(table(), table()) == []
    assert len(compare_tables(table(0.8), table(0.81))) == 1


def test_decision_H_picks_stratum_or_pooled_values():
    stratified, intervals = frozen_thresholds(table(), {"whole_mask": "stratified", "lesion": "stratified"})
    assert stratified["lesion"]["tau_C"] == {"single": 0.72, "multi": 0.41}
    assert intervals["lesion"]["tau_C"]["multi"] == pytest.approx([0.40, 0.42])
    pooled, _ = frozen_thresholds(table(), {"whole_mask": "pooled", "lesion": "stratified"})
    assert pooled["whole_mask"]["tau_D"] == {"single": 0.9, "multi": 0.9}


def test_thresholds_that_break_the_shortcuts_are_refused():
    good, _ = frozen_thresholds(table(), {"whole_mask": "stratified", "lesion": "stratified"})
    assert shortcut_problems(good, SETTINGS) == []
    bad = {"lesion": {"tau_L": {"single": 0.05, "multi": 0.8}, "tau_C": {"single": 0.0, "multi": 0.4}}}
    problems = shortcut_problems(bad, SETTINGS)
    assert any("candidate cut" in p for p in problems) and any("tau_C" in p for p in problems)
