"""Unit tests for the E4 `validate` helpers that decide the gate, on small constructed tables."""

import pandas as pd

from memtk.e4.validate import criterion_caught, gate_verdict

CRITERIA = {"recall": {"P1": {"detector": "both", "min": 1.0},
                       "P6": {"detector": "lesion", "min": 0.8, "variants": ["P6-remove", "P6-add"]}}}


def gate(passes: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"criterion": [f"c{i}" for i in range(len(passes))], "pass": passes})


def test_verdict_pass_accepted_and_fail():
    assert gate_verdict(gate([True, True]), None)[0] == "PASS"
    verdict, table = gate_verdict(gate([True, False]), {"criteria": ["c1"], "reason": "r"})
    assert verdict == "ACCEPTED" and table["accepted"].tolist() == [False, True]
    assert gate_verdict(gate([True, False]), None)[0] == "FAIL"
    assert gate_verdict(gate([False, False]), {"criteria": ["c1"], "reason": "r"})[0] == "FAIL"   # one not listed
    assert gate_verdict(gate([True, False]), {"criteria": ["c0", "c1"], "reason": "r"})[0] == "FAIL"  # lists a pass


def test_criterion_caught_uses_each_variants_detector_and_stays_boolean():
    outcomes = pd.DataFrame({"variant": ["P1", "P1", "P6-add", "P6-remove", "P3"],
                             "both_caught": [True, False, False, False, True],
                             "e2_caught": [True, True, True, False, True]})
    caught = criterion_caught(outcomes, CRITERIA)
    assert caught.dtype == bool                                 # an object column turned ~True into -2 (truthy)
    assert caught.tolist() == [True, False, True, False, False]  # P3 has no criterion
    assert (~caught).tolist() == [False, True, False, True, True]
