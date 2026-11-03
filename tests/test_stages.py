"""Shared stage helpers: decisions by dotted key, the overwrite and input guards, random picks."""

import argparse

import numpy as np
import pandas as pd
import pytest

from memtk.stages import add_stage, decided, pick, refuse_overwrite, require

CFG = {"_config_path": "test.yaml", "brain": {"method": "low_threshold_all"},
       "calibration": {"strata": {"lesion": None}}}


def test_decided_reads_dotted_keys():
    assert decided(CFG, "brain.method", "e0b_audit.py brain-review") == "low_threshold_all"


@pytest.mark.parametrize("key", ["calibration.strata.lesion", "calibration.missing"])
def test_decided_stops_while_null_and_names_the_review_command(key):
    with pytest.raises(SystemExit, match=f"{key} is not set.*'e3_calibrate.py calibrate'"):
        decided(CFG, key, "e3_calibrate.py calibrate")


def test_refuse_overwrite_and_require(tmp_path):
    out = tmp_path / "out.csv"
    refuse_overwrite([out], overwrite=False)          # nothing there yet: fine
    out.write_text("x")
    with pytest.raises(SystemExit, match="already exists"):
        refuse_overwrite([out], overwrite=False)
    refuse_overwrite([out], overwrite=True)
    require(out, "brain")
    with pytest.raises(SystemExit, match="run the 'lesions' stage first"):
        require(tmp_path / "missing.csv", "lesions")


def test_pick_is_seeded_and_keeps_the_original_order():
    df = pd.DataFrame({"v": range(20)})
    a, b = pick(np.random.default_rng(1), df, 5), pick(np.random.default_rng(1), df, 5)
    assert a.equals(b) and len(a) == 5 and a["v"].is_monotonic_increasing
    assert len(pick(np.random.default_rng(1), df, 50)) == 20
    assert pick(np.random.default_rng(1), df.iloc[:0], 5).empty


def test_add_stage_shares_the_options():
    parser = argparse.ArgumentParser()
    stages = parser.add_subparsers(dest="stage")
    add_stage(stages, "review", "a review stage", run=print)
    add_stage(stages, "check", "a check", run=print, dataset=False, overwrite=False)
    args = parser.parse_args(["review", "--dataset", "d", "--pool", "p", "--overwrite"])
    assert (args.dataset, args.pool, args.overwrite, args.run) == ("d", "p", True, print)
    with pytest.raises(SystemExit):
        parser.parse_args(["check", "--overwrite"])
