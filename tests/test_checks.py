"""E0b consistency checks: they pass on consistent outputs and fail on damaged copies."""

import json

import numpy as np
import pandas as pd
import pytest

from memtk.checks import (CheckReport, check_labels_vs_masks, check_lesion_tables, check_qc_table,
                          check_run_logs)
from memtk.io import LabelStack, LabelStackWriter
from memtk.lesions import attach_or_drop, lesion_features

SHAPE = (24, 24)


def make_masks():
    """Three small masks: two lesions + a stray, one lesion + an isolated speck, two equal lesions."""
    a = np.zeros(SHAPE, np.uint8); a[2:6, 2:6] = 1; a[10:13, 10:13] = 1; a[3, 7] = 1
    b = np.zeros(SHAPE, np.uint8); b[5:9, 5:10] = 1; b[20, 20] = 1
    c = np.zeros(SHAPE, np.uint8); c[15:17, 2:4] = 1; c[1:3, 18:20] = 1
    return {"s1": a, "s2": b, "s3": c}


def outputs(tmp_path, masks):
    """Lesion table, slice table and label stack, made the way the `lesions` stage makes them."""
    writer, lesion_rows, slice_rows = LabelStackWriter(SHAPE), [], []
    for key, mask in masks.items():
        res = attach_or_drop(mask, min_area=4, gap=2)
        writer.add(key, res.labels)
        for f in lesion_features(res, np.ones(SHAPE, bool), (0, SHAPE[0] - 1, 0, SHAPE[1] - 1)):
            lesion_rows.append({"lesion_id": f"{key}/L{f['label']}", "key": key, **f})
        slice_rows.append({"key": key, "n_lesions": res.n_lesions})
    return pd.DataFrame(lesion_rows), pd.DataFrame(slice_rows), LabelStack(writer.save(tmp_path / "labels.npz"))


def failed(report):
    return [name for name, passed, _ in report.results if not passed]


def test_consistent_outputs_pass(tmp_path):
    masks = make_masks()
    lesions, slices, labels = outputs(tmp_path, masks)
    report = CheckReport()
    check_lesion_tables(report, "toy", lesions, slices, labels)
    check_labels_vs_masks(report, "toy", masks, labels, list(masks), min_area=4, gap=2)
    assert report.all_passed, report.to_text()


# each damage takes and returns (lesion table, slice table)
@pytest.mark.parametrize("damage, expected", [
    (lambda les, sl: (les.assign(area=les["area"].where(les.index != 0, les["area"] + 1)), sl), "areas"),
    (lambda les, sl: (les.assign(centroid_row=les["centroid_row"] + 0.5), sl), "centroids"),
    (lambda les, sl: (les.assign(lesion_id=les["lesion_id"].str.replace("/L1", "/L9")), sl), "parses"),
    (lambda les, sl: (les.iloc[1:], sl), "rows per slice"),
    (lambda les, sl: (les, sl.assign(n_lesions=sl["n_lesions"] + 1)), "rows per slice"),
])
def test_damaged_tables_fail(tmp_path, damage, expected):
    lesions, slices, labels = outputs(tmp_path, make_masks())
    damaged_lesions, damaged_slices = damage(lesions, slices)
    report = CheckReport()
    check_lesion_tables(report, "toy", damaged_lesions, damaged_slices, labels)
    assert any(expected in name for name in failed(report)), report.to_text()


def test_swapped_numbering_and_damaged_masks_fail(tmp_path):
    masks = make_masks()
    lesions, slices, labels = outputs(tmp_path, masks)
    writer = LabelStackWriter(SHAPE)
    for key in labels.keys:
        m = labels[key]
        writer.add(key, np.where(m == 1, 2, np.where(m == 2, 1, m)) if key == "s1" else m)   # L1 and L2 swapped
    swapped = LabelStack(writer.save(tmp_path / "swapped.npz"))
    report = CheckReport()
    check_lesion_tables(report, "toy", lesions, slices, swapped)
    check_labels_vs_masks(report, "toy", masks, swapped, list(masks), min_area=4, gap=2)
    assert any("areas" in n for n in failed(report)) and any("reproduces" in n for n in failed(report))
    report = CheckReport()                                              # labels on an emptied mask
    check_labels_vs_masks(report, "toy", {**masks, "s2": masks["s2"] * 0}, labels, list(masks), min_area=4, gap=2)
    assert any("inside the mask" in n for n in failed(report))


def test_qc_table_fails_on_a_wrong_class():
    table = pd.DataFrame({"sample_id": ["a", "b", "c"], "n_lesions": [1, 2, 0], "n_specks": [0, 12, 0],
                          "mri_failed": [False, False, False],
                          "mask_class": ["clean", "noisy", "no_lesion"], "qc_class": ["clean", "noisy", "no_lesion"]})
    report = CheckReport()
    check_qc_table(report, table, ["a", "b", "c"], max_specks=9)
    assert report.all_passed, report.to_text()
    report = CheckReport()
    check_qc_table(report, table.assign(mask_class=["clean", "minor", "no_lesion"]), ["a", "b", "c"], max_specks=9)
    assert any("reproduced" in n for n in failed(report))


def test_run_logs_flag_stale_settings_and_wrong_order(tmp_path):
    cfg = {"lesions": {"min_lesion_area": 4}, "qc": {"max_isolated_specks": 9}}
    def log(name, ts, config):
        path = tmp_path / name
        path.write_text(json.dumps({"timestamp": ts, "config": config}))
        return path
    logs = {"lesions": (log("l.json", "2026-11-02T10:00:00", cfg), ["lesions.min_lesion_area"]),
            "qc": (log("q.json", "2026-11-02T11:00:00", cfg), ["qc.max_isolated_specks"])}
    report = CheckReport()
    check_run_logs(report, cfg, logs)
    assert report.all_passed, report.to_text()
    stale = {"lesions": (log("l2.json", "2026-11-02T10:00:00", {**cfg, "lesions": {"min_lesion_area": 5}}),
                         ["lesions.min_lesion_area"]),
             "qc": (log("q2.json", "2026-11-02T09:00:00", cfg), ["qc.max_isolated_specks"])}
    report = CheckReport()
    check_run_logs(report, cfg, stale)
    assert any("current config" in n for n in failed(report)) and any("ran after" in n for n in failed(report))
