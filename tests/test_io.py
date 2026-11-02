"""Mask and label stacks: storage of many 2D masks / label maps, looked up by key."""

import numpy as np
import pytest

import pandas as pd

from memtk.io import LabelStack, LabelStackWriter, MaskStack, MaskStackWriter, read_table


def test_round_trip_keeps_every_mask_exactly(tmp_path):
    rng = np.random.default_rng(0)
    masks = {f"file_{i}": rng.random((5, 7)) > 0.5 for i in range(4)}   # 35 px: not a multiple of 8
    masks["empty"] = np.zeros((5, 7), dtype=bool)
    writer = MaskStackWriter((5, 7))
    for key, mask in masks.items():
        writer.add(key, mask)
    stack = MaskStack(writer.save(tmp_path / "stack.npz"))
    assert len(stack) == len(masks) and stack.keys == list(masks)
    for key, mask in masks.items():
        assert stack[key].dtype == bool
        assert np.array_equal(stack[key], mask)


def test_keys_are_strings_not_numbers(tmp_path):
    writer = MaskStackWriter((2, 2))
    writer.add("1033", np.eye(2, dtype=bool))
    writer.add("file_1033", np.zeros((2, 2), dtype=bool))
    stack = MaskStack(writer.save(tmp_path / "stack.npz"))
    assert "1033" in stack and "file_1033" in stack
    assert stack["1033"].sum() == 2 and stack["file_1033"].sum() == 0


def test_label_stack_round_trip_keeps_every_label(tmp_path):
    rng = np.random.default_rng(1)
    maps = {f"Mets_00{i}/z{i}": rng.integers(0, 4, (6, 9)) * (rng.random((6, 9)) > 0.7) for i in range(3)}
    maps["empty"] = np.zeros((6, 9), dtype=int)
    maps["many"] = np.arange(54).reshape(6, 9)                         # labels up to 53
    writer = LabelStackWriter((6, 9))
    for key, labels in maps.items():
        writer.add(key, labels)
    stack = LabelStack(writer.save(tmp_path / "labels.npz"))
    assert stack.keys == list(maps) and "empty" in stack
    for key, labels in maps.items():
        assert stack[key].dtype == np.int32
        assert np.array_equal(stack[key], labels)


def test_label_stack_refuses_bad_input(tmp_path):
    writer = LabelStackWriter((3, 3))
    with pytest.raises(ValueError):
        writer.add("a", np.zeros((3, 4), dtype=int))
    with pytest.raises(ValueError):
        writer.add("a", np.full((3, 3), -1))
    writer.add("a", np.zeros((3, 3), dtype=int))
    writer.add("a", np.ones((3, 3), dtype=int))
    with pytest.raises(ValueError):
        writer.save(tmp_path / "labels.npz")


def test_read_table_keeps_numeric_looking_ids_as_text(tmp_path):
    path = tmp_path / "lesions.csv"
    pd.DataFrame({"sample_id": ["0", "1033", "007"], "lesion_id": ["0/L1", "1033/L1", "007/L2"],
                  "area": [46, 12, 5], "centroid_row_brain": [0.5, None, 0.2]}).to_csv(path, index=False)
    table = read_table(path)
    assert table["sample_id"].tolist() == ["0", "1033", "007"]     # not 0, 1033, 7
    assert table["area"].dtype.kind == "i"                          # other columns keep their types
    assert table["centroid_row_brain"].isna().tolist() == [False, True, False]


def test_wrong_shape_and_duplicate_keys_are_refused(tmp_path):
    writer = MaskStackWriter((4, 4))
    with pytest.raises(ValueError):
        writer.add("a", np.zeros((4, 5), dtype=bool))
    writer.add("a", np.zeros((4, 4), dtype=bool))
    writer.add("a", np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError):
        writer.save(tmp_path / "stack.npz")
