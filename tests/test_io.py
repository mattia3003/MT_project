"""Mask stacks: bit-packed storage of many 2D boolean masks, looked up by key."""

import numpy as np
import pytest

from memtk.io import MaskStack, MaskStackWriter


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


def test_wrong_shape_and_duplicate_keys_are_refused(tmp_path):
    writer = MaskStackWriter((4, 4))
    with pytest.raises(ValueError):
        writer.add("a", np.zeros((4, 5), dtype=bool))
    writer.add("a", np.zeros((4, 4), dtype=bool))
    writer.add("a", np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError):
        writer.save(tmp_path / "stack.npz")
