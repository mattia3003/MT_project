"""Helpers shared by the E4 stages: the frozen E3 file (detector settings and thresholds come only
from there) with its hash for the run logs, and the random streams of the planted set. E4 reads T,
H-eval and the planted set; S is never loaded."""

from __future__ import annotations

import hashlib

import numpy as np

from memtk import config
from memtk.calibration import read_frozen
from memtk.stages import require

# One random stream per choice of the planted set (decision I), so changing one draw leaves the others unchanged
STREAMS = ("bases", "direction", "angle", "remove", "add")


def load_frozen(cfg: dict, dataset: str) -> tuple[dict, str]:
    """The frozen E3 file of `dataset` and its SHA-256, recorded in every E4 run log so that
    `check` can confirm the thresholds did not change between stages."""
    path = config.frozen_file(cfg, dataset)
    require(path, "e3_calibrate.py freeze")
    return read_frozen(path), hashlib.sha256(path.read_bytes()).hexdigest()


def streams(seed: int) -> dict[str, np.random.Generator]:
    """One generator per name in STREAMS, spawned from the seed."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(child) for name, child in zip(STREAMS, children)}
