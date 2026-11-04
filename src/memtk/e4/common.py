"""Helpers shared by the E4 stages: the frozen E3 file (detector settings and thresholds come only
from there) with its hash for the run logs, the random streams of the planted set, the stored
planted set, and BRAVO slices for figures. E4 reads T, H-eval and the planted set; S is never
loaded."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
import pandas as pd

from memtk import config
from memtk.calibration import read_frozen
from memtk.ids import parse_slice_key
from memtk.io import LabelStack, load_real, read_table
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


def load_planted(cfg: dict, dataset: str) -> tuple[pd.DataFrame, np.ndarray]:
    """The planted set written by `plant`: `planted.csv` and the label maps in its row order."""
    d = config.output_dir(cfg, dataset) / "e4"
    require(d / "planted_labels.npz", "plant")
    planted = read_table(d / "planted.csv")
    stack = LabelStack(d / "planted_labels.npz")
    return planted, np.stack([stack[k] for k in planted["key"]])


def gate_status(cfg: dict, dataset: str) -> str:
    """The E4 verdict from `results/<dataset>/e4/validate/gate.csv`: PASS (every criterion
    passed), ACCEPTED (the failed ones are a recorded deviation, `validation.accepted_deviation`)
    or FAIL. E5 runs only on PASS or ACCEPTED."""
    path = config.results_dir(cfg, dataset, "e4") / "validate" / "gate.csv"
    require(path, "e4_validate.py validate")
    gate = read_table(path)
    if gate["pass"].all():
        return "PASS"
    return "ACCEPTED" if (gate["pass"] | gate["accepted"]).all() else "FAIL"


def slice_images(cfg: dict, dataset: str, keys: Iterable[str]) -> dict[str, np.ndarray]:
    """The model-space BRAVO slice of each real slice key (figure backgrounds only), one volume
    read per patient."""
    by_patient: dict[str, list[tuple[str, int]]] = {}
    for key in set(keys):
        patient, z = parse_slice_key(key)
        by_patient.setdefault(patient, []).append((key, z))
    real_dir = config.output_dir(cfg, dataset)
    images = {}
    for patient, items in by_patient.items():
        bravo, _ = load_real(real_dir, patient)
        images.update({key: bravo[:, :, z] for key, z in items})
    return images
