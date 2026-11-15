"""Helpers shared by the E4 stages: the random streams of the planted set and the gate verdict E5
reads. E4 reads T, H-eval and the planted set; S is never loaded. `load_frozen` moved to
`memtk.calibration`, `slice_images` to `memtk.io` and `load_planted` to `memtk.sets` on 2026-11-15
(E5 needs them)."""

from __future__ import annotations

import numpy as np

from memtk import config
from memtk.io import read_table
from memtk.stages import require

# One random stream per choice of the planted set (decision I), so changing one draw leaves the others unchanged
STREAMS = ("bases", "direction", "angle", "remove", "add")


def streams(seed: int) -> dict[str, np.random.Generator]:
    """One generator per name in STREAMS, spawned from the seed."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(child) for name, child in zip(STREAMS, children)}


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
