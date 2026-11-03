"""Helpers shared by the E3 stages: the T and H-cal label maps (H-cal after decision E), and the
E2 settings in pixels. E3 reads only T and H-cal; S is never loaded."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from memtk import config
from memtk.io import LabelStack, read_table
from memtk.stages import decided


@dataclass
class SliceSet:
    """The lesion slices of one real set that take part, with their label maps."""
    name: str
    keys: list[str]
    patients: np.ndarray
    labels: np.ndarray         # (N, H, W) label maps; cleaned mask = labels > 0
    left_out: dict[str, int]   # reason -> slices left out


def load_set(cfg: dict, dataset: str, name: str) -> SliceSet:
    """The slices of set `name` (T, H-cal or H-eval) from the E0b lesion tables. Held-out sets go
    through decision E (`filters.real_sets`); T is never filtered (it is the reference). Slices
    without a lesion of at least the minimum area are left out (neither detector can score them)."""
    out_dir = config.output_dir(cfg, dataset)
    slices = read_table(out_dir / "slices.csv")
    slices = slices[slices["set"] == name]
    left_out = {}
    if name != "T":
        passes = decision_e(slices, cfg)
        left_out["removed by decision E"] = int((~passes).sum())
        slices = slices[passes]
    empty = slices["n_lesions"] == 0
    left_out["no lesion of at least the minimum area"] = int(empty.sum())
    slices = slices[~empty]
    stack = LabelStack(out_dir / "labels.npz")
    labels = np.stack([stack[k] for k in slices["key"]])
    return SliceSet(name, slices["key"].tolist(), slices["patient_id"].to_numpy(), labels, left_out)


def decision_e(slices: pd.DataFrame, cfg: dict) -> pd.Series:
    """Which real slices pass the sampling filters of decision E (`filters.real_sets`), on the
    masks as saved. `max_mask_px` comes straight from `slices.csv` (all mask pixels = lesion
    pixels with strays + dropped specks); other filters need the masks and are not used here yet."""
    passes = pd.Series(True, index=slices.index)
    for name in decided(cfg, "filters.real_sets", "e0b_audit.py filters-review"):
        if name == "max_mask_px":
            passes &= (slices["lesion_px"] + slices["speck_px"]) < cfg["filters"]["max_mask_px"]
        else:
            raise SystemExit(f"Decision E filter '{name}' is not implemented in E3 (only max_mask_px); "
                             f"add it to memtk.e3.common.decision_e.")
    return passes


def lesion_settings(cfg: dict) -> dict:
    """The E2 settings in pixels (sigma and prefilter are fractions of the field in the config),
    as keywords for `memtk.lesionmatch.match`."""
    frame = decided(cfg, "detectors.lesion.position_frame", "docs/e1-e3-plan.md (decision F)")
    if frame != "pixel":
        raise SystemExit(f"detectors.lesion.position_frame '{frame}' is not implemented (only 'pixel').")
    field = cfg["preprocessing"]["spatial_size"][0]
    p = cfg["detectors"]["lesion"]
    return {"sigma_px": p["sigma"] * field, "prefilter_px": p["prefilter"] * field,
            "candidate_min_s": p["candidate_min_s"]}
