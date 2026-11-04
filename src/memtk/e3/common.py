"""Helpers shared by the E3 stages: the stored H-cal detector tables and the E2 settings in
pixels. The T and H-cal label maps come from `memtk.sets` (H-cal after decision E). E3 reads only
T and H-cal; S is never loaded."""

from __future__ import annotations

import pandas as pd

from memtk import config
from memtk.io import read_table
from memtk.lesionmatch import LesionMatch
from memtk.stages import decided, require


def load_hcal_tables(cfg: dict, dataset: str) -> tuple[pd.DataFrame, LesionMatch, dict]:
    """The H-cal detector tables stored by `calibrate`: the E1 table, the E2 results rebuilt as a
    `LesionMatch`, and the patient of every H-cal slice."""
    d = config.output_dir(cfg, dataset) / "e3"
    require(d / "e1_hcal.csv", "calibrate")
    e1 = read_table(d / "e1_hcal.csv")
    e2 = LesionMatch(read_table(d / "e2_hcal_queries.csv"), read_table(d / "e2_hcal_lesions.csv"),
                     pd.read_parquet(d / "e2_hcal_slice_pairs.parquet"),
                     pd.read_parquet(d / "e2_hcal_lesion_pairs.parquet"),
                     float(cfg["detectors"]["lesion"]["candidate_min_s"]))
    return e1, e2, dict(zip(e1["query"], e1["patient_id"]))


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
