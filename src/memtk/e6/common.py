"""Helpers shared by the E6 stages: the `exposure:` config section (decisions R, U, V, W) checked,
the 3D labelling methods of decision W, the E6 reference tables, and scored query sets (a mock set
or S from E5's stored tables, and H-eval from E4's) with their detector tables.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from memtk import config, lesionmatch
from memtk.audit import CLASSES, heldout, load_scored
from memtk.calibration import STRATA
from memtk.io import read_table
from memtk.lesions3d import components_26, link_slices
from memtk.sets import load_set
from memtk.stages import decided, require

REVIEW = "docs/e6-e7-plan.md (decisions R, U, V, W)"
# Decision W: the 3D labelling methods, by their config name
LABELLERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {"slice_linking": link_slices,
                                                             "connectivity_26": components_26}
FEATURES = ("n_lesions3d", "volume_px", "median_area_px", "max_lesions_per_slice", "position_spread")


def exposure_config(cfg: dict) -> dict:
    """The `exposure:` section with every value set (decisions R, U, V, W) and only implemented
    choices; stops with a message otherwise."""
    exp = {key: decided(cfg, f"exposure.{key}", REVIEW) for key in
           ("classes", "primary", "expected", "top_k", "reading", "rarity", "lesions3d", "h1_size_bins_px")}
    checks = {
        f"exposure.classes must be among {CLASSES}": set(exp["classes"]) <= set(CLASSES),
        "exposure.primary.classes must be among exposure.classes": set(exp["primary"]["classes"]) <= set(exp["classes"]),
        f"exposure.primary.strata must be among {STRATA}": set(exp["primary"]["strata"]) <= set(STRATA),
        "exposure.expected.baseline: only 'H-eval' is implemented": exp["expected"]["baseline"] == "H-eval",
        f"exposure.lesions3d.main and .sensitivity must be among {sorted(LABELLERS)}":
            {exp["lesions3d"]["main"], exp["lesions3d"]["sensitivity"]} <= set(LABELLERS),
        f"exposure.rarity.features must be among {FEATURES}": set(exp["rarity"]["features"]) <= set(FEATURES),
        "exposure.rarity.log1p must be among exposure.rarity.features":
            set(exp["rarity"]["log1p"]) <= set(exp["rarity"]["features"]),
    }
    failed = [msg for msg, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("; ".join(failed))
    return exp


def reference_dir(cfg: dict, dataset: str) -> Path:
    """Where `reference` writes the E6 tables of the real dataset (3D lesions, rarity, held-out NN)."""
    return config.output_dir(cfg, dataset) / "e6"


# --- reference tables -------------------------------------------------------------------------

@dataclass
class Reference:
    """The tables `reference` wrote: 3D lesions per method, the 2D -> 3D map, slice rarity, patients
    (features, rarity, T slice share)."""
    lesions3d: pd.DataFrame
    lesion_map: pd.DataFrame
    rarity: pd.DataFrame
    patients: pd.DataFrame

    def weights(self) -> pd.Series:
        """w_p, each T patient's share of the T lesion slices, by patient ID (sorted)."""
        return self.patients.set_index("patient_id")["t_slice_share"].sort_index()


def load_reference(cfg: dict, dataset: str) -> Reference:
    """The E6 reference tables of a real dataset (stops if `reference` has not run)."""
    d = reference_dir(cfg, dataset)
    require(d / "patients.csv", "e6_exposure.py reference")
    return Reference(read_table(d / "lesions3d.csv"), read_table(d / "lesion_map.csv"),
                     read_table(d / "rarity_slices.csv"), read_table(d / "patients.csv"))


# --- scored query sets ------------------------------------------------------------------------

@dataclass
class QuerySet:
    """A scored set as E6 reads it: `table` (one row per slice or mask of the analysed set: the
    main class table joined with key, cluster, mask class, scheme), the E1 table (`margin`) and the
    E2 result (every stored candidate slice and lesion pair)."""
    name: str
    table: pd.DataFrame
    e1: pd.DataFrame
    e2: lesionmatch.LesionMatch


def load_query(name: str, score_dir: Path, mask_classes: Sequence[str], candidate_min_s: float) -> QuerySet:
    """A query set scored by E5 (`e5.score.score_query`: S or a mock set) from its stored tables,
    restricted to the given mask classes (decision L's main set)."""
    table = load_scored(name, score_dir, []).table("main")
    table = table[table["mask_class"].isin(mask_classes)].reset_index(drop=True)
    return QuerySet(name, table, read_table(score_dir / "e1_main.csv"),
                    lesionmatch.load_tables(score_dir, "main", candidate_min_s))


def load_heval(cfg: dict, dataset: str, candidate_min_s: float) -> QuerySet:
    """H-eval as E6 reads it: E4's class table (clustered by patient), E1 table and E2 tables."""
    e4_dir = config.output_dir(cfg, dataset) / "e4"
    table = heldout(cfg, dataset, load_set(cfg, dataset, "H-eval"), []).table("main")
    return QuerySet("H-eval", table, read_table(e4_dir / "e1_heval.csv"),
                    lesionmatch.load_tables(e4_dir, "heval", candidate_min_s))
