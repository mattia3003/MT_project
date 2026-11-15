"""Helpers shared by the E7 stages: the `reports:` config section (decisions X, Y, Z) checked, the
name reports show for a patient, and the H-eval chance rate of each class and stratum.
"""

from __future__ import annotations

from memtk import config
from memtk.io import read_table
from memtk.stages import decided, require

REVIEW = "docs/e6-e7-plan.md (decisions X, Y, Z)"


def reports_config(cfg: dict) -> dict:
    """The `reports:` section with every value set (decisions X, Y, Z); stops with a message
    otherwise."""
    rep = {key: decided(cfg, f"reports.{key}", REVIEW) for key in ("selection", "deletion", "correctness", "taxonomy")}
    if rep["taxonomy"] != "rule_based":
        raise SystemExit("reports.taxonomy: only 'rule_based' is implemented")
    return rep


def patient_name(patient: str) -> str:
    """The name a report shows for a patient: the dataset's own ID (e.g. `Mets_230`). The dataset's
    IDs are already pseudonymous, so reports use them as every other table does (decision X amended
    2026-11-15; first P001-P105 pseudonyms, dropped as a second naming scheme)."""
    return patient


def chance_rates(cfg: dict, dataset: str) -> dict[tuple[str, str], float]:
    """H-eval's flag rate (E4 `fp_heval.csv`) by (class, stratum), e.g. ("e2_full", "multi")."""
    path = config.results_dir(cfg, dataset, "e4") / "validate" / "fp_heval.csv"
    require(path, "e4_validate.py validate")
    fp = read_table(path)
    return {(r["class"], r["stratum"]): float(r["rate"]) for r in fp.to_dict("records")}


def class_rate(chance: dict[tuple[str, str], float], e2_class: str, stratum: str) -> float | None:
    """The chance rate of an E2 class (full or partial) in a stratum, or None."""
    return chance.get((f"e2_{e2_class}", stratum))
