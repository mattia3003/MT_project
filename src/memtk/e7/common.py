"""Helpers shared by the E7 stages: the `reports:` config section (decisions X, Y, Z) checked, the
pseudonyms reports show instead of patient IDs, and the H-eval chance rate of each class and stratum.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

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


def pseudonyms(cfg: dict, dataset: str) -> Callable[[str], str]:
    """Pseudonym of each T patient (P001, P002, ... in order of patient ID), kept in
    `<output_root>/<dataset>/e7/pseudonyms.csv` (not in git; written on first use). Patients outside
    T are shown as "held-out patient"."""
    path = config.output_dir(cfg, dataset) / "e7" / "pseudonyms.csv"
    if not path.exists():
        patients = read_table(config.output_dir(cfg, dataset) / "patients.csv")
        t = sorted(patients.loc[patients["set"] == "T", "patient_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"patient_id": t, "pseudonym": [f"P{i:03d}" for i in range(1, len(t) + 1)]}).to_csv(path, index=False)
    table = read_table(path).set_index("patient_id")["pseudonym"].to_dict()

    def name_of(patient: str) -> str:
        return table.get(patient, "held-out patient")

    return name_of


def chance_rates(cfg: dict, dataset: str) -> dict[tuple[str, str], float]:
    """H-eval's flag rate (E4 `fp_heval.csv`) by (class, stratum), e.g. ("e2_full", "multi")."""
    path = config.results_dir(cfg, dataset, "e4") / "validate" / "fp_heval.csv"
    require(path, "e4_validate.py validate")
    fp = read_table(path)
    return {(r["class"], r["stratum"]): float(r["rate"]) for r in fp.to_dict("records")}


def class_rate(chance: dict[tuple[str, str], float], e2_class: str, stratum: str) -> float | None:
    """The chance rate of an E2 class (full or partial) in a stratum, or None."""
    return chance.get((f"e2_{e2_class}", stratum))
