"""E6 stage `rehearse` (`docs/e6-e7-plan.md`, steps 3 and 3b): the E6 analysis that will run on S,
run first on mock query sets with a known answer, so every table, test, figure and the reading can
be checked before E6 reads S's tables. S is never read.

- Null case: H-cal as the query set (held-out patients, never trained on). Expected: exposure
  ratios near 1, margins and source rarity like H-eval's (AUC about 0.5), reading not "patient
  copying". H-cal's flags come in near-duplicate runs of adjacent slices of one patient, unlike S's
  independent masks, so the dispersion can exceed its null even here; the margin rule guards
  against that.
- Positive case: H-cal plus E4's gated planted copies P1, P2, P4 of 20 base patients. Expected:
  the base patients over-exposed, flags with larger margins than chance flags, reading "patient
  copying" in the cells the planted copies reach.
- Placement cases (step 3b): H-cal masks with every lesion moved, shape and size kept, to the
  position of a random T lesion (`placement`), or of a random T lesion inside S's central pixel area
  (`placement_compact`): lesions where training lesions typically are, no patient's configuration
  copied. Expected: reading not "patient copying" (ideally "typical placement"). Scored here with
  the frozen detectors; E5's rates are also computed on them (exploratory: can placement alone give
  an E5-style multi-lesion excess?).

Reads the mock sets E5 `rehearse` scored (`<output_root>/<dataset>/e5/rehearse/<case>/`); writes the
placement mocks' tables to `<output_root>/<dataset>/e6/rehearse/<case>/` and every analysis to
`results/<dataset>/e6/rehearse/<case>/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from memtk import calibration, config, lesionmatch
from memtk.audit import Scored, class_table, heldout, load_scored, query_table
from memtk.calibration import load_frozen
from memtk.e5.analyze import auc_table, rates_table
from memtk.e5.common import audit_config
from memtk.e6.common import exposure_config, load_heval, load_query, load_reference
from memtk.e6.exposure import analyze_query, draw_figures, review_text, write_tables
from memtk.io import read_table
from memtk.planted import relocate_lesions
from memtk.sets import SliceSet, load_planted, load_set
from memtk.stages import decided, refuse_overwrite, require

PLACEMENT_CASES = ("placement", "placement_compact")
EXPECTED = {
    "null": "Expected: exposure ratios near 1, margin and source-rarity AUC about 0.5, reading not 'patient copying' "
            "(H-cal's flags cluster in adjacent slices, so the dispersion alone may exceed its null).",
    "positive": "Expected: the 20 base patients of the planted copies over-exposed, margin AUC above 0.5, reading "
                "'patient copying' where the planted copies dominate the flags.",
    "placement": "Expected: no patient copied, so margins no larger than chance and reading not 'patient copying' "
                 "(ideally 'typical placement': sources more typical than chance picks).",
    "placement_compact": "Expected as 'placement', with lesions in S's compact central area: the case closest to "
                         "the placement explanation of E5's result.",
}


# --- stages -----------------------------------------------------------------------------------

def run_rehearse(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The E6 analysis on E5's null and positive mock sets and on the two placement mocks, against
    H-eval. Needs `reference`, E5 `rehearse`, decisions R, U, V, W and `exposure.placement_mock`.
    Writes `<output_root>/<dataset>/e6/rehearse/<case>/` (placement mocks) and
    `results/<dataset>/e6/rehearse/<case>/`."""
    name, _ = config.real_dataset(cfg, dataset)
    mock_root = config.output_dir(cfg, name) / "e5" / "rehearse"
    data_root = config.output_dir(cfg, name) / "e6" / "rehearse"
    out_root = config.results_dir(cfg, name, "e6") / "rehearse"
    refuse_overwrite([out_root], overwrite)
    exp = exposure_config(cfg)
    placement = decided(cfg, "exposure.placement_mock", "docs/e6-e7-plan.md (step 3b)")
    audit = audit_config(cfg)
    e3, e3_hash = load_frozen(cfg, name)
    draws, rng = int(cfg["calibration"]["bootstrap_draws"]), np.random.default_rng(cfg["seed"])
    mock_rng = np.random.default_rng([cfg["seed"], 63])   # its own stream: lesion positions of the placement mocks
    cut = float(e3["detectors"]["lesion"]["candidate_min_s"])
    ref = load_reference(cfg, name)
    h = load_heval(cfg, name, cut)
    planted, _ = load_planted(cfg, name)
    base_patients = dict(zip(planted["key"], planted["patient_id"]))

    t, hcal, heval = (load_set(cfg, name, s) for s in ("T", "H-cal", "H-eval"))
    t_lesions = read_table(config.output_dir(cfg, name) / "lesions.csv")
    t_lesions = t_lesions[t_lesions["key"].isin(t.keys)]
    built = {}
    for case in PLACEMENT_CASES:
        print(f"Placement mock '{case}': moving the lesions of {len(hcal.keys)} H-cal masks and scoring them ...")
        built[case] = placement_set(case, hcal, t, t_lesions, placement, int(cfg["lesions"]["attach_gap"]), e3,
                                    mock_rng, data_root / case)
    h5 = heldout(cfg, name, heval, [])

    run_extra = {"stage": "rehearse", "inputs": ["T", "H-cal", "H-eval tables", "planted", "E6 reference"],
                 "frozen_sha256": e3_hash}
    for case in ("null", "positive", *PLACEMENT_CASES):
        source = mock_root / case if case in ("null", "positive") else data_root / case
        require(source / "classes_main.csv", "e5_audit.py rehearse")
        q = load_query(case, source, audit["sets"]["main"], cut)
        print(f"Rehearsal '{case}': E6 analysis of {len(q.table)} mock queries ...")
        tables, reading = analyze_query(q, h, ref, e3, exp, draws, rng)
        out = out_root / case
        write_tables(tables, out)
        highlight = {base_patients[k] for k in q.table["key"] if k in base_patients}
        draw_figures(tables, ref, exp, f"E6 rehearsal '{case}'", out, highlight)
        header = (f"E6 rehearse, case '{case}': dataset '{name}', mock query set (S never read); the analysis that "
                  f"will run on S.\n{EXPECTED[case]}")
        text = review_text(header, tables, reading)
        if case == "positive":
            text += planted_exposure(tables["exposure"], tables["flag_details"], base_patients,
                                     float(exp["expected"]["smoothing_m"]))
        if case in built:
            text += e5_on_mock(load_scored(case, source, []), h5, audit, draws, rng, out, built[case])
        (out / "review.txt").write_text(text, encoding="utf-8")
        config.write_run_info(out, cfg, script=__file__, extra={**run_extra, "case": case, "reading": reading})
        print(f"\n{text}\nWrote {out}")


# --- placement mocks --------------------------------------------------------------------------

def placement_set(case: str, hcal: SliceSet, t: SliceSet, t_lesions: pd.DataFrame, placement: dict, min_gap: int,
                  e3: dict, rng: np.random.Generator, out_dir: Path) -> dict:
    """A placement mock: every H-cal mask with its lesions moved to the centroids of random T lesions
    (`placement_compact`: only T lesions inside the configured box), scored against T with the frozen
    settings and written in E5's layout (`queries.csv`, `classes_main.csv`, `e1_main.csv`, E2 tables
    `main`), H-cal patients as clusters. Masks whose lesions find no room are left out and counted."""
    pos = t_lesions[["centroid_row", "centroid_col"]].to_numpy(float)
    if case == "placement_compact":
        (r0, r1), (c0, c1) = placement["compact_rows"], placement["compact_cols"]
        inside = (pos[:, 0] >= r0) & (pos[:, 0] <= r1) & (pos[:, 1] >= c0) & (pos[:, 1] <= c1)
        pos = pos[inside]
    keys, clusters, labels = [], [], []
    for k, p, lab in zip(hcal.keys, hcal.patients, hcal.labels):
        moved = relocate_lesions(lab, pos, rng, min_gap, int(placement["max_tries"]))
        if moved is not None:
            keys.append(f"{case}__{k.replace('/', '_')}")
            clusters.append(p)
            labels.append(moved.astype(hcal.labels.dtype))
    left_out = {"no room for every lesion": len(hcal.keys) - len(keys)}
    q = SliceSet(case, keys, np.asarray(clusters), np.stack(labels), left_out)
    e1, e2, e2c = calibration.run_frozen(q.labels, q.keys, t, e3)
    out_dir.mkdir(parents=True, exist_ok=True)
    query_table(q, clusters, ["clean"] * len(keys), [case] * len(keys)).to_csv(out_dir / "queries.csv", index=False)
    class_table(q, e1, e2, e2c).to_csv(out_dir / "classes_main.csv", index=False)
    e1.to_csv(out_dir / "e1_main.csv", index=False)
    lesionmatch.save_tables(e2, out_dir, "main")
    return {"masks": len(keys), "left_out": left_out, "positions": len(pos)}


def e5_on_mock(scored: Scored, h5: Scored, audit: dict, draws: int, rng: np.random.Generator, out: Path,
               built: dict) -> str:
    """Exploratory, placement mocks only: E5's rates (excess over H-eval with intervals) and AUCs of
    the mock, written to `e5_rates.csv` and `e5_auc.csv`, with a short review block."""
    rates = rates_table(scored, h5, ["main"], audit, draws, rng)
    rates = rates[rates["set"] == "main"].reset_index(drop=True)
    aucs = auc_table(scored.table("main"), h5.table("main"), draws, rng)
    rates.to_csv(out / "e5_rates.csv", index=False)
    aucs.to_csv(out / "e5_auc.csv", index=False)
    cols = ["class", "stratum", "n_s", "flagged_s", "rate_s", "rate_h", "excess", "excess_low", "excess_high"]
    with pd.option_context("display.width", 200, "display.precision", 3):
        return (f"\nPlacement mock: {built['masks']} masks built from H-cal (left out: {built['left_out']}), lesion "
                f"positions drawn from {built['positions']} T lesions.\n"
                f"Exploratory: E5's rates on this mock (excess = mock rate - H-eval rate; can placement alone give an "
                f"E5-style excess?):\n{rates[cols].to_string(index=False)}\n\nE5's AUCs on this mock:\n"
                f"{aucs.to_string(index=False)}\n")


# --- rehearsal checks -------------------------------------------------------------------------

def planted_exposure(exposure: pd.DataFrame, details: pd.DataFrame, base_patients: dict, m: float) -> str:
    """Positive case: per cell, the base patients whose planted copies are flagged in the cell, how
    many of them have a ratio interval above 1 against the other patients, how many of the 10 highest
    ratios are base patients, and how many planted flags name their base patient."""
    rows = []
    for cell, d in exposure[exposure["m"] == m].groupby("cell", sort=False):
        f = details[(details["set"] == "query") & (details["cell"] == cell) & details["key"].isin(base_patients)]
        bases = {base_patients[k] for k in f["key"]}
        is_base = d["patient_id"].isin(bases)
        top = d.sort_values("ratio", ascending=False).head(10)
        rows.append({"cell": cell, "planted_flags": len(f),
                     "naming_base": int((f["source_patient"] == f["key"].map(base_patients)).sum()),
                     "base_patients": int(is_base.sum()),
                     "base_ratio_low_above_1": int((d.loc[is_base, "ratio_low"] > 1).sum()),
                     "other_ratio_low_above_1": int((d.loc[~is_base, "ratio_low"] > 1).sum()),
                     "base_in_top10": int(top["patient_id"].isin(bases).sum()),
                     "base_median_ratio": float(d.loc[is_base, "ratio"].median()),
                     "other_median_ratio": float(d.loc[~is_base, "ratio"].median())})
    return (f"\nRehearsal check, planted copies (base patients known):\n"
            f"{pd.DataFrame(rows).to_string(index=False)}\n")
