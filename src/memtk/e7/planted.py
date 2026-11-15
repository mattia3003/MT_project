"""E7 stage `planted` (`docs/e6-e7-plan.md`, step 4; decisions X, Y, Z): the match-report generator
built and checked on E4's planted copies, whose true source is known, before any report on S is
drawn. S is never read.

- Report data and the verdict line for every planted mask with an informative lesion, from E4's
  stored tables (`planted_outcomes.csv`, `e1_planted.csv`, `e2_planted_*`).
- Correctness (decision Z) on the gated planted masks that E2 flags (those that get a report): the
  decision-J source patient is the base patient, the source slice within the window of the base
  slice, either source names the base patient, and for P6 the right lesions; against the criteria
  in `reports.correctness`. All gated masks as a secondary row.
- Deletion test (decision Y) on the gated, E2-flagged multi-lesion planted controls and, as the
  reference, on H-eval's E2-flagged multi-lesion chance flags.
- The rule-based taxonomy's confusion table (variant x category) and one example report per variant.

Writes `results/<dataset>/e7/planted/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from memtk import config, lesionmatch
from memtk.calibration import load_frozen
from memtk.e7.common import chance_rates, class_rate, patient_name, reports_config
from memtk.ids import parse_slice_key
from memtk.io import read_table, slice_images
from memtk.reports import (FLAGGED, ReportData, contributors, correctness_row, deletion_effects, deletion_summary,
                           draw_report, report_data, taxonomy, verdict)
from memtk.sets import SliceSet, load_planted, load_set
from memtk.stages import refuse_overwrite, require


# --- stages -----------------------------------------------------------------------------------

def run_planted(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Reports, correctness, deletion test and taxonomy on E4's planted copies (and H-eval's chance
    flags for the deletion reference). Needs E4 `validate` and decisions X, Y, Z. Writes
    `results/<dataset>/e7/planted/`."""
    name, _ = config.real_dataset(cfg, dataset)
    out = config.results_dir(cfg, name, "e7") / "planted"
    refuse_overwrite([out], overwrite)
    rep = reports_config(cfg)
    e3, e3_hash = load_frozen(cfg, name)
    e4_dir = config.output_dir(cfg, name) / "e4"
    outcomes_path = config.results_dir(cfg, name, "e4") / "validate" / "planted_outcomes.csv"
    require(outcomes_path, "e4_validate.py validate")
    cut = float(e3["detectors"]["lesion"]["candidate_min_s"])
    bound = int(e3["informative_bound_px"])
    rng = np.random.default_rng(cfg["seed"])
    name_of, chance = patient_name, chance_rates(cfg, name)

    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    planted, planted_labels = load_planted(cfg, name)
    outcomes = read_table(outcomes_path)
    e2p = lesionmatch.load_tables(e4_dir, "planted", cut)
    e1p = read_table(e4_dir / "e1_planted.csv").set_index("query")
    lesions = read_table(e4_dir / "planted_lesions.csv")
    informative = lesions[lesions["area"] >= bound]
    sources = dict(zip(informative["lesion_id"].astype(str), informative["source_lesion"].fillna("").astype(str)))

    print(f"Report data for {len(outcomes)} planted masks ...")
    reports, table = planted_reports(outcomes, e1p, e2p, e3, sources, name_of, chance,
                                     int(rep["correctness"]["slice_window"]))
    correct, criteria = correctness_tables(table, rep["correctness"])
    confusion = pd.crosstab(table.loc[table["gated"] & table["flagged"], "variant"],
                            table.loc[table["gated"] & table["flagged"], "taxonomy"])

    print("Deletion test: planted controls and H-eval chance flags ...")
    labels_of_planted = dict(zip(planted["key"], planted_labels))
    controls = table[table["gated"] & table["flagged"] & (table["stratum"] == "multi")
                     & table["variant"].isin(rep["deletion"]["planted_controls"])]
    del_planted = run_deletion([reports[k] for k in controls["key"]], labels_of_planted, int(rep["deletion"]["min_paired"]),
                               t, e3)
    hcls = read_table(e4_dir / "heval_classes.csv")
    e2h = lesionmatch.load_tables(e4_dir, "heval", cut)
    e1h = read_table(e4_dir / "e1_heval.csv").set_index("query")
    hflag = hcls[hcls["e2_class"].isin(FLAGGED) & (hcls["stratum_e2"] == "multi")]
    h_reports = [report_data(r, e1h.loc[r["key"]] if r["key"] in e1h.index else None, e2h, e3)
                 for _, r in hflag.iterrows()]
    del_heval = run_deletion(h_reports, dict(zip(heval.keys, heval.labels)), int(rep["deletion"]["min_paired"]), t, e3)
    deletion = pd.concat([del_planted.assign(set="planted controls"), del_heval.assign(set="H-eval chance flags")],
                         ignore_index=True)
    summary = pd.DataFrame([{"set": s, **deletion_summary(d)} for s, d in deletion.groupby("set", sort=False)])

    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "reports.csv", index=False)
    correct.to_csv(out / "correctness.csv", index=False)
    criteria.to_csv(out / "criteria.csv", index=False)
    confusion.to_csv(out / "taxonomy_confusion.csv")
    deletion.to_csv(out / "deletion.csv", index=False)
    summary.to_csv(out / "deletion_summary.csv", index=False)
    example_reports(cfg, name, table, reports, planted, labels_of_planted, t, e1p, name_of, chance, rng, out)
    text = review_text(name, table, correct, criteria, confusion, summary)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "planted", "inputs": ["T", "H-eval", "planted"],
                                                            "frozen_sha256": e3_hash})
    print(f"\n{text}\nWrote {out}")


# --- reports and correctness ------------------------------------------------------------------

def planted_reports(outcomes: pd.DataFrame, e1p: pd.DataFrame, e2p: lesionmatch.LesionMatch, e3: dict,
                    sources: dict[str, str], name_of, chance: dict, window: int) -> tuple[dict[str, ReportData], pd.DataFrame]:
    """Report data, verdict, taxonomy and correctness of every planted mask with an informative
    lesion (one row each)."""
    n_inf = e2p.queries.set_index("query")["n_informative"]
    table = outcomes.assign(n_informative=outcomes["key"].map(n_inf).fillna(0).astype(int))
    reports, rows = {}, []
    for _, r in table[table["n_informative"] > 0].iterrows():
        d = report_data(r, e1p.loc[r["key"]] if r["key"] in e1p.index else None, e2p, e3)
        reports[d.key] = d
        own = {lid: s for lid, s in sources.items() if lid.startswith(f"{d.key}/")}
        ok = correctness_row(d, r["base_key"], r["patient_id"], own, window) if d.source else \
            {"patient_ok": False, "slice_ok": False, "either_ok": False, "lesions_ok": False}
        rows.append({"key": d.key, "variant": r["variant"], "base_key": r["base_key"], "base_patient": r["patient_id"],
                     "gated": bool(r["gated"]), "stratum": d.stratum, "e2_class": d.e2_class,
                     "flagged": d.e2_class in FLAGGED, "source": d.source, "strongest_train": d.strongest_train,
                     "e1_nn": d.e1_nn, "n_query": d.n_query, "n_kept": d.n_kept, "all_copied": d.all_copied,
                     "taxonomy": taxonomy(d),
                     "patient_ok": ok["patient_ok"], "slice_ok": ok["slice_ok"], "either_ok": ok["either_ok"],
                     "lesions_ok": ok["lesions_ok"],
                     "verdict": verdict(d, name_of, class_rate(chance, d.e2_class, d.stratum))})
    return reports, pd.DataFrame(rows)


def correctness_tables(table: pd.DataFrame, crit: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decision Z per variant: shares of correct source patient, slice within the window, either
    source naming the base, and correct lesions (the lesion variants only: P6), over the gated masks
    E2 flags (main) and over
    all gated masks (an unflagged mask counts as not named); and the criteria with pass / fail."""
    rows = []
    for variant, g in table[table["gated"]].groupby("variant"):
        f = g[g["flagged"]]
        rows.append({"variant": variant, "gated": len(g), "flagged": len(f),
                     "patient": f["patient_ok"].mean(), "slice_within_window": f["slice_ok"].mean(),
                     "either_source": f["either_ok"].mean(),
                     "lesions": f["lesions_ok"].mean() if variant in crit["lesion_variants"] else np.nan,
                     "patient_all_gated": (g["flagged"] & g["patient_ok"]).mean()})
    correct = pd.DataFrame(rows)
    by = correct.set_index("variant").to_dict("index")
    criteria = [{"criterion": f"source patient {v}", "value": by[v]["patient"], "min": float(crit["patient_min"])}
                for v in crit["patient_variants"] if v in by]
    criteria += [{"criterion": f"lesions {v}", "value": by[v]["lesions"], "min": float(crit["lesions_min"])}
                 for v in crit["lesion_variants"] if v in by]
    crit_table = pd.DataFrame(criteria)
    crit_table["pass"] = crit_table["value"] >= crit_table["min"]
    return correct, crit_table


# --- deletion test ----------------------------------------------------------------------------

def run_deletion(reports: list[ReportData], labels_of: dict[str, np.ndarray], min_paired: int, t: SliceSet,
                 e3: dict) -> pd.DataFrame:
    """Decision Y on the given reports: those with at least `min_paired` contributing pairs on their
    source slice lose their top or their smallest contributor, and are re-scored."""
    chosen = [(d, c) for d in reports if d.source and (c := contributors(d.pairs, min_paired)) is not None]
    if not chosen:
        return pd.DataFrame()
    stacked = np.stack([labels_of[d.key] for d, _ in chosen])  # pyright: ignore[reportCallIssue]  # numpy stub: np.stack overloads mis-resolve here
    return deletion_effects([d.key for d, _ in chosen], stacked,
                            [c[0] for _, c in chosen], [c[1] for _, c in chosen], [d.source for d, _ in chosen],
                            [d.source_patient for d, _ in chosen], t, e3)


# --- figures and review -----------------------------------------------------------------------

def example_reports(cfg: dict, dataset: str, table: pd.DataFrame, reports: dict[str, ReportData], planted: pd.DataFrame,
                    labels_of_planted: dict, t: SliceSet, e1p: pd.DataFrame, name_of, chance: dict,
                    rng: np.random.Generator, out: Path) -> None:
    """One report per variant (`report_<variant>.png`): a gated, E2-flagged mask, multi-lesion where
    the variant has one, picked at random (seeded); the context names the known truth."""
    base_of = dict(zip(planted["key"], planted["base_key"]))
    t_labels = dict(zip(t.keys, t.labels))
    margins = e1p["margin"].to_dict()

    def labels_of(keys):
        return {k: labels_of_planted[k] if k in labels_of_planted else t_labels[k] for k in keys}

    def images_of(keys):
        real = slice_images(cfg, dataset, {base_of.get(k, k) for k in keys})
        return {k: real[base_of.get(k, k)] for k in keys}

    for variant, g in table[table["gated"] & table["flagged"]].groupby("variant"):
        pool = g[g["stratum"] == "multi"] if (g["stratum"] == "multi").any() else g
        d = reports[pool["key"].iloc[int(rng.integers(len(pool)))]]
        base_patient, base_z = parse_slice_key(base_of[d.key])
        context = [f"truth: planted {variant} of {name_of(base_patient)}, slice {base_z} (known source)",
                   f"E1 margin to the second-best patient: {margins.get(d.key, np.nan):.2f}",
                   "E6 exposure ratio: not applicable (planted set)"]
        rate = class_rate(chance, d.e2_class, d.stratum)
        draw_report(d, labels_of, images_of, name_of, context, verdict(d, name_of, rate), out / f"report_{variant}.png",
                    f"E7 match report, planted {variant} ({d.key})")


def review_text(name: str, table: pd.DataFrame, correct: pd.DataFrame, criteria: pd.DataFrame,
                confusion: pd.DataFrame, summary: pd.DataFrame) -> str:
    """The `planted` review: correctness per variant and the criteria, the deletion test, the
    taxonomy's confusion table."""
    with pd.option_context("display.width", 220, "display.max_columns", 30, "display.precision", 3):
        verdict_line = "PASS" if criteria["pass"].all() else "FAIL"
        return (f"E7 planted: dataset '{name}' (E4's planted copies, known source; S never read)\n"
                f"{len(table)} planted masks with an informative lesion; gated {int(table['gated'].sum())}, of them "
                f"flagged by E2 {int((table['gated'] & table['flagged']).sum())}.\n\n"
                f"Correctness per variant (decision Z; shares over the gated masks E2 flags; patient_all_gated counts "
                f"an unflagged mask as not named):\n{correct.to_string(index=False)}\n\n"
                f"Criteria (thesis plan): {verdict_line}\n{criteria.to_string(index=False)}\n\n"
                f"Deletion test (decision Y; drop = before - after; flips = no longer flagged with the same source "
                f"patient after a full re-match):\n{summary.T.to_string(header=False)}\n\n"
                f"Taxonomy (rule-based) of the gated, E2-flagged masks, variant x category:\n{confusion.to_string()}\n")
