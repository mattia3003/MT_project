"""E3 stage `calibrate` (evidence for decision H: thresholds per stratum or pooled).

Runs E1 and E2 (with the decided informative-lesion bound) on H-cal against T and sets the
thresholds at the `calibration.percentile` of H-cal: tau_D of the E1 score, tau_L of the
strongest single lesion pair per slice, and tau_C of the configuration score C of the nearest
neighbour. Each for single- and multi-lesion slices and pooled, with a bootstrap-by-patient
interval and the share of each stratum that the pooled threshold flags; the E2 classes on H-cal
(full: C >= tau_C; partial: not full but a lesion pair >= tau_L); a sensitivity table (one
alternative per setting); exclusion counts; score histograms. Reads T
and H-cal only. Writes the detector tables to `<output_root>/<dataset>/e3/` and the evidence to
`results/<dataset>/e3/calibrate/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from memtk import calibration, config, lesionmatch, wholemask
from memtk.e3.bound import EDGES
from memtk.e3.common import lesion_settings
from memtk.plotting import save
from memtk.sets import SliceSet, load_set
from memtk.stages import decided, refuse_overwrite, require

# The 5% cut's bound from bound-review (2026-11-03), kept as a sensitivity next to decision G's 12 px
STRICT_BOUND = 25
SIGMA_ALT = 0.08
PERCENTILE_ALT = 99
# Decision H proposal: stratify when the pooled threshold flags a stratum outside this range of
# its slices and the multi-lesion stratum rests on at least MIN_MULTI_PATIENTS patients
FLAG_RANGE = (0.025, 0.10)
MIN_MULTI_PATIENTS = 12
STRATA = ("single", "multi")


# --- stages -----------------------------------------------------------------------------------

def run_calibrate(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """E1 and E2 on H-cal, thresholds with bootstrap-by-patient intervals, sensitivity table and
    exclusion counts: the evidence for decision H (`calibration.strata.whole_mask`,
    `calibration.strata.lesion`). Needs decision G. Writes `<output_root>/<dataset>/e3/` (E1 and
    E2 tables of H-cal) and `results/<dataset>/e3/calibrate/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    data_dir = config.output_dir(cfg, name) / "e3"
    out = config.results_dir(cfg, name, "e3") / "calibrate"
    refuse_overwrite([out, data_dir / "e1_hcal.csv"], overwrite)
    impact_path = config.results_dir(cfg, name, "e0b") / "stats" / f"bound_impact_{pool_name}.csv"
    require(impact_path, "stats")
    bound = int(decided(cfg, "calibration.informative_bound_px", "e3_calibrate.py bound-review"))
    min_area = int(decided(cfg, "lesions.min_lesion_area", "e0b_audit.py lesions-review"))
    percentile = float(cfg["calibration"]["percentile"])
    draws = int(cfg["calibration"]["bootstrap_draws"])
    wm, settings = cfg["detectors"]["whole_mask"], lesion_settings(cfg)
    rng = np.random.default_rng(cfg["seed"])

    t, hcal = load_set(cfg, name, "T"), load_set(cfg, name, "H-cal")
    patients = dict(zip(hcal.keys, hcal.patients))
    e1 = wholemask.match(hcal.labels, t.labels, t.patients, hcal.keys, t.keys,
                         wm["top_k_dice"], wm["top_k_centroid"], wm["max_shift_px"])
    e1["patient_id"] = e1["query"].map(patients)
    e1["n_lesions"] = hcal.labels.reshape(len(hcal.labels), -1).max(axis=1)
    e1["stratum"] = calibration.stratum(e1["n_lesions"].to_numpy())
    e2 = lesionmatch.match(hcal.labels, hcal.keys, t.labels, t.keys, bound, **settings)

    table = thresholds(e1, e2, patients, percentile, draws, rng)
    variants = {"main": (e2, percentile), f"percentile {PERCENTILE_ALT}": (e2, PERCENTILE_ALT)}
    field = cfg["preprocessing"]["spatial_size"][0]
    variants[f"sigma {SIGMA_ALT} ({SIGMA_ALT * field:.1f} px)"] = (
        lesionmatch.match(hcal.labels, hcal.keys, t.labels, t.keys, bound,
                          **{**settings, "sigma_px": SIGMA_ALT * field}), percentile)
    lower = EDGES[EDGES.index(bound) - 1]
    for alt in (lower, STRICT_BOUND):
        variants[f"bound {alt} px"] = (lesionmatch.match(hcal.labels, hcal.keys, t.labels, t.keys, alt, **settings),
                                       percentile)
    sensitivity = pd.concat([thresholds(e1, res, patients, pct, 0, rng).assign(variant=v)
                             for v, (res, pct) in variants.items()], ignore_index=True)
    sensitivity = sensitivity.pivot_table(index=["detector", "threshold", "stratum"], columns="variant",
                                          values="value", sort=False)[list(variants)]
    exclusions = exclusion_counts(t, hcal, min_area, (lower, bound, STRICT_BOUND), pd.read_csv(impact_path))
    classes = class_rates(e2, patients, table)
    proposal = propose_strata(table)

    data_dir.mkdir(parents=True, exist_ok=True)
    e1.to_csv(data_dir / "e1_hcal.csv", index=False)
    e2.queries.to_csv(data_dir / "e2_hcal_queries.csv", index=False)
    e2.lesions.to_csv(data_dir / "e2_hcal_lesions.csv", index=False)
    e2.slice_pairs.to_parquet(data_dir / "e2_hcal_slice_pairs.parquet", index=False)
    e2.lesion_pairs.to_parquet(data_dir / "e2_hcal_lesion_pairs.parquet", index=False)
    config.write_run_info(data_dir, cfg, script=__file__, filename="calibrate.run_info.json",
                          extra={"stage": "calibrate", "inputs": ["T", "H-cal"], "settings_px": settings,
                                 "informative_bound_px": bound})
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "thresholds.csv", index=False)
    sensitivity.to_csv(out / "sensitivity.csv")
    exclusions.to_csv(out / "exclusions.csv")
    classes.to_csv(out / "classes_hcal.csv", index=False)
    les = lesion_slices(e2, patients)
    score_figure({"tau_D": (e1, "score", "E1 score (best Dice over shifts)"),
                  "tau_L": (les, "strongest_s", "E2 strongest single lesion pair s"),
                  "tau_C": (les, "C", "E2 configuration score C of the nearest neighbour")},
                 table, out / "score_histograms.png",
                 f"{name}: H-cal scores against T by stratum, with the thresholds at the {percentile:g}th "
                 f"percentile (solid: pooled; dashed: per stratum)")
    text = review_text(name, pool_name, t, hcal, bound, settings, percentile, draws, table, classes, sensitivity,
                       exclusions, proposal)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "calibrate", "pool": pool_name,
                                                            "inputs": ["T", "H-cal"], "informative_bound_px": bound})
    print(f"\n{text}\nWrote {out}\nWrote {data_dir}")


# --- thresholds -------------------------------------------------------------------------------

def lesion_slices(e2: lesionmatch.LesionMatch, patients: dict) -> pd.DataFrame:
    """One row per H-cal slice with an informative lesion: patient, stratum (by informative
    lesions), the strongest single lesion pair over all candidate slices and the configuration
    score C of the nearest neighbour (both 0 without a candidate; C does not depend on tau_L)."""
    q = e2.queries[e2.queries["n_informative"] > 0].copy()
    strongest = e2.slice_pairs.groupby("query")["best_pair_s"].max()
    q["strongest_s"] = q["query"].map(strongest).fillna(0.0)
    q["C"] = q["query"].map(e2.slice_scores(e2.candidate_min_s).groupby("query")["C"].max()).fillna(0.0)
    q["patient_id"] = q["query"].map(patients)
    q["stratum"] = calibration.stratum(q["n_informative"].to_numpy())
    return q.set_index("query")


def threshold_rows(detector: str, name: str, scores: pd.DataFrame, pooled_col: str, stratum_col: str,
                   percentile: float, draws: int, rng: np.random.Generator) -> list[dict]:
    """Rows for one threshold: pooled (from `pooled_col`) and per stratum (from `stratum_col`),
    each with its bootstrap interval (skipped when draws is 0), slices, patients and the share of
    the slices that the pooled threshold flags (score >= threshold)."""
    pooled = calibration.percentile_threshold(scores[pooled_col].to_numpy(), percentile)
    rows = []
    for stratum, d, col in [("all", scores, pooled_col), *((s, scores[scores["stratum"] == s], stratum_col)
                                                           for s in STRATA)]:
        value = calibration.percentile_threshold(d[col].to_numpy(), percentile) if len(d) else np.nan
        low, high = (calibration.bootstrap_interval(d[col].to_numpy(), d["patient_id"].to_numpy(), percentile,
                                                    draws, rng) if draws and len(d) else (np.nan, np.nan))
        rows.append({"detector": detector, "threshold": name, "stratum": stratum, "value": value,
                     "ci_low": low, "ci_high": high, "slices": len(d), "patients": d["patient_id"].nunique(),
                     "flagged_at_pooled": float((d[pooled_col].fillna(0) >= pooled).mean()) if len(d) else np.nan})
    return rows


def thresholds(e1: pd.DataFrame, e2: lesionmatch.LesionMatch, patients: dict, percentile: float, draws: int,
               rng: np.random.Generator) -> pd.DataFrame:
    """tau_D, tau_L and tau_C at `percentile`, pooled and per stratum, with intervals (skipped
    when draws is 0)."""
    les = lesion_slices(e2, patients)
    rows = threshold_rows("whole_mask", "tau_D", e1, "score", "score", percentile, draws, rng)
    rows += threshold_rows("lesion", "tau_L", les, "strongest_s", "strongest_s", percentile, draws, rng)
    rows += threshold_rows("lesion", "tau_C", les, "C", "C", percentile, draws, rng)
    return pd.DataFrame(rows)


def stratum_threshold(table: pd.DataFrame, name: str, slices: pd.DataFrame) -> pd.Series:
    """Each slice's threshold `name` under stratified thresholds (by its stratum), by query."""
    values = table[table["threshold"] == name].set_index("stratum")["value"]
    return slices["stratum"].map(values)


def class_rates(e2: lesionmatch.LesionMatch, patients: dict, table: pd.DataFrame) -> pd.DataFrame:
    """The E2 classes of the H-cal slices (all chance) per stratum, under the pooled and the
    stratified tau_L and tau_C: slices, and the shares flagged (full or partial), full (C >=
    tau_C), partial (not full, a lesion pair >= tau_L) and with every lesion copied."""
    slices = lesion_slices(e2, patients)
    pooled = table[table["stratum"] == "all"].set_index("threshold")["value"]
    rows = []
    for version, tau_L, tau_C in (
            ("pooled", float(pooled["tau_L"]), float(pooled["tau_C"])),
            ("stratified", stratum_threshold(table, "tau_L", slices), stratum_threshold(table, "tau_C", slices))):
        summary = e2.summary(tau_L, tau_C).set_index("query").reindex(slices.index)
        for stratum in ("all", *STRATA):
            d = summary if stratum == "all" else summary[slices["stratum"] == stratum]
            rows.append({"thresholds": version, "stratum": stratum, "slices": len(d),
                         "flagged": float((d["class"] != "novel").mean()), "full": float((d["class"] == "full").mean()),
                         "partial": float((d["class"] == "partial").mean()),
                         "all_lesions_copied": float(d["all_copied"].mean())})
    return pd.DataFrame(rows)


def propose_strata(table: pd.DataFrame) -> dict[str, str]:
    """Decision H proposal per detector: "stratified" when the pooled threshold flags a stratum
    outside FLAG_RANGE of its slices (for any of the detector's thresholds) and the multi-lesion
    stratum has at least MIN_MULTI_PATIENTS patients; "pooled" otherwise."""
    out = {}
    for detector, d in table.groupby("detector", sort=False):
        strata = d[d["stratum"].isin(STRATA)]
        off = ((strata["flagged_at_pooled"] < FLAG_RANGE[0]) | (strata["flagged_at_pooled"] > FLAG_RANGE[1])).any()
        enough = (strata.loc[strata["stratum"] == "multi", "patients"] >= MIN_MULTI_PATIENTS).all()
        out[str(detector)] = "stratified" if off and enough else "pooled"
    return out


def exclusion_counts(t: SliceSet, hcal: SliceSet, min_area: int, bounds, impact: pd.DataFrame) -> pd.DataFrame:
    """Per bound: lesions kept and slices left without an informative lesion, for T and H-cal
    (from the label maps) and for S main (from the E0 bound-impact table; no detector on S)."""
    rows = []
    for bound in bounds:
        row = {"bound": bound}
        for s in (t, hcal):
            all_lesions = lesionmatch.lesion_records(s.labels, s.keys, min_area)
            kept = lesionmatch.lesion_records(s.labels, s.keys, bound)
            row[f"{s.name} lesions kept"] = f"{len(kept.area)} of {len(all_lesions.area)}"
            row[f"{s.name} slices without"] = int((kept.n_informative == 0).sum())
        s_main = impact[(impact["set"] == "S main") & (impact["bound"] == bound)]
        row["S main share without"] = float(s_main["share_slices_without"].iloc[0]) if len(s_main) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("bound")


# --- review text and figure -------------------------------------------------------------------

def review_text(name: str, pool_name: str, t: SliceSet, hcal: SliceSet, bound: int, settings: dict,
                percentile: float, draws: int, table: pd.DataFrame, classes: pd.DataFrame,
                sensitivity: pd.DataFrame, exclusions: pd.DataFrame, proposal: dict) -> str:
    """The readable summary with the proposal for decision H."""
    shown = table.copy()
    shown["interval"] = [f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(shown["ci_low"], shown["ci_high"])]
    shown = shown[["detector", "threshold", "stratum", "value", "interval", "slices", "patients", "flagged_at_pooled"]]
    left = "; ".join(f"{s.name}: " + ", ".join(f"{v} {k}" for k, v in s.left_out.items()) for s in (t, hcal))
    return (
        f"E3 calibrate: dataset '{name}' (S cost from the E0 table of pool '{pool_name}'; no detector on S)\n\n"
        f"Sets: T {len(t.keys)} slices, H-cal {len(hcal.keys)} slices of {len(set(hcal.patients))} patients "
        f"(left out: {left}).\n"
        f"E1: top-k by Dice and by centroid, shift search; E2: informative lesions >= {bound} px (decision G), "
        f"sigma {settings['sigma_px']:.1f} px, prefilter {settings['prefilter_px']:.1f} px.\n"
        f"Strata: single / multi by lesions >= the minimum area for E1, by informative lesions for E2.\n\n"
        f"Thresholds at the {percentile:g}th percentile of H-cal, with {draws}-draw bootstrap-by-patient 95% "
        f"intervals; flagged_at_pooled = share of the stratum's slices at or above the pooled threshold:\n"
        f"{shown.round(3).to_string(index=False)}\n\n"
        f"E2 classes of the H-cal slices (all chance) under the pooled and the stratified thresholds: full = "
        f"C(q, NN) >= tau_C (the configuration is closer than chance); partial = not full, but a lesion pair "
        f">= tau_L; all_lesions_copied = every informative lesion of both slices paired with s >= tau_L "
        f"(descriptive):\n{classes.round(3).to_string(index=False)}\n\n"
        f"Sensitivity (threshold values; one alternative per setting):\n{sensitivity.round(3).to_string()}\n\n"
        f"Exclusions by the bound (slices left without an informative lesion are still scored by E1):\n"
        f"{exclusions.to_string()}\n\n"
        f"Proposal for decision H (stratified when the pooled threshold flags a stratum outside "
        f"{FLAG_RANGE[0]:.1%}-{FLAG_RANGE[1]:.0%} of its slices and the multi stratum has >= "
        f"{MIN_MULTI_PATIENTS} patients; pooled otherwise):\n"
        + "".join(f"  calibration.strata.{d}: {v}\n" for d, v in proposal.items())
        + f"\nLook at score_histograms.png, then set calibration.strata.whole_mask and calibration.strata.lesion "
        f"in configs/default.yaml (dated comment) and run 'e3_calibrate.py freeze'.\n")


def score_figure(scores: dict, table: pd.DataFrame, path, title: str):
    """One panel per threshold (`scores`: threshold -> (slice table, score column, axis label)):
    H-cal score histograms by stratum with the pooled and stratum thresholds."""
    fig = Figure(figsize=(4.5 * len(scores), 3.8))
    colours = {"single": "tab:blue", "multi": "tab:orange"}
    for ax, (thr, (df, col, label)) in zip(fig.subplots(1, len(scores)), scores.items()):
        bins = np.linspace(0, 1, 41)
        for s in STRATA:
            ax.hist(df.loc[df["stratum"] == s, col].fillna(0), bins=bins, histtype="step", color=colours[s],
                    density=True, label=f"{s} (n={int((df['stratum'] == s).sum())})")
        rows = table[table["threshold"] == thr].set_index("stratum")["value"]
        ax.axvline(rows["all"], color="black", lw=1.5, label=f"pooled {rows['all']:.3f}")
        for s in STRATA:
            ax.axvline(rows[s], color=colours[s], ls="--", lw=1, label=f"{s} {rows[s]:.3f}")
        ax.set_xlabel(label, fontsize=8)
        ax.set_title(thr, fontsize=9)
        ax.legend(fontsize=6)
    return save(fig, path, title)
