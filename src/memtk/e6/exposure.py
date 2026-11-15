"""E6 analysis (`docs/e6-e7-plan.md`, decisions R, U, V, W): a scored query set against H-eval and
the E6 reference tables. `analyze_query` (tables and the reading) and `draw_figures` are the one
analysis: `rehearse` runs them on E5's mock sets first, and the `exposure` stage (step 5) runs them
unchanged on S, so the analysis of S is the analysis rehearsed. Nothing here is tuned; every choice
comes from the `exposure:` section.

Per cell (class x stratum, decision R): exposure ratios per T patient (and under the sensitivity
smoothing), the concentration test, chained groups per patient; the margin to the second-best
patient and the rarity of the source slice of each flag, query flags against H-eval chance flags
(AUC, decision U); source-slice rarity against all other T slices and the patient-level Spearman of
rarity against the exposure ratio (H3, decision V). Once: H1, Spearman(span, lesion credits) per
size bin against H-eval, for both 3D lesion methods (decision W). The reading of decision U for the
primary cells.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from memtk import calibration, lesionmatch
from memtk.audit import auc, auc_bootstrap, chained, interval, sources
from memtk.calibration import STRATA
from memtk.e6.common import QuerySet, Reference
from memtk.exposure import (concentration_test, credit_counts, credit_draws, e2_margins, exposure_table,
                            h1_table, lesion_credits, patient_of_keys, spearman_test)
from memtk.ids import parse_slice_key
from memtk.plotting import save

# Per class: (stratum column, nearest-neighbour patient column of the held-out masks, E1 or E2 scale)
CELL_COLUMNS = {"e1_copy": ("stratum_e1", "e1_nn_patient", "e1"), "e2_full": ("stratum_e2", "e2_nn_patient", "e2"),
                "e2_partial": ("stratum_e2", "e2_nn_patient", "e2"), "e2_flagged": ("stratum_e2", "e2_nn_patient", "e2")}
TOP_PATIENTS = 10
MAX_GAP = 2   # chained groups (E5 decision N)


# --- analysis ---------------------------------------------------------------------------------

def analyze_query(q: QuerySet, h: QuerySet, ref: Reference, e3: dict, exp: dict, draws: int,
                  rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every E6 table of query set `q` against H-eval `h`, by name, and the reading (decision U)."""
    weights = ref.weights()
    rarity = ref.rarity.set_index("key")
    p_rarity = ref.patients.set_index("patient_id")["rarity"].reindex(weights.index).to_numpy()
    m_main, m_sens = float(exp["expected"]["smoothing_m"]), float(exp["expected"]["smoothing_m_sensitivity"])
    primary = [(c, s) for c in exp["primary"]["classes"] for s in exp["primary"]["strata"]]
    q_margin, h_margin = flag_margins(q), flag_margins(h)
    exposure_rows, conc_rows, margin_rows, h3_rows, details = [], [], [], [], []
    readings = {}
    for cls in exp["classes"]:
        stratum_col, nn_col, scale = CELL_COLUMNS[cls]
        for s in STRATA:
            cell = f"{cls} {s}"
            qf, hf = flags(q.table, cls, s), flags(h.table, cls, s)
            hn = h.table[(h.table[stratum_col] == s) & (h.table[nn_col].fillna("") != "")]
            main_table, main_conc, ratio_draws = cell_exposure(qf, hn[nn_col], hn["cluster"], weights, m_main, exp,
                                                               draws, rng)
            sens_table, sens_conc, _ = cell_exposure(qf, hn[nn_col], hn["cluster"], weights, m_sens, exp, draws, rng)
            for m, table, conc in ((m_main, main_table, main_conc), (m_sens, sens_table, sens_conc)):
                exposure_rows.append(table.assign(cell=cell, m=m))
                conc_rows.append({"cell": cell, "m": m, "primary": (cls, s) in primary, **conc})
            qf = qf.assign(margin=qf["key"].map(q_margin[scale]), source_rarity=qf["source"].map(rarity[f"rarity_{scale}"]))
            hf = hf.assign(margin=hf["key"].map(h_margin[scale]), source_rarity=hf["source"].map(rarity[f"rarity_{scale}"]))
            details += [qf.assign(cell=cell, set="query"), hf.assign(cell=cell, set="H-eval")]
            m_auc = auc_row(qf, hf, "margin", draws, rng)
            r_auc = auc_row(qf, hf, "source_rarity", draws, rng)
            src_slices = set(qf["source"])
            t_src = rarity[rarity.index.isin(src_slices)][f"rarity_{scale}"].dropna()
            t_other = rarity[~rarity.index.isin(src_slices)][f"rarity_{scale}"].dropna()
            t_auc = auc_row(pd.DataFrame({"v": t_src.to_numpy(), "cluster": patient_of_keys(t_src.index.to_series()).to_numpy()}),
                            pd.DataFrame({"v": t_other.to_numpy(), "cluster": patient_of_keys(t_other.index.to_series()).to_numpy()}),
                            "v", draws, rng)
            margin_rows.append({"cell": cell, "primary": (cls, s) in primary, "flags_q": len(qf), "flags_h": len(hf),
                                **{f"margin_{k}": v for k, v in m_auc.items()},
                                **{f"rarity_{k}": v for k, v in r_auc.items()},
                                "source_slices": len(t_src), **{f"rarity_vs_t_{k}": v for k, v in t_auc.items()}})
            if len(qf):
                test = spearman_test(p_rarity, main_table["ratio"].to_numpy(), int(exp["rarity"]["permutations"]),
                                     draws, rng, y_draws=ratio_draws)
            else:
                test = {"n": 0, "rho": np.nan, "rho_low": np.nan, "rho_high": np.nan, "p_permutation": np.nan}
            h3_rows.append({"cell": cell, "primary": (cls, s) in primary, **test})
            if (cls, s) in primary:
                readings[cell] = cell_reading(main_conc, margin_rows[-1], float(exp["reading"]["auc_null"]))

    tables = {
        "exposure": pd.concat(exposure_rows, ignore_index=True),
        "concentration": pd.DataFrame(conc_rows),
        "margins": pd.DataFrame(margin_rows),
        "h3": pd.DataFrame(h3_rows),
        "h1": h1_tables(q, h, ref, e3, exp, draws, rng),
        "flag_details": pd.concat(details, ignore_index=True)[["set", "cell", "key", "cluster", "source",
                                                               "source_patient", "margin", "source_rarity"]],
    }
    tables["top_patients"] = top_patients(tables["exposure"], ref, primary, m_main)
    return tables, overall_reading(readings)


def cell_exposure(qf: pd.DataFrame, nn_patients: pd.Series, nn_clusters: pd.Series, weights: pd.Series, m: float,
                  exp: dict, draws: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Decision R for one cell and smoothing m: the exposure table with chained groups per patient
    and the null band, the concentration test (with the null's configured percentile of the
    dispersion), and the ratio draws."""
    table, ratio_draws, pi_draws = exposure_table(qf["source_patient"].to_numpy(), qf["cluster"].to_numpy(),
                                                  nn_patients.to_numpy(), nn_clusters.to_numpy(), weights, m, draws, rng)
    conc, band = concentration_test(table["observed"].to_numpy(), table["pi"].to_numpy(), pi_draws,
                                    int(exp["top_k"]), rng)
    d_null = conc.pop("dispersion_null")
    conc["dispersion_null_pct"] = float(np.percentile(d_null, float(exp["reading"]["dispersion_percentile"])))
    groups = chained(qf["source"], MAX_GAP)
    table["groups"] = table["patient_id"].map(groups.groupby(qf["source_patient"]).nunique()).fillna(0).astype(int)
    table[["null_low", "null_high"]] = band.to_numpy()
    return table, conc, ratio_draws


def flags(t: pd.DataFrame, cls: str, stratum: str) -> pd.DataFrame:
    """The flags of one cell: key, cluster, source slice (E5 decision J) and its patient."""
    src = sources(t, cls)
    src = src[t.loc[src.index, CELL_COLUMNS[cls][0]] == stratum]
    return pd.DataFrame({"key": t.loc[src.index, "key"].to_numpy(), "cluster": t.loc[src.index, "cluster"].to_numpy(),
                         "source": src.to_numpy(), "source_patient": [parse_slice_key(k)[0] for k in src]})


def flag_margins(q: QuerySet) -> dict[str, pd.Series]:
    """Each query's margin to the second-best patient, by key, on both scales: E1 the stored
    `margin`; E2 C(NN) minus the best C of another patient, for the E2-flagged queries only."""
    flagged = set(q.table.loc[q.table["e2_class"].isin(["full", "partial"]), "key"])
    e2 = q.e2
    sub = lesionmatch.LesionMatch(e2.queries, e2.lesions, e2.slice_pairs[e2.slice_pairs["query"].isin(flagged)],
                                  e2.lesion_pairs[e2.lesion_pairs["query"].isin(flagged)], e2.candidate_min_s)
    return {"e1": q.e1.set_index("query")["margin"], "e2": e2_margins(sub)}


def auc_row(a: pd.DataFrame, b: pd.DataFrame, col: str, draws: int, rng: np.random.Generator) -> dict:
    """AUC = P(a value of `a` > one of `b`), ties one half, with a two-level bootstrap interval
    (each side resampled by its `cluster` column); medians of both sides."""
    a, b = a.dropna(subset=[col]), b.dropna(subset=[col])
    out = {"median_q": float(a[col].median()) if len(a) else np.nan,
           "median_h": float(b[col].median()) if len(b) else np.nan, "auc": np.nan, "auc_low": np.nan, "auc_high": np.nan}
    if len(a) and len(b):
        boot = auc_bootstrap(a[col].to_numpy(), a["cluster"].to_numpy(), b[col].to_numpy(), b["cluster"].to_numpy(),
                             draws, rng)
        low, high = interval(boot)
        out.update({"auc": auc(a[col].to_numpy(), b[col].to_numpy()), "auc_low": low, "auc_high": high})
    return out


def cell_reading(conc: dict, margins: dict, auc_null: float) -> dict:
    """Decision U for one primary cell: patient copying if the dispersion lies above the configured
    percentile of its null and the margin AUC interval lies above auc_null; typical placement if the
    margin AUC interval lies wholly below auc_null (flags fit their best patient less uniquely than
    chance flags; added 2026-11-15 after the placement mock), or if it reaches auc_null or below and
    the source-rarity AUC interval lies below it; otherwise undetermined."""
    beyond = conc["dispersion"] > conc["dispersion_null_pct"]
    margin_above = margins["margin_auc_low"] > auc_null
    margin_below = margins["margin_auc_high"] < auc_null
    rarity_below = margins["rarity_auc_high"] < auc_null
    if beyond and margin_above:
        reading = "patient copying"
    elif margin_below or (not margin_above and rarity_below):
        reading = "typical placement"
    else:
        reading = "undetermined"
    return {"reading": reading, "dispersion_beyond_null": bool(beyond), "margin_auc_above": bool(margin_above),
            "margin_auc_below": bool(margin_below), "rarity_auc_below": bool(rarity_below)}


def overall_reading(readings: dict[str, dict]) -> dict:
    """The reading over the primary cells: their common reading, or "mixed" with each cell's."""
    labels = {r["reading"] for r in readings.values()}
    return {"reading": labels.pop() if len(labels) == 1 else "mixed", "cells": readings}


def h1_tables(q: QuerySet, h: QuerySet, ref: Reference, e3: dict, exp: dict, draws: int,
              rng: np.random.Generator) -> pd.DataFrame:
    """Decision W for both 3D lesion methods: lesion credits of the query set and H-eval (pairs
    s >= tau_L over every stored slice pair), Spearman(span, credits) per size bin and weighted."""
    edges = [int(b) for b in exp["h1_size_bins_px"]]
    parts = []
    for method in (exp["lesions3d"]["main"], exp["lesions3d"]["sensitivity"]):
        les = ref.lesions3d[(ref.lesions3d["method"] == method) & ref.lesions3d["informative"]].reset_index(drop=True)
        ids = pd.Index(les["lesion3d_id"])
        of = ref.lesion_map.set_index("lesion_id")[f"lesion3d_id_{method}"]
        lowers = calibration.area_bins(les["largest_informative_px"].to_numpy(), edges).to_numpy(dtype=float)
        bins = np.array([calibration.bin_label(int(lo), edges) for lo in lowers])
        counts, boot = {}, {}
        for name, qs in (("q", q), ("h", h)):
            scorable = qs.table[qs.table["n_informative"] > 0]
            tau = pd.Series(calibration.frozen_threshold(e3, "lesion", "tau_L", scorable["n_informative"].to_numpy()),
                            index=scorable["key"].to_numpy())
            pairs = qs.e2.lesion_pairs[qs.e2.lesion_pairs["query"].isin(tau.index)]
            credits = lesion_credits(pairs, tau, of)
            counts[name] = credit_counts(credits, ids)
            clusters = pd.Series(scorable["cluster"].to_numpy(), index=scorable["key"].to_numpy())
            if clusters.nunique() == len(clusters):
                boot[name] = credit_draws(credits, ids, draws, rng, n_queries=len(clusters))
            else:
                boot[name] = credit_draws(credits, ids, draws, rng, clusters=clusters)
        table = h1_table(les["span"].to_numpy(), bins, counts["q"], counts["h"], boot["q"], boot["h"],
                         order=[calibration.bin_label(e, edges) for e in edges])
        plain = spearman_test(les["span"].to_numpy(), counts["q"], 1000, draws, rng)
        table = pd.concat([table, pd.DataFrame([{"bin": "all (plain rho_s)", "lesions3d": len(les),
                                                 "credited_s": int((counts["q"] > 0).sum()), "rho_s": plain["rho"]}])],
                          ignore_index=True)
        parts.append(table.assign(method=method))
    return pd.concat(parts, ignore_index=True)


def top_patients(exposure: pd.DataFrame, ref: Reference, primary: list[tuple[str, str]], m: float) -> pd.DataFrame:
    """The most exposed patients per primary cell (by the lower bound of the ratio interval), with
    their features (E6 step 6)."""
    feats = ref.patients.set_index("patient_id")
    rows = []
    for cls, s in primary:
        d = exposure[(exposure["cell"] == f"{cls} {s}") & (exposure["m"] == m)]
        d = d.sort_values(["ratio_low", "observed"], ascending=False).head(TOP_PATIENTS)
        rows.append(d.join(feats, on="patient_id"))
    return pd.concat(rows, ignore_index=True)


# --- figures ----------------------------------------------------------------------------------

def draw_figures(tables: dict[str, pd.DataFrame], ref: Reference, exp: dict, prefix: str, out: Path,
                 highlight: set[str] | None = None) -> None:
    """The E6 figures of an analysed query set: observed against expected flags per patient, margin
    and source rarity of query and H-eval flags side by side, flags by patient-rarity group, H1.
    `highlight`: patients to ring in the exposure figure (the planted copies' base patients in the
    positive rehearsal)."""
    m = float(exp["expected"]["smoothing_m"])
    cells = tables["margins"].loc[tables["margins"]["primary"], "cell"].tolist()
    details = tables["flag_details"]
    flagged_in = {c: set(details.loc[(details["set"] == "query") & (details["cell"] == c), "source_patient"])
                  for c in cells}
    exposure_figure(tables["exposure"], cells, m, out / "exposure_sorted.png",
                    f"{prefix}: flags per patient, observed vs expected by chance. Dot = one T patient; grey bar = "
                    f"the flag counts chance gives that patient (95%); a dot above its bar gets more flags than chance",
                    {c: (highlight or set()) & flagged_in[c] for c in cells})
    margins_figure(tables["flag_details"], tables["margins"], cells, out / "margins.png",
                   f"{prefix}: query flags (red) vs H-eval chance flags (grey). Margin = how much better the best "
                   f"patient fits than the second-best; copying gives larger margins (AUC above 0.5)")
    rarity_figure(tables["exposure"], tables["h3"], ref, cells, m, out / "rarity_exposure.png",
                  f"{prefix}: flags by patient-rarity group (fifths of the 105 patients, most typical left). "
                  f"Bar = observed / expected flags of the group; 1 = as chance predicts (H3)")
    h1_figure(tables["h1"], out / "h1_span.png", f"{prefix}: Spearman(span, lesion credits) per size bin (H1)")


def exposure_figure(exposure: pd.DataFrame, cells: list[str], m: float, path: Path, title: str,
                    highlight: dict[str, set[str]]) -> Path:
    """Per primary cell: each T patient's observed flags against its expected flags (pi x flags),
    with the 95% range of counts its null gives (grey bar) and the diagonal observed = expected;
    patients whose ratio interval lies above 1 in red, the cell's highlighted patients ringed."""
    fig = Figure(figsize=(6 * max(len(cells), 1), 5))
    for ax, cell in zip(np.atleast_1d(fig.subplots(1, max(len(cells), 1))), cells):
        d = exposure[(exposure["cell"] == cell) & (exposure["m"] == m)]
        e = d["expected"].to_numpy()
        ax.vlines(e, d["null_low"] * e, d["null_high"] * e, color="0.85", lw=1.5, zorder=1, label="chance range (95%)")
        top = max(float(d["observed"].max()), float((d["null_high"] * e).max()), 1.0) * 1.2
        ax.plot([0, top], [0, top], color="k", lw=0.8, label="observed = expected")
        above = (d["ratio_low"] > 1).to_numpy()
        ax.scatter(e[~above], d["observed"][~above], s=14, color="tab:blue", zorder=3, label="patient")
        ax.scatter(e[above], d["observed"][above], s=20, color="tab:red", zorder=3, label="ratio interval above 1")
        ring = d["patient_id"].isin(highlight.get(cell, set())).to_numpy()
        if ring.any():
            ax.scatter(e[ring], d["observed"][ring], s=70, facecolors="none", edgecolors="tab:orange", lw=1.2,
                       zorder=4, label="planted base patient")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_xlabel("expected flags (chance, from H-eval)", fontsize=8)
        ax.set_ylabel("observed flags", fontsize=8)
        ax.set_title(f"{cell}: {int(d['observed'].sum())} flags, {int(above.sum())} patients clearly above chance",
                     fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    return save(fig, path, title)


def margins_figure(details: pd.DataFrame, margins: pd.DataFrame, cells: list[str], path: Path, title: str) -> Path:
    """Per primary cell: margin (top) and source rarity (bottom) of the query flags and the H-eval
    chance flags as side-by-side boxes (middle half of the values, line = median) with every flag
    as a dot; the AUC with its interval in the title."""
    fig = Figure(figsize=(5 * max(len(cells), 1), 7))
    axes = fig.subplots(2, max(len(cells), 1), squeeze=False)
    rng = np.random.default_rng(0)   # jitter of the dots only
    for j, cell in enumerate(cells):
        d = details[details["cell"] == cell]
        r = margins.set_index("cell").loc[cell]
        for i, (col, key, label) in enumerate((("margin", "margin", "margin to the second-best patient"),
                                                ("source_rarity", "rarity", "rarity of the source slice"))):
            ax = axes[i, j]
            groups = [d.loc[d["set"] == s, col].dropna().to_numpy() for s in ("query", "H-eval")]
            ax.boxplot(groups, positions=[1, 2], widths=0.5, showfliers=False)
            for x, values, color in zip((1, 2), groups, ("tab:red", "0.4")):
                ax.scatter(x + rng.uniform(-0.18, 0.18, len(values)), values, s=6, alpha=0.5, color=color)
            ax.set_xticks([1, 2], [f"query flags (n={len(groups[0])})", f"H-eval chance flags (n={len(groups[1])})"],
                          fontsize=8)
            ax.set_ylabel(label, fontsize=8)
            ax.set_title(f"{cell}: AUC {r[f'{key}_auc']:.2f} [{r[f'{key}_auc_low']:.2f}, {r[f'{key}_auc_high']:.2f}]",
                         fontsize=9)
    return save(fig, path, title)


def rarity_figure(exposure: pd.DataFrame, h3: pd.DataFrame, ref: Reference, cells: list[str], m: float, path: Path,
                  title: str) -> Path:
    """Per primary cell: the T patients in five groups by rarity (fifths), and for each group the
    summed observed flags over the summed expected flags; the patient-level Spearman (H3) in the
    title."""
    rarity = ref.patients.set_index("patient_id")["rarity"]
    fig = Figure(figsize=(5 * max(len(cells), 1), 4))
    for ax, cell in zip(np.atleast_1d(fig.subplots(1, max(len(cells), 1))), cells):
        d = exposure[(exposure["cell"] == cell) & (exposure["m"] == m)].copy()
        d["group"] = pd.qcut(d["patient_id"].map(rarity), 5, labels=["most typical", "2", "3", "4", "rarest"])
        g = d.groupby("group", observed=False)[["observed", "expected"]].sum()
        ax.bar(range(len(g)), g["observed"] / g["expected"], color="tab:purple")
        ax.axhline(1, color="k", lw=0.8)
        ax.set_xticks(range(len(g)), g.index.astype(str), fontsize=8)
        ax.set_ylabel("observed / expected flags", fontsize=8)
        r = h3.set_index("cell").loc[cell]
        ax.set_title(f"{cell}: Spearman rho {r['rho']:.2f} [{r['rho_low']:.2f}, {r['rho_high']:.2f}]", fontsize=9)
    return save(fig, path, title)


def h1_figure(h1: pd.DataFrame, path: Path, title: str) -> Path:
    """rho_s and rho_h per size bin and method, with the interval of their difference."""
    d = h1[~h1["bin"].isin(["weighted", "all (plain rho_s)"])]
    fig = Figure(figsize=(10, 3.5))
    for ax, (method, g) in zip(fig.subplots(1, 2), d.groupby("method", sort=False)):
        x = np.arange(len(g))
        ax.plot(x, g["rho_s"], "o", color="tab:red", label="query set")
        ax.plot(x, g["rho_h"], "s", color="tab:olive", label="H-eval")
        ax.set_xticks(x, g["bin"], fontsize=7)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_title(str(method), fontsize=9)
        ax.set_ylabel("Spearman(span, credits)", fontsize=8)
        ax.legend(fontsize=7)
    return save(fig, path, title)


# --- review -----------------------------------------------------------------------------------

def review_text(header: str, tables: dict[str, pd.DataFrame], reading: dict) -> str:
    """The review of an analysed query set: concentration, margins and rarity, H3, H1, top patients,
    and the reading of decision U for the primary cells."""
    conc = tables["concentration"]
    keep = ["cell", "m", "primary", "flags", "patients_flagged", "dispersion", "dispersion_null_median",
            "dispersion_null_pct", "dispersion_p", *[c for c in conc.columns if c.startswith("top")],
            "above_null_band", "above_null_band_expected"]
    mar = tables["margins"]
    mcols = ["cell", "primary", "flags_q", "flags_h", "margin_median_q", "margin_median_h", "margin_auc", "margin_auc_low",
             "margin_auc_high", "rarity_median_q", "rarity_median_h", "rarity_auc", "rarity_auc_low", "rarity_auc_high",
             "source_slices", "rarity_vs_t_auc", "rarity_vs_t_auc_low", "rarity_vs_t_auc_high"]
    tcols = ["cell", "patient_id", "observed", "groups", "heldout_nn", "expected", "ratio", "ratio_low", "ratio_high",
             "n_lesions3d", "max_lesions_per_slice", "rarity"]
    cells = "\n".join(f"  {c}: {r['reading']} (dispersion beyond null {r['dispersion_beyond_null']}, margin AUC above "
                      f"0.5 {r['margin_auc_above']}, margin AUC below 0.5 {r['margin_auc_below']}, source-rarity AUC "
                      f"below 0.5 {r['rarity_auc_below']})" for c, r in reading["cells"].items())
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.precision", 3):
        return (f"{header}\n\n"
                f"Concentration of flags over T patients (decision R; null = flags drawn from pi, pi redrawn by "
                f"H-eval patient):\n{conc[keep].to_string(index=False)}\n\n"
                f"Margin to the second-best patient and source-slice rarity, flags against H-eval chance flags "
                f"(decision U; AUC = P(query > H-eval)); source slices against all other T slices (H3, slice level):\n"
                f"{mar[mcols].to_string(index=False)}\n\n"
                f"H3, patient level: Spearman(patient rarity, exposure ratio), permutation p, interval with patients "
                f"and flags resampled:\n{tables['h3'].to_string(index=False)}\n\n"
                f"H1: Spearman(span, lesion credits) per size bin, query set (rho_s) vs H-eval (rho_h); H1 holds if "
                f"the weighted difference's interval is above 0 (decision W):\n{tables['h1'].to_string(index=False)}\n\n"
                f"Most exposed patients (primary cells, by the lower bound of the ratio):\n"
                f"{tables['top_patients'][tcols].to_string(index=False)}\n\n"
                f"Reading (decision U, primary cells): {reading['reading'].upper()}\n{cells}\n")


def write_tables(tables: dict[str, pd.DataFrame], out: Path) -> None:
    """Every table as `<name>.csv` in `out`."""
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
