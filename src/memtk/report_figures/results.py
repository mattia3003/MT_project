"""Report-figures stage `results`: compact figures for the report's Results sections 3.1-3.7
(`docs/report-synthesis.md`, figure register).

Each figure is a smaller, legible redraw of what an experiment already showed, from the tables that
experiment stored; nothing is recomputed or re-scored, and S is read only through E5's stored tables
(the stage checks E5's lock first). Image tiles read single slices (model-space BRAVO, S samples as
saved) only to show them; the E8 example tiles reuse E8's own measurement (`e8.paired.pair_images`).
Reads `results/` and `processed_data/`; writes `results/<pool>/report/results/` (PNG + PDF per figure,
`figures.csv`, run log).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from memtk import config
from memtk.calibration import load_frozen
from memtk.e5.common import lock_unchanged
from memtk.e8.common import paired_config, real_slices, synthetic_samples
from memtk.e8.paired import Z_DISPLAY, pair_images, settings
from memtk.io import LabelStack, read_table, slice_images
from memtk.paired import align
from memtk.plotting import fill, zoom_window
from memtk.report_figures.common import COLORS, figure, index_row, save_both, style, write_index
from memtk.stages import refuse_overwrite, require
from memtk.synthetic import load_sample

CHANCE_TAUS = (0.5, 0.7, 0.77)             # chance-rate lines shown; 0.77 = the provisional tau_L (bold)
CHANCE_EXAMPLES = 2                        # exact single-lesion chance matches shown, plus the top multi-lesion one
RECALL_VARIANTS = ("P3", "P4", "P5-erode", "P5-dilate", "P6-remove", "P6-add")   # P1, P2 are 1 everywhere
EXPOSURE_CELL, EXPOSURE_M = "e1_copy multi", 52.5   # E6's primary cell with the excess; decision R's smoothing
EXAMPLE_REPORT = "50_exposed_e1_copy_multi.png"   # E7 report used in the 3.6 figure
SCORE_PANELS = (("e1_score", "stratum_e1", ("whole_mask", "tau_D"), "E1 whole-mask score"),
                ("e2_C", "stratum_e2", ("lesion", "tau_C"), "E2 configuration score C"),
                ("e2_strongest_s", "stratum_e2", ("lesion", "tau_L"), "E2 strongest lesion pair s"))
HELDOUT_OUTLINE = "gold"                   # held-out masks on grey MRI (grey would not show)
SSIM_RANGE = (-0.2, 1.0)                   # display range of SSIM maps, as in E8's examples


# --- stage ------------------------------------------------------------------------------------

def run_results(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The Results figures of the report (sections 3.1-3.7), each from the stored tables of its
    experiment. Needs E0b stats, E3 bound-review, E4 validate, E5 analyze (lock unchanged), E6
    exposure, E7 reports, E8 rehearse and paired. Writes `results/<pool>/report/results/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "report") / "results"
    refuse_overwrite([out], overwrite)
    hashes = lock_unchanged(cfg, name, pool_name)
    res_real, res_pool = (config.resolve_path(cfg["results_root"]) / n for n in (name, pool_name))
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    for path, stage in ((res_pool / "e7" / "reports" / "report_index.csv", "e7_reports.py reports"),
                        (res_pool / "e8" / "paired" / "measures.csv", "e8_paired.py paired"),
                        (res_pool / "e6" / "exposure" / "flag_details.csv", "e6_exposure.py exposure")):
        require(path, stage)
    main_keys = s_main_keys(syn_dir)
    e3, _ = load_frozen(cfg, name)

    rows = []
    with style():
        rows.append(lesions_figure(real_dir, syn_dir, main_keys, out))
        rows.append(chance_figure(cfg, name, res_real / "e3" / "bound_review" / "chance_rates.csv",
                                  real_dir / "e4" / "e1_heval.csv", int(e3["informative_bound_px"]), out))
        rows.append(recall_figure(res_real / "e4" / "validate" / "recall_by_group.csv", out))
        rows.append(scores_figure(syn_dir / "e5" / "classes_main.csv", main_keys,
                                  real_dir / "e4" / "heval_classes.csv", res_pool / "e5" / "analyze" / "auc.csv",
                                  e3, out))
        rows.append(exposure_figure(res_pool / "e6" / "exposure", out))
        rows.append(h1_figure(res_pool / "e6" / "exposure" / "h1_rates.csv", out))
        rows.append(report_figure(cfg, name, pool_name, pool_cfg, res_pool, out))
        rows.append(ssim_figure(cfg, name, pool_name, e3, res_pool / "e8" / "paired",
                                res_real / "e8" / "rehearse", out))
    write_index(rows, out)
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "results", "pool": pool_name, **hashes})
    print(pd.DataFrame(rows)[["figure", "section"]].to_string(index=False))
    print(f"Wrote {out}")


# --- inputs and shared drawing ------------------------------------------------------------------

def s_main_keys(syn_dir: Path) -> set[str]:
    """Keys of S main (mask class clean or minor; E0b decision D, E5 decision L)."""
    qc = read_table(syn_dir / "qc.csv")
    return set(qc.loc[qc["mask_class"].isin(["clean", "minor"]), "key"])


def image_tile(ax, image: np.ndarray, outlines: list[tuple[np.ndarray, str]], title: str,
               display: tuple[float, float] = (0.0, 1.0), cmap: str = "gray") -> None:
    """One image tile without axes: the image in a fixed display range, coloured mask outlines."""
    ax.imshow(image, cmap=cmap, vmin=display[0], vmax=display[1], interpolation="nearest")
    for mask, color in outlines:
        if mask.any():
            ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=0.9)
    ax.set_title(title, fontsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])


def box_strip(ax, data: list[np.ndarray], colors: list[str], rng: np.random.Generator, alpha: float = 0.5) -> None:
    """Box plots (median, quartiles, whiskers at 1.5 IQR, no fliers) with every value as a jittered point."""
    ax.boxplot(data, widths=0.5, showfliers=False, medianprops={"color": "k"})
    for i, (v, color) in enumerate(zip(data, colors), start=1):
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(v)), v, s=2.5, alpha=alpha if len(v) > 200 else 0.8, lw=0,
                   color=color)


# --- 3.1: synthetic vs real lesions --------------------------------------------------------------

def lesions_figure(real_dir: Path, syn_dir: Path, main_keys: set[str], out: Path) -> dict:
    """(a) Lesion area (cumulative share, log scale) of T, H (H-cal + H-eval) and S main; (b-e) the
    density of lesion centroids of T and S main in pixels (top) and relative to each slice's brain box
    (bottom; lesions of slices without a brain left out), one colour scale per row."""
    real, syn = read_table(real_dir / "lesions.csv"), read_table(syn_dir / "lesions.csv")
    sets = {"T": real[real["set"] == "T"], "H": real[real["set"].isin(["H-cal", "H-eval"])],
            "S": syn[syn["key"].isin(main_keys)]}
    fig = figure(10.0)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.25, 1, 1])
    ax_area = fig.add_subplot(grid[:, 0])
    for label, les in sets.items():
        x = np.sort(les["area"].to_numpy())
        ax_area.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=COLORS[label],
                     label=f"{'S main' if label == 'S' else label}: median {np.median(x):.0f} px")
    ax_area.set_xscale("log")
    ax_area.set_xlabel("lesion area (px, log scale)")
    ax_area.set_ylabel("share of lesions with at most this area")
    ax_area.axhline(0.5, color="0.7", lw=0.6, ls=":")
    ax_area.legend(frameon=False, loc="lower right", fontsize=6.5)
    ax_area.set_title("(a) lesion area")
    frames = (("", (0, 128), 32, "pixels", "row (px)", "column (px)"),
              ("_brain", (-0.2, 1.2), 28, "brain box", "row (fraction)", "column (fraction)"))
    for r, (suffix, lim, n_bins, frame, ylab, xlab) in enumerate(frames):
        hists = {}
        for k in ("T", "S"):
            d = sets[k][[f"centroid_row{suffix}", f"centroid_col{suffix}"]].dropna()
            hists[k] = np.histogram2d(d.iloc[:, 0], d.iloc[:, 1], bins=n_bins, range=[lim, lim])[0] / len(d)
        norm = Normalize(0, max(float(h.max()) for h in hists.values()))
        axes = [fig.add_subplot(grid[r, c]) for c in (1, 2)]
        for c, (ax, k) in enumerate(zip(axes, ("T", "S"))):
            ax.imshow(hists[k], cmap="magma", norm=norm, extent=(lim[0], lim[1], lim[1], lim[0]))
            ax.set_title(f"({'bcde'[2 * r + c]}) {'T' if k == 'T' else 'S main'}, {frame}")
            ax.set_xlabel(xlab, fontsize=6.5)
            ax.set_ylabel(ylab if c == 0 else "", fontsize=6.5)
            ax.tick_params(labelsize=6)
        fig.colorbar(ScalarMappable(norm=norm, cmap="magma"), ax=axes, shrink=0.85, label="share of lesions")
    name = "3-1_lesions_synthetic_vs_real"
    save_both(fig, out, name)
    return index_row(name, "3.1", "lesion area of T, H and S main; lesion-centroid density of T and S main in "
                     "pixels and relative to the brain box",
                     [real_dir / "lesions.csv", syn_dir / "lesions.csv", syn_dir / "qc.csv"])


# --- 3.2: chance rates, the bound and what chance matches look like ------------------------------------

def chance_figure(cfg: dict, name: str, path: Path, e1_heval_path: Path, bound: int, out: Path) -> dict:
    """(a) Chance lesion-match rate of H-cal lesions by area bin for a few candidate tau_L (the provisional
    one bold), the 5% and 10% cuts and the chosen bound (decision G). (b) What chance matches look like:
    held-out H-eval slices (never trained on) next to the training slice E1 matches them to, the exact
    single-lesion matches (score 1.0) and the highest-scoring multi-lesion one."""
    rates = read_table(path)
    bins = rates.drop_duplicates("bin")[["bin", "lower", "lesions"]].reset_index(drop=True)
    x = np.arange(len(bins))
    fig = figure(11.0)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.35, 1])  # pyright: ignore[reportGeneralTypeIssues]  # matplotlib stub: an array of SubFigures for 2 rows
    ax = top.subplots()
    shades = dict(zip(CHANCE_TAUS, ("0.75", "0.5", "k")))
    for tau in CHANCE_TAUS:
        r = rates[np.isclose(rates["tau_L"].astype(float), tau)].set_index("bin").reindex(bins["bin"])
        bold = np.isclose(tau, max(CHANCE_TAUS))
        ax.plot(x, r["rate"], marker="o", ms=3, color=shades[tau], lw=1.8 if bold else 0.9,
                label=f"τ_L = {tau}" + (" (provisional)" if bold else ""))
    for cut in (0.05, 0.10):
        ax.axhline(cut, color="0.5", lw=0.6, ls="--")
        ax.text(len(bins) - 0.5, cut, f"{cut:.0%}", va="bottom", ha="right", fontsize=6, color="0.4")
    first = int(np.flatnonzero(bins["lower"].astype(int) >= bound)[0])
    ax.axvspan(first - 0.5, len(bins) - 0.5, color="tab:green", alpha=0.08, lw=0)
    ax.text(first - 0.4, 0.97, f"informative (≥ {bound} px)", transform=ax.get_xaxis_transform(), fontsize=6,
            va="top", color="tab:green")
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(bins["bin"], bins["lesions"])], fontsize=6)
    ax.set_xlabel("H-cal lesion area")
    ax.set_ylabel("chance-match rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1, 0.5))
    ax.set_title("(a) share of held-out lesions that find a training lesion this similar by chance")

    e1 = read_table(e1_heval_path)
    exact = e1[(e1["stratum"] == "single") & (e1["score"].astype(float) >= 0.999)].head(CHANCE_EXAMPLES)
    multi = e1[e1["stratum"] == "multi"].sort_values("score").tail(1)
    examples = pd.concat([exact, multi])
    real_labels = LabelStack(config.output_dir(cfg, name) / "labels.npz")
    images = slice_images(cfg, name, [*examples["query"], *examples["nn"]])
    tiles = bottom.subplots(1, 2 * len(examples))
    for i, r in enumerate(examples.to_dict("records")):
        q, t = str(r["query"]), str(r["nn"])
        dr, dc = int(r["shift_row"]), int(r["shift_col"])
        q_mask, t_mask = align(real_labels[q] > 0, dr, dc), real_labels[t] > 0
        window = zoom_window([q_mask, t_mask])
        q_px = int((real_labels[q] > 0).sum())
        image_tile(tiles[2 * i], align(images[q], dr, dc)[window], [(q_mask[window], HELDOUT_OUTLINE)],
                   f"held-out\n{q}\n{q_px} px")
        image_tile(tiles[2 * i + 1], images[t][window], [(t_mask[window], COLORS["T"])],
                   f"training\n{t}\nE1 score {float(r['score']):.2f}")
    bottom.suptitle("(b) chance matches: H-eval slices (never trained on, yellow) and the training slice E1 "
                    "matches them to (blue)", fontsize=8)
    out_name = "3-2_chance_rates"
    save_both(fig, out, out_name)
    return index_row(out_name, "3.2", "chance lesion-match rate by lesion area and the 12 px bound; examples of "
                     "H-eval chance matches (exact single-lesion and the top multi-lesion)", [path, e1_heval_path])


# --- 3.3: recall by lesion size ----------------------------------------------------------------

def recall_figure(path: Path, out: Path) -> dict:
    """Share of planted masks caught (decision J) by the size of the base's largest lesion, per
    variant, for E1, E2 and either (P1 and P2 are 100% everywhere and left out)."""
    table = read_table(path)
    table = table[table["by"] == "size"]
    fig = figure(8.0)
    axes = fig.subplots(2, 3, sharey=True)
    for ax, variant in zip(axes.ravel(), RECALL_VARIANTS):
        d = table[table["variant"] == variant].reset_index(drop=True)
        x = np.arange(len(d))
        for col, label, ls in (("e1", "E1", "-"), ("e2", "E2", "-"), ("any", "either", ":")):
            ax.plot(x, d[col].astype(float), marker="o" if ls == "-" else None, ms=3, ls=ls, color=COLORS[label],
                    label=label)
        ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(d["group"], d["masks"])], fontsize=6)
        ax.set_title(variant)
        ax.set_ylim(-0.03, 1.03)
    for ax in axes[:, 0]:
        ax.set_ylabel("share caught")
    axes[0, 0].legend(frameon=False, loc="lower right")
    name = "3-3_recall_by_size"
    save_both(fig, out, name)
    return index_row(name, "3.3", "planted copies caught by size of the largest base lesion, per variant and "
                     "detector (P1, P2: 100% everywhere)", [path])


# --- 3.4: score distributions, multi-lesion --------------------------------------------------------

def scores_figure(s_path: Path, main_keys: set[str], h_path: Path, auc_path: Path, e3: dict, out: Path) -> dict:
    """Multi-lesion score distributions of S main and H-eval (density) for the E1 score, E2's C and the
    strongest lesion pair, with the frozen threshold and the AUC (patient bootstrap) in each title."""
    s = read_table(s_path)
    s, h, auc = s[s["key"].isin(main_keys)], read_table(h_path), read_table(auc_path)
    bins = np.linspace(0, 1, 41)
    fig = figure(5.4)
    axes = fig.subplots(1, 3)
    for ax, (col, stratum_col, (det, thr), label) in zip(axes, SCORE_PANELS):
        for t, set_name, color in ((s, "S main", COLORS["S"]), (h, "H-eval", COLORS["H-eval"])):
            v = t.loc[t[stratum_col] == "multi", col].astype(float).dropna()
            ax.hist(v, bins=bins, density=True, histtype="step", color=color, lw=1.1, label=set_name)
        ax.axvline(float(e3["thresholds"][det][thr]["multi"]), ls="--", color="k", lw=0.8)
        a = auc[(auc["score"] == col) & (auc["stratum"] == "multi")].iloc[0].to_dict()
        ax.set_title(f"{label}\nAUC {float(a['auc']):.2f} [{float(a['auc_low']):.2f}, {float(a['auc_high']):.2f}]")
        ax.set_xlabel("score")
        ax.set_yticks([])
    axes[0].set_ylabel("density")
    axes[-1].legend(frameon=False, loc="upper left")
    name = "3-4_score_distributions_multi"
    save_both(fig, out, name)
    return index_row(name, "3.4", "multi-lesion score distributions, S main vs H-eval, frozen thresholds dashed, "
                     "AUC with patient-bootstrap interval", [s_path, h_path, auc_path])


# --- 3.5: exposure, margins and rarity; H1 ------------------------------------------------------------

def exposure_figure(e6_dir: Path, out: Path) -> dict:
    """(a) Observed against expected flags per T patient in the E1 multi-lesion cell (decision R), with
    each patient's 95% chance band; patients above their band in red. (b, c) Margin to the second-best
    patient and source rarity of the S flags against H-eval's chance flags (decision U), with the AUCs."""
    exposure, details = read_table(e6_dir / "exposure.csv"), read_table(e6_dir / "flag_details.csv")
    m = read_table(e6_dir / "margins.csv").set_index("cell").loc[EXPOSURE_CELL].to_dict()
    d = exposure[(exposure["cell"] == EXPOSURE_CELL) & np.isclose(exposure["m"].astype(float), EXPOSURE_M)]
    e, o = d["expected"].astype(float).to_numpy(), d["observed"].astype(float).to_numpy()
    low, high = d["null_low"].astype(float).to_numpy() * e, d["null_high"].astype(float).to_numpy() * e
    above = o > high
    fig = figure(9.0)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.vlines(e, low, high, color="0.85", lw=1.6, zorder=1, label="chance band (95%)")
    top = max(float(o.max()), float(high.max()), 1.0) * 1.2
    ax.plot([0, top], [0, top], color="k", lw=0.7, label="observed = expected")
    ax.scatter(e[~above], o[~above], s=9, color=COLORS["T"], zorder=3, label="T patient")
    ax.scatter(e[above], o[above], s=13, color=COLORS["S"], zorder=3, label=f"above its band ({int(above.sum())})")
    for scale in (ax.set_xscale, ax.set_yscale):
        scale("symlog", linthresh=1)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("expected flags (chance, from H-eval)")
    ax.set_ylabel("observed S flags")
    ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    ax.set_title(f"(a) E1 copy, multi-lesion: {int(o.sum()):,} flags")
    rng = np.random.default_rng(0)                      # point jitter only
    for j, (col, label, auc) in enumerate((("margin", "margin to\nsecond-best patient", "margin"),
                                           ("source_rarity", "rarity of\nthe source slice", "rarity"))):
        ax = fig.add_subplot(grid[j, 1])
        data = [details.loc[(details["cell"] == EXPOSURE_CELL) & (details["set"] == s), col].astype(float).to_numpy()
                for s in ("query", "H-eval")]
        box_strip(ax, data, [COLORS["S"], COLORS["H-eval"]], rng, alpha=0.2)
        ax.set_xticks([1, 2], [f"S flags ({len(data[0]):,})", f"H-eval chance flags ({len(data[1])})"], fontsize=6.5)
        ax.set_ylabel(label, fontsize=6.5)
        ax.set_title(f"({'bc'[j]}) {auc}: AUC {float(m[f'{auc}_auc']):.2f} "
                     f"[{float(m[f'{auc}_auc_low']):.2f}, {float(m[f'{auc}_auc_high']):.2f}]")
    name = "3-5a_exposure_margins"
    save_both(fig, out, name)
    return index_row(name, "3.5", "E1 multi-lesion cell: observed vs expected flags per patient with chance bands; "
                     "margin and source rarity of S flags vs H-eval chance flags (typical placement)",
                     [e6_dir / f for f in ("exposure.csv", "flag_details.csv", "margins.csv")])


def h1_figure(path: Path, out: Path) -> dict:
    """H1 (decision W): matches per 3D lesion by slice span, relative to the size bin's average, S
    against H-eval, one panel per lesion-size bin (slice-to-slice linking)."""
    rates = read_table(path)
    rates = rates[rates["method"] == "slice_linking"]
    bins = list(rates["bin"].drop_duplicates())
    fig = figure(5.2)
    axes = fig.subplots(1, len(bins))
    for k, (ax, b) in enumerate(zip(axes, bins)):
        for s, color, label in (("query", COLORS["S"], "S"), ("H-eval", COLORS["H-eval"], "H-eval")):
            d = rates[(rates["bin"] == b) & (rates["set"] == s)].reset_index(drop=True)
            x = np.arange(len(d)) + (0.08 if s == "H-eval" else -0.08)
            y, lo, hi = (d[c].astype(float).to_numpy() for c in ("relative", "relative_low", "relative_high"))
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", ms=3, capsize=1.5, lw=0.9, color=color, label=label)
            ax.set_xticks(np.arange(len(d)), [str(v).replace(">= ", "≥") for v in d["span"]], fontsize=6)
        ax.axhline(1, color="0.7", lw=0.6)
        ax.set_title(f"lesions {str(b).replace('>= ', '≥ ')}")
        ax.set_xlabel("slice span")
        if k == 0:
            ax.set_ylabel("matches per lesion\n(1 = bin average)")
            ax.legend(frameon=False, loc="upper left")
    name = "3-5b_h1_rates"
    save_both(fig, out, name)
    return index_row(name, "3.5", "H1: matches per 3D lesion by slice span relative to the size bin's average, "
                     "S vs H-eval", [path])


# --- 3.6: a match report and its patient --------------------------------------------------------------

def report_figure(cfg: dict, name: str, pool_name: str, pool_cfg: dict, res_pool: Path, out: Path) -> dict:
    """E7 report 50 in compact form (the synthetic mask on its MRI moved by E1's shift, the named
    training slice, the masks overlaid) with the largest lesion pair's factors, and the source patient's
    profile (S flags per slice along its metastasis, with the lesion area)."""
    index = read_table(res_pool / "e7" / "reports" / "report_index.csv")
    r = index[index["report"] == EXAMPLE_REPORT].iloc[0].to_dict()
    key, patient = str(r["key"]), str(r["source_patient"])
    syn_dir = config.output_dir(cfg, pool_name)
    fig = figure(12.0)
    grid = fig.add_gridspec(3, 3, height_ratios=[1, 0.16, 0.85])
    source = match_tiles(fig, [grid[0, j] for j in range(3)], grid[1, :], cfg, name, pool_name, pool_cfg, key)
    prof = read_table(res_pool / "e7" / "reports" / "patient_profiles.csv")
    ax = fig.add_subplot(grid[2, :])
    profile_bars(ax, prof[prof["patient_id"] == patient], f"slice z of {patient} (one metastasis over {{n}} slices)",
                 "lesion area (px)")
    ax.set_title(f"(d) {patient}: S flags per slice along its metastasis")
    out_name = "3-6_report50_profile"
    save_both(fig, out, out_name)
    return index_row(out_name, "3.6", f"E7 report 50 ({key} vs {source}) compact, and {patient}'s profile",
                     [res_pool / "e7" / "reports" / f for f in ("report_index.csv", "patient_profiles.csv")]
                     + [syn_dir / "e5" / "e1_main.csv", syn_dir / "e5" / "e2_main_lesion_pairs.parquet",
                        "the sample's NIfTI", "T BRAVO"])


def match_tiles(fig, cells: list, note_cell, cfg: dict, name: str, pool_name: str, pool_cfg: dict, key: str) -> str:
    """A match report in compact form, drawn into three gridspec `cells` and a `note_cell`: (a) the S
    sample's MRI and mask moved by E1's shift, (b) E1's nearest training slice (the slice the shift belongs
    to), (c) the masks overlaid with the E1 score; the note gives the largest lesion pair's three factors
    against that slice (E2's lesion-pair table). Returns the training slice's key."""
    syn_dir = config.output_dir(cfg, pool_name)
    e1_table = read_table(syn_dir / "e5" / "e1_main.csv")
    e1 = e1_table[e1_table["query"] == key].iloc[0].to_dict()
    source = str(e1["nn"])
    pairs = pd.read_parquet(syn_dir / "e5" / "e2_main_lesion_pairs.parquet")
    pairs = pairs[(pairs["query"] == key) & (pairs["train"] == source)].sort_values("s")
    files = read_table(syn_dir / "synthetic_index.csv").set_index("sample_id")["filename"]
    q_image, _ = load_sample(config.resolve_path(pool_cfg["dir"]) / files[key], pool_cfg)
    q_labels = LabelStack(syn_dir / "labels.npz")[key]
    t_labels = LabelStack(config.output_dir(cfg, name) / "labels.npz")[source]
    t_image = slice_images(cfg, name, [source])[source]
    dr, dc = int(e1["shift_row"]), int(e1["shift_col"])
    q_moved, q_img_moved = align(q_labels > 0, dr, dc), align(q_image, dr, dc)
    window = zoom_window([q_moved, t_labels > 0])
    image_tile(fig.add_subplot(cells[0]), q_img_moved[window], [(q_moved[window], COLORS["S"])],
               f"(a) synthetic {key}\nmoved ({dr:+d}, {dc:+d})")
    image_tile(fig.add_subplot(cells[1]), t_image[window], [((t_labels > 0)[window], COLORS["T"])],
               f"(b) named source\n{source}")
    ax = fig.add_subplot(cells[2])
    q, t = q_moved[window], (t_labels > 0)[window]
    image_tile(ax, np.ones(q.shape), [], f"(c) overlay\nE1 score {float(e1['score']):.2f}")
    fill(ax, [(q & t, "tab:green"), (q & ~t, COLORS["S"]), (t & ~q, COLORS["T"])], alpha=0.9)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.6")
    if len(pairs):
        pair = pairs.iloc[-1].to_dict()
        factors = (f"Largest lesion pair: shape {float(pair['shape']):.2f} × position {float(pair['position']):.2f} × "
                   f"size {float(pair['size']):.2f} = s {float(pair['s']):.2f}")
    else:
        factors = "No lesion pair with s ≥ 0.1 against this slice"
    note = fig.add_subplot(note_cell)
    note.axis("off")
    note.text(0.5, 0.5, f"overlay: green = both, red = synthetic only, blue = source only.   {factors}", ha="center",
              va="center", fontsize=6.5, transform=note.transAxes)
    return source


def profile_bars(ax, prof: pd.DataFrame, xlabel: str, area_label: str, legend_loc: str = "upper right") -> None:
    """A patient profile (E7's `patient_profiles` rows of one patient): S flags naming each slice, E1 and
    E2 side by side, with the slice's lesion area (summed over its lesions) on a second axis. `xlabel`
    may contain `{n}`, the number of slices spanned."""
    per_z = prof.groupby("z")[["e1_flags", "e2_flags", "area"]].sum().astype(float)
    zs = np.arange(int(per_z.index.min()), int(per_z.index.max()) + 1)
    per_z = per_z.reindex(zs, fill_value=0)
    ax.bar(zs - 0.2, per_z["e1_flags"], width=0.4, color=COLORS["E1"], label="E1 flags naming the slice")
    ax.bar(zs + 0.2, per_z["e2_flags"], width=0.4, color=COLORS["E2"], label="E2 flags (full or partial)")
    ax.set_xlabel(xlabel.format(n=len(zs)))
    ax.set_ylabel("S flags (all strata)")
    ax.set_xticks(zs[::2] if len(zs) <= 30 else zs[zs % 10 == 0])
    twin = ax.twinx()
    twin.plot(zs, per_z["area"], color="0.4", lw=0.9, ls=":", marker=".", ms=3)
    twin.set_ylabel(area_label, color="0.4")
    twin.spines["right"].set_visible(True)
    ax.legend(frameon=False, loc=legend_loc)


# --- 3.7: paired-MRI SSIM ------------------------------------------------------------------------

def ssim_figure(cfg: dict, name: str, pool_name: str, e3: dict, paired_dir: Path, rehearse_dir: Path,
                out: Path) -> dict:
    """(a) Lesion-neighbourhood SSIM of S group A and B next to H-eval flagged and novel (pairs of kind
    t), with the AUCs of the pre-stated test. (b) Example pairs, each drawn by E8's own measurement: a
    typical group-A case (median SSIM), the strongest group-A case and, for scale, a rehearsal positive
    (a planted P1 copy whose image is the base patient's BRAVO two slices away; median of the mock)."""
    s, h = read_table(paired_dir / "measures.csv"), read_table(rehearse_dir / "heval" / "measures.csv")
    pos = read_table(rehearse_dir / "positive" / "measures.csv")
    tests = read_table(paired_dir / "tests.csv").set_index("region").loc["lesion"].to_dict()
    groups = [(s, "A", "S group A"), (s, "B", "S group B"), (h, "A", "H-eval\nflagged"), (h, "B", "H-eval\nnovel")]
    data = [d.loc[(d["kind"] == "t") & (d["group"] == g), "ssim_lesion"].astype(float).dropna().to_numpy()
            for d, g, _ in groups]
    fig = figure(11.0)
    grid = fig.add_gridspec(3, 4, width_ratios=[1.6, 1, 1, 1])
    ax = fig.add_subplot(grid[:, 0])
    box_strip(ax, data, [COLORS["S"], COLORS["H-eval"], COLORS["S"], COLORS["H-eval"]], np.random.default_rng(0))
    ax.set_xticks(range(1, 5), [f"{lab}\n({len(v)})" for (_, _, lab), v in zip(groups, data)], fontsize=6)
    ax.set_ylabel("mean SSIM, lesion neighbourhood")
    ax.set_title(f"(a) A vs B: AUC {float(tests['auc']):.2f}\n[{float(tests['auc_low']):.2f}, "
                 f"{float(tests['auc_high']):.2f}]; H-eval {float(tests['auc_ref']):.2f}", fontsize=7)

    a = s[(s["kind"] == "t") & (s["group"] == "A")].sort_values("ssim_lesion").reset_index(drop=True)
    p_t = pos[(pos["kind"] == "t") & (pos["group"] == "A") & (pos["reason"] == "planted P1")]
    p_t = p_t.sort_values("ssim_lesion").reset_index(drop=True)
    rows = [("(b) typical S case", a.iloc[len(a) // 2].to_dict(), "S"),
            ("(c) strongest S case", a.iloc[-1].to_dict(), "S"),
            ("(d) positive mock, z + 2", p_t.iloc[len(p_t) // 2].to_dict(), "planted")]
    st = settings(cfg, paired_config(cfg), e3)
    samples = synthetic_samples(cfg, pool_name, [str(r["query"]) for _, r, kind in rows if kind == "S"])
    real = real_slices(cfg, name, [*[str(r["comparison"]) for _, r, _ in rows],
                                   *[str(r["image"]) for _, r, kind in rows if kind == "planted"]])
    pool_labels = LabelStack(config.output_dir(cfg, pool_name) / "labels.npz")
    real_labels = LabelStack(config.output_dir(cfg, name) / "labels.npz")
    for i, (label, r, kind) in enumerate(rows):
        comparison = str(r["comparison"])
        if kind == "S":
            image, saved = samples[str(r["query"])]
            labels = pool_labels[str(r["query"])]
        else:  # P1: the planted mask is the base mask; its image is the base patient's BRAVO two slices away
            image, saved, labels = real[str(r["image"])][0], real[comparison][1], real_labels[comparison]
        t_image, t_saved = real[comparison]
        imgs, _, _ = pair_images(image, saved, labels, t_image, t_saved, int(r["shift_row"]), int(r["shift_col"]), st)
        if imgs is None:
            continue
        region = imgs.masks["lesion"]
        who = str(r["query"]) if kind == "S" else f"BRAVO {r['image']}"
        tiles = ((imgs.query, f"{label}\n{who}", "gray", Z_DISPLAY),
                 (imgs.comparison, f"comparison\n{comparison}", "gray", Z_DISPLAY),
                 (imgs.smap, f"SSIM map\nlesion region {float(r['ssim_lesion']):.2f}", "viridis", SSIM_RANGE))
        for j, (img, title, cmap, display) in enumerate(tiles):
            image_tile(fig.add_subplot(grid[i, j + 1]), img, [(region, "orange")], title, display, cmap)
    out_name = "3-7_ssim_lesion"
    save_both(fig, out, out_name)
    return index_row(out_name, "3.7", "lesion-neighbourhood SSIM: S group A vs B, H-eval flagged vs novel; example "
                     "pairs (typical and strongest S case, rehearsal positive) with SSIM maps",
                     [paired_dir / "measures.csv", paired_dir / "tests.csv", rehearse_dir / "heval" / "measures.csv",
                      rehearse_dir / "positive" / "measures.csv", "S samples' NIfTI", "T BRAVO"])
