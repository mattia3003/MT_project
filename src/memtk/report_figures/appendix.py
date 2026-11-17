"""Report-figures stage `appendix`: the figures the report shows only in its appendix
(`docs/report-synthesis.md`, section 3 and the appendix plan), at print size and resolution.

The appendix is slim (decided 2026-11-17): only figures that back a decision or a claim in the text and
are not already shown in the main text; the rest is a number in the text. Most are compact redraws in
the report style (16 cm wide, 6.5-8 pt text) from the tables the experiments stored. Two image sheets
(brain-mask methods, near-exact copies) are the experiments' own functions with fewer examples, shrunk to
the column width with their text kept at its point size (`plotting.resolution(..., max_width_in)`).
Example tiles are picked with this stage's seeded generator. Nothing is re-scored, and no experiment
stage is re-run (that would rewrite run logs that E5's lock checks).

Appendix B (E0, E3: data, preprocessing, calibration), C (E4: validation), D (E5, E6: audit and
exposure), E (E7, E8: match reports, paired MRI). S is read only through E5's stored tables (E5's lock
is checked) and, for the image tiles, its saved samples. Writes `results/<pool>/report/appendix/` (PNG
and PDF per figure, `figures.csv`, run log).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from memtk import calibration, config
from memtk.e0b.brain import method_grid
from memtk.e3.common import load_hcal_tables
from memtk.e4.validate import criterion_caught, criterion_variants
from memtk.e5.analyze import examples_figure
from memtk.e5.common import lock_unchanged
from memtk.e6.common import exposure_config, load_reference
from memtk.io import LabelStack, load_real, read_table, slice_images
from memtk.lesions import brain_mask
from memtk.plotting import resolution, zoom_window
from memtk.report_figures.common import COLORS, DPI, WIDTH_IN, figure, index_row, save_both, style, write_index
from memtk.report_figures.results import image_tile, match_tiles, profile_bars
from memtk.sets import load_planted, load_set
from memtk.stages import decided, pick, refuse_overwrite
from memtk.synthetic import load_sample

BRAIN_EXAMPLES = (2, 1)            # T slices where plain Otsu leaves a lesion mostly outside, and random ones
BRAIN_METHODS_SHOWN = ("otsu", "otsu_all", "low_threshold_all")
SPECK_GAPS = (("gap 1", 1, 1), ("gap 2", 2, 1), ("gap 3", 3, 1), ("gap 4-5", 4.5, 2), ("gap 6-10", 8, 5))
MISSES_SHOWN = 4                   # missed planted copies shown (two per row)
NEAR_EXACT_SHOWN = 6               # near-exact S masks shown (highest E1 score first)
PRIMARY_CELLS = (("e1_copy multi", "E1 copy, multi-lesion"), ("e2_full multi", "E2 full, multi-lesion"))
SENSITIVITY_ROWS = (("main", "main", "main analysis"), ("main", "percentile_99", "99th-percentile thresholds"),
                    ("main", "sigma_0.08", "σ = 0.08 × 128 px"), ("main", "bound_10", "informative bound 10 px"),
                    ("main", "bound_25", "informative bound 25 px"), ("main", "drop_only", "drop-only lesions"),
                    ("main+noisy", "main", "noisy masks included"))
REHEARSALS = (("null", "null mock (H-cal)"), ("positive", "positive mock (planted)"),
              ("placement_compact", "placement mock (compact)"))
REPORT_KEY = "02_near-exact.png"   # the E7 report redrawn in appendix E (E2's 1 px rounding limit, caught by E1)
PROFILE_PATIENT = "Mets_058"       # the patient profile in appendix E (17 metastases)
PROFILE_EXAMPLES = 3               # S masks shown next to the patient's most-matched slice
WITHIN_REGIONS = (("lesion", "lesion neighbourhood", "k"), ("brain", "whole brain", "0.55"),
                  ("ring", "ring", "tab:green"))


# --- stage ------------------------------------------------------------------------------------

def run_appendix(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The appendix figures at print size, 300 dpi. Needs E0b (brain, stats, lesions-review), E3 calibrate,
    E4 validate, E5 analyze (lock unchanged), E6 rehearse and exposure, E7 reports, E8 paired. Writes
    `results/<pool>/report/appendix/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "report") / "appendix"
    refuse_overwrite([out], overwrite)
    hashes = lock_unchanged(cfg, name, pool_name)
    rows = []
    with style(), resolution(DPI, for_print=True, max_width_in=WIDTH_IN):
        rows += e0_figures(cfg, name, pool_name, out)
        rows += e3_figures(cfg, name, out)
        rows += e4_figures(cfg, name, out)
        s = SLoader(cfg, name, pool_name, pool_cfg)
        rows += e5_figures(cfg, pool_name, s, out)
        rows += e6_figures(cfg, name, pool_name, out)
        rows += e7_figures(cfg, name, pool_name, pool_cfg, s, out)
        rows += e8_figures(cfg, pool_name, out)
    write_index(rows, out)
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "appendix", "pool": pool_name, **hashes})
    print(pd.DataFrame(rows)[["figure", "section"]].to_string(index=False))
    print(f"Wrote {out}")


def rng_for(cfg: dict, figure_name: str) -> np.random.Generator:
    """A generator seeded by the project seed and the figure's name, so each figure's picks stay the
    same when other figures are added or removed."""
    return np.random.default_rng([int(cfg["seed"]), *figure_name.encode()])


# --- appendix B: data and preprocessing (E0) --------------------------------------------------------

def e0_figures(cfg: dict, name: str, pool_name: str, out: Path) -> list[dict]:
    """B1 native vs model-space lesion area, B2 brain area by slice height, B3 component sizes, B4 specks
    by gap, B5 brain-mask methods."""
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    stats = config.results_dir(cfg, name, "e0b") / "stats"
    review = config.results_dir(cfg, name, "e0b") / "lesions_review"
    min_area = int(decided(cfg, "lesions.min_lesion_area", "e0b_audit.py lesions-review"))
    gap = int(decided(cfg, "lesions.attach_gap", "e0b_audit.py lesions-review"))
    rows = [area_ratio_figure(stats / f"native_vs_model_{name}.csv", out)]

    path = stats / f"brain_by_height_{name}.csv"
    real_px = read_table(real_dir / "brain.csv")["brain_px"].astype(float)
    syn = read_table(syn_dir / "brain.csv").merge(read_table(syn_dir / "qc.csv")[["sample_id", "mask_class"]], on="sample_id")
    syn = syn[syn["mask_class"].isin(["clean", "minor"]) & ~syn["mri_failed"].astype(str).str.lower().eq("true")]
    rows.append(brain_height_figure(path, real_px, syn["brain_px"].astype(float), out,
                                    [path, real_dir / "brain.csv", syn_dir / "brain.csv"]))

    path = review / f"component_sizes_{pool_name}.csv"
    rows.append(component_figure(read_table(path), min_area, path, out))
    path = review / f"speck_gaps_{pool_name}.csv"
    rows.append(speck_gap_figure(read_table(path), gap, path, out))
    rows.append(brain_methods_figure(cfg, real_dir, out))
    return rows


def area_ratio_figure(path: Path, out: Path) -> dict:
    """(a) Model-space lesion area against the native area scaled to 128 x 128 (log-log; dashed = plain
    downscaling). (b) The enlargement factor by native lesion size (boxes, median written above)."""
    table = read_table(path)
    fig = figure(6.0)
    left, right = fig.subplots(1, 2)
    left.scatter(table["expected_area"].astype(float), table["model_area"].astype(float), s=2, alpha=0.25,
                 lw=0, color=COLORS["T"])
    lim = (0.5, float(max(table["expected_area"].astype(float).max(), table["model_area"].astype(float).max())) * 1.2)
    left.plot(lim, lim, color="k", lw=0.8, ls="--", label="plain downscaling")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("native area scaled to 128 × 128 (px)")
    left.set_ylabel("model-space area (px)")
    left.legend(frameon=False, loc="upper left")
    left.set_title("(a) every real lesion")
    order = sorted(table["size_bin"].unique(), key=lambda b: float(str(b).lstrip(">").split("-")[0]))
    groups = [table.loc[table["size_bin"] == b, "ratio"].astype(float).to_numpy() for b in order]
    right.boxplot(groups, widths=0.55, showfliers=False, medianprops={"color": "k"})
    for i, g in enumerate(groups, start=1):
        q1, q3 = np.percentile(g, [25, 75])
        top = float(g[g <= q3 + 1.5 * (q3 - q1)].max())   # matplotlib's default upper whisker
        right.text(i, top + 0.04, f"{np.median(g):.2f}", ha="center", va="bottom", fontsize=6)
    right.axhline(1, color="k", lw=0.8, ls="--")
    right.set_xticks(range(1, len(order) + 1), order, rotation=35, fontsize=6.5)
    right.set_xlabel("native lesion area (native px)")
    right.set_ylabel("enlargement factor")
    right.set_title("(b) enlargement by lesion size (median above)")
    save_both(fig, out, "B1_native_vs_model")
    return index_row("B1_native_vs_model", "App. B (2.3)", "lesion area in model space against plain downscaling; "
                     "enlargement factor by native lesion size", [path])


def brain_height_figure(path: Path, real_px: pd.Series, syn_px: pd.Series, out: Path, sources: list) -> dict:
    """(a) Brain area of real slices by relative position in the head (median, middle half) with the
    synthetic median as a line. (b) Brain-area distributions of real lesion slices and S main samples."""
    table = pd.read_csv(path, index_col="position")
    fig = figure(6.0)
    left, right = fig.subplots(1, 2, width_ratios=[1.7, 1], sharey=True)
    left.plot(table.index, table["median"], color=COLORS["T"], label="real slices: median")
    left.fill_between(table.index, table["p25"], table["p75"], color=COLORS["T"], alpha=0.2, lw=0,
                      label="real slices: middle 50%")
    left.axhline(syn_px.median(), color=COLORS["S"], lw=1.1, label=f"synthetic median ({syn_px.median():.0f} px)")
    left.set_xlabel("relative slice position in the head (0 = lowest)")
    left.set_ylabel("brain area (px)")
    left.legend(frameon=False, loc="lower center", fontsize=6.5)
    left.set_title("(a) real brain area by height")
    bins = np.linspace(0, float(max(real_px.max(), syn_px.max())), 41)
    for values, color, label in ((real_px, COLORS["T"], "real lesion slices"), (syn_px, COLORS["S"], "synthetic")):
        right.stairs(np.histogram(values, bins=bins)[0] / len(values), bins, orientation="horizontal", color=color,
                     lw=1.1, label=f"{label} (median {values.median():.0f})")
    right.set_xlabel("share")
    right.legend(frameon=False, loc="center right", fontsize=6.5)   # the empty band between the two peaks
    right.set_title("(b) distribution")
    save_both(fig, out, "B2_brain_by_height")
    return index_row("B2_brain_by_height", "App. B (3.1)", "real brain area by position in the head against synthetic "
                     "brains; brain-area distributions", sources)


def component_figure(table: pd.DataFrame, min_area: int, path: Path, out: Path) -> dict:
    """Share of 8-connected mask components by size (1-10 px, larger) per set, log scale, with the
    minimum lesion area (decision B) marked."""
    sizes = [c for c in table.columns if c not in ("group", "components")]
    fig = figure(6.0)
    ax = fig.subplots()
    groups = list(table["group"])
    width = 0.8 / len(groups)
    colors = {"T": COLORS["T"], "H-cal": "0.65", "H-eval": "0.35", "S": COLORS["S"]}
    for k, g in enumerate(groups):
        r = table[table["group"] == g].iloc[0]
        share = r[sizes].astype(float).to_numpy() / float(r["components"])
        ax.bar(np.arange(len(sizes)) + (k - (len(groups) - 1) / 2) * width, share, width=width,
               color=colors.get(g, "0.5"), label=f"{g} ({int(r['components']):,} components)")
    ax.axvline(sizes.index(str(min_area)) - 0.5, color="k", ls="--", lw=0.8)
    ax.text(sizes.index(str(min_area)) - 0.4, 0.5, f"lesion ≥ {min_area} px", fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlabel("component size (px)")
    ax.set_ylabel("share of components (log)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncols=2)
    ax.set_ylim(top=3)
    save_both(fig, out, "B3_component_sizes")
    return index_row("B3_component_sizes", "App. B (2.3)", "component sizes per set (decision B)", [path])


def speck_gap_figure(table: pd.DataFrame, gap: int, path: Path, out: Path) -> dict:
    """Synthetic specks (components below the minimum lesion area) by their gap to the nearest lesion,
    per 1 px of gap (wider bins divided by their width): the halo at 1-2 px against the flat noise level
    (decision C)."""
    r = table[table["group"] == "S"].iloc[0]
    x = np.array([g[1] for g in SPECK_GAPS])
    y = np.array([float(r[col]) / width for col, _, width in SPECK_GAPS])
    fig = figure(5.5, width_in=10 / 2.54)
    ax = fig.subplots()
    ax.bar(x, y, width=[w * 0.85 for _, _, w in SPECK_GAPS], color=COLORS["S"])
    ax.axvline(gap + 0.5, color="k", ls="--", lw=0.8)
    ax.text(gap + 0.6, 0.92, f"attached ≤ {gap} px", fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.set_xlabel("gap to the nearest lesion (px)")
    ax.set_ylabel("synthetic specks per 1 px of gap")
    ax.set_title(f"S: {int(r['gap >=11']):,} more specks at ≥ 11 px", fontsize=7)
    save_both(fig, out, "B4_specks_by_gap")
    return index_row("B4_specks_by_gap", "App. B (2.3)", "synthetic specks by gap to the nearest lesion (decision C)",
                     [path])


def brain_methods_figure(cfg: dict, real_dir: Path, out: Path) -> dict:
    """T slices where plain Otsu leaves a lesion more than half outside the brain, and a random one,
    under plain Otsu, Otsu keeping every region and the chosen fixed threshold (decision A); E0b's own
    sheet, shrunk to the column width."""
    low = float(decided(cfg, "brain.low_threshold", "e0b_audit.py brain-review"))
    slices = read_table(real_dir / "slices.csv")
    labels = LabelStack(real_dir / "labels.npz")
    t = slices[slices["set"] == "T"]
    rng = rng_for(cfg, "B5")
    failing, examples = [], []
    for pid in rng.permutation(t["patient_id"].unique()):
        bravo, _ = load_real(real_dir, str(pid))
        for key, z in zip(t.loc[t["patient_id"] == pid, "key"], t.loc[t["patient_id"] == pid, "z"]):
            image, lab = bravo[:, :, int(z)], labels[str(key)]
            brain = brain_mask(image, "otsu", low)
            outside = [((lab == n) & ~brain).sum() > 0.5 * (lab == n).sum() for n in range(1, int(lab.max()) + 1)]
            if any(outside) and len(failing) < BRAIN_EXAMPLES[0]:
                failing.append((str(key), image, lab))
        if len(failing) >= BRAIN_EXAMPLES[0]:
            break
    for key in pick(rng, t[~t["key"].isin([f[0] for f in failing])], BRAIN_EXAMPLES[1])["key"]:
        pid, z = str(key).split("/z")
        examples.append((str(key), load_real(real_dir, pid)[0][:, :, int(z)], labels[str(key)]))
    method_grid(failing + examples, out / "B5_brain_methods.png", "", low, methods=BRAIN_METHODS_SHOWN)
    return index_row("B5_brain_methods", "App. B (2.3)", f"brain masks by {', '.join(BRAIN_METHODS_SHOWN)} on "
                     f"{BRAIN_EXAMPLES[0]} slices where plain Otsu fails and {BRAIN_EXAMPLES[1]} random T slice "
                     "(decision A)", [real_dir / "volumes", real_dir / "labels.npz"])


# --- appendix B: calibration (E3) -----------------------------------------------------------------

def e3_figures(cfg: dict, name: str, out: Path) -> list[dict]:
    """B6 H-cal score histograms by stratum with the frozen stratum thresholds (decision H)."""
    path = config.results_dir(cfg, name, "e3") / "calibrate" / "thresholds.csv"
    e1, e2, patients = load_hcal_tables(cfg, name)
    les = calibration.lesion_slices(e2, patients)
    thresholds = read_table(path)
    panels = (("tau_D", e1, "score", "(a) E1 whole-mask score"), ("tau_L", les, "strongest_s", "(b) E2 strongest lesion pair s"),
              ("tau_C", les, "C", "(c) E2 configuration score C"))
    fig = figure(5.5)
    bins = np.linspace(0, 1, 41)
    for k, (ax, (thr, d, col, title)) in enumerate(zip(fig.subplots(1, 3), panels)):
        values = thresholds[thresholds["threshold"] == thr].set_index("stratum")["value"].astype(float)
        for stratum, color in (("single", "0.4"), ("multi", "tab:purple")):
            v = d.loc[d["stratum"] == stratum, col].astype(float).fillna(0.0)
            ax.hist(v, bins=bins, density=True, histtype="step", color=color, lw=1.1,
                    label=f"{stratum}-lesion ({len(v)} slices)")
            ax.axvline(float(values[stratum]), color=color, ls="--", lw=0.9)
        tau = thr.replace("tau_", "τ_")
        ax.set_title(f"{title}\n{tau}: single {float(values['single']):.2f}, multi {float(values['multi']):.2f}", fontsize=7)
        ax.set_xlabel("score")
        ax.set_yticks([])
        if k == 0:
            ax.set_ylabel("density")
            ax.legend(frameon=False, loc="upper left", fontsize=6.5)
    save_both(fig, out, "B6_score_histograms")
    return [index_row("B6_score_histograms", "App. B (2.6, 3.2)", "H-cal scores against T by stratum with the frozen "
                      "95th-percentile thresholds (dashed)", [path, "e3 H-cal tables"])]


# --- appendix C: validation (E4) --------------------------------------------------------------------

def e4_figures(cfg: dict, name: str, out: Path) -> list[dict]:
    """C1 a few gated planted copies missed by their criterion's detector, each as base, planted mask and
    the nearest neighbour found."""
    e4_res = config.results_dir(cfg, name, "e4")
    criteria = config.value(cfg, "validation.criteria")
    path = e4_res / "validate" / "planted_outcomes.csv"
    outcomes = read_table(path)
    outcomes["gated"] = outcomes["gated"].astype(str).str.lower().eq("true")
    misses = outcomes[outcomes["gated"] & outcomes["variant"].isin(criterion_variants(criteria))
                      & ~criterion_caught(outcomes, criteria)]
    chosen = pick(rng_for(cfg, "C1"), misses, MISSES_SHOWN)
    t = load_set(cfg, name, "T")
    planted, planted_labels = load_planted(cfg, name)
    labels = {**dict(zip(t.keys, t.labels)), **dict(zip(planted["key"], planted_labels))}
    records = chosen.to_dict("records")
    nn = [str(r["e2_nn"]) if isinstance(r["e2_nn"], str) and r["e2_nn"] else str(r["e1_nn"]) for r in records]
    images = slice_images(cfg, name, [*chosen["base_key"], *nn])
    n_rows = int(np.ceil(len(chosen) / 2))
    fig = figure(3.5 * n_rows)
    grid = fig.add_gridspec(n_rows, 7, width_ratios=[1, 1, 1, 0.12, 1, 1, 1])
    for k, (r, found) in enumerate(zip(records, nn)):
        base, key = str(r["base_key"]), str(r["key"])
        flagged = str(r["e1_copy"]).lower() == "true" or r["e2_class"] in ("full", "partial")
        shown = [(labels[base], images[base], f"base\n{base}"),
                 (labels[key], images[base], f"{r['variant']}: {'wrong patient' if flagged else 'not flagged'}\n"
                                             f"E1 {float(r['e1_score']):.2f}, E2 {r['e2_class']}"),
                 (labels.get(found), images.get(found), f"nearest found\n{found}")]
        window = zoom_window([m > 0 for m, _, _ in shown if m is not None])
        for j, (m, image, title) in enumerate(shown):
            ax = fig.add_subplot(grid[k // 2, 4 * (k % 2) + j])
            if m is None or image is None:
                ax.axis("off")
                continue
            image_tile(ax, image[window], [((m > 0)[window], "red")], title)
            ax.title.set_fontsize(6)
    save_both(fig, out, "C1_misses")
    return [index_row("C1_misses", "App. C (3.3)", f"{len(chosen)} of the {len(misses)} gated planted masks missed by "
                      "their criterion's detector: base, planted mask, nearest neighbour found", [path])]


# --- S and real slices for the sheets of appendices D and E -----------------------------------------

class SLoader:
    """Label maps and images of T, H-eval and S keys, as E5 and E7 load them for their sheets (S
    samples read from the pool files, real slices from the model-space volumes)."""

    def __init__(self, cfg: dict, name: str, pool_name: str, pool_cfg: dict):
        self.cfg, self.name, self.pool_cfg = cfg, name, pool_cfg
        t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
        self.real = {**dict(zip(t.keys, t.labels)), **dict(zip(heval.keys, heval.labels))}
        self.pool = LabelStack(config.output_dir(cfg, pool_name) / "labels.npz")
        index = read_table(config.output_dir(cfg, pool_name) / "synthetic_index.csv")
        self.files = dict(zip(index["sample_id"], index["filename"]))
        self.pool_dir = config.resolve_path(pool_cfg["dir"])

    def labels_of(self, keys) -> dict:
        """Label map per key (real slice or S sample)."""
        return {k: self.real[k] if k in self.real else self.pool[k] for k in keys}

    def images_of(self, keys) -> dict:
        """Image per key: model-space BRAVO of a real slice, the synthetic MRI of an S sample."""
        real = slice_images(self.cfg, self.name, [k for k in keys if k in self.real])
        return {k: real[k] if k in real else load_sample(self.pool_dir / self.files[k], self.pool_cfg)[0] for k in keys}


# --- appendix D: audit and exposure (E5, E6) -------------------------------------------------------

def e5_figures(cfg: dict, pool_name: str, s: SLoader, out: Path) -> list[dict]:
    """D1 the excess of the two multi-lesion primary cells and their single-lesion partners under every
    sensitivity variant, D2 the near-exact copies with the highest E1 score next to their source slice."""
    a = config.results_dir(cfg, pool_name, "e5") / "analyze"
    rates = read_table(a / "rates.csv")
    fig = figure(6.5)
    axes = fig.subplots(1, 2, sharey=True)
    y = np.arange(len(SENSITIVITY_ROWS))[::-1]
    for ax, (cls, title) in zip(axes, (("e1_copy", "(a) E1 copy"), ("e2_full", "(b) E2 full"))):
        for stratum, color, dy in (("single", "0.5", -0.15), ("multi", COLORS["S"], 0.15)):
            for yi, (set_name, variant, _) in zip(y, SENSITIVITY_ROWS):
                r = rates[(rates["class"] == cls) & (rates["stratum"] == stratum) & (rates["set"] == set_name)
                          & (rates["variant"] == variant)].iloc[0]
                e, lo, hi = (100 * float(r[c]) for c in ("excess", "excess_low", "excess_high"))
                ax.errorbar(e, yi + dy, xerr=[[e - lo], [hi - e]], fmt="o", ms=3, capsize=1.5, lw=0.9, color=color,
                            label=f"{stratum}-lesion" if yi == y[0] else None)
        ax.axvline(0, color="k", lw=0.7)
        ax.axhspan(y[0] - 0.45, y[0] + 0.45, color="0.93", zorder=0)
        ax.set_xlabel("excess over H-eval (percentage points)")
        ax.set_title(title)
    axes[0].set_yticks(y, [label for _, _, label in SENSITIVITY_ROWS])
    axes[1].legend(frameon=False, loc="upper left", fontsize=6.5)
    save_both(fig, out, "D1_sensitivity")
    main = read_table(a / "flagged.csv")
    main = main[main["set"] == "main"]
    near = main[main["near_exact"].astype(str).str.lower().eq("true")].sort_values("e1_score", ascending=False)
    examples_figure(near.head(NEAR_EXACT_SHOWN), "e1_nn", s.labels_of, s.images_of, out / "D2_near_exact.png", "")
    return [index_row("D1_sensitivity", "App. D (3.4, 4.6)", "excess over H-eval with 95% intervals for E1 copy and "
                      "E2 full by stratum under every sensitivity variant (shaded: the main analysis)", [a / "rates.csv"]),
            index_row("D2_near_exact", "App. D (3.4)", f"the {NEAR_EXACT_SHOWN} near-exact masks with the highest E1 "
                      f"score (of {len(near)}; decision Q) next to their source slice", [a / "flagged.csv"])]


def e6_figures(cfg: dict, name: str, pool_name: str, out: Path) -> list[dict]:
    """D3 flags by patient-rarity group, observed over expected (H3), D4 margin and rarity AUCs of the
    rehearsal mocks next to S's (decision U)."""
    m = float(exposure_config(cfg)["expected"]["smoothing_m"])
    d = config.results_dir(cfg, pool_name, "e6") / "exposure"
    exposure, h3 = read_table(d / "exposure.csv"), read_table(d / "h3.csv")
    rarity = load_reference(cfg, name).patients.set_index("patient_id")["rarity"]
    fig = figure(5.5)
    axes = fig.subplots(1, 2, sharey=True)
    for ax, (cell, title), letter in zip(axes, PRIMARY_CELLS, "ab"):
        e = exposure[(exposure["cell"] == cell) & np.isclose(exposure["m"].astype(float), m)].copy()
        e["group"] = pd.qcut(e["patient_id"].map(rarity).astype(float), 5, labels=["most\ntypical", "2", "3", "4", "rarest"])
        g = e.groupby("group", observed=False)[["observed", "expected"]].sum().astype(float)
        ax.bar(range(len(g)), g["observed"] / g["expected"], color="tab:purple", width=0.6)
        ax.axhline(1, color="k", lw=0.7)
        ax.set_xticks(range(len(g)), g.index.astype(str))
        ax.set_xlabel("T patients in fifths by rarity")
        r = h3.set_index("cell").to_dict("index")[cell]
        rho, lo, hi = (round(float(r[c]), 2) + 0.0 for c in ("rho", "rho_low", "rho_high"))   # + 0.0: no "-0.00"
        ax.set_title(f"({letter}) {title}\nSpearman ρ {rho:.2f} [{lo:.2f}, {hi:.2f}]", fontsize=7)
    axes[0].set_ylabel("observed / expected flags")
    save_both(fig, out, "D3_rarity_exposure")
    rows = [index_row("D3_rarity_exposure", "App. D (3.5)", "flags by patient-rarity fifth, observed / expected, with "
                      "the patient-level Spearman (H3), both primary cells", [d / "exposure.csv", d / "h3.csv"])]

    rehearse = config.results_dir(cfg, name, "e6") / "rehearse"
    sets = [(read_table(rehearse / mock / "margins.csv"), label) for mock, label in REHEARSALS]
    sets.append((read_table(d / "margins.csv"), "S main"))
    fig = figure(5.5)
    axes = fig.subplots(1, 2, sharey=True)
    y = np.arange(len(sets))[::-1]
    for ax, (col, title) in zip(axes, (("margin", "(a) margin to the second-best patient"),
                                       ("rarity", "(b) rarity of the source slice"))):
        for (cell, label), color, dy in zip(PRIMARY_CELLS, (COLORS["E1"], COLORS["E2"]), (0.12, -0.12)):
            for yi, (margins, _) in zip(y, sets):
                r = margins.set_index("cell").to_dict("index")[cell]
                a, lo, hi = (float(r[f"{col}_auc{s}"]) for s in ("", "_low", "_high"))
                ax.errorbar(a, yi + dy, xerr=[[a - lo], [hi - a]], fmt="o", ms=3, capsize=1.5, lw=0.9, color=color,
                            label=label if yi == y[0] else None)
        ax.axvline(0.5, color="k", lw=0.7)
        ax.axhspan(y[-1] - 0.45, y[-1] + 0.45, color="0.93", zorder=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel("AUC, flags vs H-eval chance flags")
        ax.set_title(title)
    axes[0].set_yticks(y, [label for _, label in sets])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False, fontsize=6.5)
    save_both(fig, out, "D4_rehearsal_margins")
    rows.append(index_row("D4_rehearsal_margins", "App. D (2.10, 3.5)", "margin and source-rarity AUCs (flags vs H-eval "
                          "chance flags) of the rehearsal mocks and of S, both primary cells (decision U)",
                          [rehearse / mock / "margins.csv" for mock, _ in REHEARSALS] + [d / "margins.csv"]))
    return rows


# --- appendix E: match reports and the paired MRI (E7, E8) ----------------------------------------

def e7_figures(cfg: dict, name: str, pool_name: str, pool_cfg: dict, s: SLoader, out: Path) -> list[dict]:
    """E1 E7 report 02 in compact form (E1 finds the slice exactly; E2's centred crops miss by 1 px), E2
    the profile of `PROFILE_PATIENT`: S flags per slice, and its most-matched slice next to S masks naming
    it."""
    reports = config.results_dir(cfg, pool_name, "e7") / "reports"
    syn_dir = config.output_dir(cfg, pool_name)
    index = read_table(reports / "report_index.csv")
    key = str(index.loc[index["report"] == REPORT_KEY, "key"].iloc[0])
    fig = figure(6.2)
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 0.1])
    source = match_tiles(fig, [grid[0, j] for j in range(3)], grid[1, :], cfg, name, pool_name, pool_cfg, key)
    save_both(fig, out, "E1_report_02")
    rows = [index_row("E1_report_02", "App. E (3.6)", f"E7 report 02 compact: {key} against {source} (E1 1.00, E2 "
                      "novel: the 1 px limit of centred crops)", [reports / "report_index.csv", syn_dir / "e5" / "e1_main.csv",
                                                                  syn_dir / "e5" / "e2_main_lesion_pairs.parquet"])]

    prof = read_table(reports / "patient_profiles.csv")
    prof = prof[prof["patient_id"] == PROFILE_PATIENT]
    totals = prof.groupby("z")[["e1_flags", "e2_flags"]].sum().astype(float).sum(axis=1)
    slice_key = f"{PROFILE_PATIENT}/z{int(totals.idxmax())}"  # pyright: ignore[reportArgumentType]  # idxmax is typed Hashable; the index is z (int)
    flagged = read_table(config.results_dir(cfg, pool_name, "e5") / "analyze" / "flagged.csv")
    flagged = flagged[flagged["set"] == "main"]
    named = flagged[(flagged["e1_nn"] == slice_key) | (flagged["e2_source"] == slice_key)]
    examples = named.sort_values("e1_score", ascending=False)["key"].head(PROFILE_EXAMPLES).tolist()
    labels, images = s.labels_of([slice_key, *examples]), s.images_of([slice_key, *examples])
    window = zoom_window([labels[k] > 0 for k in [slice_key, *examples]])
    fig = figure(10.0)
    grid = fig.add_gridspec(2, 1 + PROFILE_EXAMPLES, height_ratios=[1, 1.1])
    ax = fig.add_subplot(grid[0, :])
    n_lesions = prof["lesion3d_id"].nunique()
    profile_bars(ax, prof, f"slice z of {PROFILE_PATIENT} ({n_lesions} metastases over {{n}} slices)",
                 "lesion area (px, summed)", legend_loc="upper center")
    ax.set_title(f"(a) {PROFILE_PATIENT}: S flags per slice")
    image_tile(fig.add_subplot(grid[1, 0]), images[slice_key][window], [((labels[slice_key] > 0)[window], COLORS["T"])],
               f"(b) most-matched slice\n{slice_key} ({len(named)} S flags)")
    for j, k in enumerate(examples, start=1):
        r = named[named["key"] == k].iloc[0]
        image_tile(fig.add_subplot(grid[1, j]), images[k][window], [((labels[k] > 0)[window], COLORS["S"])],
                   f"({'bcde'[j]}) {k}\nE1 {float(r['e1_score']):.2f}")
    save_both(fig, out, f"E2_profile_{PROFILE_PATIENT}")
    rows.append(index_row(f"E2_profile_{PROFILE_PATIENT}", "App. E (3.6)", f"profile of {PROFILE_PATIENT}: S flags per "
                          f"slice with the lesion area; its most-matched slice next to the {len(examples)} S masks naming "
                          "it with the highest E1 score", [reports / "patient_profiles.csv",
                                                            config.results_dir(cfg, pool_name, "e5") / "analyze" / "flagged.csv"]))
    return rows


def e8_figures(cfg: dict, pool_name: str, out: Path) -> list[dict]:
    """E3 within group A: mean SSIM by z offset from the source slice per region, and the source minus
    runner-up difference per region with its Wilcoxon p."""
    path = config.results_dir(cfg, pool_name, "e8") / "paired" / "within_a.csv"
    within = read_table(path)
    fig = figure(5.5)
    left, right = fig.subplots(1, 2, width_ratios=[1.6, 1])
    prof = within[within["measure"] == "mean SSIM by z offset"]
    ru = within[within["measure"] == "t minus runner-up"].set_index("region").to_dict("index")
    for k, (region, label, color) in enumerate(WITHIN_REGIONS):
        d = prof[prof["region"] == region]
        mean, lo, hi = (d[c].astype(float).to_numpy() for c in ("mean", "low", "high"))
        left.errorbar(d["dz"].astype(int) + (k - 1) * 0.12, mean, yerr=[mean - lo, hi - mean], marker="o", ms=2.5,
                      capsize=1.2, lw=1.2 if region == "lesion" else 0.8, color=color, label=label)
        r = ru[region]
        m_, l_, h_ = float(r["mean"]), float(r["low"]), float(r["high"])
        right.errorbar(k, m_, yerr=[[m_ - l_], [h_ - m_]], fmt="o", ms=3, capsize=2, color=color)
    left.axvline(0, color="0.8", lw=0.6)
    left.set_xlabel("z offset from the source slice")
    left.set_ylabel("mean SSIM (group A)")
    left.set_title("(a) SSIM by distance from the source")
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False, fontsize=6.5)
    right.axhline(0, color="0.6", lw=0.7)
    right.set_xticks(range(len(WITHIN_REGIONS)), [f"{region}\np {float(ru[region]['wilcoxon_p']):.2f}"
                                                  for region, _, _ in WITHIN_REGIONS])
    right.set_xlim(-0.6, len(WITHIN_REGIONS) - 0.4)
    right.set_ylabel("SSIM to source − to runner-up")
    right.set_title("(b) source vs runner-up patient")
    save_both(fig, out, "E3_e8_within_a")
    return [index_row("E3_e8_within_a", "App. E (3.7)", "within group A: SSIM by z offset from the source per region; "
                      "source minus runner-up per region with Wilcoxon p", [path])]
