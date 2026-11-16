"""Report-figures stage `method`: figures that show what the method does, on real data, synthetic
samples and planted copies with a known answer (`docs/report-synthesis.md`, figure register,
Method figures). No result on H-cal, H-eval or S is shown here. Text inside the figures is kept short;
the report's captions explain them.

- `2-3_preprocessing_qc`: one real slice native and in model space with E0's annotations, a zoom on
  one small lesion's resized mask (the 0.01 threshold against plain downscaling), and one synthetic
  mask per QC class after attach-or-drop (decisions A-D).
- `2-4_e1_shift_search`: a planted copy shifted by 2 px against its source before and after E1's shift
  search, with the Dice of every shift.
- `2-5_e2_lesion_pairs`: a planted copy rotated by 5 degrees: its lesions paired with the source's,
  each pair's centred crops with shape x position x size = s, and the configuration score C.
- `2-7_planted_variants`: every planted variant of the same base slice, and the same 1 px boundary
  change on a small and a large lesion (decisions I, K).
- `2-11_match_report`: the match report's images on a planted P6-add copy of that base (one lesion
  added) with its verdict line.

The three planted examples share one base slice: the base of a seeded pick among the P6-add copies.
Writes `results/<pool>/report/method/` (PNG + PDF per figure, `figures.csv`, run log). Reads stored
tables, single slices and the raw native volume of one patient; nothing is recomputed or re-scored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from memtk import config, lesionmatch
from memtk.calibration import load_frozen
from memtk.e0b.common import Slices
from memtk.e7.common import chance_rates, class_rate, patient_name
from memtk.ids import parse_lesion_id, parse_slice_key
from memtk.io import LabelStack, load_image, patient_files, read_table, slice_images
from memtk.lesions import attach_or_drop
from memtk.paired import align
from memtk.planted import morph, planted_key
from memtk.plotting import draw_box, draw_labels, draw_lesions, fill, lesion_colors, zoom_window
from memtk.report_figures.common import COLORS, figure, index_row, save_both, style, write_index
from memtk.report_figures.results import image_tile
from memtk.reports import ReportData, draw_pair_lines, pair_colors, report_data, verdict
from memtk.sets import load_planted
from memtk.stages import decided, pick, refuse_overwrite, require
from memtk.wholemask import shift_order, shifted_dice

E0_SLICE = "Mets_120/z40"          # a training slice with five lesions of 4-141 px (E0b's example view)
E0_ZOOM_LESION = 3                 # its 15 px lesion L3: the resized mask and the 0.01 threshold, zoomed
QC_PICKS = (("clean", "n_lesions >= 2 and not mri_failed"),
            ("minor", "n_specks >= 3 and n_strays >= 1 and not mri_failed"),
            ("noisy", "n_specks >= 60 and n_specks <= 150 and n_lesions >= 1 and not mri_failed"),
            ("no_lesion", "n_specks >= 2 and not mri_failed"))
EXAMPLE_VARIANT = "P6-add"         # the seeded pick that fixes the base slice of every planted example
VARIANTS = (("P1", "P1 exact"), ("P2", "P2 shift 2 px"), ("P3", "P3 shift 5 px"), ("P4", "P4 rotate 5°"),
            ("P5-erode", "P5 erode 1 px"), ("P5-dilate", "P5 dilate 1 px"), ("P6-remove", "P6 remove a lesion"),
            ("P6-add", "P6 add a lesion"))
BOUNDARY_SIZES = (14, 90)          # areas (px) of the small and the large T lesion shown eroded and dilated
PAIR_CROP = 13                     # half-width (px) of the centred crops of a lesion pair
OVERLAY = (("tab:green", "both"), (COLORS["S"], "copy only"), (COLORS["T"], "source only"))


# --- stage ------------------------------------------------------------------------------------

def run_method(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The Method figures of the report: preprocessing and QC, E1's shift search, E2's lesion pairs, the
    planted variants and the match report, the last four on planted copies of one base slice. Needs
    E0b (brain, lesions, qc), E4 plant and validate, E7 planted. Writes `results/<pool>/report/method/`."""
    name, data_cfg = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "report") / "method"
    refuse_overwrite([out], overwrite)
    planted_dir = config.results_dir(cfg, name, "e7") / "planted"
    require(planted_dir / "reports.csv", "e7_reports.py planted")
    rng = np.random.default_rng(cfg["seed"])
    e3, _ = load_frozen(cfg, name)
    base = str(pick_example(read_table(planted_dir / "reports.csv"), rng)["base_key"])
    planted = Planted(cfg, name, e3)

    rows = []
    with style():
        rows.append(preprocessing_figure(cfg, name, data_cfg, pool_name, pool_cfg, rng, out))
        rows.append(e1_figure(planted, planted_key("P2", base), out))
        rows.append(e2_figure(planted, planted_key("P4", base), out))
        rows.append(variants_figure(cfg, name, planted, base, out))
        rows.append(report_figure(cfg, name, planted, planted_key(EXAMPLE_VARIANT, base), out))
    write_index(rows, out)
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "method", "pool": pool_name, "base": base})
    print(pd.DataFrame(rows)[["figure", "section"]].to_string(index=False))
    print(f"Wrote {out}")


def pick_example(reports: pd.DataFrame, rng: np.random.Generator) -> dict:
    """The planted copy that fixes the example base: a gated, E2-flagged P6-add mask of a base with at
    least two informative lesions (three with the added one) whose report names the base slice and pairs
    every base lesion (seeded pick)."""
    r = reports[(reports["variant"] == EXAMPLE_VARIANT) & reports["gated"].astype(bool) & reports["flagged"].astype(bool)
                & (reports["n_query"].astype(int) >= 3) & (reports["source"] == reports["base_key"])
                & (reports["n_kept"].astype(int) == reports["n_query"].astype(int) - 1)]
    return pick(rng, r, 1).iloc[0].to_dict()


class Planted:
    """The E4 planted set with its stored detector results, for drawing single examples: label maps,
    the base slices' label maps and BRAVO, E1 rows and E2 report data (as E7 builds them)."""

    def __init__(self, cfg: dict, name: str, e3: dict):
        e4_dir = config.output_dir(cfg, name) / "e4"
        self.cfg, self.name, self.e3 = cfg, name, e3
        self.table, labels = load_planted(cfg, name)
        self.labels = dict(zip(self.table["key"], labels))
        self.e1 = read_table(e4_dir / "e1_planted.csv")
        self.e2 = lesionmatch.load_tables(e4_dir, "planted", float(e3["detectors"]["lesion"]["candidate_min_s"]))
        self.outcomes = read_table(config.results_dir(cfg, name, "e4") / "validate" / "planted_outcomes.csv")
        self.real_labels = LabelStack(config.output_dir(cfg, name) / "labels.npz")
        self.sources = [e4_dir / "planted_labels.npz", e4_dir / "e1_planted.csv", e4_dir / "e2_planted_lesion_pairs.parquet",
                        config.results_dir(cfg, name, "e4") / "validate" / "planted_outcomes.csv"]

    def e1_row(self, key: str) -> dict:
        """The stored E1 result of a planted mask."""
        return self.e1[self.e1["query"] == key].iloc[0].to_dict()

    def report(self, key: str) -> ReportData:
        """The report data of a planted mask, as E7 `planted` builds it."""
        row = self.outcomes[self.outcomes["key"] == key].iloc[0].copy()
        queries = self.e2.queries
        row["n_informative"] = int(queries.loc[queries["query"] == key, "n_informative"].iloc[0])
        return report_data(row, self.e1[self.e1["query"] == key].iloc[0], self.e2, self.e3)

    def image(self, key: str) -> np.ndarray:
        """The BRAVO slice of a real slice key (a planted mask is drawn on its base slice)."""
        return slice_images(self.cfg, self.name, [key])[key]


def overlay_tile(ax, a: np.ndarray, b: np.ndarray, title: str) -> None:
    """Two masks on white: green where both, red where only `a` (the copy), blue where only `b` (the
    source); a grey frame."""
    image_tile(ax, np.ones(a.shape), [], title)
    fill(ax, [(a & b, OVERLAY[0][0]), (a & ~b, OVERLAY[1][0]), (b & ~a, OVERLAY[2][0])], alpha=0.85)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.6")


def overlay_legend(fig_or_sub, loc: str = "lower center", ncols: int = 3) -> None:
    """The overlay colours as a figure legend."""
    handles = [Line2D([], [], marker="s", ls="", color=c, label=lab) for c, lab in OVERLAY]
    fig_or_sub.legend(handles=handles, loc=f"outside {loc}", ncols=ncols, frameon=False, fontsize=6.5)


# --- 2.3: preprocessing and QC ----------------------------------------------------------------------

def preprocessing_figure(cfg: dict, name: str, data_cfg: dict, pool_name: str, pool_cfg: dict,
                         rng: np.random.Generator, out: Path) -> dict:
    """Top: (a) the native slice with its annotation, (b) the same slice in model space with E0's brain
    outline, brain box and numbered lesions (the white box = the zoom of c), (c) one small lesion's
    resized mask (grey = mask value after the area resize): pixels above 0.01 are kept (cyan) against
    above 0.5 (plain downscaling, orange). Bottom: one synthetic sample per QC class after attach-or-drop
    (lesions numbered, attached strays lighter, dropped specks as x)."""
    min_area = int(decided(cfg, "lesions.min_lesion_area", "e0b_audit.py lesions-review"))
    gap = int(decided(cfg, "lesions.attach_gap", "e0b_audit.py lesions-review"))
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    patient, z = parse_slice_key(E0_SLICE)
    split = read_table(real_dir / "patients.csv").set_index("patient_id").at[patient, "orig_split"]
    bravo_path, seg_path = patient_files(data_cfg, str(split), patient)
    native = np.asarray(load_image(bravo_path).dataobj)[:, :, z].astype(float)
    native_seg = np.asarray(load_image(seg_path).dataobj)[:, :, z] > 0
    soft = np.asarray(load_image(real_dir / "volumes" / patient / "seg_soft.nii.gz").dataobj)[:, :, z]
    image, mask, brain, box = Slices(real_dir)[E0_SLICE]
    res = attach_or_drop(mask, min_area, gap)
    window = zoom_window([res.labels == E0_ZOOM_LESION])

    fig = figure(11.5)
    grid = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
    hi = float(np.percentile(native[native > 0], 99.5)) if (native > 0).any() else 1.0
    image_tile(fig.add_subplot(grid[0, 0:2]), native, [(native_seg, COLORS["S"])],
               f"(a) native {native.shape[0]}×{native.shape[1]}, annotation in red", (0.0, hi))
    sub = grid[0, 2:4].subgridspec(1, 2)
    ax = fig.add_subplot(sub[0, 0])
    draw_lesions(ax, image, res, brain, box, "(b) model space 128×128\n(white box = zoom of c)", fontsize=4)
    ax.add_patch(Rectangle((window[1].start - 0.5, window[0].start - 0.5), window[1].stop - window[1].start,
                           window[0].stop - window[0].start, fill=False, edgecolor="w", lw=1.0))
    ax = fig.add_subplot(sub[0, 1])
    ax.imshow(soft[window], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    kept, plain = soft[window] > 0.01, soft[window] > 0.5
    ax.contour(kept.astype(float), levels=[0.5], colors="cyan", linewidths=1.6)
    ax.contour(plain.astype(float), levels=[0.5], colors="orange", linewidths=1.6, linestyles="--")
    ax.set_title(f"(c) zoom on L{E0_ZOOM_LESION}: resized mask", fontsize=6.5)
    ax.set_xticks([])
    ax.set_yticks([])
    lesion = (res.labels == E0_ZOOM_LESION)[window]
    ax.legend(handles=[Line2D([], [], color="cyan", lw=1.6, label=f"> 0.01, kept ({int((kept & lesion).sum())} px)"),
                       Line2D([], [], color="orange", lw=1.6, ls="--",
                              label=f"> 0.5, plain ({int((plain & lesion).sum())} px)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=6)

    qc = read_table(syn_dir / "qc.csv")
    for col in ("n_lesions", "n_specks", "n_strays"):
        qc[col] = qc[col].astype(int)
    qc["mri_failed"] = qc["mri_failed"].astype(str).str.lower().eq("true")
    pool = Slices(syn_dir, pool_cfg)
    chosen = []
    for i, (mask_class, condition) in enumerate(QC_PICKS):
        key = str(pick(rng, qc[qc["mask_class"] == mask_class].query(condition), 1)["key"].iloc[0])
        chosen.append(key)
        image, mask, brain, box = pool[key]
        res = attach_or_drop(mask, min_area, gap)
        draw_lesions(fig.add_subplot(grid[1, i]), image, res, brain, box,
                     f"({'defg'[i]}) {mask_class.replace('_', ' ')}: {key}\n{res.n_lesions} lesions, "
                     f"{res.n_strays} strays, {res.n_specks} specks", fontsize=4)
    out_name = "2-3_preprocessing_qc"
    save_both(fig, out, out_name)
    return index_row(out_name, "2.3", f"{E0_SLICE} native vs model space with E0's annotations; the 0.01 threshold "
                     f"on one lesion; one synthetic sample per QC class ({', '.join(chosen)})",
                     [real_dir / "volumes" / patient, real_dir / "brain_masks.npz", syn_dir / "qc.csv",
                      syn_dir / "masks_raw.npz", "raw native volume"])


# --- 2.4: E1's shift search -------------------------------------------------------------------------

def e1_figure(planted: Planted, key: str, out: Path) -> dict:
    """A planted copy shifted by 2 px against its source: (a) overlaid as it is, with the plain Dice, (b)
    the Dice of the copy moved by every shift of E1's search (±3 px; the best one ringed), (c) overlaid
    after the best shift, with E1's score."""
    e1 = planted.e1_row(key)
    source = str(e1["nn"])
    q, t = planted.labels[key] > 0, planted.real_labels[source] > 0
    dr, dc = int(e1["shift_row"]), int(e1["shift_col"])
    scores = shifted_dice(q, t[None], 3)[0]
    grid_scores = np.full((7, 7), np.nan)
    for (r, c), v in zip(shift_order(3), scores):
        grid_scores[r + 3, c + 3] = v
    moved = align(q, dr, dc)
    window = zoom_window([q, t])
    fig = figure(7.0)
    axes = fig.subplots(1, 3)
    overlay_tile(axes[0], q[window], t[window], f"(a) as generated: Dice {float(e1['plain_dice']):.2f}")
    img = axes[1].imshow(grid_scores, cmap="viridis", vmin=0.5, vmax=1, extent=(-3.5, 3.5, 3.5, -3.5))
    axes[1].scatter([dc], [dr], s=110, facecolors="none", edgecolors="w", lw=1.5)
    axes[1].set_xticks(range(-3, 4))
    axes[1].set_yticks(range(-3, 4))
    axes[1].tick_params(labelsize=6)
    axes[1].set_xlabel("column shift (px)", fontsize=6.5)
    axes[1].set_ylabel("row shift (px)", fontsize=6.5)
    axes[1].set_title("(b) Dice per shift (ring: best)")
    fig.colorbar(img, ax=axes[1], orientation="horizontal", location="top", shrink=0.8, aspect=25, pad=0.02)
    overlay_tile(axes[2], moved[window], t[window], f"(c) moved ({dr:+d}, {dc:+d}): Dice {float(e1['score']):.2f}")
    overlay_legend(fig)
    out_name = "2-4_e1_shift_search"
    save_both(fig, out, out_name)
    return index_row(out_name, "2.4", f"E1 on planted {key}: plain Dice, Dice per shift, Dice after the best shift",
                     planted.sources)


# --- 2.5: E2's lesion pairs ---------------------------------------------------------------------------

def e2_figure(planted: Planted, key: str, out: Path) -> dict:
    """A planted copy rotated by 5 degrees against its source: (a) the copy and (b) the source with
    partner lesions in the same colour, (c) the overlay with lines joining paired lesions (labelled s);
    below, each pair's crops centred on their own centroids (what the shape factor compares), with
    shape x position x size = s, and C."""
    d = planted.report(key)
    q_labels, t_labels = planted.labels[key], planted.real_labels[d.source]
    image = planted.image(d.source)
    window = zoom_window([q_labels > 0, t_labels > 0])
    q_colors = lesion_colors(q_labels)
    s_colors = pair_colors(d.pairs, q_colors, t_labels)
    pairs = d.pairs.sort_values("query_lesion").reset_index(drop=True)
    fig = figure(11.0)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.15, 1])  # pyright: ignore[reportGeneralTypeIssues]  # matplotlib stub: an array of SubFigures for 2 rows
    axes = top.subplots(1, 3)
    draw_labels(axes[0], image[window], q_labels[window], q_colors, "(a) planted copy, rotated 5°", filled=False)
    draw_labels(axes[1], image[window], t_labels[window], s_colors, f"(b) source {d.source}", filled=False)
    overlay_tile(axes[2], q_labels[window] > 0, t_labels[window] > 0, "(c) overlay, lines = pair score s")
    draw_pair_lines(axes[2], d.pairs, q_labels, t_labels, window)
    crops = bottom.subplots(1, max(len(pairs), 3))
    for ax, p in zip(crops, pairs.itertuples()):
        qn, tn = parse_lesion_id(str(p.query_lesion))[1], parse_lesion_id(str(p.train_lesion))[1]
        a, b = centred_crop(q_labels == qn), centred_crop(t_labels == tn)
        overlay_tile(ax, a, b, f"L{qn} ↔ L{tn}: {p.shape:.2f} × {p.position:.2f} × {p.size:.2f}\n= s {p.s:.2f}")
    for ax in crops[len(pairs):]:
        ax.axis("off")
    terms = " + ".join(f"{s:.2f}" for s in pairs["s"])
    bottom.suptitle(f"(d) each pair, centred: shape × position × size = s;   C = ({terms}) / "
                    f"{max(d.n_query, d.n_train)} = {d.C:.2f}", fontsize=7.5)
    overlay_legend(bottom)
    out_name = "2-5_e2_lesion_pairs"
    save_both(fig, out, out_name)
    return index_row(out_name, "2.5", f"E2 on planted {key}: lesion pairs, centred crops with their factors, C",
                     planted.sources)


def centred_crop(lesion: np.ndarray) -> np.ndarray:
    """A (2 PAIR_CROP + 1)-px square crop of a lesion mask centred on its rounded centroid (as E2's shape
    factor aligns two lesions), zero-padded at the image border."""
    r, c = np.rint(np.argwhere(lesion).mean(axis=0)).astype(int)
    padded = np.pad(lesion, PAIR_CROP)
    return padded[r:r + 2 * PAIR_CROP + 1, c:c + 2 * PAIR_CROP + 1]


# --- 2.7: planted variants and what 1 px does -----------------------------------------------------------

def variants_figure(cfg: dict, name: str, planted: Planted, base_key: str, out: Path) -> dict:
    """(a) Every planted variant of one base slice: the variant mask (red, filled) over the base slice
    with the base mask outlined (blue). (b) The same 1 px change on a small and a large T lesion: the
    lesion, eroded and dilated by one pixel (3x3 cross, as P5), with the areas."""
    rows = planted.table[planted.table["base_key"] == base_key]
    base = planted.real_labels[base_key] > 0
    image = planted.image(base_key)
    masks = {str(v): planted.labels[str(k)] > 0 for k, v in zip(rows["key"], rows["variant"])}
    window = zoom_window([base, *masks.values()])
    fig = figure(13.0)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.75, 1])  # pyright: ignore[reportGeneralTypeIssues]  # matplotlib stub: an array of SubFigures for 2 rows
    for ax, (variant, label) in zip(top.subplots(2, 4).ravel(), VARIANTS):
        if variant not in masks:
            ax.axis("off")
            continue
        image_tile(ax, image[window], [(base[window], COLORS["T"])], label)
        fill(ax, [(masks[variant][window], COLORS["S"])], alpha=0.55)
    top.suptitle(f"(a) the planted variants of {base_key} (blue outline = base mask, red = variant)", fontsize=8)

    lesions = read_table(config.output_dir(cfg, name) / "lesions.csv")
    t = lesions[(lesions["set"] == "T") & (lesions["solidity"].astype(float) >= 0.85)]
    tiles = bottom.subplots(1, 2 * len(BOUNDARY_SIZES) + 1, gridspec_kw={"width_ratios": [1, 1, 0.25, 1, 1]})
    tiles[2].axis("off")
    chosen = []
    for j, size in enumerate(BOUNDARY_SIZES):
        row = t.iloc[int(np.argmin((t["area"].astype(int) - size).abs().to_numpy()))]
        chosen.append(str(row["lesion_id"]))
        lesion = planted.real_labels[str(row["key"])] == int(row["label"])
        eroded, dilated = morph(lesion, "erode", 1), morph(lesion, "dilate", 1)
        w = zoom_window([dilated])
        n0 = int(lesion.sum())
        for k, (m, verb) in enumerate(((eroded, "eroded"), (dilated, "dilated"))):
            ax = tiles[3 * j + k]
            image_tile(ax, np.ones(m[w].shape), [(lesion[w], COLORS["T"])],
                       f"{n0} px {verb}: {int(m.sum())} px\n(× {m.sum() / n0:.2f})")
            fill(ax, [(m[w], COLORS["S"])], alpha=0.6)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("0.6")
    bottom.suptitle("(b) one pixel of boundary on a small and a large lesion (blue = lesion, red = after the change)",
                    fontsize=8)
    out_name = "2-7_planted_variants"
    save_both(fig, out, out_name)
    return index_row(out_name, "2.7", f"planted variants of {base_key}; 1 px erosion and dilation of T lesions "
                     f"{', '.join(chosen)}", [*planted.sources, config.output_dir(cfg, name) / "lesions.csv"])


# --- 2.11: the match report -----------------------------------------------------------------------------

def report_figure(cfg: dict, name: str, planted: Planted, key: str, out: Path) -> dict:
    """The images of a match report on a planted P6-add copy (one lesion added to the base): (a) the copy,
    (b) the source the report names with partner lesions in the same colours, (c) where the source lies in
    the brain, (d) the overlay with lines joining paired lesions; below, the report's verdict line."""
    d = planted.report(key)
    q_labels, t_labels = planted.labels[key], planted.real_labels[d.source]
    image = planted.image(d.source)
    window = zoom_window([q_labels > 0, t_labels > 0])
    q_colors = lesion_colors(q_labels)
    s_colors = pair_colors(d.pairs, q_colors, t_labels)
    fig = figure(6.6)
    axes = fig.subplots(1, 4)
    draw_labels(axes[0], image[window], q_labels[window], q_colors, "(a) planted copy,\none lesion added", filled=False)
    draw_labels(axes[1], image[window], t_labels[window], s_colors, f"(b) named source\n{d.source}", filled=False)
    draw_labels(axes[2], image, t_labels, s_colors, "(c) source in the brain\n(box = zoom)", filled=False)
    draw_box(axes[2], (window[0].start, window[0].stop - 1, window[1].start, window[1].stop - 1))
    overlay_tile(axes[3], q_labels[window] > 0, t_labels[window] > 0, "(d) overlay,\nlines = pair score s")
    draw_pair_lines(axes[3], d.pairs, q_labels, t_labels, window)
    line = verdict(d, patient_name, class_rate(chance_rates(cfg, name), d.e2_class, d.stratum))
    fig.text(0.5, 0.01, "Verdict: " + line, ha="center", va="bottom", fontsize=6.5, wrap=True)
    out_name = "2-11_match_report"
    save_both(fig, out, out_name)
    return index_row(out_name, "2.11", f"the match report's images and verdict on planted {key}", planted.sources)
