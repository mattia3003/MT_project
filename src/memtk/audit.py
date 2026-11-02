"""Evidence and overview figures for E0b: candidate comparisons, tables, proposals.

Review stages never write production data. They compute candidates on the relevant data and
propose a config value, which the user sets by hand after viewing the evidence. Production
stages use the overview figures here to show what they computed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy import ndimage

from memtk.lesions import BRAIN_METHODS, brain_mask, fraction_outside, label_components
from memtk.plotting import hide_unused, save, show_slice, tile_figure

# Outline colours used in every E0b figure
BRAIN_COLOR, LESION_COLOR, BOX_COLOR = "cyan", "red", "yellow"


# --- brain-review: candidate brain-mask methods ----------------------------------------------

def evaluate_brain_methods(image: np.ndarray, mask: np.ndarray, min_area: int, low_threshold: float,
                           methods=tuple(BRAIN_METHODS)) -> list[dict]:
    """For one slice and each method: brain area, and how many of the mask's real-sized
    components (>= min_area px) lie partly or mostly outside the brain."""
    labels, n = label_components(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    lesion_ids = np.flatnonzero(sizes >= min_area) + 1
    rows = []
    for method in methods:
        brain = brain_mask(image, method, low_threshold)
        outside = [fraction_outside(labels == i, brain) for i in lesion_ids]
        rows.append({"method": method, "brain_px": int(brain.sum()), "n_lesions": len(lesion_ids),
                     "n_over_half_outside": sum(f > 0.5 for f in outside),
                     "n_any_outside": sum(f > 0 for f in outside)})
    return rows


def summarize_brain_methods(per_slice: pd.DataFrame) -> pd.DataFrame:
    """Per data group (T, S) and method: lesion counts, outside shares and brain-area percentiles."""
    g = per_slice.groupby(["group", "method"], sort=False)
    summary = g.agg(slices=("brain_px", "size"), lesions=("n_lesions", "sum"),
                    over_half_outside=("n_over_half_outside", "sum"), any_outside=("n_any_outside", "sum"),
                    brain_px_p1=("brain_px", lambda x: x.quantile(0.01)),
                    brain_px_p5=("brain_px", lambda x: x.quantile(0.05)),
                    brain_px_median=("brain_px", "median"))
    summary.insert(3, "share_over_half_outside", summary["over_half_outside"] / summary["lesions"])
    summary.insert(5, "share_any_outside", summary["any_outside"] / summary["lesions"])
    return summary


def propose_brain_method(summary: pd.DataFrame, group: str = "T", target: float = 0.005) -> tuple[str, str]:
    """Among methods whose share of lesions more than half outside the brain is at most `target`
    on `group`, the one with the fewest lesions even partly outside (the brain edge follows the
    tissue); if none reaches the target, the one with the lowest share more than half outside."""
    s = summary.xs(group, level="group")
    ok = s[s["share_over_half_outside"] <= target]
    if len(ok):
        method = str(ok["share_any_outside"].idxmin())
        return method, (f"fewest {group} lesions even partly outside the brain, among methods with at most "
                        f"{target:.1%} of {group} lesions more than half outside")
    method = str(s["share_over_half_outside"].idxmin())
    return method, f"no method reached {target:.1%} on {group}; lowest share of lesions more than half outside"


def method_grid(examples: list[tuple[str, np.ndarray, np.ndarray]], path: Path, title: str,
                low_threshold: float, methods=tuple(BRAIN_METHODS)) -> Path:
    """One row per example slice, one column per method: brain outline (cyan), lesions (red)."""
    fig, axes = tile_figure(len(examples), len(methods), tile=2.4)
    for r, (key, image, mask) in enumerate(examples):
        for c, method in enumerate(methods):
            brain = brain_mask(image, method, low_threshold)
            show_slice(axes[r, c], image, [(brain, BRAIN_COLOR), (mask > 0, LESION_COLOR)],
                       f"{key}\n{method}: {int(brain.sum())} px")
    return save(fig, path, title)


# --- brain-review: does the low threshold sit in a gap between background and tissue? ---------

# Multiples of brain.low_threshold at which the brain area is recomputed
THRESHOLD_FACTORS = (0.25, 0.5, 1.0, 2.0, 3.0)


def threshold_sensitivity(image: np.ndarray, mask: np.ndarray, low_threshold: float) -> dict:
    """For one slice: area of the `low_threshold_all` brain at each multiple of the threshold,
    and the background noise level (99.9th percentile more than 3 px from the brain and more
    than 1 px from the mask: stage 2 paints a bright dot at every mask speck, which is not
    noise). If the threshold sits in an empty gap between background and tissue, the area
    barely changes when it moves."""
    brain = brain_mask(image, "low_threshold_all", low_threshold)
    near = ndimage.binary_dilation(brain, iterations=3) | ndimage.binary_dilation(mask > 0, iterations=1)
    background = image[~near]
    return {**{f"area_x{f:g}": int(brain_mask(image, "low_threshold_all", low_threshold * f).sum())
               for f in THRESHOLD_FACTORS},
            "background_p999": float(np.percentile(background, 99.9)) if background.size else np.nan}


def summarize_threshold_sensitivity(per_slice: pd.DataFrame, low_threshold: float,
                                    min_brain_px: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per group: (1) the median and 95th-percentile absolute change of brain area at each
    multiple of the threshold, relative to the threshold itself (slices with a brain of at least
    `min_brain_px`, where a relative change is meaningful); (2) the background noise level of
    the median slice and of a near-worst slice."""
    changes, levels = [], []
    for group, d in per_slice.groupby("group", sort=False):
        d = d[d["area_x1"] >= min_brain_px]
        for f in THRESHOLD_FACTORS:
            rel = (d[f"area_x{f:g}"] - d["area_x1"]) / d["area_x1"]
            changes.append({"group": group, "threshold": round(low_threshold * f, 6), "slices": len(d),
                            "median_change": rel.median(), "p95_abs_change": rel.abs().quantile(0.95)})
        levels.append({"group": group, "background_p999_median": d["background_p999"].median(),
                       "background_p999_worst1pct": d["background_p999"].quantile(0.99)})
    return (pd.DataFrame(changes).set_index(["group", "threshold"]),
            pd.DataFrame(levels).set_index("group"))


# --- brain-review: near-blank (failed) synthetic MRIs ----------------------------------------

def sorted_sheet(items: list[tuple[str, str, np.ndarray, np.ndarray]], path: Path, title: str,
                 n_cols: int = 8) -> Path:
    """Tiles in the given order, each (key, label, image, mask), with the mask outlined in red.

    Images are shown in [0, 1] like every other figure, so a near-blank image looks dark.
    """
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = tile_figure(n_rows, n_cols, tile=1.9)
    for ax, (key, label, image, mask) in zip(axes.ravel(), items):
        show_slice(ax, image, [(mask > 0, LESION_COLOR)], f"{key}\n{label}")
    hide_unused(axes, len(items))
    return save(fig, path, title)


def tissue_counts(t_tissue: pd.Series, s_tissue: pd.Series, cuts=(20, 50, 100, 150, 200, 300, 400)) -> pd.DataFrame:
    """For each candidate cut on a tissue measure: synthetic samples that would count as failed,
    and real T lesion slices below it (which must stay rare: small real slices exist)."""
    return pd.DataFrame({"cut": cuts,
                         "S_below": [int((s_tissue < c).sum()) for c in cuts],
                         "S_share_below": [float((s_tissue < c).mean()) for c in cuts],
                         "T_slices_below": [int((t_tissue < c).sum()) for c in cuts]})


def tissue_histogram(t_tissue: pd.Series, s_tissue: pd.Series, path: Path, title: str) -> Path:
    """Tissue area of T lesion slices and synthetic samples, on log axes so the low tail shows."""
    fig = Figure(figsize=(7.5, 3.4))
    ax = fig.subplots()
    bins = np.logspace(0, np.log10(16384), 60)
    ax.hist(s_tissue.clip(lower=1), bins=bins, color="0.3", alpha=0.7, label=f"S ({len(s_tissue)})")
    ax.hist(t_tissue.clip(lower=1), bins=bins, histtype="step", color="tab:blue", lw=1.5,
            label=f"T lesion slices ({len(t_tissue)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("brain tissue (px; 0 shown as 1)")
    ax.set_ylabel("count (log)")
    ax.legend(fontsize=8)
    return save(fig, path, title)


# --- brain: overview of the production brain masks -------------------------------------------

def summarize_brains(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per group: slices, slices without any brain, brain-area percentiles and median box size.

    Expects the columns of a `brain.csv` (brain_px, box_r0..box_c1 with -1 for no brain)."""
    has_box = df["box_r0"] >= 0
    d = df.assign(box_h=(df["box_r1"] - df["box_r0"] + 1).where(has_box),
                  box_w=(df["box_c1"] - df["box_c0"] + 1).where(has_box))
    return d.groupby(by, sort=False).agg(
        slices=("brain_px", "size"), no_brain=("brain_px", lambda x: int((x == 0).sum())),
        brain_px_p1=("brain_px", lambda x: x.quantile(0.01)), brain_px_p5=("brain_px", lambda x: x.quantile(0.05)),
        brain_px_median=("brain_px", "median"), box_h_median=("box_h", "median"), box_w_median=("box_w", "median"))


def brain_sheet(items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple | None]], path: Path,
                title: str, n_cols: int = 8) -> Path:
    """Tiles of (title, image, mask, brain, box): brain outline (cyan), brain box (dashed yellow),
    mask (red). The box is inclusive (r0, r1, c0, c1), or None to draw no box."""
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = tile_figure(n_rows, n_cols, tile=1.9)
    for ax, (tile_title, image, mask, brain, box) in zip(axes.ravel(), items):
        show_slice(ax, image, [(brain, BRAIN_COLOR), (mask > 0, LESION_COLOR)], tile_title)
        if box is not None:
            r0, r1, c0, c1 = box
            ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0 + 1, r1 - r0 + 1, fill=False,
                                   edgecolor=BOX_COLOR, linestyle="--", linewidth=0.6))
    hide_unused(axes, len(items))
    return save(fig, path, title)
