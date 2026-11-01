"""Evidence for the E0b review stages: candidate comparisons, tables, figures and proposals.

Review stages never write production data. They compute candidates on the relevant data and
propose a config value, which the user sets by hand after viewing the evidence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from memtk.lesions import BRAIN_METHODS, brain_mask, fraction_outside, label_components
from memtk.plotting import hide_unused, save, show_slice, tile_figure

# Outline colours used in every E0b figure
BRAIN_COLOR, LESION_COLOR = "cyan", "red"


# --- brain-review: candidate brain-mask methods ----------------------------------------------

def evaluate_brain_methods(image: np.ndarray, mask: np.ndarray, min_area: int,
                           methods=tuple(BRAIN_METHODS)) -> list[dict]:
    """For one slice and each method: brain area, and how many of the mask's real-sized
    components (>= min_area px) lie partly or mostly outside the brain."""
    labels, n = label_components(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    lesion_ids = np.flatnonzero(sizes >= min_area) + 1
    rows = []
    for method in methods:
        brain = brain_mask(image, method)
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
                methods=tuple(BRAIN_METHODS)) -> Path:
    """One row per example slice, one column per method: brain outline (cyan), lesions (red)."""
    fig, axes = tile_figure(len(examples), len(methods), tile=2.4)
    for r, (key, image, mask) in enumerate(examples):
        for c, method in enumerate(methods):
            brain = brain_mask(image, method)
            show_slice(axes[r, c], image, [(brain, BRAIN_COLOR), (mask > 0, LESION_COLOR)],
                       f"{key}\n{method}: {int(brain.sum())} px")
    return save(fig, path, title)


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
    """For each candidate min_tissue_px: synthetic samples that would count as failed, and real
    T lesion slices with that little tissue (which must stay rare: small real slices exist)."""
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
