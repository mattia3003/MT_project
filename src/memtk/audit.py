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

from memtk.lesions import (BRAIN_METHODS, brain_mask, chessboard_distance, fraction_outside, grow, label_components,
                           speck_gaps)
from memtk.plotting import hide_unused, save, show_slice, tile_figure

# Outline colours used in every E0b figure
BRAIN_COLOR, LESION_COLOR, BOX_COLOR, HIGHLIGHT_COLOR = "cyan", "red", "yellow", "lime"


# --- brain-review: candidate brain-mask methods ----------------------------------------------

def evaluate_brain_methods(image: np.ndarray, mask: np.ndarray, min_area: int, low_threshold: float,
                           methods=tuple(BRAIN_METHODS)) -> list[dict]:
    """For one slice and each method: brain area, and how many of the mask's real-sized
    components (>= min_area px) lie partly or mostly outside the brain."""
    labels, n = label_components(mask)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    sizes[0] = 0                                     # background is not a component
    lesion_ids = np.flatnonzero(sizes >= min_area)
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
    background = image[~(grow(brain, 3) | grow(mask > 0, 1))]
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


# --- lesions-review: minimum lesion area and attach gap --------------------------------------

def component_table(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Every 8-connected component of a mask as (size, row, col) of its first pixel."""
    comp, n = label_components(mask)
    if n == 0:
        return []
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    labels, first = np.unique(comp.ravel(), return_index=True)
    width = mask.shape[1]
    return [(int(sizes[lab]), *divmod(int(f), width))  # pyright: ignore[reportReturnType]  # unpacked divmod is 2 ints
            for lab, f in zip(labels, first) if lab > 0]


def gap_to_nearest_lesion(mask: np.ndarray, row: int, col: int, min_area: int) -> int:
    """Gap (background px, Chebyshev) from the component containing (row, col) to the nearest
    other component of at least `min_area` px; -1 if there is none."""
    comp, n = label_components(mask)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    sizes[0] = 0
    target = comp == comp[row, col]
    others = np.isin(comp, np.flatnonzero(sizes >= min_area)) & ~target
    if not others.any():
        return -1
    return int(chessboard_distance(others)[target].min()) - 1


def propose_min_area(t_sizes: pd.Series, max_share: float = 0.01) -> tuple[int, str]:
    """The largest minimum area m such that at most `max_share` of the real T components are
    smaller than m (they would count as specks)."""
    m = 1
    while (t_sizes < m + 1).mean() <= max_share:
        m += 1
    return m, (f"largest m with at most {max_share:.0%} of T components smaller than m "
               f"({(t_sizes < m).mean():.2%} are)")


def size_counts(components: pd.DataFrame, max_size: int = 10) -> pd.DataFrame:
    """Per group: number and share of components of each size 1..max_size, and larger."""
    sizes = components["size"].clip(upper=max_size + 1)
    table = pd.crosstab(components["group"], sizes).reindex(columns=range(1, max_size + 2), fill_value=0)
    table.columns = [*map(str, range(1, max_size + 1)), f">{max_size}"]
    table.insert(0, "components", table.sum(axis=1))
    return table


def mask_specks(group: str, key: str, mask: np.ndarray, min_area: int) -> list[dict]:
    """The specks of one mask at one minimum area, with their gap to the nearest lesion."""
    return [{"group": group, "key": key, "min_area": min_area, "size": s, "gap": g, "row": r, "col": c}
            for s, g, r, c in speck_gaps(mask, min_area)]


def summarize_attach_or_drop(components: pd.DataFrame, specks: pd.DataFrame, min_areas, gaps) -> pd.DataFrame:
    """Per group, minimum area and attach gap: lesions (which depend on the minimum area only),
    slices left without a lesion, stray components and pixels attached, isolated specks and the
    masks that have at least one. Gap 0 is drop-only."""
    rows = []
    per_mask = components.groupby(["group", "key"], sort=False)["size"]
    for m in min_areas:
        n_lesions = per_mask.agg(lambda s: int((s >= m).sum())).rename("n_lesions").reset_index()
        for group, nl in n_lesions.groupby("group", sort=False):
            sp = specks[(specks["group"] == group) & (specks["min_area"] == m)]
            for g in gaps:
                stray = (sp["gap"] >= 1) & (sp["gap"] <= g)
                rows.append({"group": group, "min_area": m, "gap": g, "slices": len(nl),
                             "lesions": int(nl["n_lesions"].sum()), "no_lesion": int((nl["n_lesions"] == 0).sum()),
                             "strays": int(stray.sum()), "stray_px": int(sp.loc[stray, "size"].sum()),
                             "isolated_specks": int((~stray).sum()),
                             "masks_with_isolated": int(sp.loc[~stray, "key"].nunique())})
    return pd.DataFrame(rows).set_index(["group", "min_area", "gap"])


def gap_counts(specks: pd.DataFrame, bins=((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 6))) -> pd.DataFrame:
    """Per group: specks by gap to the nearest lesion, and specks in masks without a lesion."""
    cols = {"no lesion": specks["gap"] < 0}
    for lo, hi in bins:
        name = str(lo) if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 6 else f">={lo}")
        cols[f"gap {name}"] = specks["gap"].between(lo, hi)
    return pd.DataFrame(cols).groupby(specks["group"], sort=False).sum()


def size_figure(components: pd.DataFrame, min_area: int, path: Path, title: str, max_size: int = 30) -> Path:
    """Share of components by size (1..max_size, then larger) per group, log scale, with the
    proposed minimum lesion area marked."""
    fig = Figure(figsize=(9, 3.6))
    ax = fig.subplots()
    sizes = np.arange(1, max_size + 2)
    for k, (group, d) in enumerate(components.groupby("group", sort=False)):
        share = d["size"].clip(upper=max_size + 1).value_counts(normalize=True).reindex(sizes, fill_value=0)
        offset = sizes + (k - 1.5) * 0.2  # pyright: ignore[reportOperatorIssue]  # numpy stubs: array + float is valid
        ax.bar(offset, share.to_numpy(), width=0.2, label=f"{group} ({len(d)})")
    ax.axvline(min_area - 0.5, color="k", ls="--", lw=1)
    ax.text(min_area - 0.4, ax.get_ylim()[1] * 0.5, f" lesion >= {min_area} px", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [*map(str, sizes[:-1]), f">{max_size}"], fontsize=7)
    ax.set_xlabel("component size (px)")
    ax.set_ylabel("share of components (log)")
    ax.legend(fontsize=8)
    return save(fig, path, title)


def crop_sheet(items: list[tuple[str, np.ndarray, np.ndarray, int, int]], path: Path, title: str,
               size: int = 32, n_cols: int = 8) -> Path:
    """Tiles of (title, image, mask, row, col): a size x size crop centred on (row, col), with the
    component containing that pixel outlined in lime and the rest of the mask in red."""
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = tile_figure(n_rows, n_cols, tile=1.9)
    h = size // 2
    for ax, (tile_title, image, mask, r, c) in zip(axes.ravel(), items):
        comp, _ = label_components(mask)
        target = comp == comp[r, c]

        def crop(a: np.ndarray) -> np.ndarray:
            return np.pad(a, h)[r:r + size, c:c + size]

        show_slice(ax, crop(image), [(crop((mask > 0) & ~target), LESION_COLOR), (crop(target), HIGHLIGHT_COLOR)],
                   tile_title)
    hide_unused(axes, len(items))
    return save(fig, path, title)


# --- lesions: overview of the production lesion tables ---------------------------------------

def summarize_lesions(slices: pd.DataFrame, lesions: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per group (column `by` in both tables): slices and lesions, slices without a lesion, share
    of slices with a small lesion, lesion area percentiles, attached strays, isolated specks
    (and the share of them outside the brain), lesions more than half outside the brain."""
    s, les = slices.groupby(by, sort=False), lesions.groupby(by, sort=False)
    return pd.DataFrame({
        "slices": s.size(), "lesions": s["n_lesions"].sum(), "lesions_per_slice": s["n_lesions"].mean(),
        "no_lesion": s["n_lesions"].agg(lambda x: int((x == 0).sum())),
        "share_with_lesion_lt40": s["has_lesion_lt40"].mean(),
        "area_p5": les["area"].quantile(0.05), "area_median": les["area"].median(),
        "area_p95": les["area"].quantile(0.95),
        "strays": s["n_strays"].sum(), "stray_px": s["stray_px"].sum(),
        "specks": s["n_specks"].sum(), "slices_with_specks": s["n_specks"].agg(lambda x: int((x > 0).sum())),
        "specks_outside_brain_share": s["specks_outside_brain"].sum() / s["n_specks"].sum(),
        "lesions_over_half_outside": les["frac_outside_brain"].agg(lambda x: int((x > 0.5).sum())),
    })


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
