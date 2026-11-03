"""E0b stage `stats`: the E0 description of the real sets and the synthetic main audit.

`run_stats` writes the description table, distribution figures (lesions per slice, area,
centroid density), the bound-impact table (input for E3), adjacent-slice Dice, native vs
model-space lesion area and brain size by slice height. No decision is made here.

Below the stage: the computations that read volumes, then tables, then figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from tqdm import tqdm

from memtk import config, ids
from memtk.e0b.common import load_real
from memtk.io import LabelStack, load_image, patient_files, read_table
from memtk.lesions import brain_mask, count_components
from memtk.plotting import SET_COLORS, save, tile_figure
from memtk.stages import decided, refuse_overwrite, require


# --- stage ------------------------------------------------------------------------------------

CANDIDATE_BOUNDS = (5, 8, 10, 12, 15, 20, 25, 30, 40)


NATIVE_SIZE_BINS = [0, 5, 10, 20, 50, 100, 200, 500, np.inf]


def run_stats(cfg: dict, dataset: str | None = None, pool: str | None = None,
              overwrite: bool = False) -> None:
    """The E0 description of the real sets and the synthetic main audit (mask class clean or minor):
    description table, native sizes, lesions per slice, lesion area, centroid density in pixels
    and brain-relative, bound impact (input for E3), adjacent-slice Dice, native vs model lesion
    area, brain size by slice height. Needs the `qc` stage; reads the real volumes and native
    masks again. Writes `results/<dataset>/e0b/stats/`."""
    small_px = int(cfg["qc"]["small_lesion_px"])
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, name, "e0b") / "stats"
    refuse_overwrite([out], overwrite)
    require(syn_dir / "qc.csv", "qc")

    # the sets: real lesion slices per set, the main audit (mask_class clean or minor) and all of S
    real_sl, real_les = read_table(real_dir / "slices.csv"), read_table(real_dir / "lesions.csv")
    syn_sl = read_table(syn_dir / "slices.csv").merge(read_table(syn_dir / "qc.csv")[["key", "mask_class"]], on="key")
    syn_les = read_table(syn_dir / "lesions.csv")
    main_sl = syn_sl[syn_sl["mask_class"].isin(["clean", "minor"])]
    sets = {s: (real_sl[real_sl["set"] == s], real_les[real_les["set"] == s]) for s in ("T", "H-cal", "H-eval")}
    sets["S main"] = (main_sl, syn_les[syn_les["key"].isin(main_sl["key"])])
    sets["S all"] = (syn_sl, syn_les)

    describe = describe_sets(sets, small_px)
    patients = read_table(real_dir / "patients.csv")
    native = pd.crosstab(patients["native_dim0"].astype(str) + "x" + patients["native_dim1"].astype(str),
                         patients["set"]).rename_axis("native size")
    bounds = bound_impact(sets, CANDIDATE_BOUNDS)
    without = bounds.pivot(index="bound", columns="set", values="share_slices_without")[list(sets)]
    kept = bounds.pivot(index="bound", columns="set", values="share_lesions_kept")[list(sets)]

    # figures compare the real sets with the main audit; the density pools H-cal and H-eval
    shown = ["T", "H-cal", "H-eval", "S main"]
    lesions_per_slice_figure({s: sets[s][0]["n_lesions"] for s in shown}, out / f"lesions_per_slice_{pool_name}.png",
                             f"Lesions per slice: real lesion slices per set and the synthetic main audit "
                             f"({pool_name}, mask class clean or minor)")
    area_figure({s: sets[s][1]["area"] for s in shown}, out / f"lesion_area_{pool_name}.png",
                "Lesion area: cumulative share of lesions (lesions >= the minimum lesion area)")
    h_les = pd.concat([sets["H-cal"][1], sets["H-eval"][1]])
    centroid_density_figure({"T": sets["T"][1], "H": h_les, "S main": sets["S main"][1]},
                            out / f"centroid_density_{pool_name}.png",
                            "Lesion centroids: T, H (H-cal + H-eval) and the synthetic main audit; top row in "
                            "pixels, bottom row relative to the brain box (0 = first, 1 = last row/column)")

    # adjacency: Dice between consecutive lesion slices of a patient, on the cleaned masks
    pairs = adjacency_pairs(real_sl, LabelStack(real_dir / "labels.npz"))
    adjacency = (pairs.groupby(["set", "patient_id"], sort=False)["dice"]
                 .agg(pairs="size", median="median", min="min", max="max").reset_index())
    adjacency_sets = pd.DataFrame({
        "patients": adjacency.groupby("set", sort=False).size(),
        "pairs": pairs.groupby("set", sort=False).size(),
        "median_of_patient_medians": adjacency.groupby("set", sort=False)["median"].median(),
        "median_all_pairs": pairs.groupby("set", sort=False)["dice"].median(),
        "share_pairs_dice_ge_0.5": pairs.groupby("set", sort=False)["dice"].agg(lambda d: (d >= 0.5).mean()),
    })
    adjacency_figure(pairs, out / f"adjacency_dice_{name}.png",
                     f"{name}: Dice between consecutive lesion slices of a patient (cleaned masks)")

    # native vs model-space area, on slices with exactly one component in both spaces
    areas = native_vs_model(real_sl, patients, config.real_dataset(cfg, dataset)[1], cfg["preprocessing"])
    ratio_by_size = areas.groupby("size_bin", observed=True)["ratio"].describe()[["count", "25%", "50%", "75%"]]
    area_ratio_figure(areas, out / f"native_vs_model_{name}.png",
                      f"{name}: lesion area after conversion to 128x128 against plain downscaling, on slices "
                      f"with one lesion in both spaces")

    # brain area of every real slice by relative position in the head, against synthetic brains
    heights = real_brain_heights(real_dir, patients, set(real_sl["key"]),
                                 decided(cfg, "brain.method", "e0b_audit.py brain-review"),
                                 float(decided(cfg, "brain.low_threshold", "e0b_audit.py brain-review")))
    by_height = brain_by_height(heights)
    syn_brain = read_table(syn_dir / "brain.csv").merge(main_sl[["sample_id"]], on="sample_id")
    syn_brain_px = syn_brain.loc[~syn_brain["mri_failed"].astype(bool), "brain_px"]
    real_lesion_px = heights.loc[heights["lesion_slice"], "brain_px"]
    s_median = float(syn_brain_px.median())
    small_positions = [round(float(p), 3) for p in by_height.index[by_height["median"] <= s_median]]
    brain_height_figure(by_height, real_lesion_px, syn_brain_px, out / f"brain_by_height_{pool_name}.png",
                        f"Brain area: real slices by position in the head ({name}) against synthetic brains "
                        f"({pool_name}, main audit, failed MRIs left out)")

    describe.round(3).to_csv(out / f"description_{pool_name}.csv")
    native.to_csv(out / f"native_sizes_{name}.csv")
    bounds.round(4).to_csv(out / f"bound_impact_{pool_name}.csv", index=False)
    adjacency.round(4).to_csv(out / f"adjacency_dice_{name}.csv", index=False)
    areas.round(4).to_csv(out / f"native_vs_model_{name}.csv", index=False)
    by_height.round(1).to_csv(out / f"brain_by_height_{name}.csv")
    text = (f"E0b stats: dataset '{name}', pool '{pool_name}' (S main = mask class clean or minor)\n\n"
            f"Data description:\n{describe.round(3).to_string()}\n\n"
            f"Native in-plane sizes (patients per set):\n{native.to_string()}\n\n"
            f"Bound impact (input for E3, not decided here), by candidate informative-lesion bound:\n"
            f"Share of slices left without an informative lesion:\n{without.round(3).to_string()}\n"
            f"Share of lesions kept:\n{kept.round(3).to_string()}\n"
            f"Patients with at least one informative slice, per bound: see bound_impact_{pool_name}.csv.\n\n"
            f"Adjacency: Dice between consecutive lesion slices of a patient (cleaned masks):\n"
            f"{adjacency_sets.round(3).to_string()}\n\n"
            f"Lesion area after conversion / native area scaled to the model grid (plain downscaling = 1), "
            f"{len(areas)} slices with one lesion in both spaces, by native lesion area:\n"
            f"{ratio_by_size.round(2).to_string()}\n\n"
            f"Brain area by relative slice position (0 = first, 1 = last slice with brain), all "
            f"{len(heights)} real slices:\n{by_height.round(0).to_string()}\n"
            f"Synthetic brains (main audit, failed MRIs left out): median {s_median:.0f} px, IQR "
            f"{syn_brain_px.quantile(0.25):.0f}-{syn_brain_px.quantile(0.75):.0f} px. Real brains are this small or "
            f"smaller (median) only at positions {small_positions}. Brain at most {s_median:.0f} px: "
            f"{(syn_brain_px <= s_median).mean():.1%} of synthetic samples, "
            f"{(real_lesion_px <= s_median).mean():.1%} of real lesion slices.\n")
    (out / "summary.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "stats", "pool": pool_name})
    print(f"\n{text}\nWrote {out}")


# --- computations on the volumes --------------------------------------------------------------

def adjacency_pairs(slices: pd.DataFrame, labels: LabelStack) -> pd.DataFrame:
    """Dice of every pair of consecutive lesion slices (z, z + 1) of a patient, on the cleaned
    masks (labels > 0)."""
    rows = []
    for (set_name, pid), group in slices.groupby(["set", "patient_id"], sort=False):
        zs = set(group["z"])
        for z in sorted(zs):
            if z + 1 in zs:
                a, b = labels[ids.slice_key(str(pid), z)] > 0, labels[ids.slice_key(str(pid), z + 1)] > 0
                rows.append({"set": set_name, "patient_id": pid, "z": z, "dice": mask_dice(a, b)})
    return pd.DataFrame(rows)


def native_vs_model(slices: pd.DataFrame, patients: pd.DataFrame, dataset_cfg: dict, prep: dict) -> pd.DataFrame:
    """Per real slice with exactly one component in both spaces: native area, the native area
    scaled to the model grid (plain downscaling), the model-space area, their ratio and a native
    size bin. A single speck counts too: enlargement matters most for tiny components."""
    h, w = prep["spatial_size"]
    orig_split = patients.set_index("patient_id")["orig_split"]
    single = slices[slices["n_components"] == 1]
    rows = []
    for pid, group in tqdm(single.groupby("patient_id"), desc="native masks"):
        _, mask_path = patient_files(dataset_cfg, str(orig_split[pid]), str(pid))
        native_mask = np.asanyarray(load_image(mask_path).dataobj) > 0
        scale = (h / native_mask.shape[0]) * (w / native_mask.shape[1])
        for key, set_name, z, model_px in zip(group["key"], group["set"], group["z"],
                                              group["lesion_px"] + group["speck_px"]):
            native_slice = np.take(native_mask, int(z), axis=prep["slice_axis"])
            if count_components(native_slice) == 1:
                rows.append({"key": key, "set": set_name, "native_area": int(native_slice.sum()),
                             "expected_area": int(native_slice.sum()) * scale, "model_area": int(model_px)})
    areas = pd.DataFrame(rows)
    areas["ratio"] = areas["model_area"] / areas["expected_area"]
    edges = NATIVE_SIZE_BINS
    labels = [f"{lo + 1}-{hi}" if hi != np.inf else f">{lo}" for lo, hi in zip(edges[:-1], edges[1:])]
    areas["size_bin"] = pd.cut(areas["native_area"], edges, labels=labels)
    return areas


def real_brain_heights(real_dir: Path, patients: pd.DataFrame, lesion_keys: set, method: str,
                       low: float) -> pd.DataFrame:
    """Brain area of every real slice between a patient's first and last slice with brain, with
    its relative position there (0 = first, 1 = last) and whether it is a lesion slice."""
    rows = []
    for pid, set_name in tqdm(list(zip(patients["patient_id"], patients["set"])), desc="brain by height"):
        bravo, _ = load_real(real_dir, str(pid))
        brain_px = [int(brain_mask(bravo[:, :, z], method, low).sum()) for z in range(bravo.shape[2])]
        with_brain = [z for z, px in enumerate(brain_px) if px > 0]
        z0, z1 = with_brain[0], with_brain[-1]
        for z in range(z0, z1 + 1):
            rows.append({"patient_id": pid, "set": set_name, "z": z, "brain_px": brain_px[z],
                         "rel_pos": (z - z0) / (z1 - z0), "lesion_slice": ids.slice_key(str(pid), z) in lesion_keys})
    return pd.DataFrame(rows)


# --- tables -----------------------------------------------------------------------------------

def describe_sets(sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]], small_lesion_px: int) -> pd.DataFrame:
    """Data description, one column per set, from its (slices, lesions) tables: patients and
    slices, lesions per slice, single/multi-lesion shares, lesion area, the share of slices with
    a small lesion, and the spread of lesion centroids in pixels and brain-relative."""
    cols = {}
    for name, (sl, les) in sets.items():
        n = sl["n_lesions"]
        col = {"patients": sl["patient_id"].nunique() if "patient_id" in sl else np.nan,
               "slices": len(sl), "lesions": int(n.sum()), "lesions_per_slice": n.mean(),
               "max_lesions_per_slice": int(n.max()), "share_no_lesion": (n == 0).mean(),
               "share_single_lesion": (n == 1).mean(), "share_multi_lesion": (n >= 2).mean(),
               "area_p25": les["area"].quantile(0.25), "area_median": les["area"].median(),
               "area_p75": les["area"].quantile(0.75),
               f"share_with_lesion_lt{small_lesion_px}": sl["has_lesion_lt40"].astype(bool).mean()}
        for frame, suffix in (("", "px"), ("_brain", "brain")):
            for axis in ("row", "col"):
                values = les[f"centroid_{axis}{frame}"]
                col[f"centroid_{axis}_mean_{suffix}"] = values.mean()
                col[f"centroid_{axis}_sd_{suffix}"] = values.std()
        if "patient_id" in sl:
            per_patient = sl.groupby("patient_id").size()
            col.update(slices_per_patient_min=per_patient.min(), slices_per_patient_median=per_patient.median(),
                       slices_per_patient_max=per_patient.max())
        cols[name] = col
    return pd.DataFrame(cols)


def bound_impact(sets: dict[str, tuple[pd.DataFrame, pd.DataFrame]], bounds) -> pd.DataFrame:
    """For each candidate informative-lesion bound and set: lesions kept (area >= bound) and
    their share, slices left with no informative lesion and their share, and (real sets)
    patients that keep at least one informative slice."""
    rows = []
    for name, (sl, les) in sets.items():
        for b in bounds:
            kept = les[les["area"] >= b]
            with_informative = sl["key"].isin(kept["key"])
            row = {"set": name, "bound": b, "lesions_kept": len(kept), "share_lesions_kept": len(kept) / len(les),
                   "slices_without": int((~with_informative).sum()),
                   "share_slices_without": float((~with_informative).mean())}
            if "patient_id" in sl:
                row["patients_with_informative"] = sl.loc[with_informative, "patient_id"].nunique()
                row["patients"] = sl["patient_id"].nunique()
            rows.append(row)
    return pd.DataFrame(rows)


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice overlap of two boolean masks (1.0 if both are empty)."""
    total = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / total if total else 1.0


def brain_by_height(heights: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    """Brain-area quartiles of real slices per relative slice position (n_bins bins, indexed by
    the bin centre), with the number of slices and lesion slices per bin. Expects columns
    `rel_pos` (0 = first, 1 = last slice with brain), `brain_px` and `lesion_slice`."""
    position = np.minimum((heights["rel_pos"] * n_bins).astype(int), n_bins - 1)
    table = (heights.groupby(position)["brain_px"]
             .quantile([0.25, 0.5, 0.75])  # pyright: ignore[reportArgumentType]  # pandas stubs: a list of quantiles is valid
             .unstack())
    table.columns = ["p25", "median", "p75"]
    table["slices"] = heights.groupby(position).size()
    table["lesion_slices"] = heights.groupby(position)["lesion_slice"].sum()
    table.index = (table.index + 0.5) / n_bins
    return table.rename_axis("position")


# --- figures ----------------------------------------------------------------------------------

def cumulative_lines(ax, values: dict[str, pd.Series], unit: str, fmt: str = "g") -> None:
    """Cumulative distribution per set on one axis: the share of values at or below each value.
    Overlapping sets stay readable (lines, no bars) and medians read off at the dotted 0.5."""
    for name, v in values.items():
        x = np.sort(v.to_numpy())
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=SET_COLORS.get(name),
                label=f"{name} ({len(x)}; median {np.median(x):{fmt}}{unit})")
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.legend(fontsize=7)


def lesions_per_slice_figure(counts: dict[str, pd.Series], path: Path, title: str, max_count: int = 8) -> Path:
    """Share of slices by number of lesions (0..max_count, then more), one bar group per set."""
    fig = Figure(figsize=(8, 3.6))
    ax = fig.subplots()
    xs = np.arange(max_count + 2)
    width = 0.8 / len(counts)
    for k, (name, n) in enumerate(counts.items()):
        share = n.clip(upper=max_count + 1).value_counts(normalize=True).reindex(xs, fill_value=0)
        offset = xs + (k - (len(counts) - 1) / 2) * width  # pyright: ignore[reportOperatorIssue]  # numpy stubs: array + float is valid
        ax.bar(offset, share.to_numpy(), width=width, color=SET_COLORS.get(name), label=f"{name} ({len(n)})")
    ax.set_xticks(xs, [*map(str, xs[:-1]), f">{max_count}"])
    ax.set_xlabel("lesions per slice")
    ax.set_ylabel("share of slices")
    ax.legend(fontsize=8)
    return save(fig, path, title)


def area_figure(areas: dict[str, pd.Series], path: Path, title: str) -> Path:
    """Cumulative distribution of lesion area per set (no binning, so integer areas give no
    artefacts)."""
    fig = Figure(figsize=(8, 3.6))
    ax = fig.subplots()
    cumulative_lines(ax, areas, " px")
    ax.set_xscale("log")
    ax.set_xlabel("lesion area (px, log scale)")
    ax.set_ylabel("share of lesions with at most this area")
    return save(fig, path, title)


def centroid_density_figure(lesions: dict[str, pd.DataFrame], path: Path, title: str) -> Path:
    """2D density of lesion centroids per set, in two frames: pixels of the 128x128 grid (top
    row) and brain-relative, as a fraction of the brain box (bottom row; lesions of slices
    without a brain left out). Shares per bin; one colour scale per row."""
    frames = [("", (0, 128), 32, "pixels (row, col)"), ("_brain", (-0.2, 1.2), 28, "fraction of brain box")]
    fig, axes = tile_figure(len(frames), len(lesions), tile=3.0)
    for r, (frame, lim, n_bins, label) in enumerate(frames):
        hists = {}
        for name, les in lesions.items():
            d = les[[f"centroid_row{frame}", f"centroid_col{frame}"]].dropna()
            h, _, _ = np.histogram2d(d.iloc[:, 0], d.iloc[:, 1], bins=n_bins, range=[lim, lim])
            hists[name] = h / max(len(d), 1)
        scale = ScalarMappable(norm=Normalize(0, max(float(h.max()) for h in hists.values())), cmap="magma")
        for c, (name, h) in enumerate(hists.items()):
            ax = axes[r, c]
            ax.imshow(h, cmap=scale.cmap, norm=scale.norm, extent=(lim[0], lim[1], lim[1], lim[0]))
            ax.set_title(f"{name}: {label}", fontsize=8)
            ax.tick_params(labelsize=6)
        fig.colorbar(scale, ax=axes[r, :].tolist(), shrink=0.8, label="share of lesions")
    return save(fig, path, title)


def adjacency_figure(pairs: pd.DataFrame, path: Path, title: str) -> Path:
    """Dice between consecutive lesion slices as cumulative curves per set: all slice pairs
    (left) and the median per patient (right). A curve further right = more similar neighbours."""
    fig = Figure(figsize=(10, 3.6))
    left, right = fig.subplots(1, 2)
    per_patient = pairs.groupby(["set", "patient_id"], sort=False)["dice"].median().reset_index()
    for ax, data, what in ((left, pairs, "all slice pairs"), (right, per_patient, "median per patient")):
        cumulative_lines(ax, {str(s): d["dice"] for s, d in data.groupby("set", sort=False)}, "", ".2f")
        ax.set_xlabel("Dice between consecutive lesion slices")
        ax.set_ylabel(f"share of {what.split()[-1]} with at most this Dice")
        ax.set_title(what, fontsize=9)
    return save(fig, path, title)


def area_ratio_figure(table: pd.DataFrame, path: Path, title: str) -> Path:
    """Left: model-space lesion area against the native area scaled to 128x128 (log-log; the
    dashed line is plain downscaling). Right: the enlargement factor (model area / scaled native
    area) by native lesion size, as boxes (middle 50% of lesions, median written above)."""
    fig = Figure(figsize=(10, 3.8))
    left, right = fig.subplots(1, 2)
    left.scatter(table["expected_area"], table["model_area"], s=3, alpha=0.3, color="tab:blue")
    lim = (0.5, float(max(table["expected_area"].max(), table["model_area"].max())) * 1.2)
    left.plot(lim, lim, color="k", lw=0.8, ls="--", label="plain downscaling (no enlargement)")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("native area scaled to 128x128 (px)")
    left.set_ylabel("model-space area (px)")
    left.legend(fontsize=8)
    groups = [(str(b), g.to_numpy()) for b, g in table.groupby("size_bin", observed=True)["ratio"]]
    right.boxplot([g for _, g in groups], showfliers=False)
    tops = []
    for i, (_, g) in enumerate(groups, start=1):
        q1, q3 = np.percentile(g, [25, 75])
        tops.append(float(g[g <= q3 + 1.5 * (q3 - q1)].max()))  # matplotlib's default upper whisker
        right.text(i, tops[-1] + 0.05, f"{np.median(g):.2f}x", ha="center", va="bottom", fontsize=7)
    right.set_ylim(top=max(tops) + 0.35)  # room for the highest label
    right.set_xticks(range(1, len(groups) + 1), [b for b, _ in groups], fontsize=7, rotation=30)
    right.axhline(1, color="k", lw=0.8, ls="--")
    right.set_xlabel("native lesion area (native px)")
    right.set_ylabel("enlargement factor")
    right.set_title("box = middle 50% of lesions, line = median, whiskers = typical range", fontsize=8)
    return save(fig, path, title)


def brain_height_figure(table: pd.DataFrame, real_lesion_px: pd.Series, synthetic_px: pd.Series, path: Path,
                        title: str) -> Path:
    """Left: brain area of real slices by relative slice position (median and interquartile band,
    from `brain_by_height`), with the synthetic median as a red line: where it crosses the curve,
    real brains are as small as a typical synthetic one. Right: brain-area distributions of real
    lesion slices and synthetic samples, on the same vertical scale."""
    fig = Figure(figsize=(10, 4))
    left, right = fig.subplots(1, 2, width_ratios=[2, 1], sharey=True)
    left.plot(table.index, table["median"], color="tab:blue", label="real slices: median brain area")
    left.fill_between(table.index, table["p25"], table["p75"], color="tab:blue", alpha=0.2,
                      label="real slices: middle 50%")
    left.axhline(synthetic_px.median(), color="tab:red", lw=1.2,
                 label=f"synthetic brains: median ({synthetic_px.median():.0f} px)")
    left.set_xlabel("relative slice position in the head (0 = first, 1 = last slice with brain)")
    left.set_ylabel("brain area (px)")
    left.legend(fontsize=7, loc="upper left")
    bins = np.linspace(0, float(max(real_lesion_px.max(), synthetic_px.max())), 41)
    for values, color, label in ((real_lesion_px, "tab:blue", "real lesion slices"),
                                 (synthetic_px, "tab:red", "synthetic samples")):
        share = np.histogram(values, bins=bins)[0] / len(values)
        right.stairs(share, bins, orientation="horizontal", color=color, lw=1.5,
                     label=f"{label} ({len(values)}; median {values.median():.0f} px)")
    right.set_xlabel("share of slices / samples")
    right.set_title("brain area distribution", fontsize=8)
    right.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))  # below: the curves fill the panel
    return save(fig, path, title)
