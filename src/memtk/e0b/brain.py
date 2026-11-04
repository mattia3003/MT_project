"""E0b brain stages: brain masks of real and synthetic MRIs, and the failed-MRI rule.

- `run_brain_review` (decision A): compares candidate brain-mask methods on T and a sample of
  S, checks that `brain.low_threshold` sits in a gap between background and tissue, and shows
  the near-blank synthetic MRIs for the failed-MRI cut-off. Writes `results/<dataset>/e0b/
  brain_review/`.
- `run_brain` (needs decision A): brain mask, box and tissue area of every real lesion slice and
  every synthetic sample, the failed-MRI flag, and the synthetic masks as saved. Writes
  `brain.csv` and `brain_masks.npz` per dataset and pool, `masks_raw.npz` per pool, and
  `results/<dataset|pool>/e0b/brain/`.

Below the two stages: the per-slice measurements, tables and figures they use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm import tqdm

from memtk import config, ids
from memtk.e0b.common import BOX_COLUMNS
from memtk.io import MaskStackWriter, load_real, read_table
from memtk.lesions import BRAIN_METHODS, brain_box, brain_mask, fraction_outside, grow, label_components, tissue_px
from memtk.plotting import (BRAIN_COLOR, LESION_COLOR, draw_box, hide_unused, save, show_slice,
                            tile_figure)
from memtk.stages import decided, pick, refuse_overwrite
from memtk.synthetic import load_sample

# Multiples of brain.low_threshold at which the brain area is recomputed (threshold sensitivity)
THRESHOLD_FACTORS = (0.25, 0.5, 1.0, 2.0, 3.0)
# Candidate failed-MRI cut-offs (tissue px) shown in the review
TISSUE_CUTS = (20, 50, 100, 150, 200, 300, 400)


# --- stages ----------------------------------------------------------------------------------

def run_brain_review(cfg: dict, dataset: str | None = None, pool: str | None = None, n_synthetic: int = 2000,
                     min_area: int = 4, overwrite: bool = False) -> None:
    """Evidence for decision A (`brain.method`, `brain.min_tissue_px`), using T only on the real
    side: every T lesion slice and `n_synthetic` random synthetic samples under each candidate
    method; the brain area at multiples of `brain.low_threshold`; the tissue area of every T
    lesion slice and every synthetic sample. `min_area` is a provisional lesion size (decision B
    comes later). Writes `results/<dataset>/e0b/brain_review/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    real_dir = config.output_dir(cfg, name)
    out = config.results_dir(cfg, name, "e0b") / "brain_review"
    refuse_overwrite([out], overwrite)
    rng = np.random.default_rng(cfg["seed"])
    pool_dir = config.resolve_path(pool_cfg["dir"])
    low = float(decided(cfg, "brain.low_threshold", "e0b_audit.py brain-review"))

    slices = read_table(real_dir / "lesion_slices.csv")
    t_slices = slices[slices["set"] == "T"]  # decisions use T only; H-cal/H-eval stay untouched
    index = read_table(config.output_dir(cfg, pool_name) / "synthetic_index.csv")
    index = index[index["read_ok"]]

    # every T lesion slice, and a random sample of S, under each candidate method and at multiples
    # of the low threshold; tissue area of every T lesion slice and every synthetic sample (one
    # fixed measure, for the failed-MRI cut)
    rows, sens, t_tissue = [], [], []
    for pid, group in tqdm(t_slices.groupby("patient_id"), desc="T lesion slices"):
        bravo, seg = load_real(real_dir, str(pid))
        for z in group["z"]:
            key = ids.slice_key(str(pid), int(z))
            t_tissue.append({"key": key, "pid": pid, "z": int(z), "tissue_px": tissue_px(bravo[:, :, z], low)})
            for r in evaluate_brain_methods(bravo[:, :, z], seg[:, :, z], min_area, low):
                rows.append({"group": "T", "key": key, "pid": pid, "z": int(z), **r})
            sens.append({"group": "T", **threshold_sensitivity(bravo[:, :, z], seg[:, :, z], low)})
    sampled = set(rng.choice(len(index), size=min(n_synthetic, len(index)), replace=False).tolist())
    s_tissue = []
    for i, (sid, fname) in enumerate(tqdm(list(zip(index["sample_id"], index["filename"])), desc="S pool")):
        image, mask = load_sample(pool_dir / fname, pool_cfg)
        s_tissue.append(tissue_px(image, low))
        if i in sampled:
            for r in evaluate_brain_methods(image, mask, min_area, low):
                rows.append({"group": "S", "key": sid, "filename": fname, **r})
            sens.append({"group": "S", **threshold_sensitivity(image, mask, low)})
    per_slice = pd.DataFrame(rows)
    t_tissue_df = pd.DataFrame(t_tissue)
    index = index.assign(tissue_px=s_tissue)
    summary = summarize_brain_methods(per_slice)
    method, reason = propose_brain_method(summary)
    area_change, levels = summarize_threshold_sensitivity(pd.DataFrame(sens), low)

    # example figures: slices where plain Otsu fails, small synthetic brains, and random ones
    otsu = per_slice[per_slice["method"] == "otsu"]
    t_otsu, s_otsu = otsu[otsu["group"] == "T"], otsu[otsu["group"] == "S"]
    small_brain_px = float(t_otsu["brain_px"].quantile(0.05))
    t_picks = pd.concat([pick(rng, t_otsu[t_otsu["n_over_half_outside"] > 0], 4), pick(rng, t_otsu, 2)])
    real_examples = []
    for pid, z, key in zip(t_picks["pid"], t_picks["z"], t_picks["key"]):
        bravo, seg = load_real(real_dir, str(pid))
        real_examples.append((str(key), bravo[:, :, int(z)], seg[:, :, int(z)]))
    s_picks = pd.concat([pick(rng, s_otsu[s_otsu["brain_px"] < small_brain_px], 3),
                         pick(rng, s_otsu[s_otsu["n_over_half_outside"] > 0], 2), pick(rng, s_otsu, 2)])
    synth_examples = [(str(sid), *load_sample(pool_dir / fname, pool_cfg))
                      for sid, fname in zip(s_picks["key"], s_picks["filename"])]
    method_grid(real_examples, out / "methods_real.png",
                "T: first 4 rows have a lesion more than half outside the plain Otsu brain; "
                "cyan = brain, red = lesion", low)
    method_grid(synth_examples, out / f"methods_synthetic_{pool_name}.png",
                f"S: rows 1-3 small brains (< T 5th percentile, {small_brain_px:.0f} px), "
                f"4-5 a lesion outside, 6-7 random", low)

    # near-blank synthetic MRIs: the least tissue, then samples just above candidate cuts, then
    # the T lesion slices with the least tissue (small real slices exist and must not fail)
    by_tissue = index.sort_values("tissue_px", kind="stable")
    parts = [by_tissue.iloc[:32]] + [by_tissue[by_tissue["tissue_px"] >= c].iloc[:4] for c in (50, 100, 200, 400)]
    items = [(str(sid), f"tissue {t} px, max {m:.2f}", *load_sample(pool_dir / f, pool_cfg))
             for part in parts
             for sid, t, m, f in zip(part["sample_id"], part["tissue_px"], part["img_max"], part["filename"])]
    for pid, z, key, t in t_tissue_df.nsmallest(8, "tissue_px")[["pid", "z", "key", "tissue_px"]].itertuples(index=False):
        bravo, seg = load_real(real_dir, str(pid))
        items.append((f"T {key}", f"tissue {t} px", bravo[:, :, int(z)], seg[:, :, int(z)]))
    sorted_sheet(items, out / f"near_blank_{pool_name}.png",
                 "Synthetic MRIs by brain tissue: rows 1-4 the 32 lowest; rows 5-6: 4 each just above "
                 "50, 100, 200, 400 px; row 7: the 8 T lesion slices with the least tissue")
    tissue_histogram(t_tissue_df["tissue_px"], index["tissue_px"], out / f"tissue_hist_{pool_name}.png",
                     "Brain tissue per slice: all synthetic samples vs T lesion slices")
    counts = tissue_counts(t_tissue_df["tissue_px"], index["tissue_px"])

    summary.round(4).to_csv(out / f"brain_methods_{pool_name}.csv")
    counts.to_csv(out / f"tissue_counts_{pool_name}.csv", index=False)
    area_change.round(4).to_csv(out / f"threshold_sensitivity_{pool_name}.csv")
    text = (f"E0b brain-review: dataset '{name}' (T only), pool '{pool_name}' "
            f"({len(sampled)} random samples for the methods), seed {cfg['seed']}\n"
            f"Lesions = components >= {min_area} px (provisional; decision B comes later)\n"
            f"brain.low_threshold = {low} (used by the low_threshold methods and the tissue area)\n\n"
            f"Does brain.low_threshold sit in an empty gap between background and tissue?\n"
            f"Brain area (low_threshold_all) at multiples of the threshold, change relative to {low} "
            f"(slices with a brain >= 500 px):\n{area_change.round(4).to_string()}\n"
            f"Background noise (99.9th percentile, > 3 px from the brain and > 1 px from the mask), for the "
            f"median slice and a near-worst slice (99th percentile):\n{levels.round(4).to_string()}\n"
            f"Read: area changes of a few percent and background levels well below the threshold mean it "
            f"separates background from tissue. Large changes, or background near the threshold, mean it "
            f"must be revisited for this data (set brain.low_threshold).\n\n"
            f"Candidate brain-mask methods:\n{summary.round(4).to_string()}\n\n"
            f"Proposed brain.method: {method}\n  rule: {reason}\n\n"
            f"Failed-MRI cut-off: samples and T lesion slices below each tissue cut (all {len(index)} samples)\n"
            f"{counts.to_string(index=False)}\n"
            f"Choose brain.min_tissue_px from near_blank_{pool_name}.png: the highest cut below which every "
            f"synthetic image is near-blank, staying clear of the smallest real slices.\n\n"
            f"Then set both values in configs/default.yaml (dated comment) and run the 'brain' stage.\n")
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "brain-review", "pool": pool_name,
                                                            "min_area": min_area, "low_threshold": low,
                                                            "n_synthetic": len(sampled)})
    print(f"\n{text}\nWrote {out}")


def run_brain(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Brain mask, box and tissue area with the decided method (decision A) for every real lesion
    slice (T, H-cal, H-eval) and every readable synthetic sample, the failed-MRI flag, and the
    synthetic masks as saved (read from the pool once, cached for later stages). Writes
    `brain.csv` + `brain_masks.npz` per dataset and pool, `masks_raw.npz` per pool, and a summary
    with outline sheets to `results/<dataset|pool>/e0b/brain/`."""
    method = decided(cfg, "brain.method", "e0b_audit.py brain-review")
    min_tissue = int(decided(cfg, "brain.min_tissue_px", "e0b_audit.py brain-review"))
    low = float(decided(cfg, "brain.low_threshold", "e0b_audit.py brain-review"))
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    real_res = config.results_dir(cfg, name, "e0b") / "brain"
    syn_res = config.results_dir(cfg, pool_name, "e0b") / "brain"
    refuse_overwrite([real_dir / "brain.csv", syn_dir / "brain.csv", real_res, syn_res], overwrite)
    rng = np.random.default_rng(cfg["seed"])
    shape = cfg["preprocessing"]["spatial_size"]
    pool_dir = config.resolve_path(pool_cfg["dir"])
    extra = {"stage": "brain", "dataset": name, "pool": pool_name, "method": method, "low_threshold": low,
             "min_tissue_px": min_tissue}

    # real: every lesion slice of every set (T, H-cal, H-eval)
    slices = read_table(real_dir / "lesion_slices.csv")
    slices = slices[slices["lesion_px_model"] > 0]
    real_brains, rows = MaskStackWriter(shape), []
    for pid, group in tqdm(slices.groupby("patient_id"), desc=f"{name} lesion slices"):
        bravo, _ = load_real(real_dir, str(pid))
        for z, set_name in zip(group["z"], group["set"]):
            key = ids.slice_key(str(pid), int(z))
            brain, row = brain_row(bravo[:, :, int(z)], method, low)
            real_brains.add(key, brain)
            rows.append({"key": key, "patient_id": pid, "set": set_name, "z": int(z), **row})
    real = pd.DataFrame(rows)

    # synthetic: every readable sample; the mask as saved is cached for later stages
    index = read_table(syn_dir / "synthetic_index.csv")
    n_unreadable = int((~index["read_ok"]).sum())
    index = index[index["read_ok"]]
    syn_brains, raw_masks, rows = MaskStackWriter(shape), MaskStackWriter(shape), []
    for sid, fname in tqdm(list(zip(index["sample_id"], index["filename"])), desc=pool_name):
        image, mask = load_sample(pool_dir / fname, pool_cfg)
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError(f"Mask of {sid} is not binary; masks_raw.npz stores masks as 0/1")
        brain, row = brain_row(image, method, low)
        syn_brains.add(ids.sample_key(str(sid)), brain)
        raw_masks.add(ids.sample_key(str(sid)), mask > 0)
        rows.append({"sample_id": sid, **row, "mri_failed": row["tissue_px"] < min_tissue})
    syn = pd.DataFrame(rows)

    real.to_csv(real_dir / "brain.csv", index=False)
    real_brains.save(real_dir / "brain_masks.npz")
    syn.to_csv(syn_dir / "brain.csv", index=False)
    syn_brains.save(syn_dir / "brain_masks.npz")
    raw_masks.save(syn_dir / "masks_raw.npz")
    for d in (real_dir, syn_dir):
        config.write_run_info(d, cfg, script=__file__, extra=extra, filename="brain.run_info.json")

    # outline sheets: random slices per set; synthetic random, smallest brains and failed MRIs
    real_picks = pd.concat([pick(rng, real[real["set"] == s], n) for s, n in (("T", 16), ("H-cal", 8), ("H-eval", 8))])
    items = []
    for key, pid, z, px, r0, r1, c0, c1 in real_picks[["key", "patient_id", "z", "brain_px", *BOX_COLUMNS]].itertuples(
            index=False):
        bravo, seg = load_real(real_dir, str(pid))
        items.append((f"{key}\nbrain {px} px", bravo[:, :, int(z)], seg[:, :, int(z)],
                      brain_mask(bravo[:, :, int(z)], method, low), (r0, r1, c0, c1) if r0 >= 0 else None))
    brain_sheet(items, real_res / "outlines.png",
                f"{name}, brain method {method}: rows 1-2 T, row 3 H-cal, row 4 H-eval (random lesion slices); "
                f"cyan = brain, dashed yellow = brain box, red = lesion")
    ok, failed = syn[~syn["mri_failed"]], syn[syn["mri_failed"]]
    syn_picks = pd.concat([pick(rng, ok, 16), ok.nsmallest(8, "brain_px"), pick(rng, failed, 8)])
    files = dict(zip(index["sample_id"], index["filename"]))
    items = []
    for sid, px, r0, r1, c0, c1 in syn_picks[["sample_id", "brain_px", *BOX_COLUMNS]].itertuples(index=False):
        image, mask = load_sample(pool_dir / files[sid], pool_cfg)
        items.append((f"{sid}\nbrain {px} px", image, mask, brain_mask(image, method, low),
                      (r0, r1, c0, c1) if r0 >= 0 else None))
    brain_sheet(items, syn_res / "outlines.png",
                f"{pool_name}, brain method {method}: rows 1-2 random, row 3 smallest brains (not failed), "
                f"row 4 failed MRIs (tissue < {min_tissue} px)")

    real_summary = summarize_brains(real, "set")
    syn_summary = summarize_brains(syn.assign(group=np.where(syn["mri_failed"], "failed", "ok")), "group")
    real_text = (f"E0b brain: dataset '{name}', brain method {method}\n\n"
                 f"Lesion slices per set ({real['patient_id'].nunique()} patients):\n"
                 f"{real_summary.round(1).to_string()}\n\n"
                 f"Lesion slices with tissue < min_tissue_px ({min_tissue}): {int((real['tissue_px'] < min_tissue).sum())} "
                 f"(the rule applies to synthetic MRIs only; real slices should not fall under it)\n")
    syn_text = (f"E0b brain: pool '{pool_name}', brain method {method}, min_tissue_px {min_tissue}\n\n"
                f"Samples: {len(syn)} read, {n_unreadable} unreadable (skipped)\n"
                f"Failed MRIs (tissue < {min_tissue} px): {len(failed)} ({len(failed) / len(syn):.2%})\n\n"
                f"{syn_summary.round(1).to_string()}\n")
    for res, text in ((real_res, real_text), (syn_res, syn_text)):
        res.mkdir(parents=True, exist_ok=True)
        (res / "summary.txt").write_text(text, encoding="utf-8")
        config.write_run_info(res, cfg, script=__file__, extra=extra)
    print(f"\n{real_text}\n{syn_text}\nWrote {real_dir}, {syn_dir}, {real_res}, {syn_res}")


# --- per-slice measurements ------------------------------------------------------------------

def brain_row(image: np.ndarray, method: str, low_threshold: float) -> tuple[np.ndarray, dict]:
    """Brain mask of one slice and its brain.csv columns: brain area, tissue area and brain box."""
    brain = brain_mask(image, method, low_threshold)
    box = brain_box(brain)
    return brain, {"brain_px": int(brain.sum()), "tissue_px": tissue_px(image, low_threshold),
                   **dict(zip(BOX_COLUMNS, box if box is not None else (-1, -1, -1, -1)))}


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


# --- tables ----------------------------------------------------------------------------------

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


def tissue_counts(t_tissue: pd.Series, s_tissue: pd.Series, cuts=TISSUE_CUTS) -> pd.DataFrame:
    """For each candidate cut on the tissue area: synthetic samples that would count as failed,
    and real T lesion slices below it (which must stay rare: small real slices exist)."""
    return pd.DataFrame({"cut": cuts,
                         "S_below": [int((s_tissue < c).sum()) for c in cuts],
                         "S_share_below": [float((s_tissue < c).mean()) for c in cuts],
                         "T_slices_below": [int((t_tissue < c).sum()) for c in cuts]})


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


# --- figures ---------------------------------------------------------------------------------

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


def sorted_sheet(items: list[tuple[str, str, np.ndarray, np.ndarray]], path: Path, title: str,
                 n_cols: int = 8) -> Path:
    """Tiles in the given order, each (key, label, image, mask), with the mask outlined in red.
    Images are shown in [0, 1] like every other figure, so a near-blank image looks dark."""
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = tile_figure(n_rows, n_cols, tile=1.9)
    for ax, (key, label, image, mask) in zip(axes.ravel(), items):
        show_slice(ax, image, [(mask > 0, LESION_COLOR)], f"{key}\n{label}")
    hide_unused(axes, len(items))
    return save(fig, path, title)


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


def brain_sheet(items: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple | None]], path: Path,
                title: str, n_cols: int = 8) -> Path:
    """Tiles of (title, image, mask, brain, box): brain outline (cyan), brain box (dashed yellow),
    mask (red). The box is inclusive (r0, r1, c0, c1), or None to draw no box."""
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = tile_figure(n_rows, n_cols, tile=1.9)
    for ax, (tile_title, image, mask, brain, box) in zip(axes.ravel(), items):
        show_slice(ax, image, [(brain, BRAIN_COLOR), (mask > 0, LESION_COLOR)], tile_title)
        draw_box(ax, box)
    hide_unused(axes, len(items))
    return save(fig, path, title)
