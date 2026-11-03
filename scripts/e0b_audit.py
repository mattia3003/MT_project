"""E0b: data audit and mask quality control.

Review stages only produce evidence and a proposed config value. Production stages apply your
decision and refuse to run while the value they need is still null in the config.

Stages, run in this order. Each one can be run and checked on its own:
  brain-review   evidence for decision A (brain-mask method, failed-MRI tissue cut-off), using T,
                 a random sample of S for the methods and all of S for tissue area
                 -> results/<dataset>/e0b/brain_review/
  brain          (needs decision A) brain mask, box and tissue area of every real lesion slice
                 (T, H-cal, H-eval) and every synthetic sample, the failed-MRI flag, and the
                 synthetic masks as saved
                 -> <output_root>/<dataset>/brain.csv + brain_masks.npz,
                    <output_root>/<pool>/brain.csv + brain_masks.npz + masks_raw.npz,
                    results/<dataset|pool>/e0b/brain/ (summary and outline sheets)
  lesions-review evidence for decisions B and C (minimum lesion area, attach gap): component
                 sizes of all real lesion slices and synthetic masks, attach-or-drop counts for
                 nearby settings, specks by gap; proposal from T only
                 -> results/<dataset>/e0b/lesions_review/
  lesions        (needs decisions B, C) attach-or-drop of every real lesion slice and every
                 synthetic mask as saved: one row per lesion (ID, features in three position
                 frames), one row per slice, label maps; a label volume per real patient
                 -> <output_root>/<dataset|pool>/lesions.csv + slices.csv + labels.npz,
                    <output_root>/<dataset>/volumes/<pid>/labels.nii.gz,
                    results/<dataset|pool>/e0b/lesions/summary.txt + overview.png
  view           (helper, after lesions) one slice at full size with its lesion rows; optionally
                 a synthetic sample's image and labels as NIfTI for ITK-SNAP
                 -> results/<dataset|pool>/e0b/views/<key>.png
  qc-review      evidence for decision D (isolated-speck cut-off between minor and noisy masks):
                 masks by number of isolated specks, sheets per stratum, clean masks with a
                 lesion outside the brain -> results/<pool>/e0b/qc_review/
  qc             (needs decision D) one row per synthetic sample: mask class (from the mask alone),
                 failed-MRI flag, combined QC class, edge and small-lesion flags; summary, old QC
                 list cross-table, the edge check on the real sets, example sheet
                 -> <output_root>/<pool>/qc.csv, results/<pool>/e0b/qc/
  stats          (after qc) E0 description of real sets and the synthetic main audit: description
                 table, lesions per slice, lesion area, centroid density in pixels and brain-
                 relative, impact of candidate informative-lesion bounds (input for E3)
                 -> results/<dataset>/e0b/stats/
  (planned: filters-review, check)

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py brain-review
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py view --sample file_2_1033
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py view --patient Mets_120 --z 40
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from memtk import audit, config, ids, qc
from memtk.io import LabelStack, LabelStackWriter, MaskStack, MaskStackWriter, load_image, read_table, save_nifti
from memtk.lesions import (LesionLabels, attach_or_drop, brain_box, brain_mask, fraction_outside, label_components,
                           lesion_features, tissue_px)
from memtk.synthetic import load_sample

BOX_COLUMNS = ["box_r0", "box_r1", "box_c0", "box_c1"]


def load_real(real_dir: Path, pid: str) -> tuple[np.ndarray, np.ndarray]:
    """(bravo, seg) model-space volumes of one patient, shape (128, 128, D)."""
    vol_dir = real_dir / "volumes" / pid
    return (np.asanyarray(load_image(vol_dir / "bravo.nii.gz").dataobj),
            np.asanyarray(load_image(vol_dir / "seg.nii.gz").dataobj))


def decided(cfg: dict, section: str, key: str, review_stage: str):
    """A decision value from the config; stops with instructions while it is still null."""
    value = cfg[section][key]
    if value is None:
        raise SystemExit(f"{section}.{key} is not set in {cfg['_config_path']}. Run "
                         f"'e0b_audit.py {review_stage}', view its evidence and set the value first.")
    return value


def run_brain_review(cfg: dict, args: argparse.Namespace) -> None:
    name, _ = config.real_dataset(cfg, args.dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    real_dir = config.output_dir(cfg, name)
    out = config.results_dir(cfg, name, "e0b") / "brain_review"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to regenerate.")
    rng = np.random.default_rng(cfg["seed"])
    pool_dir = config.resolve_path(pool_cfg["dir"])
    low = float(decided(cfg, "brain", "low_threshold", "brain-review"))

    slices = read_table(real_dir / "lesion_slices.csv")
    t_slices = slices[slices["set"] == "T"]  # decisions use T only; H-cal/H-eval stay untouched
    index = read_table(config.output_dir(cfg, pool_name) / "synthetic_index.csv")
    index = index[index["read_ok"]]

    def load_t(pid: str):
        return load_real(real_dir, pid)

    # every T lesion slice, and a random sample of S, under each candidate method and at multiples
    # of the low threshold; tissue area of every T lesion slice and every synthetic sample (one
    # fixed measure, for the failed-MRI cut)
    rows, sens, t_tissue = [], [], []
    for pid, group in tqdm(t_slices.groupby("patient_id"), desc="T lesion slices"):
        bravo, seg = load_t(str(pid))
        for z in group["z"]:
            key = ids.slice_key(str(pid), int(z))
            t_tissue.append({"key": key, "pid": pid, "z": int(z), "tissue_px": tissue_px(bravo[:, :, z], low)})
            for r in audit.evaluate_brain_methods(bravo[:, :, z], seg[:, :, z], args.min_area, low):
                rows.append({"group": "T", "key": key, "pid": pid, "z": int(z), **r})
            sens.append({"group": "T", **audit.threshold_sensitivity(bravo[:, :, z], seg[:, :, z], low)})
    sampled = set(rng.choice(len(index), size=min(args.n_synthetic, len(index)), replace=False).tolist())
    s_tissue = []
    for i, (sid, fname) in enumerate(tqdm(list(zip(index["sample_id"], index["filename"])), desc="S pool")):
        image, mask = load_sample(pool_dir / fname, pool_cfg)
        s_tissue.append(tissue_px(image, low))
        if i in sampled:
            for r in audit.evaluate_brain_methods(image, mask, args.min_area, low):
                rows.append({"group": "S", "key": sid, "filename": fname, **r})
            sens.append({"group": "S", **audit.threshold_sensitivity(image, mask, low)})
    per_slice = pd.DataFrame(rows)
    t_tissue_df = pd.DataFrame(t_tissue)
    index = index.assign(tissue_px=s_tissue)
    summary = audit.summarize_brain_methods(per_slice)
    method, reason = audit.propose_brain_method(summary)
    area_change, levels = audit.summarize_threshold_sensitivity(pd.DataFrame(sens), low)

    # example figures: slices where plain Otsu fails, small synthetic brains, and random ones
    otsu = per_slice[per_slice["method"] == "otsu"]
    t_otsu, s_otsu = otsu[otsu["group"] == "T"], otsu[otsu["group"] == "S"]
    small_brain_px = float(t_otsu["brain_px"].quantile(0.05))

    def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[rng.choice(len(df), size=min(n, len(df)), replace=False)] if len(df) else df

    t_picks = pd.concat([pick(t_otsu[t_otsu["n_over_half_outside"] > 0], 4), pick(t_otsu, 2)])
    real_examples = []
    for pid, z, key in zip(t_picks["pid"], t_picks["z"], t_picks["key"]):
        bravo, seg = load_t(str(pid))
        real_examples.append((str(key), bravo[:, :, int(z)], seg[:, :, int(z)]))
    s_picks = pd.concat([pick(s_otsu[s_otsu["brain_px"] < small_brain_px], 3),
                         pick(s_otsu[s_otsu["n_over_half_outside"] > 0], 2), pick(s_otsu, 2)])
    synth_examples = [(str(sid), *load_sample(pool_dir / fname, pool_cfg))
                      for sid, fname in zip(s_picks["key"], s_picks["filename"])]

    audit.method_grid(real_examples, out / "methods_real.png",
                      "T: first 4 rows have a lesion more than half outside the plain Otsu brain; "
                      "cyan = brain, red = lesion", low)
    audit.method_grid(synth_examples, out / f"methods_synthetic_{pool_name}.png",
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
        bravo, seg = load_t(str(pid))
        items.append((f"T {key}", f"tissue {t} px", bravo[:, :, int(z)], seg[:, :, int(z)]))
    audit.sorted_sheet(items, out / f"near_blank_{pool_name}.png",
                       "Synthetic MRIs by brain tissue: rows 1-4 the 32 lowest; rows 5-6: 4 each just above "
                       "50, 100, 200, 400 px; row 7: the 8 T lesion slices with the least tissue")
    audit.tissue_histogram(t_tissue_df["tissue_px"], index["tissue_px"], out / f"tissue_hist_{pool_name}.png",
                           "Brain tissue per slice: all synthetic samples vs T lesion slices")
    counts = audit.tissue_counts(t_tissue_df["tissue_px"], index["tissue_px"])

    summary.round(4).to_csv(out / f"brain_methods_{pool_name}.csv")
    counts.to_csv(out / f"tissue_counts_{pool_name}.csv", index=False)
    area_change.round(4).to_csv(out / f"threshold_sensitivity_{pool_name}.csv")
    text = (f"E0b brain-review: dataset '{name}' (T only), pool '{pool_name}' "
            f"({len(sampled)} random samples for the methods), seed {cfg['seed']}\n"
            f"Lesions = components >= {args.min_area} px (provisional; decision B comes later)\n"
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
                                                            "min_area": args.min_area, "low_threshold": low,
                                                            "n_synthetic": len(sampled)})
    print(f"\n{text}\nWrote {out}")


def brain_of(image: np.ndarray, method: str, low: float) -> tuple[np.ndarray, tuple | None, dict]:
    """Brain mask, box and the brain.csv columns of one slice."""
    brain = brain_mask(image, method, low)
    box = brain_box(brain)
    return brain, box, {"brain_px": int(brain.sum()), "tissue_px": tissue_px(image, low),
                        **dict(zip(BOX_COLUMNS, box if box is not None else (-1, -1, -1, -1)))}


def run_brain(cfg: dict, args: argparse.Namespace) -> None:
    method = decided(cfg, "brain", "method", "brain-review")
    min_tissue = int(decided(cfg, "brain", "min_tissue_px", "brain-review"))
    low = float(decided(cfg, "brain", "low_threshold", "brain-review"))
    name, _ = config.real_dataset(cfg, args.dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    real_res = config.results_dir(cfg, name, "e0b") / "brain"
    syn_res = config.results_dir(cfg, pool_name, "e0b") / "brain"
    existing = [p for p in (real_dir / "brain.csv", syn_dir / "brain.csv", real_res, syn_res) if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{existing[0]} already exists; pass --overwrite to regenerate.")
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
            brain, _, row = brain_of(bravo[:, :, int(z)], method, low)
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
        brain, _, row = brain_of(image, method, low)
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
    def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))] if len(df) else df

    real_picks = pd.concat([pick(real[real["set"] == s], n) for s, n in (("T", 16), ("H-cal", 8), ("H-eval", 8))])
    items = []
    for key, pid, z, px, r0, r1, c0, c1 in real_picks[["key", "patient_id", "z", "brain_px", *BOX_COLUMNS]].itertuples(
            index=False):
        bravo, seg = load_real(real_dir, str(pid))
        items.append((f"{key}\nbrain {px} px", bravo[:, :, int(z)], seg[:, :, int(z)],
                      brain_mask(bravo[:, :, int(z)], method, low), (r0, r1, c0, c1) if r0 >= 0 else None))
    audit.brain_sheet(items, real_res / "outlines.png",
                      f"{name}, brain method {method}: rows 1-2 T, row 3 H-cal, row 4 H-eval (random lesion slices); "
                      f"cyan = brain, dashed yellow = brain box, red = lesion")
    ok, failed = syn[~syn["mri_failed"]], syn[syn["mri_failed"]]
    syn_picks = pd.concat([pick(ok, 16), ok.nsmallest(8, "brain_px"), pick(failed, 8)])
    files = dict(zip(index["sample_id"], index["filename"]))
    items = []
    for sid, px, r0, r1, c0, c1 in syn_picks[["sample_id", "brain_px", *BOX_COLUMNS]].itertuples(index=False):
        image, mask = load_sample(pool_dir / files[sid], pool_cfg)
        items.append((f"{sid}\nbrain {px} px", image, mask, brain_mask(image, method, low),
                      (r0, r1, c0, c1) if r0 >= 0 else None))
    audit.brain_sheet(items, syn_res / "outlines.png",
                      f"{pool_name}, brain method {method}: rows 1-2 random, row 3 smallest brains (not failed), "
                      f"row 4 failed MRIs (tissue < {min_tissue} px)")

    real_summary = audit.summarize_brains(real, "set")
    syn_summary = audit.summarize_brains(syn.assign(group=np.where(syn["mri_failed"], "failed", "ok")), "group")
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


def run_lesions_review(cfg: dict, args: argparse.Namespace) -> None:
    name, _ = config.real_dataset(cfg, args.dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, name, "e0b") / "lesions_review"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to regenerate.")
    if not (syn_dir / "masks_raw.npz").exists():
        raise SystemExit(f"{syn_dir / 'masks_raw.npz'} not found; run the 'brain' stage first.")
    rng = np.random.default_rng(cfg["seed"])
    pool_dir = config.resolve_path(pool_cfg["dir"])

    # every component of every real lesion slice (all sets) and every synthetic mask as saved
    slices = read_table(real_dir / "lesion_slices.csv")
    slices = slices[slices["lesion_px_model"] > 0]
    masks: dict[str, tuple[str, np.ndarray]] = {}
    for pid, group in tqdm(slices.groupby("patient_id"), desc=f"{name} lesion slices"):
        _, seg = load_real(real_dir, str(pid))
        for z, set_name in zip(group["z"], group["set"]):
            masks[ids.slice_key(str(pid), int(z))] = (str(set_name), seg[:, :, int(z)] > 0)
    raw = MaskStack(syn_dir / "masks_raw.npz")
    rows = [{"group": g, "key": k, "size": s, "row": r, "col": c}
            for k, (g, m) in masks.items() for s, r, c in audit.component_table(m)]
    for sid in tqdm(raw.keys, desc=f"{pool_name} masks"):
        rows += [{"group": "S", "key": sid, "size": s, "row": r, "col": c} for s, r, c in audit.component_table(raw[sid])]
    components = pd.DataFrame(rows)

    # decision B from T only; H-cal, H-eval and S are described with the same settings
    t_sizes = components.loc[components["group"] == "T", "size"]
    m, m_reason = audit.propose_min_area(t_sizes)
    min_areas = [a for a in (m - 1, m, m + 1) if a >= 1]
    gaps = (0, 1, 2)

    # specks per minimum area, with their gap to the nearest lesion; masks without a component
    # below the area have no specks and are skipped
    smallest = components.groupby("key", sort=False)["size"].min()
    speck_rows = []
    for a in min_areas:
        for key in tqdm(smallest.index[smallest < a], desc=f"specks at min area {a}"):
            group, mask = masks[key] if key in masks else ("S", raw[key])
            speck_rows += audit.mask_specks(group, key, mask, a)
    specks = pd.DataFrame(speck_rows, columns=["group", "key", "min_area", "size", "gap", "row", "col"])
    sizes_table = audit.size_counts(components)
    sensitivity = audit.summarize_attach_or_drop(components, specks, min_areas, gaps)
    gap_table = audit.gap_counts(specks[specks["min_area"] == m])

    # figures: size distribution; the smallest real components; specks at gap 1-3 from a lesion
    def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))] if len(df) else df

    index = read_table(syn_dir / "synthetic_index.csv")
    files = dict(zip(index["sample_id"], index["filename"]))

    def image_of(group: str, key: str) -> np.ndarray:
        if group == "S":
            return load_sample(pool_dir / files[key], pool_cfg)[0]
        pid, z = ids.parse_slice_key(key)
        return load_real(real_dir, pid)[0][:, :, z]

    def crops(df: pd.DataFrame, label) -> list:
        # pandas stubs type namedtuple fields as any scalar; these columns hold str and int
        return [(label(rec), image_of(rec.group, rec.key),  # pyright: ignore[reportArgumentType]
                 masks[rec.key][1] if rec.key in masks else raw[rec.key],  # pyright: ignore[reportArgumentType]
                 int(rec.row), int(rec.col))  # pyright: ignore[reportArgumentType]
                for rec in df.itertuples(index=False)]

    audit.size_figure(components, m, out / f"component_sizes_{pool_name}.png",
                      f"Component sizes (8-connected) of real lesion slices per set and synthetic masks as saved; "
                      f"dashed: proposed minimum lesion area {m} px")
    t_comp = components[components["group"] == "T"]
    items = []
    for s in range(1, m + 2):
        items += crops(pick(t_comp[t_comp["size"] == s], 8),
                       lambda rec: f"{rec.key}\n{rec.size} px at ({rec.row}, {rec.col})")
    audit.crop_sheet(items, out / "small_components_T.png",
                     f"T: the smallest real components, one row per size 1-{m + 1} px (8 random each); "
                     f"32x32 crops of the 128x128 slice centred on the component (lime); other lesions of the "
                     f"slice inside the crop in red. (row, col) = model-space position of its first pixel")
    at_m = specks[specks["min_area"] == m]
    items = []
    for g in (1, 2, 3):
        items += crops(pick(at_m[(at_m["group"] == "S") & (at_m["gap"] == g)], 8),
                       lambda rec: f"{rec.key}\n{rec.size} px, gap {rec.gap} at ({rec.row}, {rec.col})")
    items += crops(pick(at_m[(at_m["group"] == "T") & at_m["gap"].between(1, 3)], 8),
                   lambda rec: f"T {rec.key}\n{rec.size} px, gap {rec.gap} at ({rec.row}, {rec.col})")
    audit.crop_sheet(items, out / f"specks_by_gap_{pool_name}.png",
                     f"Specks (< {m} px, lime) next to a lesion (red): rows 1-3 synthetic at gap 1, 2, 3 px; "
                     f"row 4 T at gap 1-3 px. 32x32 crops")

    # every small T component, findable in a viewer: model-space and approximate native position
    # (native index ~ (model index + 0.5) * native size / 128 - 0.5; z is the same in both)
    patients = read_table(real_dir / "patients.csv").set_index("patient_id")
    h, w = cfg["preprocessing"]["spatial_size"]
    small = t_comp[t_comp["size"] <= m + 1].sort_values(["size", "key"]).copy()
    pid_z = [ids.parse_slice_key(k) for k in small["key"]]
    small.insert(1, "patient_id", [p for p, _ in pid_z])
    small.insert(2, "z", [z for _, z in pid_z])
    small["gap_to_lesion"] = [audit.gap_to_nearest_lesion(masks[k][1], r, c, m)
                              for k, r, c in zip(small["key"], small["row"], small["col"])]
    dims = patients.loc[small["patient_id"], ["native_dim0", "native_dim1"]].to_numpy()
    small["native_row"] = np.round((small["row"] + 0.5) * dims[:, 0] / h - 0.5).astype(int)
    small["native_col"] = np.round((small["col"] + 0.5) * dims[:, 1] / w - 0.5).astype(int)
    small.drop(columns="group").to_csv(out / "small_components_T.csv", index=False)

    sizes_table.to_csv(out / f"component_sizes_{pool_name}.csv")
    sensitivity.to_csv(out / f"attach_or_drop_{pool_name}.csv")
    gap_table.to_csv(out / f"speck_gaps_{pool_name}.csv")
    q = t_sizes.quantile([0.01, 0.05, 0.5])
    text = (f"E0b lesions-review: dataset '{name}' (decisions from T only), pool '{pool_name}' (masks as saved), "
            f"seed {cfg['seed']}\n\n"
            f"Component sizes (8-connected), all components; per size 1-10 px and larger:\n{sizes_table.to_string()}\n"
            f"T percentiles: 1st {q[0.01]:.1f}, 5th {q[0.05]:.1f}, median {q[0.5]:.1f} px\n\n"
            f"Proposed lesions.min_lesion_area: {m}\n  rule: {m_reason}\n"
            f"  view small_components_T.png: do real components below {m} px look like lesions or like "
            f"annotation fragments? small_components_T.csv lists every T component up to {m + 1} px with "
            f"its model-space and approximate native position, to look it up in a viewer.\n\n"
            f"Attach-or-drop at minimum area {', '.join(map(str, min_areas))} and gap {', '.join(map(str, gaps))} "
            f"(gap 0 = drop-only). Lesion counts depend on the minimum area only; the gap only decides "
            f"whether a speck is attached (stray) or dropped (isolated):\n{sensitivity.to_string()}\n\n"
            f"Specks at minimum area {m} by gap (background px) to the nearest lesion:\n{gap_table.to_string()}\n\n"
            f"lesions.attach_gap: the plan's default is 2. Choose from specks_by_gap_{pool_name}.png: attach a "
            f"gap only if specks at that distance look like part of the lesion, not like separate noise.\n\n"
            f"Then set both values in configs/default.yaml (dated comment) and run the 'lesions' stage.\n")
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "lesions-review", "pool": pool_name,
                                                            "proposed_min_area": m})
    print(f"\n{text}\nWrote {out}")


def boxes_of(brain_csv: pd.DataFrame, key_col: str) -> dict[str, tuple[int, int, int, int] | None]:
    """Brain box per slice key from a brain.csv (None where the slice has no brain)."""
    return {str(k): (int(r0), int(r1), int(c0), int(c1)) if r0 >= 0 else None
            for k, r0, r1, c0, c1 in brain_csv[[key_col, *BOX_COLUMNS]].itertuples(index=False)}


class Slices:
    """Image, mask as saved, brain mask and brain box of any slice of one source, by slice key.
    Real data when `pool_cfg` is None (volumes in `out_dir`), else a synthetic pool."""

    def __init__(self, out_dir: Path, pool_cfg: dict | None = None):
        self.out_dir, self.pool_cfg = out_dir, pool_cfg
        self.brains = MaskStack(out_dir / "brain_masks.npz")
        self.boxes = boxes_of(read_table(out_dir / "brain.csv"), "key" if pool_cfg is None else "sample_id")
        if pool_cfg is not None:
            self.raw = MaskStack(out_dir / "masks_raw.npz")
            index = read_table(out_dir / "synthetic_index.csv")
            self.files = dict(zip(index["sample_id"], index["filename"]))
            self.pool_dir = config.resolve_path(pool_cfg["dir"])

    def __contains__(self, key: str) -> bool:
        return key in self.brains

    def __getitem__(self, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple | None]:
        if self.pool_cfg is None:
            pid, z = ids.parse_slice_key(key)
            bravo, seg = load_real(self.out_dir, pid)
            image, mask = bravo[:, :, z], seg[:, :, z] > 0
        else:
            image, mask = load_sample(self.pool_dir / self.files[key], self.pool_cfg)[0], self.raw[key]
        return image, mask, self.brains[key], self.boxes[key]


class Labeller:
    """Attach-or-drop with the decided settings, collecting the lesion and slice rows and the
    label maps of one source (a dataset or pool). Real and synthetic data take the same path."""

    def __init__(self, source: str, min_area: int, gap: int, small_lesion_px: int, shape):
        self.source, self.min_area, self.gap, self.small_lesion_px = source, min_area, gap, small_lesion_px
        self.lesion_rows: list[dict] = []
        self.slice_rows: list[dict] = []
        self.labels = LabelStackWriter(shape)

    def add(self, key: str, ident: dict, mask: np.ndarray, brain: np.ndarray, box) -> LesionLabels:
        res = attach_or_drop(mask, self.min_area, self.gap)
        features = lesion_features(res, brain, box)
        for f in features:
            self.lesion_rows.append({"source": self.source, "lesion_id": ids.lesion_id(key, f["label"]),
                                     "key": key, **ident, **f})
        specks, n_specks = label_components(res.dropped)
        areas = [f["area"] for f in features]
        self.slice_rows.append({
            "source": self.source, "key": key, **ident,
            "n_components": res.n_lesions + res.n_strays + res.n_specks, "n_lesions": res.n_lesions,
            "lesion_px": int((res.labels > 0).sum()), "largest_lesion_px": max(areas, default=0),
            "has_lesion_lt40": any(a < self.small_lesion_px for a in areas),
            "n_strays": res.n_strays, "stray_px": int(res.attached.sum()),
            "n_specks": res.n_specks, "speck_px": int(res.dropped.sum()),
            "specks_outside_brain": sum(fraction_outside(specks == i, brain) > 0.5 for i in range(1, n_specks + 1)),
        })
        self.labels.add(key, res.labels)
        return res

    def save(self, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
        lesions = pd.DataFrame(self.lesion_rows)
        lesions.insert(1, "lesion_idx", np.arange(len(lesions)))
        slices = pd.DataFrame(self.slice_rows)
        lesions.to_csv(out_dir / "lesions.csv", index=False)
        slices.to_csv(out_dir / "slices.csv", index=False)
        self.labels.save(out_dir / "labels.npz")
        return lesions, slices


def run_lesions(cfg: dict, args: argparse.Namespace) -> None:
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    small_lesion_px = int(cfg["qc"]["small_lesion_px"])
    name, _ = config.real_dataset(cfg, args.dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    real_res = config.results_dir(cfg, name, "e0b") / "lesions"
    syn_res = config.results_dir(cfg, pool_name, "e0b") / "lesions"
    existing = [p for p in (real_dir / "lesions.csv", syn_dir / "lesions.csv", real_res, syn_res) if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{existing[0]} already exists; pass --overwrite to regenerate.")
    for d in (real_dir, syn_dir):
        if not (d / "brain_masks.npz").exists():
            raise SystemExit(f"{d / 'brain_masks.npz'} not found; run the 'brain' stage first.")
    shape = cfg["preprocessing"]["spatial_size"]
    rng = np.random.default_rng(cfg["seed"])
    extra = {"stage": "lesions", "dataset": name, "pool": pool_name, "min_lesion_area": min_area,
             "attach_gap": gap, "small_lesion_px": small_lesion_px}
    real_src, syn_src = Slices(real_dir), Slices(syn_dir, pool_cfg)

    # real: every lesion slice of every set; also a label volume per patient for viewers
    real = Labeller(name, min_area, gap, small_lesion_px, shape)
    slices = read_table(real_dir / "lesion_slices.csv")
    slices = slices[slices["lesion_px_model"] > 0].sort_values(["patient_id", "z"])
    for pid, group in tqdm(slices.groupby("patient_id"), desc=f"{name} lesion slices"):
        _, seg = load_real(real_dir, str(pid))
        volume = np.zeros(seg.shape, dtype=np.uint16)
        for z, set_name in zip(group["z"], group["set"]):
            key = ids.slice_key(str(pid), int(z))
            res = real.add(key, {"patient_id": pid, "set": set_name, "z": int(z)}, seg[:, :, int(z)] > 0,
                           real_src.brains[key], real_src.boxes[key])
            volume[:, :, int(z)] = res.labels
        save_nifti(volume, real_dir / "volumes" / str(pid) / "labels.nii.gz")
    real_lesions, real_slices = real.save(real_dir)

    # synthetic: every sample, from the masks as saved
    syn = Labeller(pool_name, min_area, gap, small_lesion_px, shape)
    for sid in tqdm(syn_src.raw.keys, desc=pool_name):
        syn.add(ids.sample_key(sid), {"sample_id": sid}, syn_src.raw[sid], syn_src.brains[sid], syn_src.boxes[sid])
    syn_lesions, syn_slices = syn.save(syn_dir)
    for d in (real_dir, syn_dir):
        config.write_run_info(d, cfg, script=__file__, extra=extra, filename="lesions.run_info.json")

    # overview sheets: random slices, slices with many lesions, and (synthetic) masks with both
    # strays and isolated specks, so every element of the drawing appears
    def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))] if len(df) else df

    def sheet(src: Slices, picks: pd.DataFrame, path: Path, title: str) -> None:
        items = []
        for key, n_les, n_specks in picks[["key", "n_lesions", "n_specks"]].itertuples(index=False):
            image, mask, brain, box = src[str(key)]
            counts = f"{n_les} lesion{'' if n_les == 1 else 's'}, {n_specks} speck{'' if n_specks == 1 else 's'}"
            items.append((f"{key}\n{counts}", image, attach_or_drop(mask, min_area, gap), brain, box))
        audit.lesion_sheet(items, path, title)

    legend = ("each lesion in its own colour with its number (L1 = largest), attached strays lighter, "
              "dropped specks x (orange), brain cyan, brain box dashed yellow")
    t = real_slices[real_slices["set"] == "T"]
    sheet(real_src, pd.concat([pick(t, 16), t.nlargest(8, "n_lesions"),
                               *(pick(real_slices[real_slices["set"] == s], 8) for s in ("H-cal", "H-eval"))]),
          real_res / "overview.png",
          f"{name}: rows 1-2 random T, row 3 T with the most lesions, row 4 H-cal, row 5 H-eval. {legend}")
    both = syn_slices[(syn_slices["n_strays"] > 0) & (syn_slices["n_specks"] > 0)]
    sheet(syn_src, pd.concat([pick(syn_slices, 16), syn_slices.nlargest(8, "n_lesions"), pick(both, 8)]),
          syn_res / "overview.png",
          f"{pool_name}: rows 1-2 random, row 3 the most lesions, row 4 with both strays and isolated specks. "
          f"{legend}")

    settings = (f"min_lesion_area {min_area} px, attach_gap {gap} px; a lesion counts as small below "
                f"{small_lesion_px} px")
    texts = []
    for res_dir, source, lesions, slices_df, by in (
            (real_res, name, real_lesions, real_slices, "set"),
            (syn_res, pool_name, syn_lesions.assign(group="S"), syn_slices.assign(group="S"), "group")):
        summary = audit.summarize_lesions(slices_df, lesions, by)
        text = (f"E0b lesions: '{source}', {settings}\n\n{summary.round(3).to_string()}\n\n"
                f"Columns: slices; lesions and per slice; slices without a lesion; slices with a lesion under "
                f"{small_lesion_px} px; lesion area percentiles; attached strays (components, px); isolated "
                f"specks (total, slices with one, share outside the brain); lesions more than half outside "
                f"the brain.\n")
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "summary.txt").write_text(text, encoding="utf-8")
        config.write_run_info(res_dir, cfg, script=__file__, extra=extra)
        texts.append(text)
    print("\n" + "\n".join(texts) + f"\nWrote {real_dir}, {syn_dir}, {real_res}, {syn_res}")


def run_qc_review(cfg: dict, args: argparse.Namespace) -> None:
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    edge_frac = float(cfg["qc"]["edge_outside_frac"])
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    syn_dir = config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, pool_name, "e0b") / "qc_review"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to regenerate.")
    if not (syn_dir / "lesions.csv").exists():
        raise SystemExit(f"{syn_dir / 'lesions.csv'} not found; run the 'lesions' stage first.")
    rng = np.random.default_rng(cfg["seed"])

    # decision D views synthetic masks only; no detector has touched S yet
    index = read_table(syn_dir / "synthetic_index.csv")
    slices = (read_table(syn_dir / "slices.csv")
              .merge(read_table(syn_dir / "brain.csv")[["sample_id", "mri_failed"]], on="sample_id")
              .merge(index[["sample_id", "legacy_excluded"]], on="sample_id"))
    slices["stratum"] = audit.speck_stratum(slices["n_specks"])
    lesions = read_table(syn_dir / "lesions.csv")
    summary = audit.summarize_speck_strata(slices, lesions)
    exact = slices["n_specks"].clip(upper=21).value_counts().reindex(range(22), fill_value=0)
    exact.index = [*map(str, range(21)), ">20"]

    def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))] if len(df) else df

    src = Slices(syn_dir, pool_cfg)

    def tiles(picks: pd.DataFrame, label) -> list:
        items = []
        for rec in picks.itertuples(index=False):
            image, mask, brain, box = src[str(rec.key)]
            items.append((label(rec), image, attach_or_drop(mask, min_area, gap), brain, box))
        return items

    legend = ("lesions coloured with their number, attached strays lighter, isolated specks orange x, "
              "brain cyan, brain box dashed yellow")
    ok = slices[~slices["mri_failed"]]
    items = []
    for stratum in audit.SPECK_STRATA:
        items += tiles(pick(ok[ok["stratum"] == stratum], 12),
                       lambda r: f"{r.key}\n{r.n_specks} speck{'' if r.n_specks == 1 else 's'}, {r.n_lesions} les.")
    audit.lesion_sheet(items, out / f"speck_strata_{pool_name}.png",
                       f"{pool_name}: 12 random masks per number of isolated specks, one row per stratum "
                       f"({', '.join(audit.SPECK_STRATA)}); failed MRIs left out. {legend}", n_cols=12)

    outside = lesions[lesions["frac_outside_brain"] > edge_frac].groupby("key")["lesion_id"].apply(
        lambda s: ", ".join(i.rsplit("/", 1)[1] for i in s))
    clean = ok[(ok["n_specks"] == 0) & ok["key"].isin(outside.index)]
    audit.lesion_sheet(tiles(pick(clean, 24), lambda r: f"{r.key}\noutside: {outside[r.key]}"),
                       out / f"edge_clean_{pool_name}.png",
                       f"{pool_name}: 24 random clean masks (no isolated speck) with a lesion more than "
                       f"{edge_frac:.0%} outside the brain (named in the tile title); {len(clean)} such masks "
                       f"in total. {legend}")

    summary.to_csv(out / f"speck_strata_{pool_name}.csv")
    exact.rename("masks").to_csv(out / f"speck_counts_{pool_name}.csv", index_label="isolated_specks")
    text = (f"E0b qc-review: pool '{pool_name}' (masks as saved; attach-or-drop at {min_area} px, gap {gap}), "
            f"seed {cfg['seed']}\n\n"
            f"Masks by number of isolated specks (all {len(slices)} masks):\n{summary.round(3).to_string()}\n"
            f"Columns: masks, share, cumulative share; failed MRIs; masks without a lesion; lesions per mask; "
            f"share of specks outside the brain; share of lesions with an attached stray, and grown > 25% by "
            f"strays; share of lesions more than half outside the brain; share excluded by the old QC list.\n\n"
            f"Exact counts, masks with 0..20 isolated specks and more:\n"
            f"{exact.to_frame('masks').T.to_string()}\n\n"
            f"Proposed qc.max_isolated_specks: 3 (the plan's suggestion). A mask with 1..N isolated specks is "
            f"minor (kept; specks are not lesions), more than N is noisy (left out of the main audit).\n"
            f"  Choose N from speck_strata_{pool_name}.png: the largest number at which masks still look like "
            f"clean masks with a few stray dots, before they turn into fields of noise.\n\n"
            f"Edge check (qc.edge_outside_frac = {edge_frac}): {len(clean)} clean masks have a lesion more than "
            f"half outside the brain; see edge_clean_{pool_name}.png. The 'qc' stage flags them; whether a flag "
            f"excludes is decided there (recommendation: flag only, see docs/report-notes.md).\n\n"
            f"Then set qc.max_isolated_specks in configs/default.yaml (dated comment) and run the 'qc' stage.\n")
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "qc-review", "pool": pool_name})
    print(f"\n{text}\nWrote {out}")


QC_COLUMNS = ["source", "key", "sample_id", "qc_class", "mask_class", "mri_failed", "edge", "n_edge_lesions",
              "has_lesion_lt40", "n_lesions", "n_specks", "n_strays", "stray_px", "legacy_excluded"]


def run_qc(cfg: dict, args: argparse.Namespace) -> None:
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    max_specks = int(decided(cfg, "qc", "max_isolated_specks", "qc-review"))
    edge_frac, small_px = float(cfg["qc"]["edge_outside_frac"]), int(cfg["qc"]["small_lesion_px"])
    name, _ = config.real_dataset(cfg, args.dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, pool_name, "e0b") / "qc"
    existing = [p for p in (syn_dir / "qc.csv", out) if p.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"{existing[0]} already exists; pass --overwrite to regenerate.")
    for d in (real_dir, syn_dir):
        if not (d / "lesions.csv").exists():
            raise SystemExit(f"{d / 'lesions.csv'} not found; run the 'lesions' stage first.")
    rng = np.random.default_rng(cfg["seed"])

    # one row per synthetic sample: mask class, failed MRI and the combined class; flags as columns
    index = read_table(syn_dir / "synthetic_index.csv")
    table = (read_table(syn_dir / "slices.csv")
             .merge(read_table(syn_dir / "brain.csv")[["sample_id", "mri_failed"]], on="sample_id")
             .merge(index[["sample_id", "legacy_excluded", "legacy_reason"]], on="sample_id"))
    table["mask_class"] = qc.mask_class(table["n_lesions"], table["n_specks"], max_specks)
    table["qc_class"] = qc.qc_class(table["mri_failed"], table["mask_class"])
    edge = qc.edge_lesions(read_table(syn_dir / "lesions.csv"), edge_frac)
    table["n_edge_lesions"] = table["key"].map(edge).fillna(0).astype(int)
    table["edge"] = table["n_edge_lesions"] > 0
    table[QC_COLUMNS].to_csv(syn_dir / "qc.csv", index=False)
    config.write_run_info(syn_dir, cfg, script=__file__, filename="qc.run_info.json",
                          extra={"stage": "qc", "pool": pool_name, "max_isolated_specks": max_specks,
                                 "edge_outside_frac": edge_frac})

    by_qc = audit.summarize_qc(table, "qc_class", qc.QC_CLASSES)
    by_mask = audit.summarize_qc(table, "mask_class", qc.MASK_CLASSES)
    failed_by_mask = pd.crosstab(table["mask_class"], table["mri_failed"]).reindex(list(qc.MASK_CLASSES), fill_value=0)
    # old reasons read e.g. "Excessive noise fragments: 11 distinct blobs"; group by the part before ':'
    old_reason = table["legacy_reason"].str.split(":").str[0].fillna("not on the old list")
    old_qc = pd.crosstab(table["qc_class"], old_reason, margins=True,
                         margins_name="total").reindex([*qc.QC_CLASSES, "total"], fill_value=0)

    # the same edge check on the real sets: if real slices failed it, the rule would be too strict
    real_lesions, real_slices = read_table(real_dir / "lesions.csv"), read_table(real_dir / "slices.csv")
    real_edge = qc.edge_lesions(real_lesions, edge_frac)
    real_check = pd.DataFrame({
        "slices": real_slices.groupby("set", sort=False).size(),
        "lesions": real_lesions.groupby("set", sort=False).size(),
        "edge_lesions": real_lesions[real_lesions["frac_outside_brain"] > edge_frac].groupby("set").size(),
        "edge_slices": real_slices[real_slices["key"].isin(real_edge.index)].groupby("set").size(),
    }).fillna(0).astype(int)

    # a sheet with random masks of each class
    src = Slices(syn_dir, pool_cfg)

    def pick(df: pd.DataFrame, n: int) -> pd.DataFrame:
        return df.iloc[np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))] if len(df) else df

    items = []
    for cls in qc.QC_CLASSES:
        for key, n_les, n_specks, n_edge in pick(table[table["qc_class"] == cls], 8)[
                ["key", "n_lesions", "n_specks", "n_edge_lesions"]].itertuples(index=False):
            image, mask, brain, box = src[str(key)]
            counts = f"{n_les} lesion{'' if n_les == 1 else 's'}, {n_specks} speck{'' if n_specks == 1 else 's'}"
            flags = f", {n_edge} at edge" if n_edge else ""
            items.append((f"{cls}: {key}\n{counts}{flags}", image, attach_or_drop(mask, min_area, gap), brain, box))
    audit.lesion_sheet(items, out / f"qc_classes_{pool_name}.png",
                       f"{pool_name}: 8 random samples per QC class (rows: {', '.join(qc.QC_CLASSES)}). Lesions "
                       f"coloured with their number, strays lighter, isolated specks orange x, brain cyan, brain "
                       f"box dashed yellow")

    by_qc.to_csv(out / f"qc_classes_{pool_name}.csv")
    old_qc.to_csv(out / f"old_qc_crosstab_{pool_name}.csv")
    real_check.to_csv(out / f"edge_check_real_{name}.csv")
    text = (f"E0b qc: pool '{pool_name}'; noisy = more than {max_specks} isolated specks; edge = a lesion more "
            f"than {edge_frac} outside the brain; small lesion = under {small_px} px\n\n"
            f"QC class per sample (precedence: {' > '.join(qc.QC_CLASSES)}):\n{by_qc.round(4).to_string()}\n\n"
            f"Mask class (from the mask alone; failed MRIs keep their mask class):\n{by_mask.round(4).to_string()}\n"
            f"Failed MRIs by mask class:\n{failed_by_mask.to_string()}\n\n"
            f"Columns: masks, share; masks with an edge-flagged lesion; masks without a lesion under {small_px} px; "
            f"attached strays (components, px); masks excluded by the old QC list.\n\n"
            f"Old QC list (reason) against the QC class:\n{old_qc.to_string()}\n\n"
            f"Edge check on the real sets ({name}), same rule:\n{real_check.to_string()}\n\n"
            f"Flags do not exclude: an edge-flagged or small-lesion-free mask keeps its class. Which classes "
            f"enter the main audit is set when E5 is planned (recommendation: mask_class clean or minor).\n")
    (out / "summary.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "qc", "pool": pool_name})
    print(f"\n{text}\nWrote {syn_dir / 'qc.csv'}, {out}")


# Candidate informative-lesion bounds (px) for the bound-impact table; the bound itself is set in E3
CANDIDATE_BOUNDS = (5, 8, 10, 12, 15, 20, 25, 30, 40)


def run_stats(cfg: dict, args: argparse.Namespace) -> None:
    small_px = int(cfg["qc"]["small_lesion_px"])
    name, _ = config.real_dataset(cfg, args.dataset)
    pool_name, _ = config.synthetic_pool(cfg, args.pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, name, "e0b") / "stats"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to regenerate.")
    if not (syn_dir / "qc.csv").exists():
        raise SystemExit(f"{syn_dir / 'qc.csv'} not found; run the 'qc' stage first.")

    # the sets: real lesion slices per set, the main audit (mask_class clean or minor) and all of S
    real_sl, real_les = read_table(real_dir / "slices.csv"), read_table(real_dir / "lesions.csv")
    syn_sl = read_table(syn_dir / "slices.csv").merge(read_table(syn_dir / "qc.csv")[["key", "mask_class"]], on="key")
    syn_les = read_table(syn_dir / "lesions.csv")
    main_sl = syn_sl[syn_sl["mask_class"].isin(["clean", "minor"])]
    sets = {s: (real_sl[real_sl["set"] == s], real_les[real_les["set"] == s]) for s in ("T", "H-cal", "H-eval")}
    sets["S main"] = (main_sl, syn_les[syn_les["key"].isin(main_sl["key"])])
    sets["S all"] = (syn_sl, syn_les)

    describe = audit.describe_sets(sets, small_px)
    patients = read_table(real_dir / "patients.csv")
    native = pd.crosstab(patients["native_dim0"].astype(str) + "x" + patients["native_dim1"].astype(str),
                         patients["set"]).rename_axis("native size")
    bounds = audit.bound_impact(sets, CANDIDATE_BOUNDS)
    without = bounds.pivot(index="bound", columns="set", values="share_slices_without")[list(sets)]
    kept = bounds.pivot(index="bound", columns="set", values="share_lesions_kept")[list(sets)]

    # figures compare the real sets with the main audit; the density pools H-cal and H-eval
    shown = ["T", "H-cal", "H-eval", "S main"]
    audit.lesions_per_slice_figure({s: sets[s][0]["n_lesions"] for s in shown}, out / f"lesions_per_slice_{pool_name}.png",
                                   f"Lesions per slice: real lesion slices per set and the synthetic main audit "
                                   f"({pool_name}, mask class clean or minor)")
    audit.area_figure({s: sets[s][1]["area"] for s in shown}, out / f"lesion_area_{pool_name}.png",
                      "Lesion area: cumulative share of lesions (lesions >= the minimum lesion area)")
    h_les = pd.concat([sets["H-cal"][1], sets["H-eval"][1]])
    audit.centroid_density_figure({"T": sets["T"][1], "H": h_les, "S main": sets["S main"][1]},
                                  out / f"centroid_density_{pool_name}.png",
                                  "Lesion centroids: T, H (H-cal + H-eval) and the synthetic main audit; top row in "
                                  "pixels, bottom row relative to the brain box (0 = first, 1 = last row/column)")

    describe.round(3).to_csv(out / f"description_{pool_name}.csv")
    native.to_csv(out / f"native_sizes_{name}.csv")
    bounds.round(4).to_csv(out / f"bound_impact_{pool_name}.csv", index=False)
    text = (f"E0b stats: dataset '{name}', pool '{pool_name}' (S main = mask class clean or minor)\n\n"
            f"Data description:\n{describe.round(3).to_string()}\n\n"
            f"Native in-plane sizes (patients per set):\n{native.to_string()}\n\n"
            f"Bound impact (input for E3, not decided here), by candidate informative-lesion bound:\n"
            f"Share of slices left without an informative lesion:\n{without.round(3).to_string()}\n"
            f"Share of lesions kept:\n{kept.round(3).to_string()}\n"
            f"Patients with at least one informative slice, per bound: see bound_impact_{pool_name}.csv.\n")
    (out / "summary.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "stats", "pool": pool_name})
    print(f"\n{text}\nWrote {out}")


# lesions.csv columns in the view's table, with short headers
VIEW_COLUMNS = {"lesion_id": "lesion", "area": "area px", "stray_px": "strays px", "centroid_row": "row",
                "centroid_col": "col", "centroid_row_brain": "row in brain", "centroid_col_brain": "col in brain",
                "eccentricity": "eccentr.", "solidity": "solidity", "frac_outside_brain": "outside brain"}


def run_view(cfg: dict, args: argparse.Namespace) -> None:
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    if args.sample is not None:
        source, pool_cfg = config.synthetic_pool(cfg, args.pool)
        key = ids.sample_key(args.sample)
    elif args.patient is not None and args.z is not None:
        source, pool_cfg = config.real_dataset(cfg, args.dataset)[0], None
        key = ids.slice_key(args.patient, args.z)
    else:
        raise SystemExit("Give --sample <id>, or --patient <id> and --z <0-based slice index>.")
    out_dir = config.output_dir(cfg, source)
    if not (out_dir / "lesions.csv").exists():
        raise SystemExit(f"{out_dir / 'lesions.csv'} not found; run the 'lesions' stage first.")
    src = Slices(out_dir, pool_cfg)
    if key not in src:
        raise SystemExit(f"'{key}' is not a {'sample' if pool_cfg else 'lesion slice'} of '{source}'.")
    path = config.results_dir(cfg, source, "e0b") / "views" / f"{key.replace('/', '_')}.png"
    if path.exists() and not args.overwrite:
        raise SystemExit(f"{path} already exists; pass --overwrite to regenerate.")

    image, mask, brain, box = src[key]
    res = attach_or_drop(mask, min_area, gap)
    if not np.array_equal(res.labels, LabelStack(out_dir / "labels.npz")[key]):
        print("Warning: labels differ from labels.npz; the lesion settings changed since the 'lesions' stage.")
    lesions = read_table(out_dir / "lesions.csv")
    table = lesions.loc[lesions["key"] == key, list(VIEW_COLUMNS)].rename(columns=VIEW_COLUMNS)
    audit.lesion_view(image, res, brain, box, table, path,
                      f"{source}: {key} | {res.n_lesions} lesions, {res.n_strays} strays ({int(res.attached.sum())} "
                      f"px), {res.n_specks} isolated specks | L-number per lesion, lighter = attached stray, "
                      f"orange x = dropped speck, cyan = brain, dashed yellow = brain box")
    print(f"Wrote {path}")
    if args.nifti:
        if pool_cfg is None:
            pid, _ = ids.parse_slice_key(key)
            print(f"Real slice: open {out_dir / 'volumes' / pid / 'labels.nii.gz'} next to bravo.nii.gz "
                  f"(ITK-SNAP slice {args.z + 1}).")
        else:
            stem = path.with_suffix("")
            save_nifti(image[:, :, None].astype(np.float32), f"{stem}_image.nii.gz")
            save_nifti(res.labels[:, :, None].astype(np.uint16), f"{stem}_labels.nii.gz")
            print(f"Wrote {stem}_image.nii.gz and {stem}_labels.nii.gz (open the labels as segmentation)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    p = stages.add_parser("brain-review", help="evidence for decision A: brain-mask method, failed-MRI cut-off")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--n-synthetic", type=int, default=2000,
                   help="random synthetic samples to evaluate (default: 2000)")
    p.add_argument("--min-area", type=int, default=4,
                   help="components of at least this many px count as lesions here (provisional; default: 4)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_brain_review)

    p = stages.add_parser("brain", help="brain masks, boxes and failed-MRI flags for all data (needs decision A)")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_brain)

    p = stages.add_parser("lesions-review", help="evidence for decisions B, C: minimum lesion area, attach gap")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_lesions_review)

    p = stages.add_parser("lesions", help="lesion table, slice table and label maps for all data (needs B, C)")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_lesions)

    p = stages.add_parser("qc-review", help="evidence for decision D: isolated-speck cut-off (minor vs noisy)")
    config.add_config_arg(p)
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_qc_review)

    p = stages.add_parser("qc", help="QC class per synthetic sample (needs D)")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_qc)

    p = stages.add_parser("stats", help="E0 description: tables, distribution figures, bound impact (after 'qc')")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_stats)

    p = stages.add_parser("view", help="one slice at full size with its lesion rows (helper, after 'lesions')")
    config.add_config_arg(p)
    p.add_argument("--sample", help="synthetic sample ID (full file stem, e.g. file_2_1033)")
    p.add_argument("--patient", help="real patient ID (with --z)")
    p.add_argument("--z", type=int, help="0-based slice index of a real patient (ITK-SNAP shows z + 1)")
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--nifti", action="store_true",
                   help="synthetic: also write the image and its labels as NIfTI for ITK-SNAP")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_view)

    args = parser.parse_args()
    args.func(config.load_config(args.config), args)


if __name__ == "__main__":
    main()
