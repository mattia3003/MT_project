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
  (planned: lesions-review, lesions, view, qc-review, qc, stats, check)

Example (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py brain-review
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from memtk import audit, config
from memtk.io import MaskStackWriter, load_image
from memtk.lesions import brain_box, brain_mask, tissue_px
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

    slices = pd.read_csv(real_dir / "lesion_slices.csv")
    t_slices = slices[slices["set"] == "T"]  # decisions use T only; H-cal/H-eval stay untouched
    index = pd.read_csv(config.output_dir(cfg, pool_name) / "synthetic_index.csv", keep_default_na=False)
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
            key = f"{pid}/z{z}"
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
    slices = pd.read_csv(real_dir / "lesion_slices.csv")
    slices = slices[slices["lesion_px_model"] > 0]
    real_brains, rows = MaskStackWriter(shape), []
    for pid, group in tqdm(slices.groupby("patient_id"), desc=f"{name} lesion slices"):
        bravo, _ = load_real(real_dir, str(pid))
        for z, set_name in zip(group["z"], group["set"]):
            key = f"{pid}/z{z}"
            brain, _, row = brain_of(bravo[:, :, int(z)], method, low)
            real_brains.add(key, brain)
            rows.append({"key": key, "patient_id": pid, "set": set_name, "z": int(z), **row})
    real = pd.DataFrame(rows)

    # synthetic: every readable sample; the mask as saved is cached for later stages
    index = pd.read_csv(syn_dir / "synthetic_index.csv", keep_default_na=False)
    n_unreadable = int((~index["read_ok"]).sum())
    index = index[index["read_ok"]]
    syn_brains, raw_masks, rows = MaskStackWriter(shape), MaskStackWriter(shape), []
    for sid, fname in tqdm(list(zip(index["sample_id"], index["filename"])), desc=pool_name):
        image, mask = load_sample(pool_dir / fname, pool_cfg)
        if not np.isin(mask, (0.0, 1.0)).all():
            raise ValueError(f"Mask of {sid} is not binary; masks_raw.npz stores masks as 0/1")
        brain, _, row = brain_of(image, method, low)
        syn_brains.add(str(sid), brain)
        raw_masks.add(str(sid), mask > 0)
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

    args = parser.parse_args()
    args.func(config.load_config(args.config), args)


if __name__ == "__main__":
    main()
