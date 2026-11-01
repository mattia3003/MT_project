"""E0b: data audit and mask quality control.

Review stages only produce evidence and a proposed config value. Production stages apply your
decision and refuse to run while the value they need is still null in the config.

Stages, run in this order. Each one can be run and checked on its own:
  brain-review   evidence for decision A (brain-mask method, failed-MRI tissue cut-off), using T,
                 a random sample of S for the methods and all of S for tissue area
                 -> results/<dataset>/e0b/brain_review/
  (planned: brain, lesions-review, lesions, view, qc-review, qc, stats, check)

Example (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py brain-review
"""

import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from memtk import audit, config
from memtk.io import load_image
from memtk.lesions import tissue_px
from memtk.synthetic import load_sample


def run_brain_review(cfg: dict, args: argparse.Namespace) -> None:
    name, _ = config.real_dataset(cfg, args.dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    real_dir = config.output_dir(cfg, name)
    out = config.results_dir(cfg, name, "e0b") / "brain_review"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to regenerate.")
    rng = np.random.default_rng(cfg["seed"])
    pool_dir = config.resolve_path(pool_cfg["dir"])

    slices = pd.read_csv(real_dir / "lesion_slices.csv")
    t_slices = slices[slices["set"] == "T"]  # decisions use T only; H-cal/H-eval stay untouched
    index = pd.read_csv(config.output_dir(cfg, pool_name) / "synthetic_index.csv", keep_default_na=False)
    index = index[index["read_ok"]]

    def load_t(pid: str):
        vol_dir = real_dir / "volumes" / pid
        return (np.asanyarray(load_image(vol_dir / "bravo.nii.gz").dataobj),
                np.asanyarray(load_image(vol_dir / "seg.nii.gz").dataobj))

    # every T lesion slice, and a random sample of S, under each candidate method; tissue area of
    # every T lesion slice and every synthetic sample (one fixed measure, for the failed-MRI cut)
    rows, t_tissue = [], []
    for pid, group in tqdm(t_slices.groupby("patient_id"), desc="T lesion slices"):
        bravo, seg = load_t(str(pid))
        for z in group["z"]:
            key = f"{pid}/z{z}"
            t_tissue.append({"key": key, "pid": pid, "z": int(z), "tissue_px": tissue_px(bravo[:, :, z])})
            for r in audit.evaluate_brain_methods(bravo[:, :, z], seg[:, :, z], args.min_area):
                rows.append({"group": "T", "key": key, "pid": pid, "z": int(z), **r})
    sampled = set(rng.choice(len(index), size=min(args.n_synthetic, len(index)), replace=False).tolist())
    s_tissue = []
    for i, (sid, fname) in enumerate(tqdm(list(zip(index["sample_id"], index["filename"])), desc="S pool")):
        image, mask = load_sample(pool_dir / fname, pool_cfg)
        s_tissue.append(tissue_px(image))
        if i in sampled:
            for r in audit.evaluate_brain_methods(image, mask, args.min_area):
                rows.append({"group": "S", "key": sid, "filename": fname, **r})
    per_slice = pd.DataFrame(rows)
    t_tissue_df = pd.DataFrame(t_tissue)
    index = index.assign(tissue_px=s_tissue)
    summary = audit.summarize_brain_methods(per_slice)
    method, reason = audit.propose_brain_method(summary)

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
                      "cyan = brain, red = lesion")
    audit.method_grid(synth_examples, out / f"methods_synthetic_{pool_name}.png",
                      f"S: rows 1-3 small brains (< T 5th percentile, {small_brain_px:.0f} px), "
                      f"4-5 a lesion outside, 6-7 random")

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
    text = (f"E0b brain-review: dataset '{name}' (T only), pool '{pool_name}' "
            f"({len(sampled)} random samples for the methods), seed {cfg['seed']}\n"
            f"Lesions = components >= {args.min_area} px (provisional; decision B comes later)\n\n"
            f"Candidate brain-mask methods:\n{summary.round(4).to_string()}\n\n"
            f"Proposed brain.method: {method}\n  rule: {reason}\n\n"
            f"Failed-MRI cut-off: samples and T lesion slices below each tissue cut (all {len(index)} samples)\n"
            f"{counts.to_string(index=False)}\n"
            f"Choose brain.min_tissue_px from near_blank_{pool_name}.png: the highest cut below which every "
            f"synthetic image is near-blank, staying clear of the smallest real slices.\n\n"
            f"Then set both values in configs/default.yaml (dated comment) and run the 'brain' stage.\n")
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "brain-review", "pool": pool_name,
                                                            "min_area": args.min_area,
                                                            "n_synthetic": len(sampled)})
    print(f"\n{text}\nWrote {out}")


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

    args = parser.parse_args()
    args.func(config.load_config(args.config), args)


if __name__ == "__main__":
    main()
