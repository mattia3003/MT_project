"""E0a: prepare the data for the memorization audit.

Stages, run in this order. Each one can be run and checked on its own:
  split   seeded patient split of a real dataset -> splits/<dataset>_seed<seed>.csv
  real    real volumes into the 128x128 model space -> <output_root>/<dataset>/
          volumes/<pid>/{bravo,seg_soft,seg}.nii.gz, patients.csv, lesion_slices.csv
  synthetic  index of a synthetic pool (files are read in place, not copied)
          -> <output_root>/<pool>/synthetic_index.csv
  check   consistency checks of all outputs above, plus a visual sample grid
          -> results/<dataset>/e0a/check_report.txt, check_grid.png (exit code 1 on failure)

Example (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e0a_prepare_data.py split
"""

import argparse
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from memtk import checks, config
from memtk.io import load_image, patient_files, save_nifti
from memtk.split import make_split, summarize_split
from memtk.synthetic import index_pool
from memtk.transforms import convert_volume_pair, lesion_px_per_slice


def run_split(cfg: dict, args: argparse.Namespace) -> None:
    name, dataset_cfg = config.real_dataset(cfg, args.dataset)
    out_path = config.split_file(cfg, name)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists. The split is fixed for the whole project; "
                         f"pass --overwrite only if you really mean to replace it.")

    print(f"Splitting '{name}' with seed {cfg['seed']} (counting lesion slices per patient)...")
    df = make_split(dataset_cfg, cfg["seed"], cfg["preprocessing"]["slice_axis"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    config.write_run_info(out_path.parent, cfg, script=__file__, filename=f"{out_path.stem}.run_info.json")

    print(f"\n{summarize_split(df).to_string()}\n")
    print(f"Wrote {out_path}")


def run_real(cfg: dict, args: argparse.Namespace) -> None:
    name, dataset_cfg = config.real_dataset(cfg, args.dataset)
    prep = cfg["preprocessing"]
    axis = prep["slice_axis"]
    split_path = config.split_file(cfg, name)
    if not split_path.exists():
        raise SystemExit(f"{split_path} not found; run the 'split' stage first.")
    out_dir = config.output_dir(cfg, name)
    volumes_dir = out_dir / "volumes"
    if volumes_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{volumes_dir} already exists; pass --overwrite to regenerate.")
        shutil.rmtree(volumes_dir)  # no stale patient folders survive a regeneration

    split_df = pd.read_csv(split_path)
    patient_rows, slice_rows = [], []
    columns = split_df[["patient_id", "orig_split", "set"]].itertuples(index=False, name=None)
    for pid, orig_split, set_name in tqdm(columns, total=len(split_df), desc=f"Converting {name}"):
        pid, orig_split, set_name = str(pid), str(orig_split), str(set_name)
        image_path, mask_path = patient_files(dataset_cfg, orig_split, pid)
        mask_native = np.asanyarray(load_image(mask_path).dataobj)
        bravo, seg_soft, seg = convert_volume_pair(image_path, mask_path, prep)

        vol_dir = volumes_dir / pid
        save_nifti(bravo, vol_dir / "bravo.nii.gz")
        save_nifti(seg_soft, vol_dir / "seg_soft.nii.gz")
        save_nifti(seg, vol_dir / "seg.nii.gz")

        px_native = lesion_px_per_slice(mask_native, axis)
        px_model = lesion_px_per_slice(seg, axis)
        in_plane = [s for i, s in enumerate(mask_native.shape) if i != axis]
        patient_rows.append({
            "patient_id": pid, "orig_split": orig_split, "set": set_name,
            "native_dim0": in_plane[0], "native_dim1": in_plane[1], "n_slices": mask_native.shape[axis],
            "n_lesion_slices_native": int((px_native > 0).sum()),
            "n_lesion_slices_model": int((px_model > 0).sum()),
            "lesion_px_native": int(px_native.sum()), "lesion_px_model": int(px_model.sum()),
        })
        # Every slice with a lesion in either space, so a lesion lost or gained by conversion shows up
        for z in np.flatnonzero((px_native > 0) | (px_model > 0)):
            slice_rows.append({
                "dataset": name, "patient_id": pid, "orig_split": orig_split,
                "set": set_name, "z": int(z),
                "lesion_px_native": int(px_native[z]), "lesion_px_model": int(px_model[z]),
            })

    patients = pd.DataFrame(patient_rows)
    slices = pd.DataFrame(slice_rows)
    patients.to_csv(out_dir / "patients.csv", index=False)
    slices.to_csv(out_dir / "lesion_slices.csv", index=False)
    config.write_run_info(out_dir, cfg, script=__file__, extra={"stage": "real", "split_file": str(split_path)})

    summary = patients.groupby("set", sort=False)[
        ["n_lesion_slices_native", "n_lesion_slices_model", "lesion_px_native", "lesion_px_model"]].sum()
    print(f"\n{summary.to_string()}\n")
    print(f"Wrote {len(patients)} patients to {out_dir}")


def run_synthetic(cfg: dict, args: argparse.Namespace) -> None:
    name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    out_dir = config.output_dir(cfg, name)
    out_path = out_dir / "synthetic_index.csv"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists; pass --overwrite to regenerate.")

    df = index_pool(name, pool_cfg, cfg["preprocessing"]["spatial_size"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    config.write_run_info(out_dir, cfg, script=__file__, extra={"stage": "synthetic"})

    bad = df[~df["shape_ok"] | ~df["mask_binary"]]
    print(f"\nSamples: {len(df)}  by naming scheme: {df['name_scheme'].value_counts().to_dict()}")
    print(f"Wrong shape: {(~df['shape_ok']).sum()}  non-binary mask: {(~df['mask_binary']).sum()}"
          f"  empty mask: {(df['mask_px'] == 0).sum()}")
    if len(bad):
        print(f"Failing samples (first 10): {bad['sample_id'].tolist()[:10]}")
    print(f"Legacy-excluded: {df['legacy_excluded'].sum()}")
    print(f"\n{df[['mask_px', 'n_components_8conn', 'img_min', 'img_max']].describe().round(3).to_string()}\n")
    print(f"Wrote {out_path}")


def run_check(cfg: dict, args: argparse.Namespace) -> None:
    name, dataset_cfg = config.real_dataset(cfg, args.dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, args.pool)
    prep = cfg["preprocessing"]
    real_dir, pool_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    results = config.results_dir(cfg, name, "e0a")
    rng = np.random.default_rng(cfg["seed"])

    split_df = pd.read_csv(config.split_file(cfg, name))
    index_df = pd.read_csv(pool_dir / "synthetic_index.csv", keep_default_na=False)
    lesion_slices = pd.read_csv(real_dir / "lesion_slices.csv")

    report = checks.CheckReport()
    report.add(f"pool '{pool_name}' was trained on dataset '{name}'", pool_cfg.get("trained_on") == name,
               f"trained_on = {pool_cfg.get('trained_on')}")
    checks.check_split(report, split_df, dataset_cfg)
    checks.check_real_volumes(report, split_df, dataset_cfg, real_dir, prep)

    # random patients per set for the fresh and the independent re-computation
    recompute_ids: list[str] = []
    for _, group in split_df.groupby("set", sort=False):
        picks = rng.choice(group["patient_id"].to_numpy(), size=args.n_recompute, replace=False)
        recompute_ids += [str(pid) for pid in picks]
    checks.check_recompute(report, split_df, dataset_cfg, real_dir, prep, recompute_ids)
    checks.check_synthetic_index(report, index_df, pool_cfg, prep["spatial_size"], rng, args.n_synthetic)
    checks.sample_grid(lesion_slices, real_dir, index_df, pool_cfg, prep, rng, args.grid_cols,
                       results / "check_grid.png")

    text = f"E0a check: dataset '{name}', pool '{pool_name}', seed {cfg['seed']}\n\n{report.to_text()}\n"
    results.mkdir(parents=True, exist_ok=True)
    (results / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(results, cfg, script=__file__, extra={"stage": "check", "pool": pool_name})
    print(f"\n{text}\nWrote {results / 'check_report.txt'} and {results / 'check_grid.png'}")
    if not report.all_passed:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    p = stages.add_parser("split", help="seeded T / H-cal / H-eval patient split")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing split file")
    p.set_defaults(func=run_split)

    p = stages.add_parser("real", help="real volumes into the 128x128 model space")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(func=run_real)

    p = stages.add_parser("synthetic", help="index of a synthetic pool")
    config.add_config_arg(p)
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--overwrite", action="store_true", help="regenerate an existing index")
    p.set_defaults(func=run_synthetic)

    p = stages.add_parser("check", help="consistency checks of all E0a outputs")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    p.add_argument("--n-recompute", type=int, default=1,
                   help="patients per set re-converted from the raw data (default: 1)")
    p.add_argument("--n-synthetic", type=int, default=300,
                   help="synthetic samples whose index values are recomputed (default: 300)")
    p.add_argument("--grid-cols", type=int, default=6, help="samples per row in check_grid.png (default: 6)")
    p.set_defaults(func=run_check)

    args = parser.parse_args()
    args.func(config.load_config(args.config), args)


if __name__ == "__main__":
    main()
