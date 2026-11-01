"""E0a: prepare the data for the memorization audit.

Stages, run in this order. Each one can be run and checked on its own:
  split   seeded patient split of a real dataset -> splits/<dataset>_seed<seed>.csv

Example (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e0a_prepare_data.py split
"""

import argparse

from memtk import config
from memtk.split import make_split, summarize_split


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    p = stages.add_parser("split", help="seeded T / H-cal / H-eval patient split")
    config.add_config_arg(p)
    p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing split file")
    p.set_defaults(func=run_split)

    args = parser.parse_args()
    args.func(config.load_config(args.config), args)


if __name__ == "__main__":
    main()
