"""Helpers shared by the stage packages (`e0b`, `e3`, ...) and their scripts: the command-line
options every stage has, decisions read from the config, the overwrite and input guards, and
random picks for example figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from memtk import config


# --- command line ---

def add_stage(stages, name: str, help: str, run, dataset: bool = True, overwrite: bool = True):
    """A stage subcommand with the options every stage shares: --config, --dataset (real
    dataset), --pool (synthetic pool) and --overwrite. Returns its parser for extra options;
    each option reaches the stage function as the keyword of the same name."""
    p = stages.add_parser(name, help=help)
    config.add_config_arg(p)
    if dataset:
        p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    if overwrite:
        p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(run=run)
    return p


# --- decisions and guards ---

def decided(cfg: dict, key: str, review: str) -> Any:
    """A decision value from the config by dotted key (e.g. "brain.method"); stops with
    instructions while it is still null. `review` is the command of the review stage that gives
    its evidence, e.g. "e0b_audit.py brain-review"."""
    v = config.value(cfg, key)
    if v is None:
        raise SystemExit(f"{key} is not set in {cfg['_config_path']}. Run '{review}', view its "
                         f"evidence and set the value first.")
    return v


def refuse_overwrite(paths, overwrite: bool) -> None:
    """Stop if any of the stage's outputs already exists, unless overwriting was asked for."""
    existing = [p for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise SystemExit(f"{existing[0]} already exists; pass --overwrite to regenerate.")


def require(path: Path, stage: str) -> None:
    """Stop if an input made by an earlier stage is missing."""
    if not path.exists():
        raise SystemExit(f"{path} not found; run the '{stage}' stage first.")


def pick(rng: np.random.Generator, df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Up to n random rows of a table, in their original order (for example figures)."""
    if not len(df):
        return df
    return df.iloc[np.sort(rng.choice(len(df), size=min(n, len(df)), replace=False))]
