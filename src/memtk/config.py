"""Load project configs, look up datasets, resolve paths and log runs.

Relative paths in a config are resolved against the repository root, never the current
working directory, so scripts behave the same wherever they are launched from.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"

# Packages whose versions are recorded in every run_info.json
LOGGED_PACKAGES = ["memtk", "numpy", "scipy", "scikit-image", "nibabel", "monai", "torch", "pandas"]


def resolve_path(path: str | Path) -> Path:
    """Return `path` as an absolute path, interpreting relative paths from the repo root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config. The resolved config path is stored under `_config_path`."""
    path = resolve_path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def _lookup(cfg: dict, section: str, default_key: str, name: str | None) -> tuple[str, dict]:
    name = name or cfg[default_key]
    entries = cfg[section]
    if name not in entries:
        raise KeyError(f"'{name}' not found under '{section}' in {cfg['_config_path']}; "
                       f"available: {sorted(entries)}")
    return name, entries[name]


def real_dataset(cfg: dict, name: str | None = None) -> tuple[str, dict]:
    """Return (name, settings) of a real dataset; `None` selects `default_real_dataset`."""
    return _lookup(cfg, "real_datasets", "default_real_dataset", name)


def synthetic_pool(cfg: dict, name: str | None = None) -> tuple[str, dict]:
    """Return (name, settings) of a synthetic pool; `None` selects `default_synthetic_pool`."""
    return _lookup(cfg, "synthetic_pools", "default_synthetic_pool", name)


def output_dir(cfg: dict, name: str) -> Path:
    """Processed-data folder for one dataset or pool: `<output_root>/<name>`."""
    return resolve_path(cfg["output_root"]) / name


def split_file(cfg: dict, dataset_name: str) -> Path:
    """Patient-split CSV for a real dataset: `<splits_dir>/<dataset>_seed<seed>.csv`."""
    return resolve_path(cfg["splits_dir"]) / f"{dataset_name}_seed{cfg['seed']}.csv"


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="YAML config (default: configs/default.yaml)")


def _package_versions() -> dict:
    versions = {}
    for pkg in LOGGED_PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def write_run_info(out_dir: str | Path, cfg: dict, script: str, extra: dict | None = None,
                   filename: str = "run_info.json") -> Path:
    """Write a run log (time, command, seed, full config, versions) into `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": script,
        "argv": sys.argv,
        "seed": cfg.get("seed"),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "config": cfg,
        **(extra or {}),
    }
    path = out_dir / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=str)
    return path
