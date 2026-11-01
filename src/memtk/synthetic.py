"""Reading and indexing a pool of synthetic samples.

Each synthetic file holds one 2D sample with the channels stacked on the last axis, e.g.
(128, 128, 2) = [MRI, mask]; the channel order comes from the pool config. Files are read in
place and never copied. A sample's ID is its full file stem: pools may mix naming schemes
(`1033.nii` and `file_1033.nii` are different samples), so IDs are never parsed to numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm

from memtk.config import resolve_path
from memtk.io import load_image

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


def count_components(mask: np.ndarray) -> int:
    """Number of 8-connected foreground components of a 2D mask."""
    result = ndimage.label(mask > 0, structure=EIGHT_CONNECTED)
    if not isinstance(result, tuple):  # only an int when an `output` array is passed
        raise TypeError("ndimage.label returned no label array")
    return int(result[1])


def sample_id(path: str | Path) -> str:
    """Full file stem without the NIfTI extension(s): 'file_1033.nii.gz' -> 'file_1033'."""
    name = Path(path).name
    return re.sub(r"\.nii(\.gz)?$", "", name)


def name_scheme(sid: str) -> str:
    """Naming scheme of a sample ID, the trailing number replaced by '<n>' ('file_2_1033' -> 'file_2_<n>').

    In the original pool each scheme is most likely one generation round. Never empty, so it
    cannot be read back as a missing value.
    """
    return re.sub(r"\d+$", "<n>", sid)


def list_pool_files(pool_cfg: dict) -> list[Path]:
    """Sorted files of a pool. Raises if the folder is missing or two files share a sample ID."""
    pool_dir = resolve_path(pool_cfg["dir"])
    if not pool_dir.is_dir():
        raise FileNotFoundError(f"Synthetic pool folder not found: {pool_dir}")
    files = sorted(p for p in pool_dir.glob(pool_cfg["glob"]) if p.is_file())
    ids = pd.Series([sample_id(f) for f in files])
    if ids.duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {pool_dir}: {ids[ids.duplicated()].tolist()[:10]}")
    return files


def load_sample(path: str | Path, pool_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """(image, mask) of one synthetic sample as saved, float32.

    Requires a (H, W, C) array; the in-plane size is not checked here.
    """
    arr = np.asanyarray(load_image(path).dataobj).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected a (H, W, channels) array, got shape {arr.shape}: {path}")
    return arr[..., pool_cfg["image_channel"]], arr[..., pool_cfg["mask_channel"]]


def read_legacy_exclusions(path: str | Path) -> dict[str, str]:
    """Parse an old QC list ('<file> | <reason>' per line, '#' comments) into {sample_id: reason}."""
    exclusions = {}
    with open(resolve_path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            filename, _, reason = line.partition("|")
            # entries are named like '10004.seg.nii.gz': the ID is the part before the first dot
            exclusions[filename.strip().split(".")[0]] = reason.strip()
    return exclusions


def index_pool(pool_name: str, pool_cfg: dict, spatial_size: list[int]) -> pd.DataFrame:
    """One row per sample: ID, file, shape check, mask statistics and image range.

    Samples are only described, never filtered: quality control happens in E0b. A file that
    cannot be read gets a row with read_ok = False and the error message, instead of stopping
    the run; its statistics are -1 / NaN.
    """
    files = list_pool_files(pool_cfg)
    legacy_file = pool_cfg.get("legacy_exclusion_file")
    legacy = read_legacy_exclusions(legacy_file) if legacy_file else {}

    rows = []
    for path in tqdm(files, desc=f"Indexing {pool_name}"):
        sid = sample_id(path)
        # free-text columns that are usually empty go last, so empty fields only end a row
        row = {"pool": pool_name, "sample_id": sid, "filename": path.name, "name_scheme": name_scheme(sid),
               "read_ok": True, "shape": "", "shape_ok": False, "mask_binary": False,
               "mask_px": -1, "n_components_8conn": -1,
               "img_min": np.nan, "img_max": np.nan, "img_mean": np.nan,
               "legacy_excluded": sid in legacy, "legacy_reason": legacy.get(sid, ""), "read_error": ""}
        try:
            image, mask = load_sample(path, pool_cfg)
        except Exception as e:  # recorded per file, reported by the 'check' stage
            row.update(read_ok=False, read_error=f"{type(e).__name__}: {e}")
            rows.append(row)
            continue
        row.update({
            "shape": "x".join(map(str, image.shape)),
            "shape_ok": list(image.shape) == list(spatial_size),
            "mask_binary": bool(np.isin(mask, (0.0, 1.0)).all()),
            "mask_px": int((mask > 0).sum()),
            "n_components_8conn": count_components(mask),
            "img_min": float(image.min()), "img_max": float(image.max()), "img_mean": float(image.mean()),
        })
        rows.append(row)
    df = pd.DataFrame(rows)

    unmatched = sorted(set(legacy) - set(df["sample_id"]))
    if unmatched:
        print(f"Warning: {len(unmatched)} legacy exclusion entries match no file, e.g. {unmatched[:5]}")
    return df
