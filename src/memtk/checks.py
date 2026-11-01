"""Consistency checks for the E0a outputs: patient split, model-space volumes, synthetic index.

Every check is recorded in a CheckReport instead of stopping at the first failure, so one run
shows everything that is wrong. Checks are generic (no hardcoded patient or sample counts);
the counts are printed in the report for manual comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
# Figure is used directly instead of pyplot: it needs no GUI backend, so importing this module
# never changes the plotting setup of a notebook or script.
from matplotlib.figure import Figure
from tqdm import tqdm

from memtk.config import resolve_path
from memtk.io import list_patients, load_image, patient_files
from memtk.lesions import count_components
from memtk.synthetic import list_pool_files, load_sample, read_legacy_exclusions
from memtk.transforms import convert_volume_pair, lesion_px_per_slice

VOLUME_FILES = ("bravo.nii.gz", "seg_soft.nii.gz", "seg.nii.gz")


@dataclass
class CheckReport:
    results: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool | np.bool_, detail: str = "") -> None:
        self.results.append((name, bool(passed), detail))

    @property
    def all_passed(self) -> bool:
        return all(passed for _, passed, _ in self.results)

    def to_text(self) -> str:
        lines = [f"[{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else "")
                 for name, passed, detail in self.results]
        n_failed = sum(not passed for _, passed, _ in self.results)
        lines.append(f"\n{len(self.results) - n_failed} passed, {n_failed} failed")
        return "\n".join(lines)


def _examples(items: list, n: int = 5) -> str:
    return f"{len(items)} failing, e.g. {items[:n]}" if items else ""


# --- split -----------------------------------------------------------------------------------

def check_split(report: CheckReport, split_df: pd.DataFrame, dataset_cfg: dict) -> None:
    ids = split_df["patient_id"]
    report.add("split: patient IDs unique", not ids.duplicated().any(), _examples(ids[ids.duplicated()].tolist()))

    for split in (dataset_cfg["train_split"], dataset_cfg["heldout_split"]):
        on_disk = set(list_patients(dataset_cfg, split))
        in_csv = set(split_df.loc[split_df["orig_split"] == split, "patient_id"])
        report.add(f"split: '{split}' patients in CSV == patients on disk", on_disk == in_csv,
                   f"{len(in_csv)} patients" if on_disk == in_csv
                   else f"only on disk: {sorted(on_disk - in_csv)[:5]}, only in CSV: {sorted(in_csv - on_disk)[:5]}")

    train = split_df[split_df["orig_split"] == dataset_cfg["train_split"]]
    report.add("split: all training patients in T", (train["set"] == "T").all())
    heldout = split_df[split_df["orig_split"] == dataset_cfg["heldout_split"]]
    report.add("split: held-out patients only in held-out sets",
               heldout["set"].isin(dataset_cfg["heldout_sets"]).all() and not (heldout["set"] == "T").any())
    sizes = heldout["set"].value_counts()
    report.add("split: held-out sets differ by at most one patient", sizes.max() - sizes.min() <= 1,
               ", ".join(f"{k}={v}" for k, v in split_df["set"].value_counts().items()))


# --- real volumes ----------------------------------------------------------------------------

def check_real_volumes(report: CheckReport, split_df: pd.DataFrame, dataset_cfg: dict,
                       out_dir: Path, prep: dict) -> None:
    """Every volume of every patient: files, shapes, dtypes, value ranges, mask consistency,
    and both index CSVs recomputed from the volumes and the raw masks."""
    patients = pd.read_csv(out_dir / "patients.csv")
    slices = pd.read_csv(out_dir / "lesion_slices.csv")
    report.add("real: patients.csv covers exactly the split patients",
               set(patients["patient_id"]) == set(split_df["patient_id"]) and len(patients) == len(split_df),
               f"{len(patients)} rows")
    # the `set` columns are copies of the split; a regenerated split without a re-run of 'real'
    # would leave them stale
    split_set = dict(zip(split_df["patient_id"], split_df["set"]))
    stale = sorted({pid for df in (patients, slices) for pid, s in zip(df["patient_id"], df["set"])
                    if split_set.get(pid) != s})
    report.add("real: 'set' in patients.csv and lesion_slices.csv == split CSV", not stale, _examples(stale))

    axis, thr = prep["slice_axis"], prep["mask_threshold"]
    failures: dict[str, list[str]] = {k: [] for k in (
        "files", "shape", "dtype", "bravo_range", "soft_range", "seg_binary", "seg_threshold",
        "depth", "csv_model_px", "csv_native_px", "patients_csv")}
    n_slices_by_pid = dict(zip(patients["patient_id"], patients["n_slices"]))

    for pid, orig_split in tqdm(list(zip(split_df["patient_id"].astype(str), split_df["orig_split"].astype(str))),
                                desc="Checking volumes"):
        vol_dir = out_dir / "volumes" / pid
        if not all((vol_dir / f).is_file() for f in VOLUME_FILES):
            failures["files"].append(pid)
            continue
        bravo, soft, seg = (np.asanyarray(load_image(vol_dir / f).dataobj) for f in VOLUME_FILES)

        in_plane = [s for i, s in enumerate(seg.shape) if i != axis]
        if not (bravo.shape == soft.shape == seg.shape and in_plane == list(prep["spatial_size"])):
            failures["shape"].append(pid)
        if not (bravo.dtype == np.float32 and soft.dtype == np.float32 and seg.dtype == np.uint8):
            failures["dtype"].append(pid)
        if bravo.min() < 0 or bravo.max() > 1:
            failures["bravo_range"].append(pid)
        if soft.min() < 0 or soft.max() > 1:
            failures["soft_range"].append(pid)
        if not np.isin(seg, (0, 1)).all():
            failures["seg_binary"].append(pid)
        if not np.array_equal(seg, (soft > thr).astype(np.uint8)):
            failures["seg_threshold"].append(pid)
        if seg.shape[axis] != n_slices_by_pid.get(pid):
            failures["depth"].append(pid)

        # lesion_slices.csv must match pixel counts recomputed from the saved and the raw masks
        rows = slices[slices["patient_id"] == pid]
        csv_model, csv_native = np.zeros(seg.shape[axis], int), np.zeros(seg.shape[axis], int)
        csv_model[rows["z"]], csv_native[rows["z"]] = rows["lesion_px_model"], rows["lesion_px_native"]
        px_model = lesion_px_per_slice(seg, axis)
        _, mask_path = patient_files(dataset_cfg, orig_split, pid)
        px_native = lesion_px_per_slice(np.asanyarray(load_image(mask_path).dataobj), axis)
        if not np.array_equal(csv_model, px_model):
            failures["csv_model_px"].append(pid)
        if px_native.shape != csv_native.shape or not np.array_equal(csv_native, px_native):
            failures["csv_native_px"].append(pid)
        p_rows = patients[patients["patient_id"] == pid]
        if p_rows.empty:
            failures["patients_csv"].append(pid)
            continue
        p = p_rows.iloc[0]
        if (p["n_lesion_slices_model"] != (px_model > 0).sum() or p["lesion_px_model"] != px_model.sum()
                or p["n_lesion_slices_native"] != (px_native > 0).sum() or p["lesion_px_native"] != px_native.sum()):
            failures["patients_csv"].append(pid)

    labels = {
        "files": "all three volume files present",
        "shape": f"shapes equal, in-plane {prep['spatial_size']}",
        "dtype": "dtypes bravo/seg_soft float32, seg uint8",
        "bravo_range": "bravo within [0, 1]",
        "soft_range": "seg_soft within [0, 1]",
        "seg_binary": "seg only 0/1",
        "seg_threshold": f"seg == (seg_soft > {thr}) exactly",
        "depth": "depth == native number of slices",
        "csv_model_px": "lesion_slices.csv model px == recount from seg",
        "csv_native_px": "lesion_slices.csv native px == recount from raw mask",
        "patients_csv": "patients.csv totals == recount",
    }
    for key, label in labels.items():
        report.add(f"real: {label}", not failures[key], _examples(failures[key]))

    lost = slices[(slices["lesion_px_model"] == 0) & (slices["lesion_px_native"] > 0)]
    gained = slices[(slices["lesion_px_model"] > 0) & (slices["lesion_px_native"] == 0)]
    report.add("real: no lesion slice lost or gained by conversion", lost.empty and gained.empty,
               f"{len(slices)} lesion slices; lost {len(lost)}, gained {len(gained)}")


# --- independent re-computation --------------------------------------------------------------

def reference_model_space(path: str | Path, prep: dict) -> np.ndarray:
    """Model-space conversion without MONAI: nibabel load, per-volume min-max, torch area resize.

    Independent of memtk.transforms, to confirm the MONAI chain does what we believe it does
    (no re-orientation, scaling over the whole volume, area interpolation, depth kept).
    """
    vol = np.asanyarray(load_image(path).dataobj).astype(np.float32)
    vol = (vol - vol.min()) / (vol.max() - vol.min())
    size = list(prep["spatial_size"])
    size.insert(prep["slice_axis"], vol.shape[prep["slice_axis"]])
    out = F.interpolate(torch.from_numpy(vol)[None, None], size=tuple(size), mode="area")
    return out[0, 0].numpy()


def check_recompute(report: CheckReport, split_df: pd.DataFrame, dataset_cfg: dict, out_dir: Path,
                    prep: dict, patient_ids: list[str], atol: float = 1e-5) -> None:
    """Saved volumes vs a fresh MONAI run (exact) and vs the independent reference (to atol)."""
    thr = prep["mask_threshold"]
    orig_split = dict(zip(split_df["patient_id"], split_df["orig_split"]))
    not_exact, ref_max_diff, ref_seg_issues = [], 0.0, []
    for pid in patient_ids:
        image_path, mask_path = patient_files(dataset_cfg, orig_split[pid], pid)
        saved = [np.asanyarray(load_image(out_dir / "volumes" / pid / f).dataobj) for f in VOLUME_FILES]
        fresh = convert_volume_pair(image_path, mask_path, prep)
        if not all(np.array_equal(a, b) for a, b in zip(saved, fresh)):
            not_exact.append(pid)

        ref_bravo, ref_soft = reference_model_space(image_path, prep), reference_model_space(mask_path, prep)
        ref_max_diff = max(ref_max_diff, float(np.abs(ref_bravo - saved[0]).max()),
                           float(np.abs(ref_soft - saved[1]).max()))
        # a binary pixel may only differ where the soft value sits within atol of the threshold
        differs = (ref_soft > thr) != (saved[2] == 1)
        if (np.abs(saved[1][differs] - thr) >= atol).any():
            ref_seg_issues.append(pid)

    report.add("recompute: saved volumes == fresh MONAI run (bit-exact)", not not_exact,
               _examples(not_exact) or f"patients {patient_ids}")
    report.add(f"recompute: saved volumes == independent nibabel+torch reference (atol {atol})",
               ref_max_diff < atol, f"max abs difference {ref_max_diff:.2e}")
    report.add("recompute: binary masks agree with the reference", not ref_seg_issues, _examples(ref_seg_issues))


# --- synthetic index -------------------------------------------------------------------------

def check_synthetic_index(report: CheckReport, index_df: pd.DataFrame, pool_cfg: dict,
                          spatial_size: list[int], rng: np.random.Generator, n_recompute: int) -> None:
    files = list_pool_files(pool_cfg)
    report.add("synthetic: index rows == files in pool folder",
               set(index_df["filename"]) == {f.name for f in files} and len(index_df) == len(files),
               f"{len(index_df)} rows, {len(files)} files")
    report.add("synthetic: sample IDs unique", not index_df["sample_id"].duplicated().any())
    report.add("synthetic: all files readable as (H, W, channels)", index_df["read_ok"].all(),
               _examples(index_df.loc[~index_df["read_ok"], "sample_id"].tolist()))
    # content checks on readable files only, so an unreadable file is reported once, above
    readable = index_df[index_df["read_ok"]]
    report.add(f"synthetic: all readable samples {spatial_size[0]}x{spatial_size[1]}", readable["shape_ok"].all(),
               _examples(readable.loc[~readable["shape_ok"], "sample_id"].tolist()))
    report.add("synthetic: all readable masks binary", readable["mask_binary"].all(),
               _examples(readable.loc[~readable["mask_binary"], "sample_id"].tolist()))
    report.add("synthetic: no empty masks", (readable["mask_px"] > 0).all(),
               _examples(readable.loc[readable["mask_px"] == 0, "sample_id"].tolist()))

    legacy_file = pool_cfg.get("legacy_exclusion_file")
    if legacy_file:
        legacy = read_legacy_exclusions(legacy_file)
        flagged = set(index_df.loc[index_df["legacy_excluded"], "sample_id"])
        report.add("synthetic: legacy exclusion flags == legacy list", flagged == set(legacy),
                   f"{len(legacy)} entries, {len(flagged)} flagged")

    # spot-check stored values against the files
    pool_dir = resolve_path(pool_cfg["dir"])
    sample = readable.iloc[rng.choice(len(readable), size=min(n_recompute, len(readable)), replace=False)]
    mismatches = []
    for sid, fname, px, ncomp, img_max in zip(sample["sample_id"], sample["filename"], sample["mask_px"],
                                              sample["n_components_8conn"], sample["img_max"]):
        image, mask = load_sample(pool_dir / fname, pool_cfg)
        if (int((mask > 0).sum()) != px or count_components(mask) != ncomp
                or not np.isclose(float(image.max()), img_max)):
            mismatches.append(sid)
    report.add("synthetic: stored values == recomputed from files", not mismatches,
               _examples(mismatches) or f"{len(sample)} random samples")


# --- visual check ----------------------------------------------------------------------------

def sample_grid(lesion_slices: pd.DataFrame, out_dir: Path, index_df: pd.DataFrame, pool_cfg: dict,
                prep: dict, rng: np.random.Generator, n_cols: int, path: Path) -> None:
    """Random lesion slices of each real set and random synthetic samples, with mask outlines.

    All images use the same display range [0, 1], so intensity differences are visible.
    """
    sets = list(dict.fromkeys(lesion_slices["set"]))  # keeps T, H-cal, H-eval in file order
    fig = Figure(figsize=(2.2 * n_cols, 2.4 * (len(sets) + 1)))
    axes = fig.subplots(len(sets) + 1, n_cols, squeeze=False)
    axis = prep["slice_axis"]

    for r, set_name in enumerate(sets):
        rows = lesion_slices[lesion_slices["set"] == set_name]
        picks = rows.iloc[rng.choice(len(rows), size=n_cols, replace=False)]
        for c, (pid, z) in enumerate(zip(picks["patient_id"], picks["z"])):
            vol_dir = out_dir / "volumes" / pid
            bravo = np.take(np.asanyarray(load_image(vol_dir / "bravo.nii.gz").dataobj), z, axis=axis)
            seg = np.take(np.asanyarray(load_image(vol_dir / "seg.nii.gz").dataobj), z, axis=axis)
            _show(axes[r, c], bravo, seg, f"{set_name} {pid} z={z}")

    pool_dir = resolve_path(pool_cfg["dir"])
    readable = index_df[index_df["read_ok"]]
    picks = readable.iloc[rng.choice(len(readable), size=n_cols, replace=False)]
    for c, (sid, fname) in enumerate(zip(picks["sample_id"], picks["filename"])):
        image, mask = load_sample(pool_dir / fname, pool_cfg)
        _show(axes[-1, c], image, mask, f"S {sid}")

    fig.suptitle("Model space (128x128): real lesion slices per set and synthetic samples; red = mask",
                 fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)


def _show(ax, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if mask.any():
        ax.contour(mask > 0, levels=[0.5], colors="r", linewidths=0.6)
    ax.set_title(title, fontsize=7)
    ax.axis("off")
