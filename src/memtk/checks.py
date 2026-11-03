"""Consistency checks for the E0 outputs.

E0a: patient split, model-space volumes, synthetic index. E0b: lesion tables against label maps,
label maps against the masks as saved, QC classes, cached stacks, and the run logs of the
production stages against the current config.

Every check is recorded in a CheckReport instead of stopping at the first failure, so one run
shows everything that is wrong. Checks are generic (no hardcoded patient or sample counts);
the counts are printed in the report for manual comparison. The E0b checks take the stacks as
anything indexable by key, so tests can run them on small hand-built data.
"""

from __future__ import annotations

import json
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

from memtk import qc
from memtk.config import resolve_path, value
from memtk.ids import parse_lesion_id
from memtk.io import list_patients, load_image, patient_files
from memtk.lesions import attach_or_drop, count_components, label_components
from memtk.plotting import save
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

    save(fig, path, "Model space (128x128): real lesion slices per set and synthetic samples; red = mask")


def _show(ax, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    if mask.any():
        ax.contour(mask > 0, levels=[0.5], colors="r", linewidths=0.6)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


# --- E0b: lesion tables and label maps -------------------------------------------------------

def check_lesion_tables(report: CheckReport, source: str, lesions: pd.DataFrame, slices: pd.DataFrame,
                        labels) -> None:
    """lesions.csv and slices.csv of one source against its label maps (`labels[key]`): IDs,
    lesion counts, labels 1..n per slice, areas, centroids and the numbering rule (larger area
    first; equal areas: the lesion whose first pixel comes first in row-major order)."""
    report.add(f"{source}: lesion IDs unique", lesions["lesion_id"].is_unique,
               _examples(lesions.loc[lesions["lesion_id"].duplicated(), "lesion_id"].tolist()))
    bad_ids = []
    for lid, key, label in zip(lesions["lesion_id"], lesions["key"], lesions["label"]):
        try:
            if parse_lesion_id(lid) != (key, int(label)):
                bad_ids.append(lid)
        except ValueError:
            bad_ids.append(lid)
    report.add(f"{source}: every lesion ID parses to its slice key and lesion number", not bad_ids, _examples(bad_ids))
    report.add(f"{source}: one label map per slice", set(labels.keys) == set(slices["key"]) and len(labels) == len(slices),
               f"{len(labels)} maps, {len(slices)} slices")
    per_slice = lesions.groupby("key").size().reindex(slices["key"], fill_value=0).to_numpy()
    wrong_count = slices.loc[per_slice != slices["n_lesions"].to_numpy(), "key"].tolist()
    report.add(f"{source}: lesion rows per slice == n_lesions in slices.csv", not wrong_count, _examples(wrong_count))

    by_key = {key: g.sort_values("label") for key, g in lesions.groupby("key")}
    bad = {"labels": [], "areas": [], "centroids": [], "numbering": []}
    grid = None
    for key in slices["key"]:
        flat = labels[key].ravel()
        if grid is None:
            grid = np.indices(labels[key].shape).reshape(2, -1).astype(float)
        n = int(flat.max())
        g = by_key.get(key)
        rows = g["label"].tolist() if g is not None else []
        if sorted(np.unique(flat[flat > 0]).tolist()) != list(range(1, n + 1)) or rows != list(range(1, n + 1)):
            bad["labels"].append(key)
            continue
        if g is None:
            continue
        areas = np.bincount(flat, minlength=n + 1)[1:]
        centroid_row = np.bincount(flat, weights=grid[0], minlength=n + 1)[1:] / areas
        centroid_col = np.bincount(flat, weights=grid[1], minlength=n + 1)[1:] / areas
        if not np.array_equal(areas, g["area"].to_numpy()):
            bad["areas"].append(key)
        if not (np.allclose(centroid_row, g["centroid_row"]) and np.allclose(centroid_col, g["centroid_col"])):
            bad["centroids"].append(key)
        values, first = np.unique(flat, return_index=True)
        first = first[values > 0]                         # first pixel of labels 1..n, in label order
        if any(areas[i] < areas[i + 1] or (areas[i] == areas[i + 1] and first[i] > first[i + 1])
               for i in range(n - 1)):
            bad["numbering"].append(key)
    report.add(f"{source}: label maps hold labels 1..n_lesions, one row each", not bad["labels"], _examples(bad["labels"]))
    report.add(f"{source}: lesion areas == label map", not bad["areas"], _examples(bad["areas"]))
    report.add(f"{source}: lesion centroids == label map", not bad["centroids"], _examples(bad["centroids"]))
    report.add(f"{source}: numbering: larger area first, ties top-left first", not bad["numbering"],
               _examples(bad["numbering"]))


def check_labels_vs_masks(report: CheckReport, source: str, masks, labels, keys, min_area: int, gap: int) -> None:
    """Label maps against the masks as saved (`masks[key]`): lesion pixels lie inside the mask,
    every mask pixel left unlabelled belongs to a speck (< min_area), and attach-or-drop with the
    current settings reproduces the stored map exactly."""
    outside, big_unlabelled, differs = [], [], []
    for key in keys:
        mask, lab = masks[key] > 0, labels[key]
        if ((lab > 0) & ~mask).any():
            outside.append(key)
        comp, n = label_components(mask & ~(lab > 0))
        if n and (np.bincount(comp.ravel())[1:] >= min_area).any():
            big_unlabelled.append(key)
        if not np.array_equal(attach_or_drop(mask, min_area, gap).labels, lab):
            differs.append(key)
    report.add(f"{source}: lesion pixels lie inside the mask as saved", not outside, _examples(outside))
    report.add(f"{source}: unlabelled mask pixels are specks only (< {min_area} px)", not big_unlabelled,
               _examples(big_unlabelled))
    report.add(f"{source}: attach-or-drop ({min_area} px, gap {gap}) reproduces every label map", not differs,
               _examples(differs))


# --- E0b: QC classes, cached stacks, run logs ------------------------------------------------

def check_qc_table(report: CheckReport, table: pd.DataFrame, sample_ids, max_specks: int) -> None:
    """qc.csv: one row per sample, known classes, classes reproduced from the counts, counts add up."""
    report.add("qc: one row per synthetic sample", table["sample_id"].is_unique and set(table["sample_id"]) == set(sample_ids),
               f"{len(table)} rows, {len(sample_ids)} samples")
    report.add("qc: every class is known", table["qc_class"].isin(qc.QC_CLASSES).all()
               and table["mask_class"].isin(qc.MASK_CLASSES).all())
    masks = qc.mask_class(table["n_lesions"], table["n_specks"], max_specks)
    combined = qc.qc_class(table["mri_failed"].astype(bool), masks)
    wrong = table.loc[(masks != table["mask_class"]) | (combined != table["qc_class"]), "sample_id"].tolist()
    report.add(f"qc: classes reproduced from lesion and speck counts (cut-off {max_specks})", not wrong, _examples(wrong))
    counts = table["qc_class"].value_counts()
    report.add("qc: class counts add up to the pool", int(counts.sum()) == len(sample_ids),
               ", ".join(f"{c} {int(counts.get(c, 0))}" for c in qc.QC_CLASSES))


def check_label_volumes(report: CheckReport, out_dir: Path, slices: pd.DataFrame, labels) -> None:
    """Every real `volumes/<pid>/labels.nii.gz` equals the label stack on its lesion slices and
    is zero on all other slices."""
    bad = []
    for pid, group in slices.groupby("patient_id"):
        volume = np.asanyarray(load_image(out_dir / "volumes" / str(pid) / "labels.nii.gz").dataobj)
        expected = np.zeros(volume.shape, dtype=np.int64)
        for key, z in zip(group["key"], group["z"]):
            expected[:, :, int(z)] = labels[key]
        if not np.array_equal(volume, expected):
            bad.append(str(pid))
    report.add("real: labels.nii.gz == label stack (lesion slices) and 0 elsewhere", not bad, _examples(bad))


def check_stack_keys(report: CheckReport, name: str, stack, expected) -> None:
    """A cached stack holds exactly the expected keys."""
    missing, extra = set(expected) - set(stack.keys), set(stack.keys) - set(expected)
    report.add(f"{name}: keys == expected slices/samples", not missing and not extra,
               f"{len(stack)} keys" + (f"; missing {sorted(missing)[:5]}" if missing else "")
               + (f"; extra {sorted(extra)[:5]}" if extra else ""))


def check_run_logs(report: CheckReport, cfg: dict, logs: dict[str, tuple[Path, list[str]]]) -> None:
    """Production run logs against the current config. `logs` maps a stage name to (its run log,
    the config keys its outputs depend on), in run order: every key must equal the current value
    (outputs from older settings are stale), and each stage must have run after the one before."""
    previous = None
    for stage, (path, keys) in logs.items():
        if not path.exists():
            report.add(f"run log of '{stage}' exists", False, str(path))
            previous = None
            continue
        info = json.loads(path.read_text(encoding="utf-8"))
        logged = info["config"]
        stale = [f"{k}: run {value(logged, k)!r}, now {value(cfg, k)!r}" for k in keys
                 if value(logged, k) != value(cfg, k)]
        report.add(f"'{stage}' outputs made with the current config values", not stale,
                   "; ".join(stale) if stale else ", ".join(keys))
        if previous is not None:
            report.add(f"'{stage}' ran after '{previous[0]}'", str(info["timestamp"]) >= previous[1],
                       f"{info['timestamp']} vs {previous[1]}")
        previous = (stage, str(info["timestamp"]))
