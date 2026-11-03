"""E0b lesion stages: what counts as a lesion, the lesion tables, and looking at single slices.

- `run_lesions_review` (decisions B, C): minimum lesion area and attach gap, from T.
- `run_lesions` (needs B, C): lesion table, slice table and label maps for every real lesion
  slice and every synthetic mask; label volumes for ITK-SNAP.
- `run_view` (helper): one slice at full size with its lesion rows.

Below the stages: the labelling collector, review tables, the production summary and figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from tqdm import tqdm

from memtk import config, ids
from memtk.e0b.common import Slices, decided, load_real, pick, refuse_overwrite, require
from memtk.io import LabelStack, LabelStackWriter, MaskStack, read_table, save_nifti
from memtk.lesions import (LesionLabels, attach_or_drop, chessboard_distance, fraction_outside, label_components,
                           lesion_features, speck_gaps)
from memtk.plotting import (HIGHLIGHT_COLOR, LESION_COLOR, draw_lesions, hide_unused, lesion_sheet, save, show_slice,
                            tile_figure)
from memtk.synthetic import load_sample


# --- stages -----------------------------------------------------------------------------------

def run_lesions_review(cfg: dict, dataset: str | None = None, pool: str | None = None,
                       overwrite: bool = False) -> None:
    """Evidence for decisions B and C (`lesions.min_lesion_area`, `lesions.attach_gap`), proposal
    from T only: component sizes of every real lesion slice and every synthetic mask as saved,
    attach-or-drop counts at the proposed minimum area +-1 and gaps 0-2, specks by gap, crop sheets
    and a list of the smallest T components. Needs the `brain` stage (masks_raw.npz). Writes
    `results/<dataset>/e0b/lesions_review/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, name, "e0b") / "lesions_review"
    refuse_overwrite([out], overwrite)
    require(syn_dir / "masks_raw.npz", "brain")
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
            for k, (g, m) in masks.items() for s, r, c in component_table(m)]
    for sid in tqdm(raw.keys, desc=f"{pool_name} masks"):
        rows += [{"group": "S", "key": sid, "size": s, "row": r, "col": c} for s, r, c in component_table(raw[sid])]
    components = pd.DataFrame(rows)

    # decision B from T only; H-cal, H-eval and S are described with the same settings
    t_sizes = components.loc[components["group"] == "T", "size"]
    m, m_reason = propose_min_area(t_sizes)
    min_areas = [a for a in (m - 1, m, m + 1) if a >= 1]
    gaps = (0, 1, 2)

    # specks per minimum area, with their gap to the nearest lesion; masks without a component
    # below the area have no specks and are skipped
    smallest = components.groupby("key", sort=False)["size"].min()
    speck_rows = []
    for a in min_areas:
        for key in tqdm(smallest.index[smallest < a], desc=f"specks at min area {a}"):
            group, mask = masks[key] if key in masks else ("S", raw[key])
            speck_rows += mask_specks(group, key, mask, a)
    specks = pd.DataFrame(speck_rows, columns=["group", "key", "min_area", "size", "gap", "row", "col"])
    sizes_table = size_counts(components)
    sensitivity = summarize_attach_or_drop(components, specks, min_areas, gaps)
    gap_table = gap_counts(specks[specks["min_area"] == m])

    # figures: size distribution; the smallest real components; specks at gap 1-3 from a lesion

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

    size_figure(components, m, out / f"component_sizes_{pool_name}.png",
                f"Component sizes (8-connected) of real lesion slices per set and synthetic masks as saved; "
                f"dashed: proposed minimum lesion area {m} px")
    t_comp = components[components["group"] == "T"]
    items = []
    for s in range(1, m + 2):
        items += crops(pick(rng, t_comp[t_comp["size"] == s], 8),
                       lambda rec: f"{rec.key}\n{rec.size} px at ({rec.row}, {rec.col})")
    crop_sheet(items, out / "small_components_T.png",
               f"T: the smallest real components, one row per size 1-{m + 1} px (8 random each); "
               f"32x32 crops of the 128x128 slice centred on the component (lime); other lesions of the "
               f"slice inside the crop in red. (row, col) = model-space position of its first pixel")
    at_m = specks[specks["min_area"] == m]
    items = []
    for g in (1, 2, 3):
        items += crops(pick(rng, at_m[(at_m["group"] == "S") & (at_m["gap"] == g)], 8),
                       lambda rec: f"{rec.key}\n{rec.size} px, gap {rec.gap} at ({rec.row}, {rec.col})")
    items += crops(pick(rng, at_m[(at_m["group"] == "T") & at_m["gap"].between(1, 3)], 8),
                   lambda rec: f"T {rec.key}\n{rec.size} px, gap {rec.gap} at ({rec.row}, {rec.col})")
    crop_sheet(items, out / f"specks_by_gap_{pool_name}.png",
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
    small["gap_to_lesion"] = [gap_to_nearest_lesion(masks[k][1], r, c, m)
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


def run_lesions(cfg: dict, dataset: str | None = None, pool: str | None = None,
                overwrite: bool = False) -> None:
    """Attach-or-drop with the decided settings (decisions B, C) for every real lesion slice and every
    synthetic mask as saved: `lesions.csv` (one row per lesion: ID, features in three position
    frames), `slices.csv` (one row per slice), `labels.npz` per dataset and pool, a
    `labels.nii.gz` per real patient, and a summary with an overview sheet in
    `results/<dataset|pool>/e0b/lesions/`. Needs the `brain` stage."""
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    small_lesion_px = int(cfg["qc"]["small_lesion_px"])
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    real_res = config.results_dir(cfg, name, "e0b") / "lesions"
    syn_res = config.results_dir(cfg, pool_name, "e0b") / "lesions"
    refuse_overwrite([real_dir / "lesions.csv", syn_dir / "lesions.csv", real_res, syn_res], overwrite)
    for d in (real_dir, syn_dir):
        require(d / "brain_masks.npz", "brain")
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

    def sheet(src: Slices, picks: pd.DataFrame, path: Path, title: str) -> None:
        items = []
        for key, n_les, n_specks in picks[["key", "n_lesions", "n_specks"]].itertuples(index=False):
            image, mask, brain, box = src[str(key)]
            counts = f"{n_les} lesion{'' if n_les == 1 else 's'}, {n_specks} speck{'' if n_specks == 1 else 's'}"
            items.append((f"{key}\n{counts}", image, attach_or_drop(mask, min_area, gap), brain, box))
        lesion_sheet(items, path, title)

    legend = ("each lesion in its own colour with its number (L1 = largest), attached strays lighter, "
              "dropped specks x (orange), brain cyan, brain box dashed yellow")
    t = real_slices[real_slices["set"] == "T"]
    sheet(real_src, pd.concat([pick(rng, t, 16), t.nlargest(8, "n_lesions"),
                               *(pick(rng, real_slices[real_slices["set"] == s], 8) for s in ("H-cal", "H-eval"))]),
          real_res / "overview.png",
          f"{name}: rows 1-2 random T, row 3 T with the most lesions, row 4 H-cal, row 5 H-eval. {legend}")
    both = syn_slices[(syn_slices["n_strays"] > 0) & (syn_slices["n_specks"] > 0)]
    sheet(syn_src, pd.concat([pick(rng, syn_slices, 16), syn_slices.nlargest(8, "n_lesions"), pick(rng, both, 8)]),
          syn_res / "overview.png",
          f"{pool_name}: rows 1-2 random, row 3 the most lesions, row 4 with both strays and isolated specks. "
          f"{legend}")

    settings = (f"min_lesion_area {min_area} px, attach_gap {gap} px; a lesion counts as small below "
                f"{small_lesion_px} px")
    texts = []
    for res_dir, source, lesions, slices_df, by in (
            (real_res, name, real_lesions, real_slices, "set"),
            (syn_res, pool_name, syn_lesions.assign(group="S"), syn_slices.assign(group="S"), "group")):
        summary = summarize_lesions(slices_df, lesions, by)
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


VIEW_COLUMNS = {"lesion_id": "lesion", "area": "area px", "stray_px": "strays px", "centroid_row": "row",
                "centroid_col": "col", "centroid_row_brain": "row in brain", "centroid_col_brain": "col in brain",
                "eccentricity": "eccentr.", "solidity": "solidity", "frac_outside_brain": "outside brain"}


def run_view(cfg: dict, sample: str | None = None, patient: str | None = None, z: int | None = None,
             dataset: str | None = None, pool: str | None = None, nifti: bool = False,
             overwrite: bool = False) -> None:
    """One slice at full size, drawn like the lesion overview, with its lesion rows as a table:
    a synthetic sample (`sample`) or a real slice (`patient` and 0-based `z`). With `nifti`, a
    synthetic sample's image and labels are also written as NIfTI. Writes
    `results/<dataset|pool>/e0b/views/<key>.png`. Needs the `lesions` stage."""
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    if sample is not None:
        source, pool_cfg = config.synthetic_pool(cfg, pool)
        key = ids.sample_key(sample)
    elif patient is not None and z is not None:
        source, pool_cfg = config.real_dataset(cfg, dataset)[0], None
        key = ids.slice_key(patient, z)
    else:
        raise SystemExit("Give --sample <id>, or --patient <id> and --z <0-based slice index>.")
    out_dir = config.output_dir(cfg, source)
    require(out_dir / "lesions.csv", "lesions")
    src = Slices(out_dir, pool_cfg)
    if key not in src:
        raise SystemExit(f"'{key}' is not a {'sample' if pool_cfg else 'lesion slice'} of '{source}'.")
    path = config.results_dir(cfg, source, "e0b") / "views" / f"{key.replace('/', '_')}.png"
    refuse_overwrite([path], overwrite)

    image, mask, brain, box = src[key]
    res = attach_or_drop(mask, min_area, gap)
    if not np.array_equal(res.labels, LabelStack(out_dir / "labels.npz")[key]):
        print("Warning: labels differ from labels.npz; the lesion settings changed since the 'lesions' stage.")
    lesions = read_table(out_dir / "lesions.csv")
    table = lesions.loc[lesions["key"] == key, list(VIEW_COLUMNS)].rename(columns=VIEW_COLUMNS)
    lesion_view(image, res, brain, box, table, path,
                f"{source}: {key} | {res.n_lesions} lesions, {res.n_strays} strays ({int(res.attached.sum())} "
                f"px), {res.n_specks} isolated specks | L-number per lesion, lighter = attached stray, "
                f"orange x = dropped speck, cyan = brain, dashed yellow = brain box")
    print(f"Wrote {path}")
    if nifti:
        if pool_cfg is None:
            pid, slice_z = ids.parse_slice_key(key)
            print(f"Real slice: open {out_dir / 'volumes' / pid / 'labels.nii.gz'} next to bravo.nii.gz "
                  f"(ITK-SNAP slice {slice_z + 1}).")
        else:
            stem = path.with_suffix("")
            save_nifti(image[:, :, None].astype(np.float32), f"{stem}_image.nii.gz")
            save_nifti(res.labels[:, :, None].astype(np.uint16), f"{stem}_labels.nii.gz")
            print(f"Wrote {stem}_image.nii.gz and {stem}_labels.nii.gz (open the labels as segmentation)")


# --- labelling (production) -------------------------------------------------------------------

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


# --- lesions-review: tables -------------------------------------------------------------------

def component_table(mask: np.ndarray) -> list[tuple[int, int, int]]:
    """Every 8-connected component of a mask as (size, row, col) of its first pixel."""
    comp, n = label_components(mask)
    if n == 0:
        return []
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    labels, first = np.unique(comp.ravel(), return_index=True)
    width = mask.shape[1]
    return [(int(sizes[lab]), *divmod(int(f), width))  # pyright: ignore[reportReturnType]  # unpacked divmod is 2 ints
            for lab, f in zip(labels, first) if lab > 0]


def gap_to_nearest_lesion(mask: np.ndarray, row: int, col: int, min_area: int) -> int:
    """Gap (background px, Chebyshev) from the component containing (row, col) to the nearest
    other component of at least `min_area` px; -1 if there is none."""
    comp, n = label_components(mask)
    sizes = np.bincount(comp.ravel(), minlength=n + 1)
    sizes[0] = 0
    target = comp == comp[row, col]
    others = np.isin(comp, np.flatnonzero(sizes >= min_area)) & ~target
    if not others.any():
        return -1
    return int(chessboard_distance(others)[target].min()) - 1


def propose_min_area(t_sizes: pd.Series, max_share: float = 0.01) -> tuple[int, str]:
    """The largest minimum area m such that at most `max_share` of the real T components are
    smaller than m (they would count as specks)."""
    m = 1
    while (t_sizes < m + 1).mean() <= max_share:
        m += 1
    return m, (f"largest m with at most {max_share:.0%} of T components smaller than m "
               f"({(t_sizes < m).mean():.2%} are)")


def size_counts(components: pd.DataFrame, max_size: int = 10) -> pd.DataFrame:
    """Per group: number and share of components of each size 1..max_size, and larger."""
    sizes = components["size"].clip(upper=max_size + 1)
    table = pd.crosstab(components["group"], sizes).reindex(columns=range(1, max_size + 2), fill_value=0)
    table.columns = [*map(str, range(1, max_size + 1)), f">{max_size}"]
    table.insert(0, "components", table.sum(axis=1))
    return table


def mask_specks(group: str, key: str, mask: np.ndarray, min_area: int) -> list[dict]:
    """The specks of one mask at one minimum area, with their gap to the nearest lesion."""
    return [{"group": group, "key": key, "min_area": min_area, "size": s, "gap": g, "row": r, "col": c}
            for s, g, r, c in speck_gaps(mask, min_area)]


def summarize_attach_or_drop(components: pd.DataFrame, specks: pd.DataFrame, min_areas, gaps) -> pd.DataFrame:
    """Per group, minimum area and attach gap: lesions (which depend on the minimum area only),
    slices left without a lesion, stray components and pixels attached, isolated specks and the
    masks that have at least one. Gap 0 is drop-only."""
    rows = []
    per_mask = components.groupby(["group", "key"], sort=False)["size"]
    for m in min_areas:
        n_lesions = per_mask.agg(lambda s: int((s >= m).sum())).rename("n_lesions").reset_index()
        for group, nl in n_lesions.groupby("group", sort=False):
            sp = specks[(specks["group"] == group) & (specks["min_area"] == m)]
            for g in gaps:
                stray = (sp["gap"] >= 1) & (sp["gap"] <= g)
                rows.append({"group": group, "min_area": m, "gap": g, "slices": len(nl),
                             "lesions": int(nl["n_lesions"].sum()), "no_lesion": int((nl["n_lesions"] == 0).sum()),
                             "strays": int(stray.sum()), "stray_px": int(sp.loc[stray, "size"].sum()),
                             "isolated_specks": int((~stray).sum()),
                             "masks_with_isolated": int(sp.loc[~stray, "key"].nunique())})
    return pd.DataFrame(rows).set_index(["group", "min_area", "gap"])


def gap_counts(specks: pd.DataFrame, bins=((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10 ** 6))) -> pd.DataFrame:
    """Per group: specks by gap to the nearest lesion, and specks in masks without a lesion."""
    cols = {"no lesion": specks["gap"] < 0}
    for lo, hi in bins:
        name = str(lo) if lo == hi else (f"{lo}-{hi}" if hi < 10 ** 6 else f">={lo}")
        cols[f"gap {name}"] = specks["gap"].between(lo, hi)
    return pd.DataFrame(cols).groupby(specks["group"], sort=False).sum()


# --- lesions: summary -------------------------------------------------------------------------

def summarize_lesions(slices: pd.DataFrame, lesions: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per group (column `by` in both tables): slices and lesions, slices without a lesion, share
    of slices with a small lesion, lesion area percentiles, attached strays, isolated specks
    (and the share of them outside the brain), lesions more than half outside the brain."""
    s, les = slices.groupby(by, sort=False), lesions.groupby(by, sort=False)
    return pd.DataFrame({
        "slices": s.size(), "lesions": s["n_lesions"].sum(), "lesions_per_slice": s["n_lesions"].mean(),
        "no_lesion": s["n_lesions"].agg(lambda x: int((x == 0).sum())),
        "share_with_lesion_lt40": s["has_lesion_lt40"].mean(),
        "area_p5": les["area"].quantile(0.05), "area_median": les["area"].median(),
        "area_p95": les["area"].quantile(0.95),
        "strays": s["n_strays"].sum(), "stray_px": s["stray_px"].sum(),
        "specks": s["n_specks"].sum(), "slices_with_specks": s["n_specks"].agg(lambda x: int((x > 0).sum())),
        "specks_outside_brain_share": s["specks_outside_brain"].sum() / s["n_specks"].sum(),
        "lesions_over_half_outside": les["frac_outside_brain"].agg(lambda x: int((x > 0.5).sum())),
    })


# --- figures ----------------------------------------------------------------------------------

def size_figure(components: pd.DataFrame, min_area: int, path: Path, title: str, max_size: int = 30) -> Path:
    """Share of components by size (1..max_size, then larger) per group, log scale, with the
    proposed minimum lesion area marked."""
    fig = Figure(figsize=(9, 3.6))
    ax = fig.subplots()
    sizes = np.arange(1, max_size + 2)
    for k, (group, d) in enumerate(components.groupby("group", sort=False)):
        share = d["size"].clip(upper=max_size + 1).value_counts(normalize=True).reindex(sizes, fill_value=0)
        offset = sizes + (k - 1.5) * 0.2  # pyright: ignore[reportOperatorIssue]  # numpy stubs: array + float is valid
        ax.bar(offset, share.to_numpy(), width=0.2, label=f"{group} ({len(d)})")
    ax.axvline(min_area - 0.5, color="k", ls="--", lw=1)
    ax.text(min_area - 0.4, ax.get_ylim()[1] * 0.5, f" lesion >= {min_area} px", fontsize=8)
    ax.set_yscale("log")
    ax.set_xticks(sizes, [*map(str, sizes[:-1]), f">{max_size}"], fontsize=7)
    ax.set_xlabel("component size (px)")
    ax.set_ylabel("share of components (log)")
    ax.legend(fontsize=8)
    return save(fig, path, title)


def crop_sheet(items: list[tuple[str, np.ndarray, np.ndarray, int, int]], path: Path, title: str,
               size: int = 32, n_cols: int = 8) -> Path:
    """Tiles of (title, image, mask, row, col): a size x size crop centred on (row, col), with the
    component containing that pixel outlined in lime and the rest of the mask in red."""
    n_rows = int(np.ceil(len(items) / n_cols))
    fig, axes = tile_figure(n_rows, n_cols, tile=1.9)
    h = size // 2
    for ax, (tile_title, image, mask, r, c) in zip(axes.ravel(), items):
        comp, _ = label_components(mask)
        target = comp == comp[r, c]

        def crop(a: np.ndarray) -> np.ndarray:
            return np.pad(a, h)[r:r + size, c:c + size]

        show_slice(ax, crop(image), [(crop((mask > 0) & ~target), LESION_COLOR), (crop(target), HIGHLIGHT_COLOR)],
                   tile_title)
    hide_unused(axes, len(items))
    return save(fig, path, title)


def lesion_view(image: np.ndarray, res: LesionLabels, brain: np.ndarray, box: tuple | None, table: pd.DataFrame,
                path: Path, title: str) -> Path:
    """One slice at full size (as `draw_lesions`), with its lesion rows as a table below."""
    fig = Figure(figsize=(8, 8 + 0.25 * (len(table) + 2)))
    img_ax, table_ax = fig.subplots(2, 1, height_ratios=[8, 0.25 * (len(table) + 2)])
    draw_lesions(img_ax, image, res, brain, box, "", fontsize=9)
    table_ax.axis("off")
    if len(table):
        cells = table.round(2).astype(str).to_numpy().tolist()
        t = table_ax.table(cellText=cells, colLabels=list(table.columns), loc="upper center", cellLoc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(7)
        t.auto_set_column_width(list(range(len(table.columns))))
    return save(fig, path, title)
