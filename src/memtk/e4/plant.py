"""E4 stage `plant` (needs decision I; evidence for decision K): the planted copies.

Draws the base masks from T (`memtk.planted.choose_bases`) and makes every variant of each
(`memtk.planted.variants_of`): P1 exact, P2 / P3 shift, P4 rotation, P5 erosion and dilation,
P6-remove, and P6-add with an informative H-eval lesion. Planted masks are labelled like real
masks and each planted lesion is traced to its source lesion. No detector runs, so the go/no-go
criteria (decision K) can be set from what the variants do before anything is scored. Settings:
the informative-lesion bound and minimum lesion area from the frozen E3 file, the attach gap
(decision C) from the config as E0b used it, the planting design from `validation` (decision I).
Writes the planted set to `<output_root>/<dataset>/e4/` and the review (tables, figures,
`review.txt`) to `results/<dataset>/e4/plant/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml
from matplotlib import patheffects
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from memtk import calibration, config, lesionmatch
from memtk.e4.common import STREAMS, load_frozen, slice_images, streams
from memtk.ids import lesion_id, parse_lesion_id, parse_slice_key
from memtk.io import LabelStackWriter, MaskStack
from memtk.planted import VARIANTS, Planted, choose_bases, variants_of
from memtk.plotting import HIGHLIGHT_COLOR, LESION_COLOR, LESION_COLORS, draw_box, hide_unused, save, show_slice, tile_figure
from memtk.sets import SliceSet, load_set
from memtk.stages import decided, pick, refuse_overwrite

# Lower edges of the base lesion area bins (informative lesions; 12 px = the frozen bound)
SIZE_BINS = (12, 20, 40, 80)
PARAM_COLUMNS = ("shift_row", "shift_col", "angle", "removed", "added", "tries")
GRID_BASES = 12          # bases in the variants grid (half single-, half multi-lesion)
SHEET_BASES = 20         # bases per variant sheet
MARGIN, MIN_CROP = 6, 24  # px around the lesions in zoomed tiles; smallest zoomed window
ORIGINAL_COLOR, PLANTED_COLOR, DONOR_COLOR = LESION_COLOR, HIGHLIGHT_COLOR, "orange"  # orange: the zoom box is yellow
MORPH_VARIANTS = ("P4", "P5-erode", "P5-dilate")


# --- stages -----------------------------------------------------------------------------------

def run_plant(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The planted set with its provenance and the review for decision K (`validation.criteria`).
    Needs the frozen E3 file, decision I (`validation.bases`, `validation.variants`) and the E0b
    `lesions` and `brain` outputs. Writes `<output_root>/<dataset>/e4/` (`planted_labels.npz`,
    `planted.csv`, `planted_lesions.csv`) and `results/<dataset>/e4/plant/`."""
    name, _ = config.real_dataset(cfg, dataset)
    data_dir = config.output_dir(cfg, name) / "e4"
    out = config.results_dir(cfg, name, "e4") / "plant"
    refuse_overwrite([out, data_dir / "planted_labels.npz"], overwrite)
    frozen, frozen_hash = load_frozen(cfg, name)

    ps = plant_set(cfg, name, frozen)
    planted = planted_table(ps)
    lesions = planted_lesion_table(ps)
    fate = lesion_fate(ps)
    summary = variant_summary(planted, fate, ps)
    change = area_change(fate)

    data_dir.mkdir(parents=True, exist_ok=True)
    writer = LabelStackWriter(ps.planted[0].labels.shape)
    for p in ps.planted:
        writer.add(p.key, p.labels)
    writer.save(data_dir / "planted_labels.npz")
    planted.to_csv(data_dir / "planted.csv", index=False)
    lesions.to_csv(data_dir / "planted_lesions.csv", index=False)
    run_extra = {"stage": "plant", "inputs": ["T", "H-eval"], "frozen_file": config.frozen_file(cfg, name).name,
                 "frozen_sha256": frozen_hash, "streams": list(STREAMS)}
    config.write_run_info(data_dir, cfg, script=__file__, filename="plant.run_info.json", extra=run_extra)

    out.mkdir(parents=True, exist_ok=True)
    ps.bases.to_csv(out / "bases.csv", index=False)
    summary.to_csv(out / "variant_summary.csv", index=False)
    change.to_csv(out / "area_change.csv", index=False)
    fate.to_csv(out / "lesion_fate.csv", index=False)
    ps.skipped.to_csv(out / "skipped_variants.csv", index=False)
    images = slice_images(cfg, name, ps.bases["key"])
    rng = np.random.default_rng(cfg["seed"])
    variants_grid(ps, images, rng, out / "variants_grid.png",
                  f"{name}: planted variants of {GRID_BASES} random bases (rows), zoomed; red outline = the base "
                  f"mask, green outline = the planted mask; filled red = a base lesion with no planted lesion traced "
                  f"to it (removed or eroded away), filled orange = an added H-eval lesion")
    for variant in VARIANTS:
        variant_sheet(ps, images, variant, rng, out / f"variant_{variant}.png",
                      f"{name}: {variant} for {SHEET_BASES} random bases. Per base: the planted mask on the slice "
                      f"(zoom box dashed), the base lesions, and the planted lesions filled in the colour of the "
                      f"lesion they come from (orange = added H-eval lesion, grey = no source) over the base lesions "
                      f"dashed in their colour; dot and arrow = a lesion centre that moved by 1 px or more")
    area_figure(fate, ps.bound, out / "area_change.png",
                f"{name}: area of every informative base lesion before and after P4, P5-erode and P5-dilate "
                f"(largest planted piece traced to it; x at the bottom = gone, or merged into a neighbouring lesion "
                f"that is traced to the neighbour); dashed = the {ps.bound} px bound")
    text = review_text(name, ps, summary, change, config.value(cfg, "validation.criteria"))
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra=run_extra)
    print(f"\n{text}\nWrote {data_dir}\nWrote {out}")


# --- the planted set --------------------------------------------------------------------------

@dataclass
class PlantedSet:
    """The bases and their planted variants, with what could not be made."""
    bases: pd.DataFrame                # key, patient_id, z, n_informative, group, stratum
    skipped_patients: list[str]        # drawn but unable to meet their quota with the z spacing
    planted: list[Planted]
    skipped: pd.DataFrame              # base_key, variant, reason
    base_labels: dict[str, np.ndarray]
    source_areas: dict[str, int]       # lesion ID (base or H-eval donor) -> area
    n_donors: int
    bound: int


def plant_set(cfg: dict, name: str, frozen: dict) -> PlantedSet:
    """Bases and variants from T and H-eval (decision I), with the bound and minimum lesion area
    of the frozen file and the attach gap of decision C. Deterministic for a given seed."""
    bound, min_area = int(frozen["informative_bound_px"]), int(frozen["min_lesion_area"])
    gap = int(decided(cfg, "lesions.attach_gap", "e0b_audit.py lesions-review"))
    bases_p = decided(cfg, "validation.bases", "docs/e4-plan.md (decision I)")
    variants_p = decided(cfg, "validation.variants", "docs/e4-plan.md (decision I)")
    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    rng = streams(cfg["seed"])

    table = pd.DataFrame({"key": t.keys, "patient_id": t.patients, "z": [parse_slice_key(k)[1] for k in t.keys],
                          "n_informative": lesionmatch.lesion_records(t.labels, t.keys, bound).n_informative})
    bases, skipped_patients = choose_bases(table, bases_p, rng["bases"])
    bases["stratum"] = calibration.stratum(bases["n_informative"].to_numpy())
    donors = informative_lesions(heval, bound)
    brains = MaskStack(config.output_dir(cfg, name) / "brain_masks.npz")
    labels = dict(zip(t.keys, t.labels))

    planted, skipped = [], []
    for key in bases["key"]:
        made, missing = variants_of(key, labels[key], brains[key], donors, variants_p, bound, min_area, gap, rng)
        planted += made
        skipped += [{"base_key": key, "variant": v, "reason": r} for v, r in missing.items()]
    base_labels = {k: labels[k] for k in bases["key"]}
    areas = {lid: int(m.sum()) for lid, m in donors}
    for key, lab in base_labels.items():
        areas.update({lesion_id(key, n): int(a) for n, a in enumerate(np.bincount(lab.ravel())) if n})
    return PlantedSet(bases, skipped_patients, planted, pd.DataFrame(skipped, columns=["base_key", "variant", "reason"]),
                      base_labels, areas, len(donors), bound)


def informative_lesions(s: SliceSet, bound: int) -> list[tuple[str, np.ndarray]]:
    """Every lesion of a set with an area of at least `bound`, as (lesion ID, boolean mask): the
    donors of P6-add."""
    out = []
    for key, lab in zip(s.keys, s.labels):
        for n, area in enumerate(np.bincount(lab.ravel())):
            if n and area >= bound:
                out.append((lesion_id(key, n), lab == n))
    return out


# --- tables -----------------------------------------------------------------------------------

def stratum_or_none(n: np.ndarray) -> np.ndarray:
    """The stratum by lesion count, "none" for a mask without such lesions (not scorable)."""
    return np.where(np.asarray(n) == 0, "none", calibration.stratum(n))


def planted_table(ps: PlantedSet) -> pd.DataFrame:
    """One row per planted mask: key, variant, base and its patient, group and stratum, lesion
    counts (E1: lesions >= the minimum area; E2: informative), strata, whether every informative
    base lesion is still a lesion >= the bound, and the variant's parameters."""
    base = ps.bases.set_index("key").to_dict("index")
    rows = []
    for p in ps.planted:
        areas = np.bincount(p.labels.ravel())[1:]
        b = base[p.base_key]
        rows.append({"key": p.key, "variant": p.variant, "base_key": p.base_key, "patient_id": b["patient_id"],
                     "z": int(b["z"]), "group": b["group"], "base_n_informative": int(b["n_informative"]),
                     "base_stratum": b["stratum"], "n_lesions": int((areas > 0).sum()),
                     "n_informative": int((areas >= ps.bound).sum()), "keeps_base_lesions": p.keeps_base_lesions,
                     **{c: p.params.get(c) for c in PARAM_COLUMNS}})
    out = pd.DataFrame(rows)
    out.insert(10, "stratum_e1", stratum_or_none(out["n_lesions"].to_numpy()))
    out.insert(11, "stratum_e2", stratum_or_none(out["n_informative"].to_numpy()))
    return out


def planted_lesion_table(ps: PlantedSet) -> pd.DataFrame:
    """One row per planted lesion: its ID, planted mask, variant, area, and the lesion it comes
    from (a base lesion, or the H-eval lesion of P6-add; "" if none) with that lesion's area."""
    rows = []
    for p in ps.planted:
        areas = np.bincount(p.labels.ravel())
        for n, source in p.sources.items():
            rows.append({"lesion_id": lesion_id(p.key, n), "key": p.key, "variant": p.variant, "base_key": p.base_key,
                         "number": n, "area": int(areas[n]), "source_lesion": source,
                         "source_area": ps.source_areas.get(source, 0)})
    return pd.DataFrame(rows)


def lesion_fate(ps: PlantedSet) -> pd.DataFrame:
    """One row per planted mask and informative base lesion: the base lesion's area, the total and
    the largest area of the planted lesions traced to it (0 if none), whether that largest piece
    is still >= the bound (`kept`), and whether the variant removed it on purpose (P6-remove)."""
    rows = []
    for p in ps.planted:
        areas = np.bincount(p.labels.ravel())
        traced = pd.Series({n: areas[n] for n in p.sources}, dtype=float).groupby(pd.Series(p.sources))
        total, largest = traced.sum(), traced.max()
        base_areas = np.bincount(ps.base_labels[p.base_key].ravel())
        for n, base_area in enumerate(base_areas):
            if not n or base_area < ps.bound:
                continue
            lid = lesion_id(p.base_key, n)
            biggest = int(largest.get(lid, 0))
            rows.append({"key": p.key, "variant": p.variant, "base_key": p.base_key, "base_lesion": lid,
                         "base_area": int(base_area), "planted_area": int(total.get(lid, 0)), "largest_piece": biggest,
                         "kept": biggest >= ps.bound, "removed": p.params.get("removed") == lid})
    return pd.DataFrame(rows)


def variant_summary(planted: pd.DataFrame, fate: pd.DataFrame, ps: PlantedSet) -> pd.DataFrame:
    """Per variant: masks made and skipped, scorable by E1 and E2, E2 strata, masks whose E2
    stratum differs from the base's, masks keeping every informative base lesion >= the bound (the
    gate rule, `validation.gate`), base lesions lost (removed on purpose excluded) and the median
    area ratio."""
    rows = []
    for variant in VARIANTS:
        d = planted[planted["variant"] == variant]
        f = fate[(fate["variant"] == variant) & ~fate["removed"]]
        rows.append({"variant": variant, "planted": len(d), "skipped": int((ps.skipped["variant"] == variant).sum()),
                     "scorable_e1": int((d["n_lesions"] > 0).sum()), "scorable_e2": int((d["n_informative"] > 0).sum()),
                     "e2_single": int((d["stratum_e2"] == "single").sum()), "e2_multi": int((d["stratum_e2"] == "multi").sum()),
                     "stratum_changed": int((d["stratum_e2"] != d["base_stratum"]).sum()),
                     "keeps_base_lesions": int(d["keeps_base_lesions"].sum()),
                     "base_lesions": len(f), "base_lesions_lost": int((~f["kept"]).sum()),
                     "median_area_ratio": float((f["largest_piece"] / f["base_area"]).median()) if len(f) else np.nan})
    return pd.DataFrame(rows)


def area_change(fate: pd.DataFrame) -> pd.DataFrame:
    """Per variant that changes lesion shape (P4, P5) and base lesion size bin: lesions, the share
    still >= the bound, and the median and range of the area ratio (largest piece / base area)."""
    f = fate[fate["variant"].isin(MORPH_VARIANTS)].copy()
    f["bin"] = [calibration.bin_label(int(b), SIZE_BINS) for b in calibration.area_bins(f["base_area"].to_numpy(), SIZE_BINS)]
    f["ratio"] = f["largest_piece"] / f["base_area"]
    out = (f.groupby(["variant", "bin"], sort=False)
           .agg(lesions=("ratio", "size"), kept=("kept", "mean"), median_ratio=("ratio", "median"),
                min_ratio=("ratio", "min"), max_ratio=("ratio", "max"))
           .reset_index())
    order = {calibration.bin_label(b, SIZE_BINS): i for i, b in enumerate(SIZE_BINS)}
    return out.sort_values(["variant", "bin"], key=lambda s: s.map(order) if s.name == "bin" else s.map(MORPH_VARIANTS.index))


# --- figures ----------------------------------------------------------------------------------

def zoom_window(masks: list[np.ndarray]) -> tuple[slice, slice]:
    """A square window around the pixels of all masks, MARGIN px wide on each side and at least
    MIN_CROP px, kept inside the image."""
    rows, cols = np.nonzero(np.any(masks, axis=0))
    h, w = masks[0].shape
    side = min(max(int(rows.max() - rows.min()), int(cols.max() - cols.min())) + 1 + 2 * MARGIN, h, w)
    side = max(side, min(MIN_CROP, h, w))
    r0 = int(np.clip(round((rows.min() + rows.max()) / 2 - side / 2), 0, h - side))
    c0 = int(np.clip(round((cols.min() + cols.max()) / 2 - side / 2), 0, w - side))
    return slice(r0, r0 + side), slice(c0, c0 + side)


def describe(p: Planted) -> str:
    """A short caption of a planted mask's parameters."""
    q = p.params
    if "shift_row" in q:
        text = f"shift ({q['shift_row']:+d}, {q['shift_col']:+d})"
    elif "angle" in q:
        text = f"rotate {q['angle']:+g} deg"
    elif "op" in q:
        text = q["op"]
    elif "removed" in q:
        text = "removed " + q["removed"].rsplit("/", 1)[1]
    elif "added" in q:
        text = "added H-eval lesion"
    else:
        text = "exact"
    return text + (", lesion lost" if not p.keeps_base_lesions and p.variant != "P6-remove" else "")


def by_base(ps: PlantedSet) -> dict[str, dict[str, Planted]]:
    """The planted masks of each base, by variant."""
    out: dict[str, dict[str, Planted]] = {k: {} for k in ps.bases["key"]}
    for p in ps.planted:
        out[p.base_key][p.variant] = p
    return out


def variants_grid(ps: PlantedSet, images: dict, rng: np.random.Generator, path, title: str):
    """One row per base (GRID_BASES, half single-, half multi-lesion), columns the original and
    every variant, zoomed to the same window per row: base outline red, planted outline green;
    base lesions with no planted lesion traced to them (removed, or eroded away) filled red, added
    H-eval lesions filled orange."""
    half = GRID_BASES // 2
    chosen = pd.concat([pick(rng, ps.bases[ps.bases["stratum"] == s], half) for s in ("single", "multi")])
    planted = by_base(ps)
    fig, axes = tile_figure(len(chosen), 1 + len(VARIANTS), tile=1.6)
    for row_axes, key in zip(axes, chosen["key"]):
        labels = ps.base_labels[key]
        base = labels > 0
        variants = planted[key]
        win = zoom_window([base, *(p.labels > 0 for p in variants.values())])
        show_slice(row_axes[0], images[key][win], [(base[win], ORIGINAL_COLOR)], f"original\n{key}")
        for ax, variant in zip(row_axes[1:], VARIANTS):
            p = variants.get(variant)
            if p is None:
                ax.text(0.5, 0.5, f"{variant}\nnot made", ha="center", va="center", fontsize=7)
                ax.axis("off")
                continue
            show_slice(ax, images[key][win], [(base[win], ORIGINAL_COLOR), ((p.labels > 0)[win], PLANTED_COLOR)],
                       f"{variant}\n{describe(p)}")
            traced = set(p.sources.values())
            gone = np.isin(labels, [n for n in range(1, int(labels.max()) + 1) if lesion_id(key, n) not in traced])
            added = np.isin(p.labels, [n for n, c in source_colors(p).items() if c == DONOR_COLOR])
            fill(ax, [(gone[win], ORIGINAL_COLOR), (added[win], DONOR_COLOR)])
    return save(fig, path, title)


def fill(ax, regions: list[tuple[np.ndarray, str]], alpha: float = 0.55) -> None:
    """Fill boolean regions in their colours, half transparent, over what is drawn."""
    overlay = np.zeros((*regions[0][0].shape, 4))
    for region, color in regions:
        overlay[region] = (*to_rgba(color)[:3], alpha)
    ax.imshow(overlay, interpolation="nearest")


def draw_labels(ax, image: np.ndarray, labels: np.ndarray, colors: dict[int, str], title: str) -> None:
    """A slice with each lesion of `labels` filled and outlined in its colour and numbered."""
    show_slice(ax, image, [], title)
    regions = []
    for n, color in colors.items():
        lesion = labels == n
        if not lesion.any():
            continue
        rgba = to_rgba(color)
        regions.append((lesion, color))
        ax.contour(lesion.astype(float), levels=[0.5], colors=[rgba], linewidths=0.6)
        rows, cols = np.nonzero(lesion)
        ax.text(cols.max() + 1, rows.min() - 0.5, f"L{n}", color=rgba, fontsize=5, fontweight="bold",
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="black")])
    if regions:
        fill(ax, regions)


def draw_change(ax, base: np.ndarray, p: Planted, win: tuple[slice, slice]) -> None:
    """Over a tile of the planted lesions (zoom window `win`): each base lesion as a dashed outline
    in its colour, and where its centre moved by at least 1 px, a dot at the old centre and an arrow
    to the centre of the largest planted lesion traced to it."""
    planted = p.labels[win]
    areas = np.bincount(planted.ravel())
    for n in range(1, int(base.max()) + 1):
        lesion = base[win] == n
        if not lesion.any():
            continue
        color = LESION_COLORS[(n - 1) % len(LESION_COLORS)]
        ax.contour(lesion.astype(float), levels=[0.5], colors=[color], linewidths=0.8, linestyles="dashed")
        pieces = [m for m, s in p.sources.items() if s == lesion_id(p.base_key, n) and m < len(areas) and areas[m]]
        if not pieces:
            continue
        old = np.argwhere(lesion).mean(axis=0)
        new = np.argwhere(planted == max(pieces, key=lambda m: areas[m])).mean(axis=0)
        if np.hypot(*(new - old)) >= 1:
            # a black edge keeps the dot and arrow visible on the lesion fill of the same colour
            ax.plot(old[1], old[0], "o", color=color, markersize=2.5, markeredgecolor="black", markeredgewidth=0.4)
            arrow = ax.annotate("", xy=(new[1], new[0]), xytext=(old[1], old[0]),
                                arrowprops={"arrowstyle": "->", "color": color, "lw": 0.9, "shrinkA": 0, "shrinkB": 0})
            if arrow.arrow_patch is not None:
                arrow.arrow_patch.set_path_effects([patheffects.withStroke(linewidth=2, foreground="black")])


def source_colors(p: Planted) -> dict[int, str]:
    """The colour of each planted lesion: that of the base lesion it comes from, DONOR_COLOR for
    an added H-eval lesion, grey without a source."""
    out = {}
    for n, source in p.sources.items():
        if not source:
            out[n] = "grey"
            continue
        key, number = parse_lesion_id(source)
        out[n] = LESION_COLORS[(number - 1) % len(LESION_COLORS)] if key == p.base_key else DONOR_COLOR
    return out


def variant_sheet(ps: PlantedSet, images: dict, variant: str, rng: np.random.Generator, path, title: str):
    """SHEET_BASES random bases with this variant, three tiles each: the planted mask on the whole
    slice with the zoom window, the base lesions, and the planted lesions coloured by their source
    with the base lesions dashed and the centre moves drawn (`draw_change`)."""
    planted = {p.base_key: p for p in ps.planted if p.variant == variant}
    chosen = pick(rng, ps.bases[ps.bases["key"].isin(planted)], SHEET_BASES)
    fig, axes = tile_figure(int(np.ceil(len(chosen) / 2)), 6, tile=1.8)
    tiles = axes.ravel()
    for k, key in enumerate(chosen["key"]):
        p, base = planted[key], ps.base_labels[key]
        win = zoom_window([base > 0, p.labels > 0])
        full, before, after = tiles[3 * k:3 * k + 3]
        show_slice(full, images[key], [((p.labels > 0), PLANTED_COLOR)], f"{p.key}\n{describe(p)}")
        draw_box(full, (win[0].start, win[0].stop - 1, win[1].start, win[1].stop - 1))
        base_colors = {n: LESION_COLORS[(n - 1) % len(LESION_COLORS)] for n in range(1, int(base.max()) + 1)}
        draw_labels(before, images[key][win], base[win], base_colors, "base")
        draw_labels(after, images[key][win], p.labels[win], source_colors(p), f"{variant} (dashed = base)")
        draw_change(after, base, p, win)
    hide_unused(axes, 3 * len(chosen))
    return save(fig, path, title)


def area_figure(fate: pd.DataFrame, bound: int, path, title: str):
    """One panel per shape-changing variant: largest planted piece against base lesion area (log
    axes), lesions that are gone drawn as x at the bottom, the bound dashed."""
    fig = Figure(figsize=(4.2 * len(MORPH_VARIANTS), 4))
    for ax, variant in zip(fig.subplots(1, len(MORPH_VARIANTS)), MORPH_VARIANTS):
        f = fate[fate["variant"] == variant]
        gone = f["largest_piece"] == 0
        ax.scatter(f.loc[~gone, "base_area"], f.loc[~gone, "largest_piece"], s=8, alpha=0.6,
                   c=np.where(f.loc[~gone, "kept"], "tab:blue", "tab:red"))
        ax.scatter(f.loc[gone, "base_area"], np.ones(int(gone.sum())), marker="x", s=14, c="tab:red")
        top = max(float(f["base_area"].max()), float(f["largest_piece"].max())) * 1.3
        ax.plot([1, top], [1, top], color="grey", lw=0.8)
        ax.axhline(bound, ls="--", color="grey", lw=0.8)
        ax.axvline(bound, ls=":", color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(bound * 0.8, top)
        ax.set_ylim(0.8, top)
        lost = int((~f["kept"]).sum())
        ax.set_title(f"{variant}: {lost} of {len(f)} lesions below {bound} px, gone or merged", fontsize=8)
        ax.set_xlabel("base lesion area (px)", fontsize=8)
        ax.set_ylabel("largest planted piece (px)", fontsize=8)
    return save(fig, path, title)


# --- review text ------------------------------------------------------------------------------

def review_text(name: str, ps: PlantedSet, summary: pd.DataFrame, change: pd.DataFrame, criteria: dict | None) -> str:
    """The readable summary, with decision K as set in the config (or where to set it)."""
    b = ps.bases
    groups = b.groupby("group").agg(patients=("patient_id", "nunique"), bases=("key", "size"),
                                    multi=("stratum", lambda s: int((s == "multi").sum())))
    reasons = ps.skipped.groupby(["variant", "reason"]).size().rename("bases").reset_index()
    return (
        f"E4 plant: dataset '{name}' (T and H-eval only; no detector runs)\n\n"
        f"Bases (decision I): {len(b)} slices of {b['patient_id'].nunique()} T patients, "
        f"{int((b['stratum'] == 'multi').sum())} multi-lesion (by informative lesions >= {ps.bound} px):\n"
        f"{groups.to_string()}\n"
        f"Drawn patients skipped (quota not met with the z spacing): {', '.join(ps.skipped_patients) or 'none'}.\n"
        f"P6-add donors: {ps.n_donors} informative H-eval lesions.\n\n"
        f"Variants (scorable_e1 / e2 = at least one lesion >= the minimum area / informative lesion; "
        f"keeps_base_lesions = every informative base lesion still >= {ps.bound} px, the gate rule for every "
        f"variant but P6-remove; "
        f"base_lesions_lost excludes the lesion P6-remove removes on purpose):\n"
        f"{summary.round(3).to_string(index=False)}\n\n"
        f"Variants not made:\n{reasons.to_string(index=False) if len(reasons) else '  none'}\n\n"
        f"Area change of the informative base lesions by size (ratio = largest planted piece / base area; "
        f"kept = share still >= {ps.bound} px):\n{change.round(2).to_string(index=False)}\n\n"
        f"Look at variants_grid.png, the variant_<P>.png sheets (colours: is every planted lesion traced to the "
        f"right base lesion?) and area_change.png.\n"
        + (f"Decision K (validation.criteria, set in the config; reasons in docs/e4-plan.md):\n"
           f"{yaml.safe_dump(criteria, sort_keys=False, default_flow_style=None)}"
           if criteria else
           f"Then set decision K, validation.criteria in configs/default.yaml (dated comment; proposal in "
           f"docs/e4-plan.md), before running 'e4_validate.py validate'.\n"))
