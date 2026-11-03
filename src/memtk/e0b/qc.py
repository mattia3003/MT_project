"""E0b quality-control stages for the synthetic masks.

- `run_qc_review` (decision D): the isolated-speck cut-off between minor and noisy masks.
- `run_qc` (needs D): one QC class per synthetic sample, with flags.

Class rules live in `memtk.qc` (library); below the stages: the review strata and summaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from memtk import config
from memtk.e0b.common import Slices, decided, pick, refuse_overwrite, require
from memtk.io import read_table
from memtk.lesions import attach_or_drop
from memtk.plotting import lesion_sheet
from memtk.qc import MASK_CLASSES, QC_CLASSES, edge_lesions, mask_class, qc_class


# --- stages -----------------------------------------------------------------------------------

def run_qc_review(cfg: dict, pool: str | None = None, overwrite: bool = False) -> None:
    """Evidence for decision D (`qc.max_isolated_specks`): synthetic masks by number of isolated
    specks, exact counts, a sheet per stratum, and clean masks with a lesion outside the brain.
    Synthetic masks only (no detector has touched S yet). Needs the `lesions` stage. Writes
    `results/<pool>/e0b/qc_review/`."""
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    edge_frac = float(cfg["qc"]["edge_outside_frac"])
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    syn_dir = config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, pool_name, "e0b") / "qc_review"
    refuse_overwrite([out], overwrite)
    require(syn_dir / "lesions.csv", "lesions")
    rng = np.random.default_rng(cfg["seed"])

    # decision D views synthetic masks only; no detector has touched S yet
    index = read_table(syn_dir / "synthetic_index.csv")
    slices = (read_table(syn_dir / "slices.csv")
              .merge(read_table(syn_dir / "brain.csv")[["sample_id", "mri_failed"]], on="sample_id")
              .merge(index[["sample_id", "legacy_excluded"]], on="sample_id"))
    slices["stratum"] = speck_stratum(slices["n_specks"])
    lesions = read_table(syn_dir / "lesions.csv")
    summary = summarize_speck_strata(slices, lesions)
    exact = slices["n_specks"].clip(upper=21).value_counts().reindex(range(22), fill_value=0)
    exact.index = [*map(str, range(21)), ">20"]


    src = Slices(syn_dir, pool_cfg)

    def tiles(picks: pd.DataFrame, label) -> list:
        items = []
        for rec in picks.itertuples(index=False):
            image, mask, brain, box = src[str(rec.key)]
            items.append((label(rec), image, attach_or_drop(mask, min_area, gap), brain, box))
        return items

    legend = ("lesions coloured with their number, attached strays lighter, isolated specks orange x, "
              "brain cyan, brain box dashed yellow")
    ok = slices[~slices["mri_failed"]]
    items = []
    for stratum in SPECK_STRATA:
        items += tiles(pick(rng, ok[ok["stratum"] == stratum], 12),
                       lambda r: f"{r.key}\n{r.n_specks} speck{'' if r.n_specks == 1 else 's'}, {r.n_lesions} les.")
    lesion_sheet(items, out / f"speck_strata_{pool_name}.png",
                 f"{pool_name}: 12 random masks per number of isolated specks, one row per stratum "
                 f"({', '.join(SPECK_STRATA)}); failed MRIs left out. {legend}", n_cols=12)

    outside = lesions[lesions["frac_outside_brain"] > edge_frac].groupby("key")["lesion_id"].apply(
        lambda s: ", ".join(i.rsplit("/", 1)[1] for i in s))
    clean = ok[(ok["n_specks"] == 0) & ok["key"].isin(outside.index)]
    lesion_sheet(tiles(pick(rng, clean, 24), lambda r: f"{r.key}\noutside: {outside[r.key]}"),
                 out / f"edge_clean_{pool_name}.png",
                 f"{pool_name}: 24 random clean masks (no isolated speck) with a lesion more than "
                 f"{edge_frac:.0%} outside the brain (named in the tile title); {len(clean)} such masks "
                 f"in total. {legend}")

    summary.to_csv(out / f"speck_strata_{pool_name}.csv")
    exact.rename("masks").to_csv(out / f"speck_counts_{pool_name}.csv", index_label="isolated_specks")
    text = (f"E0b qc-review: pool '{pool_name}' (masks as saved; attach-or-drop at {min_area} px, gap {gap}), "
            f"seed {cfg['seed']}\n\n"
            f"Masks by number of isolated specks (all {len(slices)} masks):\n{summary.round(3).to_string()}\n"
            f"Columns: masks, share, cumulative share; failed MRIs; masks without a lesion; lesions per mask; "
            f"share of specks outside the brain; share of lesions with an attached stray, and grown > 25% by "
            f"strays; share of lesions more than half outside the brain; share excluded by the old QC list.\n\n"
            f"Exact counts, masks with 0..20 isolated specks and more:\n"
            f"{exact.to_frame('masks').T.to_string()}\n\n"
            f"Proposed qc.max_isolated_specks: 3 (the plan's suggestion). A mask with 1..N isolated specks is "
            f"minor (kept; specks are not lesions), more than N is noisy (left out of the main audit).\n"
            f"  Choose N from speck_strata_{pool_name}.png: the largest number at which masks still look like "
            f"clean masks with a few stray dots, before they turn into fields of noise.\n\n"
            f"Edge check (qc.edge_outside_frac = {edge_frac}): {len(clean)} clean masks have a lesion more than "
            f"half outside the brain; see edge_clean_{pool_name}.png. The 'qc' stage flags them; whether a flag "
            f"excludes is decided there (recommendation: flag only, see docs/report-notes.md).\n\n"
            f"Then set qc.max_isolated_specks in configs/default.yaml (dated comment) and run the 'qc' stage.\n")
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "qc-review", "pool": pool_name})
    print(f"\n{text}\nWrote {out}")


QC_COLUMNS = ["source", "key", "sample_id", "qc_class", "mask_class", "mri_failed", "edge", "n_edge_lesions",
              "has_lesion_lt40", "n_lesions", "n_specks", "n_strays", "stray_px", "legacy_excluded"]


def run_qc(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """QC classes with the decided cut-off (decision D): per synthetic sample its mask class (from
    the mask alone), the failed-MRI flag, the combined class, and the edge and small-lesion
    flags (`<pool>/qc.csv`); a summary, the old-QC cross-table, the edge check on the real sets
    and an example sheet in `results/<pool>/e0b/qc/`. Needs the `lesions` stage."""
    min_area = int(decided(cfg, "lesions", "min_lesion_area", "lesions-review"))
    gap = int(decided(cfg, "lesions", "attach_gap", "lesions-review"))
    max_specks = int(decided(cfg, "qc", "max_isolated_specks", "qc-review"))
    edge_frac, small_px = float(cfg["qc"]["edge_outside_frac"]), int(cfg["qc"]["small_lesion_px"])
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, pool_name, "e0b") / "qc"
    refuse_overwrite([syn_dir / "qc.csv", out], overwrite)
    for d in (real_dir, syn_dir):
        require(d / "lesions.csv", "lesions")
    rng = np.random.default_rng(cfg["seed"])

    # one row per synthetic sample: mask class, failed MRI and the combined class; flags as columns
    index = read_table(syn_dir / "synthetic_index.csv")
    table = (read_table(syn_dir / "slices.csv")
             .merge(read_table(syn_dir / "brain.csv")[["sample_id", "mri_failed"]], on="sample_id")
             .merge(index[["sample_id", "legacy_excluded", "legacy_reason"]], on="sample_id"))
    table["mask_class"] = mask_class(table["n_lesions"], table["n_specks"], max_specks)
    table["qc_class"] = qc_class(table["mri_failed"], table["mask_class"])
    edge = edge_lesions(read_table(syn_dir / "lesions.csv"), edge_frac)
    table["n_edge_lesions"] = table["key"].map(edge).fillna(0).astype(int)
    table["edge"] = table["n_edge_lesions"] > 0
    table[QC_COLUMNS].to_csv(syn_dir / "qc.csv", index=False)
    config.write_run_info(syn_dir, cfg, script=__file__, filename="qc.run_info.json",
                          extra={"stage": "qc", "pool": pool_name, "max_isolated_specks": max_specks,
                                 "edge_outside_frac": edge_frac})

    by_qc = summarize_qc(table, "qc_class", QC_CLASSES)
    by_mask = summarize_qc(table, "mask_class", MASK_CLASSES)
    failed_by_mask = pd.crosstab(table["mask_class"], table["mri_failed"]).reindex(list(MASK_CLASSES), fill_value=0)
    # old reasons read e.g. "Excessive noise fragments: 11 distinct blobs"; group by the part before ':'
    old_reason = table["legacy_reason"].str.split(":").str[0].fillna("not on the old list")
    old_qc = pd.crosstab(table["qc_class"], old_reason, margins=True,
                         margins_name="total").reindex([*QC_CLASSES, "total"], fill_value=0)

    # the same edge check on the real sets: if real slices failed it, the rule would be too strict
    real_lesions, real_slices = read_table(real_dir / "lesions.csv"), read_table(real_dir / "slices.csv")
    real_edge = edge_lesions(real_lesions, edge_frac)
    real_check = pd.DataFrame({
        "slices": real_slices.groupby("set", sort=False).size(),
        "lesions": real_lesions.groupby("set", sort=False).size(),
        "edge_lesions": real_lesions[real_lesions["frac_outside_brain"] > edge_frac].groupby("set").size(),
        "edge_slices": real_slices[real_slices["key"].isin(real_edge.index)].groupby("set").size(),
    }).fillna(0).astype(int)

    # a sheet with random masks of each class
    src = Slices(syn_dir, pool_cfg)


    items = []
    for cls in QC_CLASSES:
        for key, n_les, n_specks, n_edge in pick(rng, table[table["qc_class"] == cls], 8)[
                ["key", "n_lesions", "n_specks", "n_edge_lesions"]].itertuples(index=False):
            image, mask, brain, box = src[str(key)]
            counts = f"{n_les} lesion{'' if n_les == 1 else 's'}, {n_specks} speck{'' if n_specks == 1 else 's'}"
            flags = f", {n_edge} at edge" if n_edge else ""
            items.append((f"{cls}: {key}\n{counts}{flags}", image, attach_or_drop(mask, min_area, gap), brain, box))
    lesion_sheet(items, out / f"qc_classes_{pool_name}.png",
                 f"{pool_name}: 8 random samples per QC class (rows: {', '.join(QC_CLASSES)}). Lesions "
                 f"coloured with their number, strays lighter, isolated specks orange x, brain cyan, brain "
                 f"box dashed yellow")

    by_qc.to_csv(out / f"qc_classes_{pool_name}.csv")
    old_qc.to_csv(out / f"old_qc_crosstab_{pool_name}.csv")
    real_check.to_csv(out / f"edge_check_real_{name}.csv")
    text = (f"E0b qc: pool '{pool_name}'; noisy = more than {max_specks} isolated specks; edge = a lesion more "
            f"than {edge_frac} outside the brain; small lesion = under {small_px} px\n\n"
            f"QC class per sample (precedence: {' > '.join(QC_CLASSES)}):\n{by_qc.round(4).to_string()}\n\n"
            f"Mask class (from the mask alone; failed MRIs keep their mask class):\n{by_mask.round(4).to_string()}\n"
            f"Failed MRIs by mask class:\n{failed_by_mask.to_string()}\n\n"
            f"Columns: masks, share; masks with an edge-flagged lesion; masks without a lesion under {small_px} px; "
            f"attached strays (components, px); masks excluded by the old QC list.\n\n"
            f"Old QC list (reason) against the QC class:\n{old_qc.to_string()}\n\n"
            f"Edge check on the real sets ({name}), same rule:\n{real_check.to_string()}\n\n"
            f"Flags do not exclude: an edge-flagged or small-lesion-free mask keeps its class. Which classes "
            f"enter the main audit is set when E5 is planned (recommendation: mask_class clean or minor).\n")
    (out / "summary.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "qc", "pool": pool_name})
    print(f"\n{text}\nWrote {syn_dir / 'qc.csv'}, {out}")


# --- qc-review: speck strata ------------------------------------------------------------------

SPECK_STRATA = ["0", "1", "2", "3", "4-5", "6-9", "10-19", ">=20"]


def speck_stratum(n_specks: pd.Series) -> pd.Series:
    """The SPECK_STRATA label of each mask's number of isolated specks."""
    return pd.cut(n_specks, bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 5.5, 9.5, 19.5, np.inf], labels=SPECK_STRATA)


def summarize_speck_strata(slices: pd.DataFrame, lesions: pd.DataFrame) -> pd.DataFrame:
    """Per speck stratum: masks (share, cumulative share), failed MRIs, masks without a lesion,
    lesions per mask, share of specks outside the brain, share of lesions with strays and grown
    by more than 25% through them, share of lesions mostly outside the brain, and the share the
    old QC list excluded. Expects slices with `stratum`, `mri_failed` and `legacy_excluded`."""
    les = lesions.merge(slices[["key", "stratum"]], on="key")
    les["grown"] = les["stray_px"] > 0.25 * (les["area"] - les["stray_px"])
    s, g = slices.groupby("stratum", observed=True), les.groupby("stratum", observed=True)
    out = pd.DataFrame({
        "masks": s.size(), "failed_mri": s["mri_failed"].sum(),
        "no_lesion": s["n_lesions"].agg(lambda x: int((x == 0).sum())),
        "lesions_per_mask": s["n_lesions"].mean(),
        "specks_outside_brain": s["specks_outside_brain"].sum() / s["n_specks"].sum(),
        "lesions_with_strays": g["stray_px"].agg(lambda x: (x > 0).mean()),
        "lesions_grown_gt25pct": g["grown"].mean(),
        "lesions_mostly_outside": g["frac_outside_brain"].agg(lambda x: (x > 0.5).mean()),
        "old_qc_excluded": s["legacy_excluded"].mean(),
    })
    out.insert(1, "share", out["masks"] / len(slices))
    out.insert(2, "cum_share", out["share"].cumsum())
    return out


# --- qc: summary ------------------------------------------------------------------------------

def summarize_qc(table: pd.DataFrame, by: str, order) -> pd.DataFrame:
    """Per class (column `by`, rows in `order`): masks and share, edge-flagged masks, masks without
    a lesion under 40 px, attached strays (components, px), masks excluded by the old QC list."""
    g = table.groupby(by, sort=False)
    out = pd.DataFrame({
        "masks": g.size(), "edge_flagged": g["edge"].sum(),
        "no_lesion_lt40": g["has_lesion_lt40"].agg(lambda x: int((~x.astype(bool)).sum())),
        "strays": g["n_strays"].sum(), "stray_px": g["stray_px"].sum(), "old_qc_excluded": g["legacy_excluded"].sum(),
    })
    out.insert(1, "share", out["masks"] / len(table))
    return out.reindex([c for c in order if c in out.index])
