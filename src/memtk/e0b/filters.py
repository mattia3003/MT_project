"""E0b stage `filters-review` (decision E): which of the sampling script's filters to apply to
H-cal and H-eval.

`run_filters_review` checks each filter on the masks as saved and cleaned for every set, and
what applying it to the real sets would cost. `filter_checks` is also what E3/E4 use to apply
the decided filters (`filters.real_sets`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from memtk import config
from memtk.e0b.common import load_real
from memtk.io import LabelStack, MaskStack, read_table
from memtk.lesions import label_components
from memtk.stages import refuse_overwrite, require


# --- stage ------------------------------------------------------------------------------------

def run_filters_review(cfg: dict, dataset: str | None = None, pool: str | None = None,
                       overwrite: bool = False) -> None:
    """Evidence for decision E (`filters.real_sets`): each of the sampling script's filters on the
    masks as saved and cleaned, for every set, and the cost of applying it to the real sets.
    Needs the `qc` stage. Writes `results/<dataset>/e0b/filters_review/`."""
    max_px, small_px = int(cfg["filters"]["max_mask_px"]), int(cfg["qc"]["small_lesion_px"])
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, name, "e0b") / "filters_review"
    refuse_overwrite([out], overwrite)
    require(syn_dir / "qc.csv", "qc")

    # every real lesion slice and every synthetic mask, each filter on the mask as saved and cleaned
    rows = []
    real_sl = read_table(real_dir / "slices.csv")
    labels, brains = LabelStack(real_dir / "labels.npz"), MaskStack(real_dir / "brain_masks.npz")
    for pid, group in tqdm(real_sl.groupby("patient_id"), desc=f"{name} lesion slices"):
        _, seg = load_real(real_dir, str(pid))
        for key, set_name, z in zip(group["key"], group["set"], group["z"]):
            rows.append({"set": set_name, "patient_id": pid, "key": key,
                         **filter_checks(seg[:, :, int(z)] > 0, labels[key], brains[key], max_px, small_px)})
    syn_sl = read_table(syn_dir / "slices.csv").merge(read_table(syn_dir / "qc.csv")[["key", "mask_class"]], on="key")
    raw, labels, brains = (MaskStack(syn_dir / "masks_raw.npz"), LabelStack(syn_dir / "labels.npz"),
                           MaskStack(syn_dir / "brain_masks.npz"))
    for key, mask_class in tqdm(list(zip(syn_sl["key"], syn_sl["mask_class"])), desc=pool_name):
        rows.append({"set": "S main" if mask_class in ("clean", "minor") else "S other", "key": key,
                     **filter_checks(raw[key], labels[key], brains[key], max_px, small_px)})
    checks = pd.DataFrame(rows)

    synthetic = checks[checks["set"].isin(["S main", "S other"])]
    groups = pd.concat([checks[checks["set"].isin(["T", "H-cal", "H-eval", "S main"])], synthetic.assign(set="S all")])
    passing = summarize_filters(groups, "set")
    cost = filter_cost(checks, ["H-cal", "H-eval", "T"])
    speck_only = {s: int((d["raw_small_lesion"] & ~d["clean_small_lesion"]).sum())
                  for s, d in groups.groupby("set", sort=False)}
    holds = [f for f in FILTERS
             if passing.at[f"raw_{f}", "S all"] >= 0.99]  # pyright: ignore[reportOperatorIssue]  # pandas stubs: a cell of a float table

    out.mkdir(parents=True, exist_ok=True)
    passing.round(4).to_csv(out / f"filter_pass_{pool_name}.csv")
    cost.round(4).to_csv(out / f"filter_cost_{name}.csv", index=False)
    text = (f"E0b filters-review: dataset '{name}', pool '{pool_name}'. The sampling script kept a mask only if "
            f"(max_mask_px) its pixel sum < {max_px}, (small_lesion) a lesion < {small_px} px, (inside_brain) every "
            f"lesion pixel inside the generated brain.\n"
            f"raw_* = on the mask as saved (every component counts, as the sampling script saw it); clean_* = on the "
            f"lesions after attach-or-drop (as the detectors see it). The brain is our brain mask, not the script's.\n\n"
            f"Share of masks passing each filter:\n{passing.round(4).to_string()}\n\n"
            f"Masks with a small 'lesion' only because of a speck (raw passes, cleaned fails small_lesion): "
            f"{speck_only}\n\n"
            f"Cost of applying each filter (cleaned masks) to the real sets: slices removed, patients left:\n"
            f"{cost.round(3).to_string(index=False)}\n\n"
            f"Filters that hold in the pool (>= 99% of synthetic masks as saved pass): {holds}\n"
            f"Decision E: which filters to apply to H-cal and H-eval (never T, the reference copies are matched "
            f"against). Set filters.real_sets in configs/default.yaml (dated comment), as a list; [] for none.\n")
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "filters-review", "pool": pool_name})
    print(f"\n{text}\nWrote {out}")


# --- filters ----------------------------------------------------------------------------------

FILTERS = ("max_mask_px", "small_lesion", "inside_brain")


def filter_checks(mask: np.ndarray, labels: np.ndarray, brain: np.ndarray, max_mask_px: int,
                  small_lesion_px: int) -> dict:
    """Whether one mask passes each sampling filter, in two versions: `raw_*` on the mask as saved
    (every component counts, as the sampling script saw it) and `clean_*` on the lesions after
    attach-or-drop (as the detectors see it). A cleaned mask without lesions fails `small_lesion`
    and passes `inside_brain`."""
    raw = mask > 0
    comp, n = label_components(raw)
    raw_sizes = np.bincount(comp.ravel(), minlength=n + 1)[1:]
    lesion = labels > 0
    lesion_sizes = np.bincount(labels.ravel())[1:]
    lesion_sizes = lesion_sizes[lesion_sizes > 0]
    return {"raw_max_mask_px": bool(raw.sum() < max_mask_px),
            "raw_small_lesion": bool((raw_sizes < small_lesion_px).any()),
            "raw_inside_brain": not bool((raw & ~brain).any()),
            "clean_max_mask_px": bool(lesion.sum() < max_mask_px),
            "clean_small_lesion": bool((lesion_sizes < small_lesion_px).any()),
            "clean_inside_brain": not bool((lesion & ~brain).any())}


def summarize_filters(checks: pd.DataFrame, by: str) -> pd.DataFrame:
    """Per set: the share of masks passing each filter in both versions, and all three together."""
    cols = [f"{v}_{f}" for v in ("raw", "clean") for f in FILTERS]
    table = checks.assign(raw_all=checks[[f"raw_{f}" for f in FILTERS]].all(axis=1),
                          clean_all=checks[[f"clean_{f}" for f in FILTERS]].all(axis=1))
    return table.groupby(by, sort=False)[[*cols, "raw_all", "clean_all"]].mean().T


def filter_cost(checks: pd.DataFrame, sets, version: str = "clean") -> pd.DataFrame:
    """For each filter (and all three) on the real sets: slices removed and their share, and
    patients left with no slice at all."""
    rows = []
    for name in sets:
        d = checks[checks["set"] == name]
        for f in (*FILTERS, "all"):
            passed = d[[f"{version}_{g}" for g in FILTERS]].all(axis=1) if f == "all" else d[f"{version}_{f}"]
            rows.append({"set": name, "filter": f, "slices": len(d), "removed": int((~passed).sum()),
                         "share_removed": float((~passed).mean()),
                         "patients_left": d.loc[passed, "patient_id"].nunique(), "patients": d["patient_id"].nunique()})
    return pd.DataFrame(rows)
