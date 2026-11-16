"""E8 stage `select` (`docs/e8-plan.md`, step 2; decisions AA, AB, AD): the S image pairs that `paired`
will measure, chosen from stored tables only. No MRI is read.

- **Group A (AA):** the S rows of E7's `report_index.csv` with a reason in `paired.group_a`, one per
  chained group; comparison slice t = the reason's decision-J source (`e1`: E1 nearest neighbour,
  `e2`: E2 source; both from E5's `flagged.csv`); bootstrap cluster = t's patient.
- **Group B (AB):** S main masks that neither detector flags, with an informative lesion and an MRI
  that did not fail; k per A mask from the same cell (informative lesions, largest-lesion bin,
  brain-area group of the pool); comparison slice = E2 nearest neighbour, E1's when C = 0; cluster =
  the comparison slice's patient.
- **Alignment (AC):** each query's shift onto its comparison slice (`common.add_shifts`); for an E1
  nearest neighbour it must equal E1's stored shift, which the review counts.
- **Within A (AD):** the runner-up patient's best slice and the source patient's slices at
  z +- 1..dz_max.

Writes `results/<pool>/e8/select/` (`pairs.csv`, `review.txt`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from memtk import config
from memtk.audit import load_scored
from memtk.calibration import load_frozen
from memtk.e5.common import lock_unchanged
from memtk.e8.common import (PAIR_COLUMNS, add_shifts, case_control_pairs, cells, comparison_patients, dz_pairs,
                             paired_config, quantile_edges, runner_up_pairs)
from memtk.ids import parse_slice_key
from memtk.io import LabelStack, read_table
from memtk.sets import load_set
from memtk.stages import refuse_overwrite, require


# --- stages -----------------------------------------------------------------------------------

def run_select(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Group A, matched group B, comparison slices, shifts and the within-A rows of S (decisions AA,
    AB, AD), from E5's and E7's stored tables with E5's lock unchanged; reads no MRI. Writes
    `results/<pool>/e8/select/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "e8") / "select"
    refuse_overwrite([out], overwrite)
    p = paired_config(cfg)
    lock = lock_unchanged(cfg, name, pool_name)
    e3, e3_hash = load_frozen(cfg, name)
    pairs, e1, counts = select_pairs(cfg, name, pool_name, p, e3)

    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "pairs.csv", index=False)
    text = review_text(pool_name, pairs, e1, counts)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "select", "inputs": ["S tables (E5, E7)", "T"],
                                                            "frozen_sha256": e3_hash, "e5_lock": lock})
    print(f"\n{text}\nWrote {out}")


# --- the pairs --------------------------------------------------------------------------------

def select_pairs(cfg: dict, dataset: str, pool: str, p: dict, e3: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The pair table of S (seeded with `cfg["seed"]`), S's E1 table and the counts for the review;
    shared by `run_select` and `check`."""
    rng = np.random.default_rng(cfg["seed"])
    max_shift = int(e3["detectors"]["whole_mask"]["max_shift_px"])
    cases, pool_rows, e1, counts = s_tables(cfg, pool, p)
    print(f"Group A: {len(cases)} masks; B pool: {len(pool_rows)} masks")
    t = load_set(cfg, dataset, "T")
    t_masks = {k: lab > 0 for k, lab in zip(t.keys, t.labels)}
    stack = LabelStack(config.output_dir(cfg, pool) / "labels.npz")

    def query_mask(key: str) -> np.ndarray:
        return stack[key] > 0

    pairs = case_control_pairs("S", cases, pool_rows, int(p["controls"]["k"]), rng)
    pairs = comparison_patients(add_shifts(pairs, query_mask, t_masks, max_shift))
    a = pairs[pairs["group"] == "A"]
    depth = read_table(config.output_dir(cfg, dataset) / "patients.csv").set_index("patient_id")["n_slices"]
    within = [runner_up_pairs(a, e1, t, query_mask, max_shift), dz_pairs(a, depth, int(p["test"]["dz_max"]))]
    return pd.concat([pairs[PAIR_COLUMNS], *within], ignore_index=True), e1, counts


# --- inputs -----------------------------------------------------------------------------------

def s_tables(cfg: dict, pool: str, p: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Group A cases and the B pool (columns `key`, `cell`, `cluster`, `comparison`; A also
    `reason`), S's E1 table indexed by query, and counts for the review."""
    report_path = config.results_dir(cfg, pool, "e7") / "reports" / "report_index.csv"
    flagged_path = config.results_dir(cfg, pool, "e5") / "analyze" / "flagged.csv"
    require(report_path, "e7_reports.py reports")
    require(flagged_path, "e5_audit.py analyze")
    out_dir, score_dir = config.output_dir(cfg, pool), config.output_dir(cfg, pool) / "e5"
    brain_px = read_table(out_dir / "brain.csv").set_index("sample_id")["brain_px"]
    failed = read_table(out_dir / "qc.csv").set_index("key")["mri_failed"].astype(bool)
    e1 = read_table(score_dir / "e1_main.csv").set_index("query")

    reports = read_table(report_path)
    a = reports[(reports["set"] == "S") & reports["reason"].isin(list(p["group_a"]))][["key", "reason"]]
    flagged = read_table(flagged_path)
    flagged = flagged[flagged["set"] == "main"].set_index("key")
    f = flagged.loc[a["key"]]
    source_col = {"e1": "e1_nn", "e2": "e2_source"}
    a = a.assign(comparison=[f.at[k, source_col[p["group_a"][r]]] for k, r in zip(a["key"], a["reason"])],
                 n_informative=f["n_informative"].to_numpy(), l1_area=f["l1_area"].to_numpy())

    s = load_scored("S", score_dir, []).table("main")
    s = s[s["mask_class"].isin(cfg["audit"]["sets"]["main"])]
    novel = (~s["e1_copy"].astype(bool)) & (s["e2_class"] == "novel") & (s["n_informative"] >= 1)
    ok_mri = ~s["key"].map(failed).astype(bool)
    b = s[novel & ok_mri]
    b = b.assign(comparison=np.where(b["e2_C"] > 0, b["e2_nn"], b["e1_nn"]))

    edges = quantile_edges(brain_px.loc[b["key"]], int(p["controls"]["brain_quantiles"]))
    a = a.assign(cell=cells(a["n_informative"], a["l1_area"], brain_px.loc[a["key"]], p["controls"], edges),
                 cluster=[parse_slice_key(k)[0] for k in a["comparison"]])
    b = b.assign(cell=cells(b["n_informative"], b["l1_area"], brain_px.loc[b["key"]], p["controls"], edges),
                 cluster=[parse_slice_key(k)[0] for k in b["comparison"]])
    counts = {"main": len(s), "novel": int(novel.sum()), "novel_mri_failed": int((novel & ~ok_mri).sum()),
              "pool": len(b), "e1_fallback": int((b["e2_C"] <= 0).sum()), "brain_edges": edges}
    return (a[["key", "reason", "cell", "cluster", "comparison"]].reset_index(drop=True),
            b[["key", "cell", "cluster", "comparison"]].reset_index(drop=True), e1, counts)


# --- review -----------------------------------------------------------------------------------

def review_text(pool: str, pairs: pd.DataFrame, e1: pd.DataFrame, counts: dict) -> str:
    """Counts of the selection: group A by reason, the B pool, matches per case and cell, cases short
    of k, agreement of the shifts with E1's stored ones, the within-A rows."""
    t = pairs[pairs["kind"] == "t"]
    a, b = t[t["group"] == "A"], t[t["group"] == "B"]
    per_case = b.groupby("case").size().reindex(a["query"], fill_value=0)
    short = per_case[per_case < per_case.max()] if len(per_case) else per_case
    on_e1 = t[t["comparison"].to_numpy() == e1.loc[t["query"], "nn"].to_numpy()]
    stored = e1.loc[on_e1["query"], ["shift_row", "shift_col"]].to_numpy()
    agree = int((on_e1[["shift_row", "shift_col"]].to_numpy() == stored).all(axis=1).sum())
    lines = [
        f"E8 select ({pool}): the image pairs `paired` will measure (decisions AA, AB, AD). No MRI read.",
        "",
        f"Group A: {len(a)} masks from {a['comparison_patient'].nunique()} source patients",
        *(f"  {r}: {n}" for r, n in a["reason"].value_counts().items()),
        f"B pool: {counts['pool']} masks (S main {counts['main']}; novel by both detectors with an informative lesion "
        f"{counts['novel']}, of which {counts['novel_mri_failed']} with a failed MRI left out); comparison slice "
        f"from E1 because C = 0: {counts['e1_fallback']}",
        f"Brain-area group edges (px, B pool): {', '.join(f'{e:.0f}' for e in counts['brain_edges'])}",
        f"Group B: {len(b)} masks matched to {int((per_case > 0).sum())} A masks; per A mask "
        f"{per_case.min()}-{per_case.max()}",
        f"A masks with fewer than {per_case.max()} matches: {len(short)}"
        + (f" ({', '.join(f'{k}: {v}' for k, v in short.items())})" if len(short) else ""),
        "",
        "Cells (n<informative lesions>|s<largest-lesion bin>|b<brain group>), A and B masks:",
        *(f"  {c}: A {int((a['cell'] == c).sum())}, B {int((b['cell'] == c).sum())}" for c in sorted(a["cell"].unique())),
        "",
        f"Whole-mask match with the comparison slice (Dice over shifts), median [IQR]: A "
        f"{a['match_dice'].median():.2f} [{a['match_dice'].quantile(0.25):.2f}, {a['match_dice'].quantile(0.75):.2f}], "
        f"B {b['match_dice'].median():.2f} [{b['match_dice'].quantile(0.25):.2f}, {b['match_dice'].quantile(0.75):.2f}]",
        f"Shifts on an E1 nearest neighbour equal to E1's stored shift: {agree} of {len(on_e1)}",
        f"Comparison patients: A {a['comparison_patient'].nunique()}, B {b['comparison_patient'].nunique()}",
        "",
        f"Within A: {int((pairs['kind'] == 'runner_up').sum())} runner-up slices (median Dice "
        f"{pairs.loc[pairs['kind'] == 'runner_up', 'match_dice'].median():.2f}), "
        f"{int((pairs['kind'] == 'dz').sum())} slices at z +- 1..{int(pairs['dz'].abs().max())}",
    ]
    return "\n".join(lines) + "\n"
