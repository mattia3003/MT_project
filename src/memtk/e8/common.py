"""Helpers shared by the E8 stages (`docs/e8-plan.md`): the `paired:` config section (decisions AA-AE)
checked, decision AB's cells, and the pair table that `rehearse` and `paired` measure.

A pair table has one row per image pair: the query image (synthetic MRI, held-out or mock BRAVO), the
T comparison slice whose BRAVO it is compared with, and the shift that aligns the query onto it.
Rows of kind `t` are the cases (group A, or a held-out set's flagged slices) and their matched controls
(group B, or the held-out novel slices); within group A, rows of kind `runner_up` (the best slice of the
best other patient) and `dz` (the source patient's slices at z +- 1..dz_max, aligned with t's shift).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from memtk import config
from memtk.ids import parse_slice_key, slice_key
from memtk.io import load_real, read_table
from memtk.paired import REGIONS, best_match, match_controls, size_cell
from memtk.sets import SliceSet
from memtk.stages import decided
from memtk.synthetic import load_sample

REVIEW = "docs/e8-plan.md (decisions AA-AE)"
SECTIONS = {"group_a": (), "controls": ("k", "lesion_count_cap", "size_bins_px", "brain_quantiles"),
            "measure": ("ssim_sigma", "data_range", "region_grow_px", "regions", "primary"),
            "test": ("reference", "auc_null", "dz_max"),
            "rehearsal": ("null_set", "positive_variants", "positive_dz")}
Values = Sequence[float] | np.ndarray | pd.Series   # one number per slice
PAIR_COLUMNS = ["set", "group", "kind", "dz", "query", "case", "reason", "cell", "cluster", "comparison",
                "comparison_patient", "shift_row", "shift_col", "match_dice"]


# --- config -----------------------------------------------------------------------------------

def paired_config(cfg: dict) -> dict:
    """The `paired:` section with every value set (decisions AA-AE); stops with a message otherwise or
    when a value is not implemented."""
    p = {section: decided(cfg, f"paired.{section}", REVIEW) for section in SECTIONS}
    for section, keys in SECTIONS.items():
        for key in keys:
            decided(cfg, f"paired.{section}.{key}", REVIEW)
    if set(p["group_a"].values()) - {"e1", "e2"}:
        raise SystemExit("paired.group_a: every reason maps to 'e1' or 'e2'")
    m = p["measure"]
    if set(m["regions"]) - set(REGIONS) or m["primary"] not in m["regions"]:
        raise SystemExit(f"paired.measure: regions must be among {REGIONS} and include the primary one")
    if p["test"]["reference"] != "H-eval":
        raise SystemExit("paired.test.reference: only 'H-eval' is implemented")
    if p["rehearsal"]["null_set"] != "H-cal":
        raise SystemExit("paired.rehearsal.null_set: only 'H-cal' (scored by E5 `rehearse`) is implemented")
    return p


# --- cells (decision AB) ----------------------------------------------------------------------

def quantile_edges(values: Values, n: int) -> list[float]:
    """The inner cut points of n equal-count groups of the values (tertiles for n = 3)."""
    return [float(x) for x in np.quantile(np.asarray(values, dtype=float), np.arange(1, n) / n)]


def cells(n_informative: Values, l1_area: Values, brain_px: Values, controls: dict,
          brain_edges: Sequence[float]) -> list[str]:
    """Decision AB's cell of each slice as text `n<informative lesions, capped>|s<size bin of the
    largest lesion>|b<brain-area group>`."""
    n = np.minimum(np.asarray(n_informative, dtype=int), int(controls["lesion_count_cap"]))
    s = size_cell(l1_area, controls["size_bins_px"])
    b = size_cell(brain_px, brain_edges)
    return [f"n{a}|s{c}|b{d}" for a, c, d in zip(n, s, b)]


# --- the pair table ---------------------------------------------------------------------------

def case_control_pairs(set_name: str, cases: pd.DataFrame, pool: pd.DataFrame, k: int,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Rows of kind `t`: every case (group A; columns `key`, `reason`, `cell`, `cluster`,
    `comparison`) and k matched controls each from the pool (group B; columns `key`, `cell`,
    `cluster`, `comparison`), drawn by `paired.match_controls`. Shift columns are added later
    (`add_shifts`)."""
    matched = match_controls(cases[["key", "cell"]], pool[["key", "cell"]], k, rng)
    a = cases.assign(group="A", case=cases["key"])
    reason = dict(zip(cases["key"], cases["reason"]))
    b = (pool.set_index("key").loc[matched["control"], ["cluster", "comparison", "cell"]].reset_index()
         .assign(group="B", case=matched["case"].to_numpy(), reason=[reason[c] for c in matched["case"]]))
    rows = pd.concat([a, b], ignore_index=True).rename(columns={"key": "query"})
    return rows.assign(set=set_name, kind="t", dz=0)


def add_shifts(pairs: pd.DataFrame, query_mask: Callable[[str], np.ndarray], t_masks: dict[str, np.ndarray],
               max_shift: int) -> pd.DataFrame:
    """`shift_row`, `shift_col`, `match_dice` of each pair: the best Dice over shifts within ±max_shift
    of the query's cleaned mask against its comparison slice's (decision AC; E1's search and tie
    rule on one candidate, so for an E1 nearest neighbour it gives E1's stored shift)."""
    out = pairs.copy()
    found = [best_match(query_mask(q), t_masks[c][None], max_shift) for q, c in zip(out["query"], out["comparison"])]
    out["match_dice"] = [f[1] for f in found]
    out["shift_row"] = [f[2] for f in found]
    out["shift_col"] = [f[3] for f in found]
    return out


def runner_up_pairs(a: pd.DataFrame, e1: pd.DataFrame, t: SliceSet, query_mask: Callable[[str], np.ndarray],
                    max_shift: int) -> pd.DataFrame:
    """Within A (decision AD): per group-A row, the T slice of the best other patient that matches
    the query best (whole-mask Dice over shifts, every slice of that patient searched). The other
    patient is the best E1 patient that is not t's patient: the E1 nearest neighbour's patient when
    it differs from t's (an E2 source), else E1's runner-up patient. `e1` is indexed by query."""
    rows = []
    for _, r in a.iterrows():
        nn_patient, runner_up = e1.loc[r["query"], ["nn_patient", "runner_up_patient"]]
        other = nn_patient if nn_patient != r["comparison_patient"] else runner_up
        idx = np.flatnonzero(t.patients == other)
        i, score, dr, dc = best_match(query_mask(r["query"]), t.labels[idx] > 0, max_shift)
        rows.append({**r.to_dict(), "kind": "runner_up", "comparison": t.keys[idx[i]], "comparison_patient": other,
                     "shift_row": dr, "shift_col": dc, "match_dice": score})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def dz_pairs(a: pd.DataFrame, depth: pd.Series, dz_max: int) -> pd.DataFrame:
    """Within A (decision AD): per group-A row, the slices of t's patient at z +- 1..dz_max that lie
    inside the volume (`depth`: slices per patient), aligned with t's shift (`match_dice` empty: the
    slice may hold no lesion)."""
    rows = []
    for _, r in a.iterrows():
        patient, z = parse_slice_key(r["comparison"])
        for d in (*range(-dz_max, 0), *range(1, dz_max + 1)):
            if 0 <= z + d < depth[patient]:
                rows.append({**r.to_dict(), "kind": "dz", "dz": d, "comparison": slice_key(patient, z + d),
                             "match_dice": np.nan})
    return pd.DataFrame(rows, columns=PAIR_COLUMNS)


def comparison_patients(pairs: pd.DataFrame) -> pd.DataFrame:
    """The table with `comparison_patient` filled from the comparison slice keys."""
    return pairs.assign(comparison_patient=[parse_slice_key(k)[0] for k in pairs["comparison"]])


# --- images -----------------------------------------------------------------------------------

def real_slices(cfg: dict, dataset: str, keys: Iterable[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(BRAVO, mask as saved) of each real slice key in model space, float64 and boolean; one volume
    read per patient (`io.load_real`)."""
    by_patient: dict[str, list[tuple[str, int]]] = {}
    for key in set(keys):
        patient, z = parse_slice_key(key)
        by_patient.setdefault(patient, []).append((key, z))
    real_dir = config.output_dir(cfg, dataset)
    out = {}
    for patient, items in by_patient.items():
        bravo, seg = load_real(real_dir, patient)
        out.update({key: (np.asarray(bravo[:, :, z], dtype=float), np.asarray(seg[:, :, z]) > 0) for key, z in items})
    return out


def synthetic_samples(cfg: dict, pool: str, keys: Iterable[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(synthetic MRI, mask as saved) of each sample ID, float64 and boolean, read from the pool files
    (`synthetic.load_sample`)."""
    _, pool_cfg = config.synthetic_pool(cfg, pool)
    index = read_table(config.output_dir(cfg, pool) / "synthetic_index.csv").set_index("sample_id")
    pool_dir = config.resolve_path(pool_cfg["dir"])
    out = {}
    for key in set(keys):
        image, mask = load_sample(pool_dir / str(index.at[key, "filename"]), pool_cfg)
        out[key] = (np.asarray(image, dtype=float), mask > 0)
    return out
