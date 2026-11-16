"""E8 stage `rehearse` (`docs/e8-plan.md`, step 3; decisions AB-AE): the measurement and analysis that
`paired` will run on S, run first where no synthetic MRI is involved. No S data is read.

- **heval (the reference):** H-eval slices flagged by either detector (cases; comparison slice = the
  decision-J source: E1 nearest neighbour for an E1 copy, else the E2 source) against matched novel
  H-eval slices (controls; E2 nearest neighbour, E1's when C = 0), real BRAVO against the BRAVO of
  the comparison slice. Its measures are stored for `paired` (the H-eval AUC of decision AD).
- **null:** the same on H-cal, judged against H-eval: no slice can carry memorized tissue, so it must
  not read "supported".
- **positive:** planted P1/P2 masks (E4, gated, flagged by E2 with the base patient as source) whose
  "MRI" is the base patient's BRAVO `positive_dz` slices away, moved by the planted shift, against
  matched novel H-cal slices: same tissue as the source, not identical; must read "supported".

Cases and controls are matched by decision AB's cells (brain-area groups from the set's controls);
bootstrap clusters are the query's patient (planted: the base patient). Writes
`results/<dataset>/e8/rehearse/{heval,null,positive}/` and the run log in `rehearse/`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from memtk import config
from memtk.audit import heldout, load_scored
from memtk.calibration import load_frozen
from memtk.e8.common import (add_shifts, case_control_pairs, cells, comparison_patients, paired_config,
                             quantile_edges, real_slices)
from memtk.e8.paired import (READINGS, Query, Settings, analyze, draw_examples, draw_groups, example_cases,
                             measure_pairs, review_lines, settings)
from memtk.ids import parse_slice_key, slice_key
from memtk.io import read_table
from memtk.lesions import brain_mask
from memtk.paired import align
from memtk.sets import load_planted, load_set
from memtk.stages import refuse_overwrite, require

FLAGGED = ("full", "partial")


# --- stages -----------------------------------------------------------------------------------

def run_rehearse(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The E8 analysis on the H-eval reference, the H-cal null and the planted positive mock (no S).
    Needs E4 `validate`, E5 `rehearse` (H-cal scored) and decisions AB-AE. Writes
    `results/<dataset>/e8/rehearse/`."""
    name, _ = config.real_dataset(cfg, dataset)
    out = config.results_dir(cfg, name, "e8") / "rehearse"
    refuse_overwrite([out], overwrite)
    p = paired_config(cfg)
    e3, e3_hash = load_frozen(cfg, name)
    s = settings(cfg, p, e3)
    results, real, mock_info = rehearse_cases(cfg, name, p, e3, s)

    primary = f"ssim_{p['measure']['primary']}"
    text, readings = [review_header(name, p, s, mock_info)], {}
    for case, r in results.items():
        readings[case] = r.reading
        write_case(out / case, r.measures, r.tables, r.reference, r.query, real, s, primary, r.labels, case)
        text += ["", *review_lines(case, r.measures, r.tables, r.reading, primary)]
    ok = readings["null"] != READINGS[0] and readings["positive"] == READINGS[0]
    text += ["", f"Rehearsal as required (null not '{READINGS[0]}', positive '{READINGS[0]}'): {'yes' if ok else 'NO'}"]
    (out / "review.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "rehearse", "inputs": ["T", "H-eval", "H-cal", "planted"],
                                                            "frozen_sha256": e3_hash, "readings": readings, "as_required": ok})
    print("\n".join(text) + f"\nWrote {out}")


# --- the rehearsal ----------------------------------------------------------------------------

@dataclass
class Case:
    """One rehearsal case: its measured pairs, tables and reading, the reference it was judged
    against, the query loader and the group labels of its figure."""
    measures: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    reading: str | None
    reference: pd.DataFrame | None
    query: Query
    labels: tuple[str, str]


def rehearse_cases(cfg: dict, name: str, p: dict, e3: dict,
                   s: Settings) -> tuple[dict[str, Case], dict[str, tuple[np.ndarray, np.ndarray]], dict]:
    """Every rehearsal case (heval, null, positive) built, measured and analysed, seeded with
    `cfg["seed"]`; also the real slices read and the positive mock's counts. Shared by
    `run_rehearse` and `check`."""
    max_shift = int(e3["detectors"]["whole_mask"]["max_shift_px"])
    draws = int(cfg["calibration"]["bootstrap_draws"])
    rng = np.random.default_rng(cfg["seed"])
    k = int(p["controls"]["k"])

    t = load_set(cfg, name, "T")
    t_masks = {key: lab > 0 for key, lab in zip(t.keys, t.labels)}
    brain_px = read_table(config.output_dir(cfg, name) / "brain.csv").set_index("key")["brain_px"]
    heval_set, hcal_set = load_set(cfg, name, "H-eval"), load_set(cfg, name, "H-cal")
    null_dir = config.output_dir(cfg, name) / "e5" / "rehearse" / "null"    # H-cal scored by E5 `rehearse`
    require(null_dir / "classes_main.csv", "e5_audit.py rehearse")
    heval_table = heldout(cfg, name, heval_set, []).table("main")
    hcal_table = load_scored("null", null_dir, []).table("main")

    print("Pairs: H-eval, H-cal, planted positive mock ...")
    labels = {**dict(zip(heval_set.keys, heval_set.labels)), **dict(zip(hcal_set.keys, hcal_set.labels))}
    heval_pairs = heldout_pairs("H-eval", heval_table, labels, brain_px, p, k, t_masks, max_shift, rng)
    null_pairs = heldout_pairs("H-cal", hcal_table, labels, brain_px, p, k, t_masks, max_shift, rng)
    mock, mock_cases, mock_info = positive_mock(cfg, name, p, s)
    pos_pairs = positive_pairs(mock, mock_cases, hcal_table, labels, brain_px, p, k, t_masks, max_shift, rng)

    keys = set().union(*(set(d["comparison"]) for d in (heval_pairs, null_pairs, pos_pairs)),
                       heval_pairs["query"], null_pairs["query"], pos_pairs.loc[pos_pairs["group"] == "B", "query"])
    print(f"Reading {len(keys)} real slices ...")
    real = real_slices(cfg, name, keys)

    def real_query(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, saved = real[key]
        return image, saved, labels[key]

    def mock_query(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if key in mock:
            return mock[key]
        return real_query(key)

    print("Measuring and analysing ...")
    heval = measure_pairs(heval_pairs, real_query, real, s)
    measured = {"heval": (heval, None, real_query, ("H-eval flagged", "H-eval novel")),
                "null": (measure_pairs(null_pairs, real_query, real, s), heval, real_query, ("H-cal flagged", "H-cal novel")),
                "positive": (measure_pairs(pos_pairs, mock_query, real, s), heval, mock_query,
                             ("planted (BRAVO z+2)", "H-cal novel"))}
    results = {}
    for case, (m, ref, query, group_labels) in measured.items():
        tables, read = analyze(m, ref, p, s, draws, rng)
        results[case] = Case(m, tables, read, ref, query, group_labels)
    return results, real, mock_info


# --- pairs ------------------------------------------------------------------------------------

def is_true(values: pd.Series) -> pd.Series:
    """Boolean column read from a CSV (bool or the text 'True')."""
    return values.astype(str) == "True"


def novel_pool(table: pd.DataFrame, brain_px: pd.Series, p: dict) -> tuple[pd.DataFrame, list[float]]:
    """Decision AB's controls of a held-out set: slices novel to both detectors with an informative
    lesion, comparison = E2 nearest neighbour (E1's when C = 0), cluster = the slice's patient,
    cells with brain-area groups from these slices; also the group edges."""
    novel = table[~is_true(table["e1_copy"]) & (table["e2_class"] == "novel") & (table["n_informative"] >= 1)]
    edges = quantile_edges(brain_px.loc[novel["key"]], int(p["controls"]["brain_quantiles"]))
    pool = novel.assign(comparison=np.where(novel["e2_C"] > 0, novel["e2_nn"], novel["e1_nn"]),
                        cell=cells(novel["n_informative"], novel["l1_area"], brain_px.loc[novel["key"]], p["controls"], edges))
    return pool[["key", "cell", "cluster", "comparison"]].reset_index(drop=True), edges


def heldout_pairs(set_name: str, table: pd.DataFrame, labels: dict[str, np.ndarray], brain_px: pd.Series, p: dict,
                  k: int, t_masks: dict[str, np.ndarray], max_shift: int, rng: np.random.Generator) -> pd.DataFrame:
    """Cases (slices flagged by either detector, with an informative lesion; comparison = decision J's
    source, E1 first) and k matched novel controls each, with shifts."""
    pool, edges = novel_pool(table, brain_px, p)
    e1 = is_true(table["e1_copy"])
    flagged = table[(e1 | table["e2_class"].isin(FLAGGED)) & (table["n_informative"] >= 1)]
    e1f = is_true(flagged["e1_copy"])
    source = np.where(e1f, flagged["e1_nn"], np.where(flagged["e2_class"] == "full", flagged["e2_nn"],
                                                      flagged["e2_strongest_train"]))
    cases = flagged.assign(comparison=source,
                           reason=np.where(e1f, "E1 copy", "E2 " + flagged["e2_class"].astype(str)),
                           cell=cells(flagged["n_informative"], flagged["l1_area"], brain_px.loc[flagged["key"]],
                                      p["controls"], edges))
    pairs = case_control_pairs(set_name, cases[["key", "reason", "cell", "cluster", "comparison"]].reset_index(drop=True),
                               pool, k, rng)
    return comparison_patients(add_shifts(pairs, lambda key: labels[key] > 0, t_masks, max_shift))


def positive_mock(cfg: dict, dataset: str, p: dict,
                  s: Settings) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], pd.DataFrame, dict]:
    """Decision AE's positive cases: planted masks of the configured variants that are gated and
    flagged by E2 with the base patient as decision-J source. Returns per case (image, mask, labels):
    the base patient's BRAVO `positive_dz` slices from the base (the other side if outside the
    volume), moved by the planted shift; the planted mask and label map. Also one row per case
    (`key`, `reason`, `cluster`, `comparison`, `n_informative`, `l1_area`, `brain_px`) and counts."""
    outcomes = read_table(config.results_dir(cfg, dataset, "e4") / "validate" / "planted_outcomes.csv")
    planted, planted_labels = load_planted(cfg, dataset)
    lab = dict(zip(planted["key"], planted_labels))
    shifts = planted.set_index("key")[["shift_row", "shift_col"]].fillna(0).astype(int)
    depth = read_table(config.output_dir(cfg, dataset) / "patients.csv").set_index("patient_id")["n_slices"]
    variants = list(p["rehearsal"]["positive_variants"])
    o = outcomes[outcomes["variant"].isin(variants) & is_true(outcomes["gated"]) & outcomes["e2_class"].isin(FLAGGED)]
    o = o.assign(source=np.where(o["e2_class"] == "full", o["e2_nn"], o["e2_strongest_train"]))
    o = o[[parse_slice_key(k)[0] for k in o["source"]] == o["patient_id"]]
    dz = int(p["rehearsal"]["positive_dz"])
    image_keys = {}
    for key, base in zip(o["key"], o["base_key"]):
        patient, z = parse_slice_key(base)
        image_keys[key] = slice_key(patient, z + dz if z + dz < depth[patient] else z - dz)
    real = real_slices(cfg, dataset, image_keys.values())
    mock: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows = []
    for key, variant, source in zip(o["key"], o["variant"], o["source"]):
        image = align(real[image_keys[key]][0], *shifts.loc[key])
        labels = lab[key]
        mock[key] = (image, labels > 0, labels)
        areas = np.bincount(labels.ravel())[1:]
        rows.append({"key": key, "reason": f"planted {variant}", "cluster": parse_slice_key(source)[0],
                     "comparison": source, "image": image_keys[key], "n_informative": int((areas >= s.bound).sum()),
                     "l1_area": int(areas[0]) if len(areas) else 0,
                     "brain_px": int(brain_mask(image, s.brain_method, s.low_threshold).sum())})
    info = {"variants": variants, "candidates": int(outcomes["variant"].isin(variants).sum()), "cases": len(rows),
            "image_offset": dz}
    return mock, pd.DataFrame(rows), info


def positive_pairs(mock: dict, c: pd.DataFrame, hcal_table: pd.DataFrame, labels: dict[str, np.ndarray],
                   brain_px: pd.Series, p: dict, k: int, t_masks: dict[str, np.ndarray], max_shift: int,
                   rng: np.random.Generator) -> pd.DataFrame:
    """The positive mock's cases (`c`, from `positive_mock`) with k matched novel H-cal controls each
    (cells with H-cal's brain groups), with shifts and `image`: the real slice whose BRAVO is the
    query image (a case: the base patient's slice `positive_dz` away, before the planted shift; a
    control: itself)."""
    pool, edges = novel_pool(hcal_table, brain_px, p)
    cases = c.assign(cell=cells(c["n_informative"], c["l1_area"], c["brain_px"], p["controls"], edges))
    pairs = case_control_pairs("planted", cases[["key", "reason", "cell", "cluster", "comparison"]], pool, k, rng)
    image_of = dict(zip(c["key"], c["image"]))
    pairs = pairs.assign(image=[image_of.get(q, q) for q in pairs["query"]])

    def mask(key: str) -> np.ndarray:
        return mock[key][1] if key in mock else labels[key] > 0

    return comparison_patients(add_shifts(pairs, mask, t_masks, max_shift))


# --- outputs ----------------------------------------------------------------------------------

def write_case(out, m: pd.DataFrame, tables: dict[str, pd.DataFrame], ref: pd.DataFrame | None, query: Query,
               real: dict, s: Settings, primary: str, group_labels: tuple[str, str], case: str) -> None:
    """The tables and figures of one rehearsal case."""
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "measures.csv", index=False)
    for tname, tab in tables.items():
        if len(tab):
            tab.to_csv(out / f"{tname}.csv", index=False)
    draw_groups(m, ref, tables["tests"], group_labels, out / "ssim_groups.png",
                f"E8 rehearse, {case}: SSIM of cases and controls with their comparison slices")
    draw_examples(m, example_cases(m, primary), query, real, s, primary, out / "examples.png",
                  f"E8 rehearse, {case}: one case per reason (median SSIM), z-scored, lesion neighbourhood outlined")


def review_header(dataset: str, p: dict, s: Settings, mock_info: dict) -> str:
    """The first lines of `review.txt`: settings and the positive mock's counts."""
    return (f"E8 rehearse ({dataset}): the analysis `paired` will run on S, on held-out and planted sets (no S read).\n"
            f"Settings: brain {s.brain_method} (> {s.low_threshold}); informative bound {s.bound} px; lesion region "
            f"+{s.grow_px} px; SSIM sigma {s.sigma}, data range {s.data_range} (z-units); regions {', '.join(s.regions)}; "
            f"primary {p['measure']['primary']}; k = {p['controls']['k']} controls per case.\n"
            f"Positive mock: {mock_info['cases']} of {mock_info['candidates']} planted {'/'.join(mock_info['variants'])} masks "
            f"(gated, E2-flagged, base patient as source); image = base BRAVO {mock_info['image_offset']} slices away, "
            f"moved by the planted shift.")
