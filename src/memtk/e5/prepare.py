"""E5 stage `prepare` (needs decisions O, P, Q; `docs/e5-plan.md`): everything E5 fixes before S is
scored, from T, H-cal and H-eval only.

- Drop-only label maps (attach gap 0) of the real sets, made again from the model-space masks;
  H-cal and H-eval must equal their E0b label maps (they have no strays), which keeps the frozen
  thresholds valid for the drop-only variant.
- The thresholds of every sensitivity variant (decision O), re-derived on H-cal exactly as E3 did
  and compared with E3's `sensitivity.csv`; the drop-only thresholds must equal the frozen ones.
- tau_D* per stratum for H2 (decision P): the H-cal E1 percentile that flags E2 flagged's H-cal
  rate.
- The held-out maxima (H-cal and H-eval) of the E1 score per stratum and size bin (near-exact,
  decision Q) and of E2 C per stratum (the beyond-held-out tier).
- H-eval scored under every variant: the baseline `analyze` compares S with (main: E4's tables).

Writes `configs/frozen/e5_<dataset>.yaml` (in git; read by `score` and `analyze`), the H-eval
tables and the drop-only T label maps to `<output_root>/<dataset>/e5/`, and the evidence to
`results/<dataset>/e5/prepare/`. S is never read.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import yaml

from memtk import calibration, config, lesionmatch, wholemask
from memtk.audit import CLASSES, cell_maxima, class_flags, class_table, share_row
from memtk.calibration import STRATA, load_frozen
from memtk.e5.common import RERUN, audit_config, audit_hash, parse_variant, refuse_after_score, variant_frozen
from memtk.ids import parse_slice_key
from memtk.io import LabelStackWriter, load_real, read_table
from memtk.sets import SliceSet, load_set, relabel
from memtk.stages import refuse_overwrite, require

# Column of E3's sensitivity.csv that holds each variant kind's thresholds
E3_COLUMN = {"percentile": "percentile {v:g}", "sigma": "sigma {v:g} ({px:.1f} px)", "bound": "bound {v:g} px"}
TOL = 1e-9


# --- stages -----------------------------------------------------------------------------------

def run_prepare(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The E5 file fixed before S is scored: sensitivity thresholds, tau_D*, held-out maxima, the
    drop-only check; and H-eval under every variant. Needs E3 `freeze`, E4 `validate` and decisions
    O, P, Q. Writes `configs/frozen/e5_<dataset>.yaml`, `<output_root>/<dataset>/e5/` and
    `results/<dataset>/e5/prepare/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    refuse_after_score(cfg, pool_name, "prepare")
    path = config.frozen_file(cfg, name, "e5")
    data_dir = config.output_dir(cfg, name) / "e5"
    out = config.results_dir(cfg, name, "e5") / "prepare"
    refuse_overwrite([path, out, data_dir / "heval_classes_drop_only.csv"], overwrite)
    audit = audit_config(cfg)
    e3, e3_hash = load_frozen(cfg, name)
    e4_dir = config.output_dir(cfg, name) / "e4"
    require(e4_dir / "heval_classes.csv", "e4_validate.py validate")
    sensitivity_path = config.results_dir(cfg, name, "e3") / "calibrate" / "sensitivity.csv"
    require(sensitivity_path, "e3_calibrate.py calibrate")
    draws, rng = int(cfg["calibration"]["bootstrap_draws"]), np.random.default_rng(cfg["seed"])
    gap = int(cfg["lesions"]["attach_gap"])
    min_area = int(e3["min_lesion_area"])
    field = int(cfg["preprocessing"]["spatial_size"][0])

    t, hcal, heval = (load_set(cfg, name, s) for s in ("T", "H-cal", "H-eval"))
    e3_dir = config.output_dir(cfg, name) / "e3"
    e1_hcal = read_table(e3_dir / "e1_hcal.csv")
    e2_hcal = lesionmatch.load_tables(e3_dir, "hcal", float(e3["detectors"]["lesion"]["candidate_min_s"]))
    patients = dict(zip(hcal.keys, hcal.patients))
    e1_heval = read_table(e4_dir / "e1_heval.csv")
    e2_heval = lesionmatch.load_tables(e4_dir, "heval", float(e3["detectors"]["lesion"]["candidate_min_s"]))
    hc_main = read_table(e4_dir / "heval_classes.csv")

    print("Drop-only label maps of the real sets ...")
    drop, drop_table = drop_only_sets(cfg, name, (t, hcal, heval), min_area, gap)
    print("Sensitivity thresholds on H-cal and H-eval under every variant ...")
    variants, threshold_table, heval_tables = {}, [], {}
    e3_sensitivity = pd.read_csv(sensitivity_path).set_index(["detector", "threshold", "stratum"])
    for vname in audit["sensitivity"]:
        variant, rows, tables = prepare_variant(vname, e3, field, (t, hcal, heval), drop, e1_hcal, e2_hcal,
                                                patients, e1_heval, e2_heval, e3_sensitivity, rng)
        variants[vname] = variant
        threshold_table += rows
        heval_tables[vname] = tables
    h2 = tau_d_star(e1_hcal, e3)
    e1_max, e2_max, maxima_table = heldout_maxima(cfg, name, e1_hcal, e1_heval, e2_hcal, patients, hc_main,
                                                  audit["near_exact"]["size_bins_px"])
    rates = heval_rates({"main": (hc_main, e3["hcal_chance_rates"]),
                         **{v: (heval_tables[v]["classes"], variants[v]["hcal_chance_rates"]) for v in variants}},
                        draws, rng)

    data_dir.mkdir(parents=True, exist_ok=True)
    writer = LabelStackWriter(t.labels.shape[1:])
    for k, m in zip(t.keys, drop["T"].labels):
        writer.add(k, m)
    writer.save(data_dir / "labels_drop_only_T.npz")
    for vname, tables in heval_tables.items():
        tables["classes"].to_csv(data_dir / f"heval_classes_{vname}.csv", index=False)
        if "e1" in tables:
            tables["e1"].to_csv(data_dir / f"e1_heval_{vname}.csv", index=False)
        if "e2" in tables:
            lesionmatch.save_tables(tables["e2"], data_dir, f"heval_{vname}")
    a_hash = audit_hash(cfg)
    run_extra = {"stage": "prepare", "inputs": ["T", "H-cal", "H-eval"], "frozen_sha256": e3_hash, "audit_sha256": a_hash}
    config.write_run_info(data_dir, cfg, script=__file__, filename="prepare.run_info.json", extra=run_extra)

    e5 = {
        "dataset": name,
        "written": datetime.now().isoformat(timespec="seconds"),
        "inputs": ["T", "H-cal", "H-eval"],
        "e3_frozen_sha256": e3_hash,
        "audit_sha256": a_hash,
        "variants": variants,
        "h2": h2,
        "near_exact": {"min_score": float(audit["near_exact"]["min_score"]),
                       "size_bins_px": [int(b) for b in audit["near_exact"]["size_bins_px"]],
                       "e1_maxima": e1_max, "e2_C_maxima": e2_max},
        "drop_only_labels": {r["set"]: {"slices": r["slices"], "changed_at_gap_0": r["changed_at_gap_0"]}
                             for r in drop_table.to_dict("records")},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# E5 values for '{name}' fixed before S is scored, written by 'e5_audit.py prepare' from T, H-cal and\n"
              f"# H-eval only. Do not edit by hand: 'score' and 'analyze' read this file and record its hash.\n")
    path.write_text(header + yaml.safe_dump(e5, sort_keys=False, allow_unicode=True), encoding="utf-8")
    config.write_run_info(path.parent, cfg, script=__file__, filename=f"e5_{name}.run_info.json", extra=run_extra)

    out.mkdir(parents=True, exist_ok=True)
    thresholds_df = pd.DataFrame(threshold_table)
    thresholds_df.to_csv(out / "thresholds.csv", index=False)
    maxima_table.to_csv(out / "heldout_maxima.csv", index=False)
    rates.to_csv(out / "heval_rates.csv", index=False)
    drop_table.to_csv(out / "drop_only.csv", index=False)
    text = review_text(name, t, hcal, heval, drop_table, thresholds_df, variants, h2, maxima_table, e2_max, rates,
                       audit["near_exact"]["size_bins_px"])
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra=run_extra)
    print(f"\n{text}\nWrote {path}\nWrote {data_dir}\nWrote {out}")


# --- drop-only label maps ---------------------------------------------------------------------

def real_masks(cfg: dict, dataset: str, keys: list[str]) -> np.ndarray:
    """The model-space masks as saved (E0a `seg` volumes, specks and strays included) of real
    slice keys, in key order; one volume read per patient."""
    by_patient: dict[str, list[tuple[int, int]]] = {}
    for i, key in enumerate(keys):
        patient, z = parse_slice_key(key)
        by_patient.setdefault(patient, []).append((i, z))
    real_dir = config.output_dir(cfg, dataset)
    masks = None
    for patient, items in by_patient.items():
        _, seg = load_real(real_dir, patient)
        if masks is None:
            masks = np.zeros((len(keys), *seg.shape[:2]), dtype=bool)
        for i, z in items:
            masks[i] = seg[:, :, z] > 0
    return masks if masks is not None else np.zeros((0, 0, 0), dtype=bool)


def drop_only_sets(cfg: dict, dataset: str, sets: tuple[SliceSet, ...], min_area: int,
                   gap: int) -> tuple[dict[str, SliceSet], pd.DataFrame]:
    """Each real set with its label maps made again at attach gap 0 (drop-only), and a table per
    set: slices, whether the configured gap reproduces the E0b label maps (the path is right), and
    the slices whose label maps change at gap 0 (H-cal and H-eval must have none: stops otherwise,
    since the frozen thresholds would no longer apply to the drop-only variant)."""
    out, rows = {}, []
    for s in sets:
        masks = real_masks(cfg, dataset, s.keys)
        again = relabel(masks, min_area, gap)
        drop = relabel(masks, min_area, 0)
        same_gap = [bool(np.array_equal(a, b)) for a, b in zip(again, s.labels)]
        changed = [k for k, a, b in zip(s.keys, drop, s.labels) if not np.array_equal(a, b)]
        rows.append({"set": s.name, "slices": len(s.keys), "reproduces_e0b_at_gap": int(sum(same_gap)),
                     "changed_at_gap_0": len(changed), "changed_keys": " ".join(changed)})
        out[s.name] = SliceSet(s.name, s.keys, s.patients, drop, dict(s.left_out))
    table = pd.DataFrame(rows)
    if (table["reproduces_e0b_at_gap"] != table["slices"]).any():
        raise SystemExit(f"Relabelling the saved masks does not reproduce the E0b label maps:\n{table.to_string()}")
    held_out = table[table["set"] != "T"]
    if held_out["changed_at_gap_0"].any():
        raise SystemExit("Held-out label maps change at attach gap 0, so the frozen thresholds do not apply to the "
                         f"drop-only variant; re-calibration would be needed (docs/e5-plan.md, decision O):\n{table.to_string()}")
    return out, table


# --- sensitivity variants ---------------------------------------------------------------------

def prepare_variant(vname: str, e3: dict, field: int, sets: tuple[SliceSet, SliceSet, SliceSet],
                    drop: dict[str, SliceSet], e1_hcal: pd.DataFrame, e2_hcal: lesionmatch.LesionMatch,
                    patients: dict, e1_heval: pd.DataFrame, e2_heval: lesionmatch.LesionMatch,
                    e3_sensitivity: pd.DataFrame, rng: np.random.Generator) -> tuple[dict, list[dict], dict]:
    """One variant of decision O: its settings, the thresholds re-derived on H-cal (stratified as
    the frozen file, decision H) with the H-cal chance rates they give, the comparison with E3
    (`sensitivity.csv`, or the frozen thresholds for drop-only), and H-eval scored under it."""
    t, hcal, heval = sets
    kind, value = parse_variant(vname)
    les = e3["detectors"]["lesion"]
    settings = {"percentile": float(e3["percentile"]), "informative_bound_px": int(e3["informative_bound_px"]),
                "sigma_px": float(les["sigma_px"]), "prefilter_px": float(les["prefilter_px"]),
                "candidate_min_s": float(les["candidate_min_s"]), "attach_gap": None}
    reference = t
    if kind == "percentile":
        settings["percentile"] = value
    elif kind == "sigma":
        settings["sigma_px"] = value * field
    elif kind == "bound":
        settings["informative_bound_px"] = int(value)
    else:
        settings["attach_gap"] = 0
        reference, hcal, heval = drop["T"], drop["H-cal"], drop["H-eval"]
    e2_settings = {"min_area": settings["informative_bound_px"], "sigma_px": settings["sigma_px"],
                   "prefilter_px": settings["prefilter_px"], "candidate_min_s": settings["candidate_min_s"]}
    wm, _ = calibration.frozen_settings(e3)

    e1_cal, e2_cal = e1_hcal, e2_hcal
    if "whole_mask" in RERUN[kind]:
        e1_cal = e1_table(hcal, reference, wm)
    if "lesion" in RERUN[kind]:
        e2_cal = lesionmatch.match(hcal.labels, hcal.keys, reference.labels, reference.keys, **e2_settings)
    table = calibration.thresholds(e1_cal, e2_cal, patients, settings["percentile"], 0, rng)
    values, _ = calibration.frozen_thresholds(table, e3["strata"])
    problems = calibration.shortcut_problems(values, e2_settings)
    if problems:
        raise SystemExit(f"Variant {vname}: " + "; ".join(problems))

    rows = []
    for detector, names in calibration.THRESHOLDS.items():
        for tname in names:
            for s in STRATA:
                if kind == "drop_only":
                    expected = e3["thresholds"][detector][tname][s]
                else:
                    column = E3_COLUMN[kind].format(v=value, px=settings["sigma_px"])
                    expected = float(e3_sensitivity.loc[(detector, tname, s), column])  # pyright: ignore[reportArgumentType]  # one cell
                v = values[detector][tname][s]
                rows.append({"variant": vname, "detector": detector, "threshold": tname, "stratum": s, "value": v,
                             "e3_value": expected, "equal": bool(abs(v - expected) <= TOL)})
    unequal = [r for r in rows if not r["equal"]]
    if unequal:
        source = "the frozen thresholds" if kind == "drop_only" else "E3 sensitivity.csv"
        raise SystemExit(f"Variant {vname}: thresholds re-derived on H-cal differ from {source}: "
                         + "; ".join(f"{r['threshold']} {r['stratum']} {r['value']} vs {r['e3_value']}" for r in unequal))

    variant = {"kind": kind, "rerun": list(RERUN[kind]), "settings": settings, "thresholds": values,
               "hcal_chance_rates": calibration.chance_rates_at(e1_cal, e2_cal, patients, values)}
    vf = variant_frozen(e3, variant)
    tables: dict = {}
    if kind == "drop_only":
        e1c, e2, e2c = calibration.run_frozen(heval.labels, heval.keys, reference, vf)
        tables.update(e1=e1c, e2=e2)
    else:
        e1c = calibration.e1_classes(e1_heval, e1_heval["n_lesions"].to_numpy(), vf)
        e2 = e2_heval
        if "lesion" in RERUN[kind]:
            e2 = lesionmatch.match(heval.labels, heval.keys, reference.labels, reference.keys, **e2_settings)
            tables["e2"] = e2
        e2c = calibration.e2_classes(e2, vf)
    tables["classes"] = class_table(heval, e1c, e2, e2c)
    return variant, rows, tables


def e1_table(s: SliceSet, reference: SliceSet, wm: dict) -> pd.DataFrame:
    """E1 of a set against a reference with the columns `calibration.thresholds` needs, as E3
    `calibrate` builds them: patient, lesion count and stratum."""
    e1 = wholemask.match(s.labels, reference.labels, reference.patients, s.keys, reference.keys, **wm)
    e1["patient_id"] = e1["query"].map(dict(zip(s.keys, s.patients)))
    e1["n_lesions"] = s.labels.reshape(len(s.labels), -1).max(axis=1)
    e1["stratum"] = calibration.stratum(e1["n_lesions"].to_numpy())
    return e1


# --- H2 and held-out maxima -------------------------------------------------------------------

def tau_d_star(e1_hcal: pd.DataFrame, e3: dict) -> dict:
    """Decision P: per stratum, the H-cal E1 percentile that flags as many H-cal slices as E2
    flagged (full or partial) does at the frozen thresholds (frozen `hcal_chance_rates`), and the
    share it actually flags (ties can move it slightly)."""
    rate = {s: float(e3["hcal_chance_rates"][s]["e2_flagged"]) for s in STRATA}
    tau, achieved = {}, {}
    for s in STRATA:
        scores = e1_hcal.loc[e1_hcal["stratum"] == s, "score"].to_numpy()
        tau[s] = calibration.percentile_threshold(scores, 100 * (1 - rate[s]))
        achieved[s] = round(float((scores >= tau[s]).mean()), 4)
    return {"rule": "E2 flagged vs E1 at tau_D*: the H-cal E1 percentile flagging E2 flagged's H-cal rate",
            "matched_rate": rate, "tau_D_star": tau, "hcal_rate_at_tau_D_star": achieved}


def heldout_maxima(cfg: dict, dataset: str, e1_hcal: pd.DataFrame, e1_heval: pd.DataFrame,
                   e2_hcal: lesionmatch.LesionMatch, patients: dict, hc_main: pd.DataFrame,
                   edges: list[int]) -> tuple[list[dict], dict, pd.DataFrame]:
    """Decision Q: the highest E1 score of the held-out slices (H-cal and H-eval together) per
    stratum and size bin of the largest lesion, and the highest E2 C(q, NN) per stratum; also each
    set on its own, for the review table."""
    largest = read_table(config.output_dir(cfg, dataset) / "slices.csv").set_index("key")["largest_lesion_px"]
    e1 = {"H-cal": e1_hcal[["query", "score", "stratum"]], "H-eval": e1_heval[["query", "score", "stratum"]]}
    e1["held-out"] = pd.concat(list(e1.values()), ignore_index=True)
    les = calibration.lesion_slices(e2_hcal, patients)
    c = {"H-cal": pd.DataFrame({"C": les["C"].to_numpy(), "stratum": les["stratum"].to_numpy()}),
         "H-eval": pd.DataFrame({"C": hc_main["e2_C"], "stratum": hc_main["stratum_e2"]}).query("stratum != 'none'")}
    c["held-out"] = pd.concat([c["H-cal"], c["H-eval"]], ignore_index=True)
    rows = []
    for name, d in e1.items():
        m = cell_maxima(d["score"].to_numpy(), d["stratum"].to_numpy(), d["query"].map(largest).to_numpy(), edges)
        rows += [{"set": name, "score": "e1", **r} for r in m.to_dict("records")]
    for name, d in c.items():
        for s, g in d.groupby("stratum"):
            rows.append({"set": name, "score": "e2_C", "stratum": s, "lower": -1, "max": float(g["C"].max()),
                         "slices": len(g)})
    table = pd.DataFrame(rows)
    held = table[table["set"] == "held-out"]
    e1_max = [{"stratum": r["stratum"], "lower": int(r["lower"]), "max": float(r["max"]), "slices": int(r["slices"])}
              for r in held[held["score"] == "e1"].to_dict("records")]
    e2_max = {str(r["stratum"]): float(r["max"]) for r in held[held["score"] == "e2_C"].to_dict("records")}
    return e1_max, e2_max, table


def heval_rates(classes: dict[str, tuple[pd.DataFrame, dict]], draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Per variant, class and stratum: the H-eval rate with its bootstrap-by-patient interval (the
    baseline S is compared with) next to the H-cal rate at the same thresholds."""
    rows = []
    for vname, (hc, hcal_rates) in classes.items():
        for cls in CLASSES:
            on, flag = class_flags(hc, cls)
            col = "stratum_e1" if cls == "e1_copy" else "stratum_e2"
            for s in (*STRATA, "all"):
                sel = on & ((hc[col] == s) | (s == "all"))
                share = share_row(flag[sel], hc.loc[sel, "patient_id"], draws, rng)
                rows.append({"variant": vname, "class": cls, "stratum": s, "slices": int(sel.sum()),
                             "patients": hc.loc[sel, "patient_id"].nunique(), "flagged": share["count"],
                             "rate": share["share"], "ci_low": share["ci_low"], "ci_high": share["ci_high"],
                             "hcal_rate": hcal_rates[s][cls] if s != "all" else np.nan})
    return pd.DataFrame(rows)


# --- review text ------------------------------------------------------------------------------

def review_text(name: str, t: SliceSet, hcal: SliceSet, heval: SliceSet, drop: pd.DataFrame,
                thresholds: pd.DataFrame, variants: dict, h2: dict, maxima: pd.DataFrame, e2_max: dict,
                rates: pd.DataFrame, edges: list[int]) -> str:
    """The readable summary of `prepare`: what was fixed before S, and what to look at."""
    wide = thresholds.pivot_table(index="variant", columns=["threshold", "stratum"], values="value", sort=False)
    e1_held = maxima[(maxima["set"] == "held-out") & (maxima["score"] == "e1")].pivot_table(
        index="lower", columns="stratum", values="max")
    e1_held.index = ["whole stratum" if i == -1 else calibration.bin_label(int(i), edges) for i in e1_held.index]
    full_c = [s for s, v in e2_max.items() if v >= 1.0]
    tier_note = (f" A held-out {', '.join(full_c)}-lesion slice already reaches C = 1, so no {', '.join(full_c)}-lesion "
                 f"mask can be beyond it (the E8 rule counts the multi-lesion tier only)." if full_c else "")
    per_set = maxima[(maxima["score"] == "e1") & (maxima["lower"] == -1)].pivot_table(
        index="set", columns="stratum", values="max")
    shown = rates.assign(ci=[f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(rates["ci_low"], rates["ci_high"])])
    shown = shown[shown["stratum"] != "all"]
    return (
        f"E5 prepare: dataset '{name}' (T, H-cal and H-eval only; S never read)\n\n"
        f"Sets: T {len(t.keys)}, H-cal {len(hcal.keys)}, H-eval {len(heval.keys)} slices.\n\n"
        f"Drop-only label maps (attach gap 0) from the masks as saved; the configured gap reproduces E0b for every "
        f"slice; held-out sets unchanged, so the frozen thresholds apply:\n"
        f"{drop[['set', 'slices', 'reproduces_e0b_at_gap', 'changed_at_gap_0', 'changed_keys']].to_string(index=False)}\n\n"
        f"Sensitivity thresholds re-derived on H-cal (stratified as frozen), every one equal to E3 sensitivity.csv "
        f"(drop-only: to the frozen thresholds):\n{wide.round(3).to_string()}\n\n"
        f"H-cal chance rates per variant:\n"
        + "".join(f"  {v}: {variants[v]['hcal_chance_rates']}\n" for v in variants)
        + f"\nH2 (decision P): tau_D* {({s: round(v, 4) for s, v in h2['tau_D_star'].items()})} flags "
        f"{h2['hcal_rate_at_tau_D_star']} of H-cal (target E2 flagged {h2['matched_rate']}).\n\n"
        f"Near-exact (decision Q): highest held-out E1 score (H-cal and H-eval) by size of the largest lesion:\n"
        f"{e1_held.round(3).to_string()}\nper set, whole stratum:\n{per_set.round(3).to_string()}\n"
        f"Highest held-out E2 C(q, NN) per stratum (beyond-held-out tier): "
        f"{({s: round(v, 3) for s, v in e2_max.items()})}.{tier_note}\n\n"
        f"H-eval rates per variant (the baseline; main = E4), bootstrap by patient, H-cal rate beside:\n"
        f"{shown[['variant', 'class', 'stratum', 'slices', 'flagged', 'rate', 'ci', 'hcal_rate']].round(3).to_string(index=False)}\n\n"
        f"Look at: the drop-only table (only T may change), the near-exact limits (a single lesion under 12 px can "
        f"never be near-exact), and the H-eval rates per variant. Next: 'e5_audit.py rehearse' (no S).\n")
