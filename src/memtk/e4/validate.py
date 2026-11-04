"""E4 stage `validate` (needs decision K): the go/no-go gate before the main audit.

Runs both detectors with the settings and thresholds of the frozen E3 file on the planted set
(positives, from `plant`) and on H-eval (negatives: never trained on, so every flag is a false
positive), all against T. A planted mask is caught when it is flagged and the source patient is
its base patient (decision J); the criteria count only the gated masks (decision I's gate rule)
and are compared with `validation.criteria` (decision K). Reads T, H-eval, the planted set and the
frozen file; S is never read. Writes the detector tables to `<output_root>/<dataset>/e4/` (the
H-eval ones are E5's baseline) and the validation tables, gate verdict, figures and
`review.txt` to `results/<dataset>/e4/validate/`.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from memtk import calibration, config, lesionmatch, wholemask
from memtk.e4.common import load_frozen, load_planted, slice_images
from memtk.e4.plant import SIZE_BINS, draw_labels, zoom_window
from memtk.ids import parse_slice_key
from memtk.lesionmatch import LesionMatch
from memtk.planted import VARIANTS
from memtk.plotting import LESION_COLORS, hide_unused, save, tile_figure
from memtk.sets import SliceSet, load_set
from memtk.stages import decided, pick, refuse_overwrite, require

DETECTORS = {"e1": "e1_caught", "e2": "e2_caught", "any": "any_caught", "both": "both_caught"}
# Detector names in `validation.criteria` -> the caught column they test
CRITERION_DETECTORS = {"whole_mask": "e1_caught", "lesion": "e2_caught", "any": "any_caught", "both": "both_caught"}
FP_CLASSES = ("e1_copy", "e2_full", "e2_partial", "e2_flagged")
STRATA = ("single", "multi")
STRATUM_COLORS = {"single": "tab:blue", "multi": "tab:orange"}
MAX_MISSES = 24        # planted masks shown in misses.png
FLAG_EXAMPLES = 4      # H-eval false positives shown per class


# --- stages -----------------------------------------------------------------------------------

def run_validate(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Both detectors on the planted set and H-eval with the frozen thresholds; recall per
    variant, false positives per stratum, the gate verdict. Needs `plant` (made with the same
    frozen file) and decisions I, J and K. Writes `<output_root>/<dataset>/e4/` (detector tables,
    H-eval classes) and `results/<dataset>/e4/validate/`."""
    name, _ = config.real_dataset(cfg, dataset)
    data_dir = config.output_dir(cfg, name) / "e4"
    out = config.results_dir(cfg, name, "e4") / "validate"
    refuse_overwrite([out, data_dir / "e1_planted.csv"], overwrite)
    criteria = decided(cfg, "validation.criteria", "e4_validate.py plant")
    if decided(cfg, "validation.gate", "docs/e4-plan.md (decision I)") != "keeps_base_lesions":
        raise SystemExit("validation.gate: only 'keeps_base_lesions' is implemented.")
    if decided(cfg, "validation.caught", "docs/e4-plan.md (decision J)") != "flagged_and_source_patient":
        raise SystemExit("validation.caught: only 'flagged_and_source_patient' is implemented.")
    frozen, frozen_hash = load_frozen(cfg, name)
    plant_info = data_dir / "plant.run_info.json"
    require(plant_info, "plant")
    if json.loads(plant_info.read_text(encoding="utf-8"))["frozen_sha256"] != frozen_hash:
        raise SystemExit("The frozen file changed since 'plant' ran; run 'e4_validate.py plant --overwrite' first.")
    draws = int(cfg["calibration"]["bootstrap_draws"])
    rng = np.random.default_rng(cfg["seed"])

    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    planted, planted_labels = load_planted(cfg, name)
    e1_p, e2_p, e2c_p = run_detectors(planted_labels, planted["key"].tolist(), t, frozen)
    e1_h, e2_h, e2c_h = run_detectors(heval.labels, heval.keys, t, frozen)
    t_labels = dict(zip(t.keys, t.labels))
    l1_area = {k: int((t_labels[k] == 1).sum()) for k in planted["base_key"].unique()}
    outcomes = planted_outcomes(planted, e1_p, e2c_p, l1_area)
    hc = heval_classes(heval, e1_h, e2_h, e2c_h)

    validation = validation_table(outcomes, draws, rng)
    groups = group_recall(outcomes)
    fp = fp_table(hc, frozen, criteria, draws, rng)
    rq2 = rq2_table(outcomes)
    deviation = config.value(cfg, "validation.accepted_deviation")
    verdict, gate = gate_verdict(gate_table(outcomes, fp, criteria), deviation)

    data_dir.mkdir(parents=True, exist_ok=True)
    for tag, e1, e2 in (("planted", e1_p, e2_p), ("heval", e1_h, e2_h)):
        e1.to_csv(data_dir / f"e1_{tag}.csv", index=False)
        e2.queries.to_csv(data_dir / f"e2_{tag}_queries.csv", index=False)
        e2.lesions.to_csv(data_dir / f"e2_{tag}_lesions.csv", index=False)
        e2.slice_pairs.to_parquet(data_dir / f"e2_{tag}_slice_pairs.parquet", index=False)
        e2.lesion_pairs.to_parquet(data_dir / f"e2_{tag}_lesion_pairs.parquet", index=False)
    hc.to_csv(data_dir / "heval_classes.csv", index=False)
    run_extra = {"stage": "validate", "inputs": ["T", "H-eval", "planted"], "frozen_sha256": frozen_hash,
                 "verdict": verdict}
    config.write_run_info(data_dir, cfg, script=__file__, filename="validate.run_info.json", extra=run_extra)

    out.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(out / "planted_outcomes.csv", index=False)
    validation.to_csv(out / "validation.csv", index=False)
    groups.to_csv(out / "recall_by_group.csv", index=False)
    fp.to_csv(out / "fp_heval.csv", index=False)
    rq2.to_csv(out / "rq2_planted.csv", index=False)
    gate.to_csv(out / "gate.csv", index=False)

    score_figure(outcomes, hc, frozen, out / "score_distributions.png",
                 f"{name}: scores of the planted masks per variant and of H-eval (negatives), by stratum; dashed = "
                 f"frozen thresholds (blue single, orange multi); x = a mask outside the gate")
    recall_figure(groups, out / "recall_by_size.png",
                  f"{name}: share of all planted masks caught (decision J) by the size of the base's largest lesion, "
                  f"per variant and detector (unscorable masks count as missed)")
    factors_figure(e2_p, outcomes, out / "factors.png",
                   f"{name}: the three factors and s of each planted lesion paired with a lesion of its own base "
                   f"slice (E2 Hungarian pairs), per variant")
    misses = outcomes[outcomes["gated"] & outcomes["variant"].isin(criterion_variants(criteria))
                      & ~criterion_caught(outcomes, criteria)]
    labels = {**t_labels, **dict(zip(heval.keys, heval.labels)), **dict(zip(planted["key"], planted_labels))}
    flags = flag_examples(hc, rng)
    images = slice_images(cfg, name, [*outcomes["base_key"], *misses["e2_nn"].replace("", np.nan).dropna(),
                                      *misses["e1_nn"].dropna(), *flags["key"], *flags["nn"]])
    if len(misses):
        misses_figure(misses, labels, images, rng, out / "misses.png",
                      f"{name}: gated planted masks missed by their criterion's detector (up to {MAX_MISSES}): base, "
                      f"planted mask, and the nearest neighbour found (E2, else E1), with scores")
    flags_figure(flags, labels, images, out / "heval_flags.png",
                 f"{name}: H-eval slices (never trained on) flagged at the frozen thresholds, {FLAG_EXAMPLES} random "
                 f"per class, next to the training slice that flagged them: what a chance match looks like")
    text = review_text(name, t, heval, outcomes, validation, groups, fp, rq2, gate, verdict, deviation, len(misses))
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra=run_extra)
    print(f"\n{text}\nWrote {data_dir}\nWrote {out}")


# --- detectors --------------------------------------------------------------------------------

def run_detectors(labels: np.ndarray, keys: list[str], t: SliceSet,
                  frozen: dict) -> tuple[pd.DataFrame, LesionMatch, pd.DataFrame]:
    """E1 (masks with a lesion only) and E2 of a stack of label maps against T with the frozen
    settings; the E1 table with its frozen classes, the E2 results, and the E2 classes."""
    wm, les = calibration.frozen_settings(frozen)
    n_lesions = labels.reshape(len(labels), -1).max(axis=1)
    scorable = n_lesions > 0
    e1 = wholemask.match(labels[scorable], t.labels, t.patients, [k for k, s in zip(keys, scorable) if s],
                         t.keys, **wm)
    e2 = lesionmatch.match(labels, keys, t.labels, t.keys, **les)
    return calibration.e1_classes(e1, n_lesions[scorable], frozen), e2, calibration.e2_classes(e2, frozen)


def planted_outcomes(planted: pd.DataFrame, e1: pd.DataFrame, e2c: pd.DataFrame, l1_area: dict) -> pd.DataFrame:
    """One row per planted mask: identity, strata, `gated` (decision I's gate rule; P6-remove
    always), the base's largest lesion area and size bin, both detectors' results (unscorable:
    class "none", never caught), and caught by decision J: flagged and the source patient is the
    base patient (E1: the nearest neighbour; E2 full: the nearest neighbour; E2 partial: the
    strongest-pair slice)."""
    out = planted[["key", "variant", "base_key", "patient_id", "group", "base_stratum", "stratum_e1", "stratum_e2",
                   "keeps_base_lesions"]].copy()
    out["gated"] = out["keeps_base_lesions"].astype(bool) | (out["variant"] == "P6-remove")
    out["base_l1_area"] = out["base_key"].map(l1_area)
    out["size_bin"] = [calibration.bin_label(int(b), SIZE_BINS)
                       for b in calibration.area_bins(out["base_l1_area"].to_numpy(), SIZE_BINS)]
    a = e1.set_index("query")
    out["e1_score"] = out["key"].map(a["score"])
    out["e1_nn"] = out["key"].map(a["nn"])
    out["e1_nn_patient"] = out["key"].map(a["nn_patient"])
    out["e1_copy"] = out["key"].map(a["copy"]).eq(True)
    b = e2c.set_index("query")
    out["e2_class"] = out["key"].map(b["class"]).fillna("none")
    for col in ("C", "nn", "nn_patient", "strongest_s", "strongest_train", "strongest_patient", "n_kept", "all_copied"):
        out[f"e2_{col}"] = out["key"].map(b[col])
    out["e1_caught"] = out["e1_copy"] & (out["e1_nn_patient"] == out["patient_id"])
    out["e2_caught"] = (((out["e2_class"] == "full") & (out["e2_nn_patient"] == out["patient_id"]))
                        | ((out["e2_class"] == "partial") & (out["e2_strongest_patient"] == out["patient_id"])))
    out["any_caught"] = out["e1_caught"] | out["e2_caught"]
    out["both_caught"] = out["e1_caught"] & out["e2_caught"]
    out["e1_nn_is_base"] = out["e1_nn"] == out["base_key"]
    out["e2_nn_is_base"] = out["e2_nn"] == out["base_key"]
    return out


def heval_classes(heval: SliceSet, e1: pd.DataFrame, e2: LesionMatch, e2c: pd.DataFrame) -> pd.DataFrame:
    """One row per H-eval slice: patient, E1 strata, score, nearest neighbour and copy flag; E2
    stratum (by informative lesions; "none" without one), class, C, nearest neighbour and
    strongest pair. Every flag is a false positive."""
    out = pd.DataFrame({"key": heval.keys, "patient_id": heval.patients})
    a = e1.set_index("query")
    out["n_lesions"] = out["key"].map(a["n_lesions"])
    out["stratum_e1"] = out["key"].map(a["stratum"])
    for col in ("score", "nn", "nn_patient", "copy"):
        out[f"e1_{col}"] = out["key"].map(a[col])
    out["n_informative"] = out["key"].map(e2.queries.set_index("query")["n_informative"]).fillna(0).astype(int)
    out["stratum_e2"] = np.where(out["n_informative"] == 0, "none", calibration.stratum(out["n_informative"].to_numpy()))
    b = e2c.set_index("query")
    out["e2_class"] = out["key"].map(b["class"]).fillna("none")
    for col in ("C", "nn", "nn_patient", "strongest_s", "strongest_train", "n_kept", "all_copied"):
        out[f"e2_{col}"] = out["key"].map(b[col])
    return out


# --- tables -----------------------------------------------------------------------------------

def share_row(flags: pd.Series, patients: pd.Series, draws: int, rng: np.random.Generator) -> dict:
    """Count, share and bootstrap-by-patient interval of a boolean column."""
    low, high = calibration.bootstrap_share(flags.to_numpy(), patients.to_numpy(), draws, rng)
    return {"count": int(flags.sum()), "share": float(flags.mean()) if len(flags) else np.nan,
            "ci_low": low, "ci_high": high}


def validation_table(outcomes: pd.DataFrame, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Per variant, subset (`gated` = the masks the criteria count; `all` = every planted mask,
    unscorable ones counted as missed) and detector (e1, e2, any, both): masks, base patients,
    caught and recall with its bootstrap-by-patient interval, per-patient recall (patients with
    every mask caught), and for e1 / e2: flagged, source-patient accuracy among flagged, the
    nearest neighbour being the base slice among caught; for e2 also full, partial, all_copied."""
    rows = []
    for variant in VARIANTS:
        for subset in ("gated", "all"):
            d = outcomes[(outcomes["variant"] == variant) & (outcomes["gated"] | (subset == "all"))]
            for det, col in DETECTORS.items():
                share = share_row(d[col], d["patient_id"], draws, rng)
                row = {"variant": variant, "subset": subset, "detector": det, "masks": len(d),
                       "patients": d["patient_id"].nunique(), "caught": share["count"], "recall": share["share"],
                       "ci_low": share["ci_low"], "ci_high": share["ci_high"],
                       "patient_recall": float(d.groupby("patient_id")[col].all().mean()) if len(d) else np.nan}
                if det in ("e1", "e2"):
                    flagged = d["e1_copy"] if det == "e1" else d["e2_class"].isin(["full", "partial"])
                    caught = d[col]
                    row.update({"flagged": int(flagged.sum()),
                                "source_accuracy": float(caught.sum() / flagged.sum()) if flagged.any() else np.nan,
                                "nn_is_base": float(d.loc[caught, f"{det}_nn_is_base"].mean()) if caught.any() else np.nan})
                if det == "e2":
                    row.update({"full": int((d["e2_class"] == "full").sum()),
                                "partial": int((d["e2_class"] == "partial").sum()),
                                "all_copied": int(d["e2_all_copied"].eq(True).sum())})
                rows.append(row)
    return pd.DataFrame(rows)


def group_recall(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Recall over all planted masks of each variant (unscorable = missed) by the base's stratum
    and by the size bin of the base's largest lesion, for e1, e2 and any."""
    rows = []
    for variant in VARIANTS:
        d = outcomes[outcomes["variant"] == variant]
        for by, col, values in (("stratum", "base_stratum", STRATA),
                                ("size", "size_bin", [calibration.bin_label(b, SIZE_BINS) for b in SIZE_BINS])):
            for value in values:
                g = d[d[col] == value]
                rows.append({"variant": variant, "by": by, "group": value, "masks": len(g),
                             **{det: float(g[DETECTORS[det]].mean()) if len(g) else np.nan for det in ("e1", "e2", "any")}})
    return pd.DataFrame(rows)


def fp_flags(hc: pd.DataFrame, cls: str) -> tuple[pd.Series, pd.Series]:
    """The H-eval slices a false-positive class is counted on, and which of them it flags."""
    if cls == "e1_copy":
        on = hc["stratum_e1"].notna()
        return on, hc["e1_copy"].eq(True)
    on = hc["stratum_e2"] != "none"
    wanted = {"e2_full": ["full"], "e2_partial": ["partial"], "e2_flagged": ["full", "partial"]}[cls]
    return on, hc["e2_class"].isin(wanted)


def fp_table(hc: pd.DataFrame, frozen: dict, criteria: dict, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """False positives on H-eval per class and stratum (single, multi, all): slices, patients,
    flagged, rate with its bootstrap-by-patient interval, the H-cal chance rate at the same
    thresholds (frozen file), and the criterion's limit and result where one is set."""
    limits = criteria.get("false_positive_max", {})
    rows = []
    for cls in FP_CLASSES:
        on, flag = fp_flags(hc, cls)
        stratum_col = "stratum_e1" if cls == "e1_copy" else "stratum_e2"
        for stratum in (*STRATA, "all"):
            sel = on & ((hc[stratum_col] == stratum) | (stratum == "all"))
            share = share_row(flag[sel], hc.loc[sel, "patient_id"], draws, rng)
            limit = limits.get(cls) if stratum != "all" else None
            rows.append({"class": cls, "stratum": stratum, "slices": int(sel.sum()),
                         "patients": hc.loc[sel, "patient_id"].nunique(), "flagged": share["count"],
                         "rate": share["share"], "ci_low": share["ci_low"], "ci_high": share["ci_high"],
                         "hcal_rate": frozen["hcal_chance_rates"][stratum][cls] if stratum != "all" else np.nan,
                         "limit": limit, "pass": (share["share"] <= limit) if limit is not None else None})
    return pd.DataFrame(rows)


def rq2_table(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Per variant, the gated masks caught by E2 only, by E1 only, by both and by neither: the
    first evidence for RQ2 (lesion-level matching finds copies whole-mask Dice misses)."""
    rows = []
    for variant in VARIANTS:
        d = outcomes[(outcomes["variant"] == variant) & outcomes["gated"]]
        e1, e2 = d["e1_caught"], d["e2_caught"]
        rows.append({"variant": variant, "masks": len(d), "e2_only": int((e2 & ~e1).sum()),
                     "e1_only": int((e1 & ~e2).sum()), "both": int((e1 & e2).sum()), "neither": int((~e1 & ~e2).sum())})
    return pd.DataFrame(rows)


def criterion_variants(criteria: dict) -> list[str]:
    """Every variant a recall criterion counts."""
    return [v for name, spec in criteria["recall"].items() for v in spec.get("variants", [name])]


def criterion_caught(outcomes: pd.DataFrame, criteria: dict) -> pd.Series:
    """Per planted mask, whether the detector of its variant's recall criterion caught it (False
    for variants without a criterion)."""
    caught = np.zeros(len(outcomes), dtype=bool)
    for name, spec in criteria["recall"].items():
        sel = outcomes["variant"].isin(spec.get("variants", [name])).to_numpy()
        caught[sel] = outcomes.loc[sel, CRITERION_DETECTORS[spec["detector"]]].to_numpy(dtype=bool)
    return pd.Series(caught, index=outcomes.index)


def gate_table(outcomes: pd.DataFrame, fp: pd.DataFrame, criteria: dict) -> pd.DataFrame:
    """One row per criterion of decision K: the recall of the gated masks of its variants with its
    detector (>= min), and each false-positive limit per stratum (<= max); value, limit, pass.
    A recall criterion without gated masks fails."""
    rows = []
    for name, spec in criteria["recall"].items():
        variants = spec.get("variants", [name])
        d = outcomes[outcomes["variant"].isin(variants) & outcomes["gated"]]
        value = float(d[CRITERION_DETECTORS[spec["detector"]]].mean()) if len(d) else np.nan
        rows.append({"criterion": f"recall {name}", "detail": f"{spec['detector']}, {' + '.join(variants)}, {len(d)} gated masks",
                     "value": value, "limit": f">= {spec['min']}", "pass": bool(len(d) and value >= spec["min"])})
    for r in fp[fp["limit"].notna()].to_dict("records"):
        rows.append({"criterion": f"false positives {r['class']} {r['stratum']}", "detail": f"H-eval, {r['slices']} slices",
                     "value": r["rate"], "limit": f"<= {r['limit']}", "pass": bool(r["pass"])})
    return pd.DataFrame(rows)


def gate_verdict(gate: pd.DataFrame, deviation: dict | None) -> tuple[str, pd.DataFrame]:
    """The verdict and the gate table with `accepted` (a failed criterion listed in
    `validation.accepted_deviation`): PASS if every criterion passes; ACCEPTED if the failed
    criteria are exactly the listed ones (a recorded deviation, decided by the user); FAIL
    otherwise, also when the list names a criterion that passed or misses one that failed."""
    listed = set(deviation["criteria"]) if deviation else set()
    gate = gate.assign(accepted=gate["criterion"].isin(listed) & ~gate["pass"])
    failed = set(gate.loc[~gate["pass"], "criterion"])
    if not failed:
        return "PASS", gate
    return ("ACCEPTED" if failed == listed else "FAIL"), gate


# --- figures ----------------------------------------------------------------------------------

def score_figure(outcomes: pd.DataFrame, hc: pd.DataFrame, frozen: dict, path, title: str):
    """Per score (E1 score, E2 C of the nearest neighbour, E2 strongest pair): the planted masks of
    each variant and the H-eval slices as jittered points by stratum, with the frozen thresholds."""
    panels = (("e1_score", "stratum_e1", ("whole_mask", "tau_D"), "E1 score (best Dice over shifts)"),
              ("e2_C", "stratum_e2", ("lesion", "tau_C"), "E2 configuration score C(q, NN)"),
              ("e2_strongest_s", "stratum_e2", ("lesion", "tau_L"), "E2 strongest lesion pair s"))
    groups = [*VARIANTS, "H-eval"]
    jitter = np.random.default_rng(0)
    fig = Figure(figsize=(15, 4.2))
    for ax, (col, stratum_col, (det, thr), label) in zip(fig.subplots(1, 3), panels):
        for i, group in enumerate(groups):
            d = hc if group == "H-eval" else outcomes[outcomes["variant"] == group]
            for k, s in enumerate(STRATA):
                g = d[(d[stratum_col] == s) & d[col].notna()]
                x = i + (k - 0.5) * 0.3 + jitter.uniform(-0.12, 0.12, len(g))
                gated = g["gated"].to_numpy() if "gated" in g else np.ones(len(g), bool)
                ax.scatter(x[gated], g.loc[gated, col], s=4, alpha=0.5, color=STRATUM_COLORS[s])
                ax.scatter(x[~gated], g.loc[~gated, col], s=10, marker="x", alpha=0.7, color=STRATUM_COLORS[s])
        for s in STRATA:
            ax.axhline(frozen["thresholds"][det][thr][s], ls="--", lw=0.8, color=STRATUM_COLORS[s])
        ax.set_xticks(range(len(groups)), groups, rotation=45, fontsize=7)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(label, fontsize=9)
    return save(fig, path, title)


def recall_figure(groups: pd.DataFrame, path, title: str):
    """One panel per variant: recall of e1, e2 and any by the size bin of the base's largest lesion."""
    fig = Figure(figsize=(14, 6))
    bins = [calibration.bin_label(b, SIZE_BINS) for b in SIZE_BINS]
    for ax, variant in zip(fig.subplots(2, 4).ravel(), VARIANTS):
        d = groups[(groups["variant"] == variant) & (groups["by"] == "size")].set_index("group").reindex(bins)
        for det, style in (("e1", "s-"), ("e2", "o-"), ("any", "k--")):
            ax.plot(range(len(bins)), d[det], style, label=det, ms=4, lw=1)
        ax.set_xticks(range(len(bins)), [f"{b}\n(n={int(n)})" for b, n in zip(bins, d["masks"].fillna(0))], fontsize=6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(variant, fontsize=9)
        ax.legend(fontsize=6)
    return save(fig, path, title)


def factors_figure(e2: LesionMatch, outcomes: pd.DataFrame, path, title: str):
    """Box plots per variant of shape, position, size and s of the E2 pairs between each planted
    mask and its own base slice: where each variant loses its score."""
    base_of = dict(zip(outcomes["key"], outcomes["base_key"]))
    pairs = e2.lesion_pairs[e2.lesion_pairs["train"] == e2.lesion_pairs["query"].map(base_of)]
    variant_of = dict(zip(outcomes["key"], outcomes["variant"]))
    pairs = pairs.assign(variant=pairs["query"].map(variant_of))
    fig = Figure(figsize=(15, 3.8))
    for ax, factor in zip(fig.subplots(1, 4), ("shape", "position", "size", "s")):
        data = [pairs.loc[pairs["variant"] == v, factor].to_numpy() for v in VARIANTS]
        ax.boxplot(data, showfliers=True, flierprops={"markersize": 2})
        ax.set_xticks(range(1, len(VARIANTS) + 1), VARIANTS, rotation=45, fontsize=7)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(factor, fontsize=9)
    return save(fig, path, title)


def lesion_colors(labels: np.ndarray) -> dict[int, str]:
    """One colour per lesion number."""
    return {n: LESION_COLORS[(n - 1) % len(LESION_COLORS)] for n in range(1, int(labels.max()) + 1)}


def misses_figure(misses: pd.DataFrame, labels: dict, images: dict, rng: np.random.Generator, path, title: str):
    """Per missed planted mask (up to MAX_MISSES): its base, the planted mask, and the nearest
    neighbour found (E2, else E1), on a shared zoom window, with the scores."""
    chosen = pick(rng, misses, MAX_MISSES)
    fig, axes = tile_figure(int(np.ceil(len(chosen) / 2)), 6, tile=1.9)
    tiles = axes.ravel()
    for k, r in enumerate(chosen.itertuples(index=False)):
        nn = r.e2_nn if isinstance(r.e2_nn, str) and r.e2_nn else r.e1_nn
        flagged = r.e1_copy or r.e2_class in ("full", "partial")
        why = "flagged, wrong patient" if flagged else "not flagged (below thresholds)"
        shown = [(r.base_key, labels[r.base_key], images[r.base_key], f"base {r.base_key}"),
                 (r.key, labels[r.key], images[r.base_key], f"{r.variant}: {why}\nE1 {r.e1_score:.2f}, E2 {r.e2_class}"),
                 (nn, labels.get(nn), images.get(nn),
                  f"NN {nn}\nC {r.e2_C:.2f}, strongest s {r.e2_strongest_s:.2f}")]
        win = zoom_window([m > 0 for _, m, _, _ in shown if m is not None])
        for ax, (_, m, image, caption) in zip(tiles[3 * k:3 * k + 3], shown):
            if m is None or image is None:
                ax.axis("off")
                continue
            draw_labels(ax, image[win], m[win], lesion_colors(m), caption)
    hide_unused(axes, 3 * len(chosen))
    return save(fig, path, title)


def flag_examples(hc: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Up to FLAG_EXAMPLES random H-eval false positives per class, each with the training slice
    that flagged it (E1: nearest neighbour; E2 full: nearest neighbour; E2 partial: strongest pair)."""
    rows = []
    for cls, sel, nn_col, score in (("E1 copy", hc["e1_copy"].eq(True), "e1_nn", "e1_score"),
                                    ("E2 full", hc["e2_class"] == "full", "e2_nn", "e2_C"),
                                    ("E2 partial", hc["e2_class"] == "partial", "e2_strongest_train", "e2_strongest_s")):
        for r in pick(rng, hc[sel], FLAG_EXAMPLES).itertuples(index=False):
            rows.append({"class": cls, "key": r.key, "nn": getattr(r, nn_col), "score": getattr(r, score)})
    return pd.DataFrame(rows, columns=["class", "key", "nn", "score"])


def flags_figure(flags: pd.DataFrame, labels: dict, images: dict, path, title: str):
    """One row per class: H-eval slice and its training slice, side by side on a shared window."""
    classes = list(dict.fromkeys(flags["class"]))
    fig, axes = tile_figure(max(len(classes), 1), 2 * FLAG_EXAMPLES, tile=1.7)
    for row_axes, cls in zip(axes, classes):
        d = flags[flags["class"] == cls]
        for k, r in enumerate(d.itertuples(index=False)):
            q, t = labels[r.key], labels[r.nn]
            win = zoom_window([q > 0, t > 0])
            draw_labels(row_axes[2 * k], images[r.key][win], q[win], lesion_colors(q), f"{cls}\nH-eval {r.key}")
            draw_labels(row_axes[2 * k + 1], images[r.nn][win], t[win], lesion_colors(t),
                        f"T {r.nn} ({parse_slice_key(r.nn)[0]})\nscore {r.score:.2f}")
        hide_unused(row_axes, 2 * len(d))
    return save(fig, path, title)


# --- review text ------------------------------------------------------------------------------

def review_text(name: str, t: SliceSet, heval: SliceSet, outcomes: pd.DataFrame, validation: pd.DataFrame,
                groups: pd.DataFrame, fp: pd.DataFrame, rq2: pd.DataFrame, gate: pd.DataFrame, verdict: str,
                deviation: dict | None, n_misses: int) -> str:
    """The readable summary, ending with the gate verdict."""
    gated = validation[validation["subset"] == "gated"].copy()
    gated["text"] = [f"{r:.2f} [{lo:.2f}, {hi:.2f}] ({c}/{m})" for r, lo, hi, c, m in
                     zip(gated["recall"], gated["ci_low"], gated["ci_high"], gated["caught"], gated["masks"])]
    recall = gated.pivot(index="variant", columns="detector", values="text").reindex(VARIANTS)[list(DETECTORS)]
    detail = gated[gated["detector"].isin(["e1", "e2"])].pivot(
        index="variant", columns="detector", values=["flagged", "source_accuracy", "nn_is_base"]).reindex(VARIANTS)
    e2 = gated[gated["detector"] == "e2"].set_index("variant")[["full", "partial", "all_copied"]].reindex(VARIANTS)
    sizes = groups[(groups["by"] == "size") & groups["variant"].isin(["P3", "P5-erode", "P5-dilate"])]
    shown_fp = fp.assign(ci=[f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(fp["ci_low"], fp["ci_high"])])
    failed = gate[~gate["pass"]]
    left = "; ".join(f"{s.name}: " + ", ".join(f"{v} {k}" for k, v in s.left_out.items()) for s in (t, heval))
    return (
        f"E4 validate: dataset '{name}' (T, H-eval and the planted set; settings and thresholds from the frozen file; "
        f"S never read)\n\n"
        f"Sets: T {len(t.keys)} slices; H-eval {len(heval.keys)} slices of {len(set(heval.patients))} patients "
        f"(left out: {left}); planted {len(outcomes)} masks, {int(outcomes['gated'].sum())} gated.\n\n"
        f"Recall of the gated masks (caught = flagged and the base patient as source, decision J), with "
        f"bootstrap-by-patient 95% intervals:\n{recall.to_string()}\n\n"
        f"Flagged, source-patient accuracy among flagged, nearest neighbour = the base slice among caught:\n"
        f"{detail.round(3).to_string()}\n\n"
        f"E2 classes of the gated masks (all_copied = every lesion of both slices paired >= tau_L):\n{e2.to_string()}\n\n"
        f"Reported only, recall of all masks by the size of the base's largest lesion (unscorable = missed):\n"
        f"{sizes.round(3).to_string(index=False)}\n\n"
        f"False positives on H-eval (every flag is chance), with the H-cal rate at the same thresholds:\n"
        f"{shown_fp[['class', 'stratum', 'slices', 'patients', 'flagged', 'rate', 'ci', 'hcal_rate', 'limit', 'pass']].round(3).to_string(index=False)}\n\n"
        f"RQ2, gated masks caught by E2 only / E1 only / both / neither:\n{rq2.to_string(index=False)}\n\n"
        f"Gate (decision K):\n{gate.round(3).to_string(index=False)}\n\n"
        f"Missed gated masks of a criterion's variants: {n_misses} (misses.png{'' if n_misses else ' not written'}).\n"
        f"Figures: score_distributions.png, recall_by_size.png, factors.png, heval_flags.png.\n\n"
        + verdict_text(verdict, failed, deviation))


def verdict_text(verdict: str, failed: pd.DataFrame, deviation: dict | None) -> str:
    """The closing lines of the review: the verdict and what it allows."""
    names = "; ".join(failed["criterion"])
    if verdict == "PASS":
        return "GATE: PASS. E5 may run.\n"
    if deviation and verdict == "ACCEPTED":
        return (f"GATE: ACCEPTED. Failed ({names}), accepted as a recorded deviation "
                f"(validation.accepted_deviation): {' '.join(str(deviation['reason']).split())} E5 may run.\n")
    mismatch = (f" validation.accepted_deviation lists {sorted(deviation['criteria'])}, which is not exactly the failed "
                f"criteria." if deviation else "")
    return (f"GATE: FAIL ({names}).{mismatch} Do not run E5. Look at misses.png and the tables; a P1 miss is a bug. "
            f"Otherwise the user decides which detector setting to change (top-k first for a missed shifted copy); "
            f"re-run E3 calibrate and freeze on H-cal, then 'e4_validate.py validate'. Never look at S.\n")
