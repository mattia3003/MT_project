"""E7 stage `reports` (`docs/e6-e7-plan.md`, step 6; decisions X, Y, Z): match reports on S with the
generator checked on the planted copies, the deletion test on S, and the taxonomy of S's flags
against H-eval's chance flags. S comes only from E5's stored tables (set main); the stage refuses if
E5's lock changed, if E6 `exposure` has not run, or if the planted copies did not pass the
correctness criteria.

- **Selection** (decision X), one report per chained group (E5 decision N): every near-exact mask
  (E1) and every E2 mask beyond the held-out maximum (E5 decision Q); the top flag of each of the most
  over-exposed patients per primary cell (E6 `top_patients.csv`); a seeded random sample of S
  multi-lesion flags and of H-eval multi-lesion chance flags as contrast.
- **Context** (Panel C): source patient, near-exact and beyond-held-out tiers, the chained group and its
  size, the margins to the second-best patient, the source patient's E6 exposure ratio, the 3D lesions
  the source slice's lesions belong to (decision W) and their span; the chance rate in the verdict.
- **Deletion test** (decision Y) on every S E2-flagged multi-lesion mask with enough contributing
  pairs, and on H-eval's for reference.
- **Taxonomy** (rule-based, fixed on the planted copies) of every E2-flagged S mask and H-eval flag.
- **Patient profiles** (descriptive, added after the S run, at the user's request): for the most
  exposed patients of E6, their 3D lesions over the slices they span with the S masks matching each
  lesion on each slice, the S flags per slice, and the most-matched slice next to its S masks; the
  patient-level view of H1.

Writes `results/<pool>/e7/reports/` (reports in `reports/`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from memtk import config, lesionmatch
from memtk.calibration import STRATA, frozen_threshold, load_frozen
from memtk.e5.common import audit_config, lock_unchanged
from memtk.e6.common import exposure_config, load_heval, load_query, load_reference
from memtk.e7.common import chance_rates, class_rate, patient_name, reports_config
from memtk.e7.planted import run_deletion
from memtk.exposure import patient_of_keys
from memtk.io import LabelStack, read_table, slice_images
from memtk.reports import (FLAGGED, ReportData, deletion_summary, draw_patient_profile, draw_report, profile_tables,
                           report_data, taxonomy, verdict)
from memtk.sets import load_set
from memtk.stages import pick, refuse_overwrite, require
from memtk.synthetic import load_sample

PROFILE_PATIENTS = 3   # patient profiles per primary cell, E6's most exposed first (descriptive)
PROFILE_EXAMPLES = 5   # S masks shown next to a profile's most-matched slice


# --- stages -----------------------------------------------------------------------------------

def run_reports(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Match reports on S (decision X), the deletion test on S (Y) and the taxonomy (Z), against H-eval.
    Needs E5 `score` and `analyze` (lock unchanged), E6 `exposure`, E7 `planted` with every criterion
    passed. Writes `results/<pool>/e7/reports/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "e7") / "reports"
    refuse_overwrite([out], overwrite)
    hashes = lock_unchanged(cfg, name, pool_name)
    rep, audit, exp = reports_config(cfg), audit_config(cfg), exposure_config(cfg)
    e3, _ = load_frozen(cfg, name)
    criteria_path = config.results_dir(cfg, name, "e7") / "planted" / "criteria.csv"
    require(criteria_path, "e7_reports.py planted")
    if not read_table(criteria_path)["pass"].astype(bool).all():
        raise SystemExit("The report generator failed a correctness criterion on the planted copies (decision Z); "
                         "no report on S is drawn before it passes.")
    e6_dir = config.results_dir(cfg, pool_name, "e6") / "exposure"
    require(e6_dir / "top_patients.csv", "e6_exposure.py exposure")
    flagged_path = config.results_dir(cfg, pool_name, "e5") / "analyze" / "flagged.csv"
    require(flagged_path, "e5_audit.py analyze")
    rng = np.random.default_rng(cfg["seed"])
    cut = float(e3["detectors"]["lesion"]["candidate_min_s"])
    name_of, chance = patient_name, chance_rates(cfg, name)

    q = load_query("S", config.output_dir(cfg, pool_name) / "e5", audit["sets"]["main"], cut)
    h = load_heval(cfg, name, cut)
    flagged = read_table(flagged_path)
    flagged = flagged[flagged["set"] == "main"].reset_index(drop=True)
    flagged = flagged.assign(group=flagged.apply(group_of, axis=1))
    e6 = {k: read_table(e6_dir / f"{k}.csv") for k in ("exposure", "top_patients", "flag_details")}
    ref = load_reference(cfg, name)
    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")

    selection = select_reports(flagged, h.table, e6["top_patients"], e6["flag_details"], rep["selection"], exp, rng)
    print(f"Drawing {len(selection)} match reports ...")
    s_lm, h_lm = subset(q.e2, set(selection["key"]) | set(flagged["key"])), subset(h.e2, set(h.table["key"]))
    pool_labels = LabelStack(config.output_dir(cfg, pool_name) / "labels.npz")
    files = read_table(config.output_dir(cfg, pool_name) / "synthetic_index.csv").set_index("sample_id")["filename"]
    pool_dir = config.resolve_path(pool_cfg["dir"])
    t_labels, h_labels = dict(zip(t.keys, t.labels)), dict(zip(heval.keys, heval.labels))

    def labels_of(keys):
        return {k: t_labels[k] if k in t_labels else h_labels[k] if k in h_labels else pool_labels[k] for k in keys}

    cache: dict[str, np.ndarray] = {}

    def images_of(keys):
        missing = [k for k in keys if k not in cache]
        real = slice_images(cfg, name, [k for k in missing if k in t_labels or k in h_labels])
        cache.update({k: real[k] if k in real else load_sample(pool_dir / files[k], pool_cfg)[0] for k in missing})
        return {k: cache[k] for k in keys}

    context_of = Context(flagged, q, h, e6, ref, exp, e3, name_of)
    chosen = set(selection["key"])
    rows_of = {(s, r["key"]): r for s, table in (("S", flagged), ("H-eval", h.table))
               for _, r in table[table["key"].isin(chosen)].iterrows()}
    e1_of = {(s, r["query"]): r for s, table in (("S", q.e1), ("H-eval", h.e1))
             for _, r in table[table["query"].isin(chosen)].iterrows()}
    index = []
    for i, r in enumerate(selection.to_dict("records")):
        from_s, ident = r["set"] == "S", (r["set"], r["key"])
        row = rows_of[ident]
        d = report_data(row, e1_of[ident], s_lm if from_s else h_lm, e3)
        line = verdict(d, name_of, class_rate(chance, d.e2_class, d.stratum))
        path = out / "reports" / f"{i + 1:02d}_{str(r['reason']).replace(' ', '_')}.png"
        draw_report(d, labels_of, images_of, name_of, context_of(row, d, from_s), line, path,
                    f"E7 match report {i + 1}: {r['reason']} ({r['set']} {r['key']})")
        index.append({"report": path.name, "set": r["set"], "key": r["key"], "reason": r["reason"], "group": r["group"],
                      "e1_score": d.e1_score, "e2_class": d.e2_class, "C": d.C, "source": d.source,
                      "source_patient": d.source_patient, "taxonomy": taxonomy(d),
                      "verdict": line})

    print("Patient profiles ...")
    profiles = patient_profiles(flagged, q, e6["top_patients"], ref, exp, e3, labels_of, images_of, out / "patients")

    print("Deletion test on S and H-eval; taxonomy ...")
    deletion, summary = deletion_tables(flagged, h.table, s_lm, h_lm, pool_labels, h_labels, t, e3,
                                        int(rep["deletion"]["min_paired"]))
    taxa = taxonomy_table(flagged, h.table, s_lm, h_lm, e3)

    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(index).to_csv(out / "report_index.csv", index=False)
    deletion.to_csv(out / "deletion.csv", index=False)
    summary.to_csv(out / "deletion_summary.csv", index=False)
    taxa.to_csv(out / "taxonomy.csv", index=False)
    profiles.to_csv(out / "patient_profiles.csv", index=False)
    text = review_text(pool_name, pd.DataFrame(index), summary, taxa)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "reports", "inputs": ["T", "H-eval", "S tables (E5)",
                                                            "E6 exposure"], **hashes})
    print(f"\n{text}\nWrote {out}")


# --- selection (decision X) -------------------------------------------------------------------

def group_of(row: pd.Series) -> str:
    """The chained group (E5 decision N) a flag belongs to: its E2 group if E2 flags it, else its E1 group."""
    g = row["e2_group"] if row["e2_class"] in FLAGGED else row["e1_group"]
    return g if isinstance(g, str) else ""


def select_reports(flagged: pd.DataFrame, heval: pd.DataFrame, top: pd.DataFrame, details: pd.DataFrame, sel: dict,
                   exp: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Decision X: the reports to draw (`set`, `key`, `reason`, `group`), one per chained group, in the
    order strong tiers, most exposed patients, random S flags, random H-eval chance flags."""
    f = flagged
    rows: list[dict] = []
    taken: set[str] = set()

    def add(d: pd.DataFrame, score: str, reason: str) -> None:
        for _, r in d.sort_values(score, ascending=False).iterrows():
            g = r["group"] or r["key"]
            if g not in taken:
                taken.add(g)
                rows.append({"set": "S", "key": r["key"], "reason": reason, "group": r["group"]})

    if sel["strong_tiers"]:
        add(f[f["near_exact"].astype(bool)], "e1_score", "near-exact")
        add(f[f["beyond_heldout_e2"].astype(bool)], "e2_C", "beyond held-out")
    for cls, s in ((c, st) for c in exp["primary"]["classes"] for st in exp["primary"]["strata"]):
        cell = f"{cls} {s}"
        score = "e1_score" if cls == "e1_copy" else "e2_C"
        for p in top.loc[top["cell"] == cell, "patient_id"].head(int(sel["top_exposed_patients"])):
            keys = details.loc[(details["set"] == "query") & (details["cell"] == cell) & (details["source_patient"] == p),
                               "key"]
            add(f[f["key"].isin(keys)].nlargest(1, score), score, f"exposed {cell}")
    multi = f[((f["stratum_e1"] == "multi") & f["e1_copy"].astype(bool))
              | ((f["stratum_e2"] == "multi") & f["e2_class"].isin(FLAGGED))]
    candidates = multi[~multi["group"].isin(taken) & ~multi["key"].isin([r["key"] for r in rows])]
    for _, r in pick(rng, candidates.drop_duplicates("group"), int(sel["random_s_multi"])).iterrows():
        taken.add(r["group"])
        rows.append({"set": "S", "key": r["key"], "reason": "random S flag", "group": r["group"]})
    hm = heval[((heval["stratum_e1"] == "multi") & heval["e1_copy"].eq(True))
               | ((heval["stratum_e2"] == "multi") & heval["e2_class"].isin(FLAGGED))]
    for _, r in pick(rng, hm, int(sel["random_heval"])).iterrows():
        rows.append({"set": "H-eval", "key": r["key"], "reason": "H-eval chance flag", "group": ""})
    return pd.DataFrame(rows)


# --- context (Panel C) ------------------------------------------------------------------------

class Context:
    """Panel C lines of a report: tiers, chained group, margins, the E6 exposure ratio of the source
    patient, the 3D lesions of the source slice."""

    def __init__(self, flagged: pd.DataFrame, q, h, e6: dict, ref, exp: dict, e3: dict, name_of) -> None:
        self.e3, self.name_of = e3, name_of
        self.group_size = flagged[flagged["group"] != ""]["group"].value_counts()
        self.margin = {"S": q.e1.set_index("query")["margin"], "H-eval": h.e1.set_index("query")["margin"]}
        d = e6["flag_details"]
        e2d = d[d["cell"].str.startswith("e2_")].drop_duplicates(["set", "key"])
        self.e2_margin = {"S": e2d[e2d["set"] == "query"].set_index("key")["margin"],
                          "H-eval": e2d[e2d["set"] == "H-eval"].set_index("key")["margin"]}
        m = float(exp["expected"]["smoothing_m"])
        self.exposure = e6["exposure"][e6["exposure"]["m"] == m].set_index(["cell", "patient_id"])
        main = exp["lesions3d"]["main"]
        self.lesion3d = ref.lesion_map.set_index("lesion_id")[f"lesion3d_id_{main}"]
        self.span = ref.lesions3d[ref.lesions3d["method"] == main].set_index("lesion3d_id")

    def __call__(self, row: pd.Series, d: ReportData, from_s: bool) -> list[str]:
        """The context lines of one report."""
        who = "S" if from_s else "H-eval"
        tau_d = float(frozen_threshold(self.e3, "whole_mask", "tau_D", np.array([int(row["n_lesions"])]))[0])
        lines = ["query: synthetic mask (S)" if from_s else
                 "query: held-out patient (H-eval), never trained on: any match here is chance"]
        e1 = f"E1: whole-mask score {d.e1_score:.2f} (copy threshold {tau_d:.2f})"
        if from_s:
            e1 += f"; near-exact (beyond every held-out score): {'yes' if row['near_exact'] else 'no'}"
            e1 += f"; E2 C beyond the held-out maximum: {'yes' if row['beyond_heldout_e2'] else 'no'}"
        lines.append(e1)
        m1, m2 = (fmt(m.get(d.key, np.nan)) for m in (self.margin[who], self.e2_margin[who]))
        lines.append(f"margin to the second-best patient: E1 {m1}, E2 {m2} (small = many patients fit about as well)")
        if from_s and row["group"]:
            p, z = row["group"].split(":z")
            lines.append(f"chained group {self.name_of(p)}:z{z}: {int(self.group_size.get(row['group'], 1))} S flags")
        if d.source:
            cell = (f"e2_full {d.stratum}" if d.e2_class == "full" else f"e2_flagged {d.stratum}" if d.e2_class in FLAGGED
                    else f"e1_copy {row['stratum_e1']}")
            key = (cell, d.source_patient)
            if from_s and key in self.exposure.index:
                e = self.exposure.loc[key]
                lines.append(f"E6 exposure of {self.name_of(d.source_patient)} ({cell}): {int(e['observed'])} flags vs "
                             f"{e['expected']:.1f} expected, ratio {e['ratio']:.1f} [{e['ratio_low']:.1f}, "
                             f"{e['ratio_high']:.1f}]")
            ids = {self.lesion3d.get(str(p.train_lesion), "") for p in d.pairs.itertuples()} - {""}
            spans = [f"z{int(self.span.at[i, 'z_first'])}-z{int(self.span.at[i, 'z_last'])} ({int(self.span.at[i, 'span'])} "
                     f"slices)" for i in sorted(ids) if i in self.span.index]
            if spans:
                lines.append(f"paired source lesions belong to 3D lesions spanning {', '.join(spans)}")
        return lines


def fmt(value: float) -> str:
    """A score with two decimals, or "-" when missing."""
    return "-" if pd.isna(value) else f"{value:.2f}"


# --- patient profiles -----------------------------------------------------------------------

def patient_profiles(flagged: pd.DataFrame, q, top: pd.DataFrame, ref, exp: dict, e3: dict, labels_of, images_of,
                     out: Path) -> pd.DataFrame:
    """One patient profile (`reports.draw_patient_profile`) per patient of `profile_data`, drawn to
    `out/<patient>.png`; returns `profile_table` of the same data."""
    method = exp["lesions3d"]["main"]
    data = profile_data(flagged, q, top, ref, exp, e3)
    for p, (slices, lesions) in data.items():
        l3 = ref.lesions3d[(ref.lesions3d["method"] == method) & (ref.lesions3d["patient_id"] == p)
                           & (ref.lesions3d["n_2d_lesions"] > 0)]
        totals = (slices["e1_flags"] + slices["e2_flags"]).to_numpy()
        slice_key = f"{p}/z{int(slices['z'].to_numpy()[totals.argmax()])}"
        named = flagged[(flagged["e1_nn"] == slice_key) | (flagged["e2_source"] == slice_key)]
        examples = [(k, slice_key) for k in named.nlargest(PROFILE_EXAMPLES, "e1_score")["key"]]
        draw_patient_profile(p, l3, slices, lesions, examples, labels_of, images_of, out / f"{p}.png",
                             f"E7 patient profile {p}: where S's matches fall over the patient's metastases (all strata; "
                             f"descriptive, added after the S run)")
    return profile_table(data)


def profile_data(flagged: pd.DataFrame, q, top: pd.DataFrame, ref, exp: dict,
                 e3: dict) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """`reports.profile_tables` for each of the PROFILE_PATIENTS most exposed patients of every
    primary cell (E6 `top_patients.csv`), with the S masks' lesion pairs >= tau_L to their lesions and
    the S flags (all strata) as input."""
    patients = list(dict.fromkeys(p for c in top["cell"].unique()
                                  for p in top.loc[top["cell"] == c, "patient_id"].head(PROFILE_PATIENTS)))
    scorable = q.table[q.table["n_informative"] > 0]
    tau = pd.Series(frozen_threshold(e3, "lesion", "tau_L", scorable["n_informative"].to_numpy()),
                    index=scorable["key"].to_numpy())
    pairs = q.e2.lesion_pairs
    pairs = pairs[pairs["query"].isin(tau.index) & patient_of_keys(pairs["train"]).isin(patients)]
    credits = pairs[pairs["s"].to_numpy() >= pairs["query"].map(tau).to_numpy()][["query", "train_lesion"]]
    e1_src = flagged.loc[flagged["e1_copy"].astype(bool), "e1_nn"]
    e2_src = flagged.loc[flagged["e2_class"].isin(FLAGGED), "e2_source"]
    column = f"lesion3d_id_{exp['lesions3d']['main']}"
    return {p: profile_tables(p, ref.lesion_map, column, credits, e1_src, e2_src) for p in patients}


def profile_table(data: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> pd.DataFrame:
    """One row per 2D lesion of the profiled patients: its 3D lesion, z, area, matching S masks and the
    flags on its slice (`patient_profiles.csv`)."""
    rows = [lesions.merge(slices, on="z", how="left").assign(patient_id=p) for p, (slices, lesions) in data.items()]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# --- deletion and taxonomy --------------------------------------------------------------------

def deletion_tables(flagged: pd.DataFrame, heval: pd.DataFrame, s_lm: lesionmatch.LesionMatch,
                    h_lm: lesionmatch.LesionMatch, pool_labels, h_labels: dict, t, e3: dict,
                    min_paired: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decision Y on every E2-flagged multi-lesion mask of S (main) and of H-eval: the per-mask table
    (`set`: S, H-eval chance flags) and the summary per set."""
    s_multi = flagged[flagged["e2_class"].isin(FLAGGED) & (flagged["stratum_e2"] == "multi")]
    s_reports = [report_data(r, None, s_lm, e3) for _, r in s_multi.iterrows()]
    h_multi = heval[heval["e2_class"].isin(FLAGGED) & (heval["stratum_e2"] == "multi")]
    h_reports = [report_data(r, None, h_lm, e3) for _, r in h_multi.iterrows()]
    deletion = pd.concat([run_deletion(s_reports, {d.key: pool_labels[d.key] for d in s_reports}, min_paired, t, e3)
                          .assign(set="S"),
                          run_deletion(h_reports, h_labels, min_paired, t, e3).assign(set="H-eval chance flags")],
                         ignore_index=True)
    summary = pd.DataFrame([{"set": s, **deletion_summary(g)} for s, g in deletion.groupby("set", sort=False)])
    return deletion, summary


def subset(e2: lesionmatch.LesionMatch, keys: set[str]) -> lesionmatch.LesionMatch:
    """An E2 result restricted to the given query slices (fast lookups for many reports)."""
    return lesionmatch.LesionMatch(e2.queries, e2.lesions, e2.slice_pairs[e2.slice_pairs["query"].isin(keys)],
                                   e2.lesion_pairs[e2.lesion_pairs["query"].isin(keys)], e2.candidate_min_s)


def taxonomy_table(flagged: pd.DataFrame, heval: pd.DataFrame, s_lm: lesionmatch.LesionMatch,
                   h_lm: lesionmatch.LesionMatch, e3: dict) -> pd.DataFrame:
    """The taxonomy (rules fixed on the planted copies) of every E2-flagged mask of S (main) and of
    H-eval, per stratum: counts and shares per category."""
    rows = []
    for set_name, table, lm in (("S", flagged, s_lm), ("H-eval", heval, h_lm)):
        f = table[table["e2_class"].isin(FLAGGED)]
        cats = pd.Series([taxonomy(report_data(r, None, lm, e3)) for _, r in f.iterrows()], index=f.index)
        for s in STRATA:
            c = cats[f["stratum_e2"] == s].value_counts()
            for cat, n in c.items():
                rows.append({"set": set_name, "stratum": s, "taxonomy": cat, "count": int(n), "share": n / c.sum()})
    return pd.DataFrame(rows)


def review_text(pool: str, index: pd.DataFrame, summary: pd.DataFrame, taxa: pd.DataFrame) -> str:
    """The `reports` review: the reports drawn, the deletion test, the taxonomy."""
    with pd.option_context("display.width", 230, "display.max_columns", 30, "display.max_colwidth", 140,
                           "display.precision", 3):
        share = taxa.pivot_table(index=["stratum", "taxonomy"], columns="set", values="share").fillna(0)
        return (f"E7 reports: pool '{pool}' (S from E5's stored tables, main set; E5's lock unchanged; the generator "
                f"passed the planted correctness criteria)\n\n"
                f"Reports drawn (decision X), by reason:\n{index['reason'].value_counts().to_string()}\n\n"
                f"{index[['report', 'key', 'source_patient', 'e2_class', 'taxonomy', 'verdict']].to_string(index=False)}"
                f"\n\nDeletion test (decision Y; drop = before - after; flips = no longer flagged with the same source "
                f"patient after a full re-match):\n{summary.T.to_string(header=False)}\n\n"
                f"Taxonomy (rule-based, fixed on the planted copies) of every E2-flagged mask, share per stratum, S vs "
                f"H-eval chance flags:\n{share.to_string()}\n")
