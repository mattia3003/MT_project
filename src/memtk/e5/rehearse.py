"""E5 stage `rehearse` (`docs/e5-plan.md`, step 3): the scoring and analysis code that will run on S,
run first on mock query sets made from real data, so that every table, test and figure can be
checked before S is read. S is never read.

- Null case: H-cal as the query set (held-out patients, never trained on). Expected: no excess
  over H-eval. H-cal set the thresholds, so it sits at about 5% by construction against about 7%
  on H-eval; the excess can come out negative. This tests the code, not a hypothesis.
- Positive case: H-cal plus the gated planted copies P1, P2 and P4 of E4 (exact, shifted 2 px,
  rotated 5 degrees; known source). Expected: a clear excess carried by the planted masks, sources
  that name their base patients, near-exact copies among P1.

Mock sets have no stray noise: their drop-only label maps are their label maps. Each planted mask
is its own cluster (like a synthetic sample), H-cal slices are clustered by patient. Writes the raw
tables to `<output_root>/<dataset>/e5/rehearse/<case>/` and the analysis to
`results/<dataset>/e5/rehearse/<case>/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from memtk import config, lesionmatch
from memtk.calibration import load_frozen
from memtk.audit import Scored, heldout, load_scored, query_table
from memtk.e5.analyze import analyze_scored, draw_figures, informative_areas, review_text, write_tables
from memtk.e5.common import audit_config, audit_hash, check_prepared, drop_only_t
from memtk.e5.score import score_query
from memtk.io import slice_images
from memtk.sets import SliceSet, load_planted, load_set
from memtk.stages import refuse_overwrite

PLANTED_VARIANTS = ("P1", "P2", "P4")


# --- stages -----------------------------------------------------------------------------------

def run_rehearse(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Score and analyse the null and positive mock query sets with the E5 code, against H-eval.
    Needs `prepare` (made with the current `audit:` section). Writes
    `<output_root>/<dataset>/e5/rehearse/` and `results/<dataset>/e5/rehearse/`."""
    name, _ = config.real_dataset(cfg, dataset)
    data_root = config.output_dir(cfg, name) / "e5" / "rehearse"
    out_root = config.results_dir(cfg, name, "e5") / "rehearse"
    refuse_overwrite([out_root, data_root], overwrite)
    audit = audit_config(cfg)
    e3, e3_hash = load_frozen(cfg, name)
    e5, e5_hash = load_frozen(cfg, name, "e5")
    check_prepared(cfg, e5, e3_hash)
    draws, rng = int(cfg["calibration"]["bootstrap_draws"]), np.random.default_rng(cfg["seed"])
    variants = list(audit["sensitivity"])
    cut = float(e3["detectors"]["lesion"]["candidate_min_s"])

    t, hcal, heval = (load_set(cfg, name, s) for s in ("T", "H-cal", "H-eval"))
    t_drop = drop_only_t(cfg, name, t)
    h = heldout(cfg, name, heval, variants)
    t_areas = informative_areas(t, int(e3["informative_bound_px"]))
    planted, planted_labels = load_planted(cfg, name)
    chosen = (planted["variant"].isin(PLANTED_VARIANTS) & planted["keeps_base_lesions"].astype(bool)).to_numpy()
    planted, planted_labels = planted[chosen].reset_index(drop=True), planted_labels[chosen]
    base_of = dict(zip(planted["key"], planted["base_key"]))
    patient_of = dict(zip(planted["key"], planted["patient_id"]))

    cases = {"null": mock_set("H-cal", hcal, None, None),
             "positive": mock_set("H-cal + planted P1, P2, P4", hcal, planted, planted_labels)}
    labels = {**dict(zip(t.keys, t.labels)), **dict(zip(hcal.keys, hcal.labels)),
              **dict(zip(planted["key"], planted_labels))}

    def labels_of(keys):
        return {k: labels[k] for k in keys}

    def images_of(keys):
        real = slice_images(cfg, name, {base_of.get(k, k) for k in keys})
        return {k: real[base_of.get(k, k)] for k in keys}

    run_extra = {"stage": "rehearse", "inputs": ["T", "H-cal", "H-eval", "planted"], "frozen_sha256": e3_hash,
                 "e5_sha256": e5_hash, "audit_sha256": audit_hash(cfg)}
    for case, (q, meta) in cases.items():
        print(f"Rehearsal '{case}': scoring {len(q.keys)} mock queries ...")
        data_dir, out = data_root / case, out_root / case
        log = score_query(q, q.labels, meta, t, t_drop, e3, e5, variants, data_dir)
        scored = load_scored(case, data_dir, variants)
        tables, reading = analyze_scored(scored, h, lesionmatch.load_tables(data_dir, "main", cut), t_areas, e3, e5,
                                         audit, draws, rng)
        write_tables(tables, out)
        draw_figures(scored, h, tables, e3, f"Rehearsal '{case}'", labels_of, images_of, out)
        extra = planted_recovery(tables["flagged"], scored, patient_of) if case == "positive" else ""
        header = (f"E5 rehearse, case '{case}': dataset '{name}', mock query set {q.name} (S never read); "
                  f"the code that will run on S.\n{EXPECTED[case]}")
        text = review_text(header, scored, tables, reading, log) + extra
        (out / "review.txt").write_text(text, encoding="utf-8")
        config.write_run_info(data_dir, cfg, script=__file__, filename="rehearse.run_info.json", extra=run_extra)
        config.write_run_info(out, cfg, script=__file__, extra=run_extra)
        print(f"\n{text}\nWrote {data_dir}\nWrote {out}")


EXPECTED = {
    "null": "Expected: no excess over H-eval and AUC about 0.5 (H-cal set the thresholds, so its rates sit near 5% "
            "against about 7% on H-eval: the excess may be negative); no near-exact copy.",
    "positive": "Expected: a clear excess carried by the planted copies; their sources name their base patients; "
                "near-exact copies among P1 (except single lesions under 12 px).",
}


# --- inputs -----------------------------------------------------------------------------------

def mock_set(name: str, hcal: SliceSet, planted: pd.DataFrame | None,
             planted_labels: np.ndarray | None) -> tuple[SliceSet, pd.DataFrame]:
    """A mock query set and its meta table: H-cal slices (clustered by patient), plus the planted
    masks if given (each its own cluster)."""
    keys, clusters, labels = list(hcal.keys), list(hcal.patients), [hcal.labels]
    scheme = ["H-cal"] * len(hcal.keys)
    if planted is not None and planted_labels is not None:
        keys += planted["key"].tolist()
        clusters += planted["key"].tolist()
        labels.append(planted_labels.astype(hcal.labels.dtype))
        scheme += ["planted"] * len(planted)
    q = SliceSet(name, keys, np.asarray(clusters), np.concatenate(labels), {})
    return q, query_table(q, clusters, ["clean"] * len(keys), scheme)


# --- rehearsal checks (the figures are `analyze.draw_figures`, shared with S) ------------------

def planted_recovery(flagged: pd.DataFrame, scored: Scored, patient_of: dict) -> str:
    """Positive case: how many planted masks were flagged, whether the source patient each detector
    names is the base patient, near-exact copies per variant, and the base patients recovered."""
    meta = scored.meta[scored.meta["scheme"] == "planted"]
    f = flagged[flagged["key"].isin(meta["key"])].copy()
    f["variant"] = f["key"].str.split("__").str[0]
    f["base_patient"] = f["key"].map(patient_of)
    rows = []
    for variant, g in f.groupby("variant"):
        n = int(meta["key"].str.startswith(f"{variant}__").sum())
        e1 = g[g["e1_copy"].eq(True)]
        e2 = g[g["e2_class"].isin(["full", "partial"])]
        rows.append({"variant": variant, "planted": n, "flagged": len(g),
                     "e1_flagged": len(e1), "e1_right_patient": int((e1["e1_nn_patient"] == e1["base_patient"]).sum()),
                     "e2_flagged": len(e2), "e2_right_patient": int((e2["e2_source_patient"] == e2["base_patient"]).sum()),
                     "near_exact": int(g["near_exact"].sum()), "beyond_heldout_e2": int(g["beyond_heldout_e2"].sum())})
    table = pd.DataFrame(rows)
    groups = f.loc[f["e2_group"] != "", "e2_group"]
    recovered = {gr.split(":")[0] for gr in groups}
    return (f"\nRehearsal check, planted copies (decision J's source; base patient known):\n{table.to_string(index=False)}\n"
            f"E2 groups of planted masks: {groups.nunique()}, naming {len(recovered)} source patients; base patients: "
            f"{len(set(patient_of.values()))}, of them recovered: {len(recovered & set(patient_of.values()))}.\n")
