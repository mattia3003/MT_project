"""E6 stage `reference` (`docs/e6-e7-plan.md`, step 2; needs decisions R, V, W): everything E6 needs
from the real data, before any query set is analysed. S is never read.

- 3D lesions of every T volume under both labelling methods of decision W (slice-to-slice linking,
  26-connectivity), on the model-space mask as saved, and the 3D lesion of every E0b 2D lesion.
- Slice rarity on both detectors' scales (decision V): each T slice against the T slices of other
  patients, with the detector settings of the frozen E3 file.
- Patient features and patient rarity (decision V), and each patient's share of the T lesion
  slices (the prior w_p of the smoothed pi_p, decision R).
- The nearest-neighbour patients of the scorable H-eval masks per detector and stratum (the counts
  pi_p is made from, decision R), from E4's `heval_classes.csv`.

Writes `<output_root>/<dataset>/e6/` (`lesions3d.csv`, `lesion_map.csv`, `rarity_slices.csv`,
`patients.csv`, `heldout_nn.csv`) and `results/<dataset>/e6/reference/` (`review.txt`,
`reference.png`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from memtk import config
from memtk.calibration import STRATA, frozen_settings, load_frozen
from memtk.e6.common import LABELLERS, exposure_config, reference_dir
from memtk.exposure import knn_rarity, patient_features, slice_rarity, slice_shares
from memtk.ids import lesion3d_id, lesion_id, parse_slice_key
from memtk.io import load_real, read_table
from memtk.lesions3d import lesion3d_table, map_2d_to_3d
from memtk.plotting import save
from memtk.sets import SliceSet, load_set
from memtk.stages import refuse_overwrite, require

# Nearest-neighbour columns of `heval_classes.csv` per detector: (stratum column, NN patient column)
NN_COLUMNS = {"e1": ("stratum_e1", "e1_nn_patient"), "e2": ("stratum_e2", "e2_nn_patient")}


# --- stages -----------------------------------------------------------------------------------

def run_reference(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The E6 reference tables of the real dataset: 3D lesions (both methods), slice and patient
    rarity, T slice shares, H-eval nearest-neighbour patients. Needs E0b `lesions`, E3 `freeze`, E4
    `validate` and decisions R, V, W. Writes `<output_root>/<dataset>/e6/` and
    `results/<dataset>/e6/reference/`."""
    name, _ = config.real_dataset(cfg, dataset)
    data_dir = reference_dir(cfg, name)
    out = config.results_dir(cfg, name, "e6") / "reference"
    refuse_overwrite([data_dir / "lesions3d.csv", out], overwrite)
    exp = exposure_config(cfg)
    e3, e3_hash = load_frozen(cfg, name)
    heval_path = config.output_dir(cfg, name) / "e4" / "heval_classes.csv"
    require(heval_path, "e4_validate.py validate")
    bound = int(e3["informative_bound_px"])
    methods = [exp["lesions3d"]["main"], exp["lesions3d"]["sensitivity"]]

    t = load_set(cfg, name, "T")
    lesions = read_table(config.output_dir(cfg, name) / "lesions.csv")
    lesions = lesions[(lesions["set"] == "T") & lesions["key"].isin(t.keys)]
    print(f"3D lesions of {len(np.unique(t.patients))} T volumes ({', '.join(methods)}) ...")
    table3d, lesion_map = lesions3d_tables(cfg, name, t, lesions, methods, bound)
    print("Slice rarity: every T slice against the other patients' slices (E1 and E2) ...")
    wm, les = frozen_settings(e3)
    rarity = slice_rarity(t, wm, les)
    main = exp["lesions3d"]["main"]
    feats = patient_features(lesion_map.rename(columns={f"lesion3d_id_{main}": "lesion3d_id"}))
    r = exp["rarity"]
    feats["rarity"] = knn_rarity(feats, r["features"], r["log1p"], int(r["patient_k"]))
    shares = slice_shares(t.patients)
    patients = feats.join(shares.rename("t_slice_share")).join(
        pd.Series(t.patients).value_counts().rename("t_slices")).rename_axis("patient_id").reset_index()
    heldout_nn = nn_counts(read_table(heval_path))

    data_dir.mkdir(parents=True, exist_ok=True)
    table3d.to_csv(data_dir / "lesions3d.csv", index=False)
    lesion_map.to_csv(data_dir / "lesion_map.csv", index=False)
    rarity.to_csv(data_dir / "rarity_slices.csv", index=False)
    patients.to_csv(data_dir / "patients.csv", index=False)
    heldout_nn.to_csv(data_dir / "heldout_nn.csv", index=False)
    run_extra = {"stage": "reference", "inputs": ["T", "H-eval tables"], "frozen_sha256": e3_hash}
    config.write_run_info(data_dir, cfg, script=__file__, filename="reference.run_info.json", extra=run_extra)

    out.mkdir(parents=True, exist_ok=True)
    reference_figure(table3d, rarity, patients, methods, bound, out / "reference.png",
                     f"E6 reference ({name}): 3D lesions, slice rarity, patient rarity")
    text = review_text(name, t, table3d, lesion_map, rarity, patients, heldout_nn, read_table(heval_path), methods,
                       bound)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra=run_extra)
    print(f"\n{text}\nWrote {data_dir}\nWrote {out}")


# --- 3D lesions -------------------------------------------------------------------------------

def lesions3d_tables(cfg: dict, dataset: str, t: SliceSet, lesions: pd.DataFrame, methods: list[str],
                     bound: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per method, the 3D lesions of every T volume (`lesions3d.lesion3d_table` with `method`,
    `n_2d_lesions` (E0b lesions mapped to it), `largest_informative_px` (its largest mapped 2D lesion
    of at least `bound` px; NaN if none) and `informative`); and the E0b 2D lesions of T with their 3D
    lesion per method (`lesion3d_id_<method>`; one volume read per patient)."""
    labels_of = dict(zip(t.keys, t.labels))
    lesion_map = lesions[["lesion_id", "key", "patient_id", "z", "label", "area",
                          "centroid_row_brain", "centroid_col_brain"]].copy()
    tables = []
    for method in methods:
        column, assigned = f"lesion3d_id_{method}", {}
        for p in np.unique(t.patients):
            _, seg = load_real(config.output_dir(cfg, dataset), p)
            labels3d = LABELLERS[method](seg)
            tables.append(lesion3d_table(labels3d, p).assign(method=method))
            for key in [k for k, q in zip(t.keys, t.patients) if q == p]:
                z = parse_slice_key(key)[1]
                for n, d in map_2d_to_3d(labels_of[key], labels3d[:, :, z]).items():
                    assigned[lesion_id(key, n)] = lesion3d_id(p, d) if d else ""
        lesion_map[column] = lesion_map["lesion_id"].map(assigned)
        if (lesion_map[column].fillna("") == "").any():
            raise ValueError(f"{method}: E0b lesions without a 3D lesion (all T lesion pixels are in the mask as saved)")
    table3d = pd.concat(tables, ignore_index=True)
    for method in methods:
        sel = table3d["method"] == method
        by3d = lesion_map.groupby(f"lesion3d_id_{method}")
        table3d.loc[sel, "n_2d_lesions"] = table3d.loc[sel, "lesion3d_id"].map(by3d.size()).fillna(0).to_numpy()
        informative = lesion_map[lesion_map["area"] >= bound].groupby(f"lesion3d_id_{method}")["area"].max()
        table3d.loc[sel, "largest_informative_px"] = table3d.loc[sel, "lesion3d_id"].map(informative).to_numpy()
    table3d["n_2d_lesions"] = table3d["n_2d_lesions"].astype(int)
    table3d["informative"] = table3d["largest_informative_px"].notna()
    lesion_map["informative"] = lesion_map["area"] >= bound
    return table3d, lesion_map


# --- held-out nearest neighbours --------------------------------------------------------------

def nn_counts(heval: pd.DataFrame) -> pd.DataFrame:
    """Per detector (e1, e2) and stratum: how many scorable H-eval masks have their nearest neighbour
    in each T patient (long table: detector, stratum, patient_id, count). Masks without a nearest
    neighbour (E2: no candidate slice) are left out."""
    rows = []
    for det, (stratum_col, nn_col) in NN_COLUMNS.items():
        for s in STRATA:
            nn = heval.loc[heval[stratum_col] == s, nn_col].fillna("")
            counts = nn[nn != ""].value_counts()
            rows += [{"detector": det, "stratum": s, "patient_id": p, "count": int(c)} for p, c in counts.items()]
    return pd.DataFrame(rows)


# --- review -----------------------------------------------------------------------------------

def reference_figure(table3d: pd.DataFrame, rarity: pd.DataFrame, patients: pd.DataFrame, methods: list[str],
                     bound: int, path: Path, title: str) -> Path:
    """Slice span of the informative 3D lesions per method; slice rarity on both scales; patient
    rarity against the number of 3D lesions."""
    fig = Figure(figsize=(14, 3.8))
    ax_span, ax_rar, ax_pat = fig.subplots(1, 3)
    bins = np.arange(0.5, table3d["span"].max() + 1.5)
    for method, color in zip(methods, ("tab:blue", "tab:orange")):
        d = table3d[(table3d["method"] == method) & table3d["informative"]]
        ax_span.hist(d["span"], bins=bins, histtype="step", color=color, label=f"{method} (n={len(d)})")
    ax_span.set_xlabel("slice span (slices)", fontsize=8)
    ax_span.set_title(f"3D lesions with a lesion >= {bound} px", fontsize=9)
    ax_span.legend(fontsize=7)
    for col, color in (("rarity_e1", "tab:blue"), ("rarity_e2", "tab:red")):
        v = rarity[col].dropna()
        ax_rar.hist(v, bins=np.linspace(0, 1, 26), histtype="step", color=color, label=f"{col} (n={len(v)})")
    ax_rar.set_xlabel("slice rarity = 1 - best score against other patients", fontsize=8)
    ax_rar.set_title("T slices", fontsize=9)
    ax_rar.legend(fontsize=7)
    ax_pat.scatter(patients["n_lesions3d"], patients["rarity"], s=10)
    ax_pat.set_xscale("log")
    ax_pat.set_xlabel("3D lesions (log)", fontsize=8)
    ax_pat.set_ylabel("patient rarity (mean distance to 5 nearest)", fontsize=8)
    ax_pat.set_title("T patients", fontsize=9)
    return save(fig, path, title)


def review_text(name: str, t: SliceSet, table3d: pd.DataFrame, lesion_map: pd.DataFrame, rarity: pd.DataFrame,
                patients: pd.DataFrame, heldout_nn: pd.DataFrame, heval: pd.DataFrame, methods: list[str],
                bound: int) -> str:
    """The `reference` review: 3D lesion counts per method and where they differ, slice and patient
    rarity, and the coverage of the held-out nearest neighbours (the data behind pi_p)."""
    lines = [f"E6 reference: '{name}' (T and stored H-eval tables only; no S)",
             f"T: {len(t.keys)} lesion slices of {len(np.unique(t.patients))} patients; left out: {t.left_out}", ""]
    lines.append("3D lesions (decision W), on the model-space masks as saved:")
    for m in methods:
        d = table3d[table3d["method"] == m]
        inf = d[d["informative"]]
        lines.append(f"  {m}: {len(d)} 3D lesions, {int((d['n_2d_lesions'] > 0).sum())} with an E0b lesion, "
                     f"{len(inf)} informative (>= {bound} px); holding two pieces in one slice: "
                     f"{int((d['max_parts'] > 1).sum())}; span of informative ones median {inf['span'].median():g}, "
                     f"quartiles {inf['span'].quantile(0.25):g} / {inf['span'].quantile(0.75):g}, max {inf['span'].max()}")
    a, b = (f"lesion3d_id_{m}" for m in methods)
    pairs = lesion_map.groupby(b)[a].nunique()
    split = pairs[pairs > 1]
    lines.append(f"  E0b 2D lesions: {len(lesion_map)} ({int(lesion_map['informative'].sum())} informative); "
                 f"{len(split)} {methods[1]} lesions are split into several {methods[0]} lesions, holding "
                 f"{int(lesion_map[b].isin(split.index).sum())} 2D lesions")
    lines.append("")
    lines.append("Slice rarity (decision V; 1 - best score against other patients' T slices):")
    for col in ("rarity_e1", "rarity_e2"):
        v = rarity[col].dropna()
        lines.append(f"  {col}: {len(v)} slices, quartiles {v.quantile(0.25):.3f} / {v.median():.3f} / "
                     f"{v.quantile(0.75):.3f}; rarity 0 (an identical slice in another patient): {int((v <= 1e-12).sum())}")
    lines.append("")
    lines.append("Patient rarity (mean distance to the 5 nearest other patients), the five rarest:")
    cols = ["patient_id", "n_lesions3d", "volume_px", "median_area_px", "max_lesions_per_slice", "position_spread", "rarity"]
    lines.append(patients.sort_values("rarity", ascending=False)[cols].head(5).to_string(index=False))
    lines.append("")
    lines.append("Held-out nearest neighbours (the counts behind pi_p, decision R):")
    for (det, s), d in heldout_nn.groupby(["detector", "stratum"]):
        col = NN_COLUMNS[str(det)][0]
        n_scorable = int((heval[col] == s).sum())
        lines.append(f"  {det} {s}: {int(d['count'].sum())} of {n_scorable} scorable H-eval masks have a nearest "
                     f"neighbour; {len(d)} of {len(patients)} T patients named; largest share "
                     f"{d['count'].max() / d['count'].sum():.3f}")
    return "\n".join(lines) + "\n"
