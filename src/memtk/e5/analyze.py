"""E5 stage `analyze` (`docs/e5-plan.md`, step 5) and the analysis it runs: a scored query set
against the held-out baseline H-eval, from the raw tables `score_query` wrote. Every choice comes
from the `audit:` section and the E5 file; nothing here is tuned. The stage refuses to run unless
`score` completed with the same `audit:` section and frozen files (the lock).

`analyze_scored` (tables) and `draw_figures` are the one analysis: `analyze` runs them on S, and
`rehearse` ran them on mock query sets first, so the analysis of S is the analysis rehearsed.

Tables (decision): rates and excess with the Holm-adjusted primary family (M); AUC with the
Mann-Whitney p (M, secondary); rates by size of the largest lesion with an H-eval rate
standardised to the query set's stratum x size mix (M, size-confound check); rates per naming
scheme (M); every sensitivity variant (O); the RQ2 cross-tabulation and H2 at equal false-positive
rate (P); chained adjacent-slice groups (N); near-exact copies and the beyond-held-out tier (Q);
`flagged.csv`, one row per flagged query; the mechanical outcome reading and the E8 exit counts.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import mannwhitneyu

from memtk import calibration, config, lesionmatch
from memtk.audit import (CLASSES, auc, auc_bootstrap, beyond_max, cell_rates, chain_groups, class_flags,
                         copied_area_share, excess_test, holm, interval, one_sided_p)
from memtk.calibration import STRATA
from memtk.e5.common import audit_config, audit_hash, score_marker
from memtk.e5.score import query_table
from memtk.ids import lesion_id, parse_slice_key
from memtk.io import LabelStack, read_table, slice_images
from memtk.plotting import draw_labels, hide_unused, lesion_colors, save, tile_figure, zoom_window
from memtk.sets import SliceSet, load_set
from memtk.stages import refuse_overwrite, require
from memtk.synthetic import load_sample

STRATUM_COL = {"e1_copy": "stratum_e1", "e2_full": "stratum_e2", "e2_partial": "stratum_e2", "e2_flagged": "stratum_e2"}
SCORES = (("e1_score", "stratum_e1", ("whole_mask", "tau_D"), "E1 score (best Dice over shifts)"),
          ("e2_C", "stratum_e2", ("lesion", "tau_C"), "E2 configuration score C(q, NN)"),
          ("e2_strongest_s", "stratum_e2", ("lesion", "tau_L"), "E2 strongest lesion pair s"))
MAX_EXAMPLES = 12    # pairs per example sheet


# --- stages -----------------------------------------------------------------------------------

def run_analyze(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Analyse S as scored by `score` (step 5): every table and figure of decisions L-Q, `flagged.csv`
    and the outcome reading, with H-eval as the baseline. Refuses unless `score` completed and the
    `audit:` section and both frozen files are those `score` recorded (the lock). Writes
    `results/<pool>/e5/analyze/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    data_dir = config.output_dir(cfg, pool_name) / "e5"
    out = config.results_dir(cfg, pool_name, "e5") / "analyze"
    marker = score_marker(cfg, pool_name)
    require(marker, "e5_audit.py score")
    refuse_overwrite([out], overwrite)
    audit = audit_config(cfg)
    e3, e3_hash = calibration.load_frozen(cfg, name)
    e5, e5_hash = calibration.load_frozen(cfg, name, "e5")
    scored_with = json.loads(marker.read_text(encoding="utf-8"))
    now = {"audit_sha256": audit_hash(cfg), "frozen_sha256": e3_hash, "e5_sha256": e5_hash}
    changed = [k for k, v in now.items() if scored_with[k] != v]
    if changed:
        raise SystemExit(f"Changed since S was scored: {', '.join(changed)}. The analysis is fixed before S (the lock, "
                         f"docs/e5-plan.md); restore the recorded version. An analysis added after scoring is "
                         f"exploratory and does not belong in this stage.")
    draws, rng = int(cfg["calibration"]["bootstrap_draws"]), np.random.default_rng(cfg["seed"])
    variants = list(audit["sensitivity"])

    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    h = heldout(cfg, name, heval, variants)
    s = load_scored("S", data_dir, variants)
    e2_main = lesionmatch.load_tables(data_dir, "main", float(e3["detectors"]["lesion"]["candidate_min_s"]))
    print(f"Analysing S ({len(s.meta)} masks) against H-eval ...")
    tables, reading = analyze_scored(s, h, e2_main, informative_areas(t, int(e3["informative_bound_px"])), e3, e5,
                                     audit, draws, rng)
    write_tables(tables, out)

    pool_labels = LabelStack(config.output_dir(cfg, pool_name) / "labels.npz")
    t_labels = dict(zip(t.keys, t.labels))
    files = read_table(config.output_dir(cfg, pool_name) / "synthetic_index.csv").set_index("sample_id")["filename"]
    pool_dir = config.resolve_path(pool_cfg["dir"])

    def labels_of(keys):
        return {k: t_labels[k] if k in t_labels else pool_labels[k] for k in keys}

    def images_of(keys):
        real = slice_images(cfg, name, [k for k in keys if k in t_labels])
        return {k: real[k] if k in real else load_sample(pool_dir / files[k], pool_cfg)[0] for k in keys}

    draw_figures(s, h, tables, e3, f"E5 analyze ({pool_name})", labels_of, images_of, out)
    log = read_table(data_dir / "score_log.csv") if (data_dir / "score_log.csv").exists() else None
    header = (f"E5 analyze: pool '{pool_name}' against '{name}' (S scored once, {scored_with['completed']}; the "
              f"analysis fixed by decisions L-Q, hashes equal to those recorded by score)")
    text = review_text(header, s, tables, reading, log)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "analyze", "inputs": ["T", "H-eval", "S tables"],
                                                            **now, "reading": reading})
    print(f"\n{text}\nWrote {out}")


# --- scored sets ------------------------------------------------------------------------------

@dataclass
class Scored:
    """A scored set: `meta` (one row per slice: key, cluster, mask_class, scheme, n_lesions,
    l1_area) and the class table of each variant (`main` and decision O)."""
    name: str
    meta: pd.DataFrame
    classes: dict[str, pd.DataFrame]

    def table(self, variant: str) -> pd.DataFrame:
        """The class table of a variant joined with `meta` (one row per slice)."""
        return self.classes[variant].drop(columns=["patient_id"]).merge(
            self.meta.drop(columns=["n_lesions"]), on="key", how="left", validate="one_to_one")


def load_scored(name: str, score_dir: Path, variants: list[str]) -> Scored:
    """A query set scored by `score_query` (its `queries.csv` and class tables)."""
    return Scored(name, read_table(score_dir / "queries.csv"),
                  {v: read_table(score_dir / f"classes_{v}.csv") for v in ["main", *variants]})


def heldout(cfg: dict, dataset: str, heval: SliceSet, variants: list[str]) -> Scored:
    """H-eval as a scored set (the baseline): its slices clustered by patient, the main class table
    from E4 and one per variant from `prepare`."""
    meta = query_table(heval, heval.patients, ["clean"] * len(heval.keys), ["H-eval"] * len(heval.keys))
    e4_dir, e5_dir = (config.output_dir(cfg, dataset) / e for e in ("e4", "e5"))
    require(e4_dir / "heval_classes.csv", "e4_validate.py validate")
    classes = {"main": read_table(e4_dir / "heval_classes.csv")}
    for v in variants:
        require(e5_dir / f"heval_classes_{v}.csv", "e5_audit.py prepare")
        classes[v] = read_table(e5_dir / f"heval_classes_{v}.csv")
    return Scored("H-eval", meta, classes)


def informative_areas(t: SliceSet, bound: int) -> pd.Series:
    """Area of every informative T lesion (>= the bound), by lesion ID (for the copied-area share)."""
    rows = {}
    for key, labels in zip(t.keys, t.labels):
        areas = np.bincount(labels.ravel())
        rows.update({lesion_id(key, n): int(a) for n, a in enumerate(areas) if n and a >= bound})
    return pd.Series(rows, dtype=int)


def subsets(table: pd.DataFrame, sets: dict) -> dict[str, pd.Series]:
    """Decision L: the rows of each analysis set (main, noisy, main + noisy) by `mask_class`;
    empty sets are left out, and main + noisy only exists when both parts do."""
    main, noisy = table["mask_class"].isin(sets["main"]), table["mask_class"].isin(sets["noisy"])
    out = {"main": main, "noisy": noisy}
    if main.any() and noisy.any():
        out["main+noisy"] = main | noisy
    return {k: v for k, v in out.items() if v.any()}


# --- analysis ---------------------------------------------------------------------------------

def analyze_scored(q: Scored, h: Scored, e2_main: lesionmatch.LesionMatch, t_areas: pd.Series, e3: dict, e5: dict,
                   audit: dict, draws: int, rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], dict]:
    """Every table of the analysis of `q` against `h` (H-eval), by name, and the outcome reading."""
    variants = ["main", *audit["sensitivity"]]
    rates = rates_table(q, h, variants, audit, draws, rng)
    qm, hm = q.table("main"), h.table("main")
    main = subsets(qm, audit["sets"])["main"]
    qmain = qm[main]
    edges = [int(e3["min_lesion_area"]), *(int(b) for b in audit["size_bins_px"])]
    groups, group_ids = groups_table(qm, hm, audit["sets"], int(audit["window"]["max_gap"]))
    near, strong = near_exact_table(qm, hm, audit["sets"], e5, int(audit["window"]["max_gap"]),
                                    int(audit["e8_min_strong"]))
    tables = {
        "rates": rates,
        "auc": auc_table(qmain, hm, draws, rng),
        "rates_by_size": size_table(qmain, hm, edges, draws, rng),
        "rates_by_scheme": scheme_table(qmain, hm, draws, rng),
        "sensitivity": rates[(rates["set"] == "main") & (rates["stratum"] != "all")].reset_index(drop=True),
        "rq2_crosstab": rq2_table(qmain, hm),
        "h2": h2_table(qmain, hm, e5["h2"]["tau_D_star"], draws, rng),
        "groups": groups,
        "near_exact": near,
        "flagged": flagged_table(qm, audit["sets"], group_ids, strong, e2_main, t_areas, e3),
    }
    return tables, outcome(rates, near, audit)


def rates_table(q: Scored, h: Scored, variants: list[str], audit: dict, draws: int,
                rng: np.random.Generator) -> pd.DataFrame:
    """Per set (decision L), variant (main and decision O), class and stratum (single, multi, all):
    the query set's and H-eval's rates and the excess with interval and one-sided p (query set
    resampled by its clusters, H-eval by patient, independently), the estimated excess count
    (excess x slices), and for the primary family (decision M; main set, main variant) the
    Holm-adjusted p and whether the cell is beyond chance."""
    rows = []
    for variant in variants:
        qt, ht = q.table(variant), h.table(variant)
        for set_name, keep in subsets(qt, audit["sets"]).items():
            for cls in CLASSES:
                on_q, flag_q = class_flags(qt, cls)
                on_h, flag_h = class_flags(ht, cls)
                col = STRATUM_COL[cls]
                for s in (*STRATA, "all"):
                    sq = keep & on_q & ((qt[col] == s) | (s == "all"))
                    sh = on_h & ((ht[col] == s) | (s == "all"))
                    if not sq.any():
                        continue
                    r = excess_test(flag_q[sq].to_numpy(), qt.loc[sq, "cluster"].to_numpy(), flag_h[sh].to_numpy(),
                                    ht.loc[sh, "cluster"].to_numpy(), draws, rng)
                    rows.append({"set": set_name, "variant": variant, "class": cls, "stratum": s, **r,
                                 "excess_count": r["n_s"] * r["excess"], "excess_count_low": r["n_s"] * r["excess_low"],
                                 "excess_count_high": r["n_s"] * r["excess_high"]})
    out = pd.DataFrame(rows)
    primary = audit["primary"]
    is_primary = ((out["set"] == "main") & (out["variant"] == "main") & out["class"].isin(primary["classes"])
                  & out["stratum"].isin(primary["strata"]))
    out["primary"] = is_primary
    out["p_holm"] = np.nan
    out.loc[is_primary, "p_holm"] = holm(out.loc[is_primary, "p"].to_numpy())
    out["beyond_chance"] = is_primary & (out["p_holm"] < float(primary["alpha"]))
    return out


def auc_table(qt: pd.DataFrame, ht: pd.DataFrame, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Decision M (secondary): per score and stratum, AUC = P(query score > H-eval score), with a
    two-level bootstrap interval and the one-sided Mann-Whitney p (which treats slices as
    independent). E2 scores of slices without a candidate count as 0."""
    rows = []
    for col, stratum_col, _, _ in SCORES:
        for s in STRATA:
            a, b = qt[qt[stratum_col] == s], ht[ht[stratum_col] == s]
            if not len(a) or not len(b):
                continue
            sa, sb = a[col].fillna(0.0).to_numpy(float), b[col].fillna(0.0).to_numpy(float)
            boot = auc_bootstrap(sa, a["cluster"].to_numpy(), sb, b["cluster"].to_numpy(), draws, rng)
            low, high = interval(boot)
            rows.append({"score": col, "stratum": s, "n_s": len(sa), "n_h": len(sb), "auc": auc(sa, sb),
                         "auc_low": low, "auc_high": high,
                         "mann_whitney_p": float(mannwhitneyu(sa, sb, alternative="greater").pvalue)})
    return pd.DataFrame(rows)


def size_table(qt: pd.DataFrame, ht: pd.DataFrame, edges: list[int], draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Decision M (size-confound check): per class and stratum, the rates and excess in each size
    bin of the largest lesion, then the excess against H-eval standardised to the query set's mix
    (size bins within a stratum; stratum x size bin for all)."""
    rows = []
    for cls in CLASSES:
        on_q, flag_q = class_flags(qt, cls)
        on_h, flag_h = class_flags(ht, cls)
        col = STRATUM_COL[cls]
        bins_q = calibration.area_bins(qt["l1_area"].to_numpy(), edges).to_numpy()
        bins_h = calibration.area_bins(ht["l1_area"].to_numpy(), edges).to_numpy()
        for s in (*STRATA, "all"):
            sq = (on_q & ((qt[col] == s) | (s == "all"))).to_numpy()
            sh = (on_h & ((ht[col] == s) | (s == "all"))).to_numpy()
            if not sq.any():
                continue
            for lower in sorted(set(bins_q[sq][~np.isnan(bins_q[sq])])):
                bq, bh = sq & (bins_q == lower), sh & (bins_h == lower)
                if not bh.any():
                    rows.append({"class": cls, "stratum": s, "bin": calibration.bin_label(int(lower), edges),
                                 "n_s": int(bq.sum()), "rate_s": float(flag_q[bq].mean()), "n_h": 0})
                    continue
                r = excess_test(flag_q[bq].to_numpy(), qt["cluster"].to_numpy()[bq], flag_h[bh].to_numpy(),
                                ht["cluster"].to_numpy()[bh], draws, rng)
                rows.append({"class": cls, "stratum": s, "bin": calibration.bin_label(int(lower), edges), **r})
            cells_q = (qt[col].astype(str) + "|" + pd.Series(bins_q, index=qt.index).astype(str)).to_numpy()
            cells_h = (ht[col].astype(str) + "|" + pd.Series(bins_h, index=ht.index).astype(str)).to_numpy()
            levels = sorted(set(cells_q[sq]))
            weights = np.array([np.sum(cells_q[sq] == c) for c in levels]) / sq.sum()
            pq, bq_boot = cell_rates(flag_q[sq].to_numpy(), cells_q[sq], qt["cluster"].to_numpy()[sq], levels, weights,
                                     draws, rng)
            ph, bh_boot = cell_rates(flag_h[sh].to_numpy(), cells_h[sh], ht["cluster"].to_numpy()[sh], levels, weights,
                                     draws, rng)
            boot = bq_boot[:, 0] - bh_boot[:, 0]
            low, high = interval(boot)
            rows.append({"class": cls, "stratum": s, "bin": "standardised", "n_s": int(sq.sum()), "rate_s": pq[0],
                         "n_h": int(sh.sum()), "rate_h": ph[0], "excess": pq[0] - ph[0], "excess_low": low,
                         "excess_high": high, "p": one_sided_p(boot)})
    return pd.DataFrame(rows)


def scheme_table(qt: pd.DataFrame, ht: pd.DataFrame, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Decision M (descriptive): per naming scheme (generation round; source set for mock sets),
    class and stratum, the rate and excess against all of H-eval."""
    rows = []
    for scheme, g in qt.groupby("scheme"):
        for cls in CLASSES:
            on_q, flag_q = class_flags(g, cls)
            on_h, flag_h = class_flags(ht, cls)
            col = STRATUM_COL[cls]
            for s in (*STRATA, "all"):
                sq, sh = on_q & ((g[col] == s) | (s == "all")), on_h & ((ht[col] == s) | (s == "all"))
                if not sq.any():
                    continue
                r = excess_test(flag_q[sq].to_numpy(), g.loc[sq, "cluster"].to_numpy(), flag_h[sh].to_numpy(),
                                ht.loc[sh, "cluster"].to_numpy(), draws, rng)
                rows.append({"scheme": scheme, "class": cls, "stratum": s, **r})
    return pd.DataFrame(rows)


def informative(t: pd.DataFrame) -> pd.Series:
    """The RQ2 subset: slices scored by both detectors (an informative lesion, so also a lesion of
    the minimum area)."""
    return (t["stratum_e2"] != "none") & t["stratum_e1"].notna()


def rq2_table(qt: pd.DataFrame, ht: pd.DataFrame) -> pd.DataFrame:
    """RQ2: on the informative subset, the E1 flag against the E2 class, counts and shares, for
    the query set and H-eval."""
    rows = []
    for name, t in (("query", qt), ("H-eval", ht)):
        d = t[informative(t)]
        counts = d.groupby([d["e1_copy"].eq(True), d["e2_class"]]).size()
        for (e1_flag, e2_class), n in counts.items():  # pyright: ignore[reportGeneralTypeIssues]  # a two-level index gives pairs
            rows.append({"set": name, "e1_copy": bool(e1_flag), "e2_class": e2_class, "slices": int(n),
                         "share": n / len(d)})
    return pd.DataFrame(rows)


def h2_table(qt: pd.DataFrame, ht: pd.DataFrame, tau_star: dict, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Decision P: on the informative subset, per comparison ((i) E2 full vs E1 copy; (ii) E2
    flagged vs E1 at tau_D*) and E2 stratum (all = H-eval standardised to the query set's stratum
    mix): each detector's excess over H-eval and the difference of excesses (E2 - E1) with its
    interval and one-sided p; H2 holds if the interval is above 0. Discordant counts beside."""
    rows = []

    def flags(t: pd.DataFrame, comparison: str) -> np.ndarray:
        if comparison == "i":
            e2, e1 = t["e2_class"] == "full", t["e1_copy"].eq(True)
        else:
            star = t["stratum_e1"].map(tau_star).astype(float)
            e2, e1 = t["e2_class"].isin(["full", "partial"]), t["e1_score"].to_numpy(float) >= star
        return np.c_[np.asarray(e2, dtype=bool), np.asarray(e1, dtype=bool)]

    q, h = qt[informative(qt)], ht[informative(ht)]
    for comparison in ("i", "ii"):
        fq, fh = flags(q, comparison), flags(h, comparison)
        for s in (*STRATA, "all"):
            sq = ((q["stratum_e2"] == s) | (s == "all")).to_numpy()
            sh = ((h["stratum_e2"] == s) | (s == "all")).to_numpy()
            if not sq.any():
                continue
            levels = list(STRATA) if s == "all" else [s]
            cells_q, cells_h = q["stratum_e2"].to_numpy()[sq], h["stratum_e2"].to_numpy()[sh]
            weights = np.array([np.sum(cells_q == c) for c in levels]) / sq.sum()
            pq, bq = cell_rates(fq[sq], cells_q, q["cluster"].to_numpy()[sq], levels, weights, draws, rng)
            ph, bh = cell_rates(fh[sh], cells_h, h["cluster"].to_numpy()[sh], levels, weights, draws, rng)
            excess, boot = pq - ph, bq - bh
            diff = boot[:, 0] - boot[:, 1]
            low, high = interval(diff)
            rows.append({"comparison": "(i) E2 full vs E1 copy" if comparison == "i" else "(ii) E2 flagged vs E1 at tau_D*",
                         "stratum": s, "n_s": int(sq.sum()), "n_h": int(sh.sum()),
                         "rate_s_e2": pq[0], "rate_h_e2": ph[0], "excess_e2": excess[0],
                         "rate_s_e1": pq[1], "rate_h_e1": ph[1], "excess_e1": excess[1],
                         "difference": excess[0] - excess[1], "difference_low": low, "difference_high": high,
                         "p": one_sided_p(diff), "h2_holds": bool(low > 0),
                         "s_e2_only": int((fq[sq, 0] & ~fq[sq, 1]).sum()), "s_e1_only": int((~fq[sq, 0] & fq[sq, 1]).sum()),
                         "h_e2_only": int((fh[sh, 0] & ~fh[sh, 1]).sum()), "h_e1_only": int((~fh[sh, 0] & fh[sh, 1]).sum())})
    return pd.DataFrame(rows)


# --- sources and groups -----------------------------------------------------------------------

def sources(t: pd.DataFrame, cls: str) -> pd.Series:
    """The source slice of each slice a class flags (decision J): E1 copy -> E1 nearest neighbour;
    E2 full -> E2 nearest neighbour; E2 partial -> the strongest-pair slice; E2 flagged -> by its
    class. Indexed like `t`, flagged rows only."""
    on, flag = class_flags(t, cls)
    d = t[on & flag]
    if cls == "e1_copy":
        return d["e1_nn"]
    return pd.Series(np.where(d["e2_class"] == "full", d["e2_nn"], d["e2_strongest_train"]), index=d.index)


def chained(src: pd.Series, max_gap: int) -> pd.Series:
    """Decision N: the group of each source slice (chained within max_gap slices of one patient),
    labelled `<patient>:z<first z>`."""
    if not len(src):
        return pd.Series(dtype=str)
    parsed = [parse_slice_key(k) for k in src]
    patients, z = [p for p, _ in parsed], [zz for _, zz in parsed]
    g = chain_groups(patients, z, max_gap)
    first = pd.DataFrame({"g": g, "p": patients, "z": z}).groupby("g").agg(p=("p", "first"), z=("z", "min"))
    labels = {i: f"{r.p}:z{r.z}" for i, r in first.iterrows()}
    return pd.Series([labels[i] for i in g], index=src.index)


def groups_table(qt: pd.DataFrame, ht: pd.DataFrame, sets: dict, max_gap: int) -> tuple[pd.DataFrame, dict]:
    """Decision N, per set and class (and for H-eval): flags, distinct source slices, chained
    groups, distinct source patients. Also the group label of every flagged query per set (E1
    copy groups and E2 flagged groups), for `flagged.csv`."""
    rows, ids = [], {}
    sets_q = {f"query {k}": v for k, v in subsets(qt, sets).items()}
    for name, t in (*((n, qt[keep]) for n, keep in sets_q.items()), ("H-eval", ht)):
        for cls in CLASSES:
            src = sources(t, cls)
            grp = chained(src, max_gap)
            rows.append({"set": name, "class": cls, "flags": len(src), "source_slices": src.nunique(),
                         "groups": grp.nunique(), "patients": len({parse_slice_key(k)[0] for k in src})})
            if cls in ("e1_copy", "e2_flagged") and name != "H-eval":
                ids[(name.removeprefix("query "), cls)] = grp
    return pd.DataFrame(rows), ids


# --- near-exact copies, the beyond-held-out tier, flagged ---------------------------------------

def strong_flags(t: pd.DataFrame, e5: dict) -> tuple[pd.Series, pd.Series]:
    """Decision Q per slice: near-exact (E1 score >= min_score and above the held-out maximum of
    its stratum and size bin) and beyond the held-out E2 C maximum of its stratum."""
    ne = e5["near_exact"]
    e1_max = pd.DataFrame(ne["e1_maxima"])
    e2_max = pd.DataFrame([{"stratum": s, "lower": -1, "max": v} for s, v in ne["e2_C_maxima"].items()])
    score = t["e1_score"].fillna(-1.0).to_numpy(float)
    near = (score >= float(ne["min_score"])) & beyond_max(score, t["stratum_e1"].fillna("none").to_numpy(),
                                                          t["l1_area"].to_numpy(), e1_max, ne["size_bins_px"])
    c = t["e2_C"].fillna(-1.0).to_numpy(float)
    tier = beyond_max(c, t["stratum_e2"].to_numpy(), t["l1_area"].to_numpy(), e2_max, ne["size_bins_px"])
    return pd.Series(near, index=t.index), pd.Series(tier, index=t.index)


def near_exact_table(qt: pd.DataFrame, ht: pd.DataFrame, sets: dict, e5: dict, max_gap: int,
                     e8_min: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decision Q per set (and H-eval, where both counts are 0 by construction): near-exact copies
    and their groups, E2 masks beyond the held-out C maximum (multi-lesion in practice) and their
    groups, the plan's plain count of E1 score > 0.9 per stratum, and the E8 exit count (groups of
    the union of both) with eligibility. Returns the table and the per-slice flags of the query."""
    near_q, tier_q = strong_flags(qt, e5)
    rows = []
    sets_q = {f"query {k}": v for k, v in subsets(qt, sets).items()}
    for name, t in (*((n, qt[keep]) for n, keep in sets_q.items()), ("H-eval", ht)):
        near, tier = strong_flags(t, e5)
        src_near, src_tier = t.loc[near, "e1_nn"], t.loc[tier, "e2_nn"]
        union = pd.concat([src_near, src_tier]).drop_duplicates()
        row = {"set": name, "near_exact": int(near.sum()), "near_exact_groups": chained(src_near, max_gap).nunique(),
               "beyond_heldout_e2": int(tier.sum()), "beyond_heldout_e2_groups": chained(src_tier, max_gap).nunique(),
               "e8_strong_groups": chained(union, max_gap).nunique()}
        row["e8_eligible"] = row["e8_strong_groups"] >= e8_min
        for s in STRATA:
            d = t[t["stratum_e1"] == s]
            row[f"e1_over_0.9_{s}"] = int((d["e1_score"] > 0.9).sum())
            row[f"e1_over_0.9_{s}_rate"] = float((d["e1_score"] > 0.9).mean()) if len(d) else np.nan
        rows.append(row)
    flags = pd.DataFrame({"near_exact": near_q, "beyond_heldout_e2": tier_q})
    return pd.DataFrame(rows), flags


def flagged_table(qt: pd.DataFrame, sets: dict, group_ids: dict, strong: pd.DataFrame, e2_main: lesionmatch.LesionMatch,
                  t_areas: pd.Series, e3: dict) -> pd.DataFrame:
    """One row per flagged query (E1 copy, E2 full or partial) of every set, main variant: set,
    identity, strata, both detectors' results, the E2 source slice and patient, the group labels
    (decision N, within its set), the copied-area share (E5 step 7, E2 flags), near-exact and
    beyond-held-out flags (decision Q). Read by E6, E7 and E8."""
    parts = []
    for set_name, keep in subsets(qt, sets).items():
        if set_name == "main+noisy":
            continue
        t = qt[keep]
        flagged = t["e1_copy"].eq(True) | t["e2_class"].isin(["full", "partial"])
        d = t[flagged].copy()
        d.insert(0, "set", set_name)
        d["e2_source"] = sources(t, "e2_flagged").reindex(d.index).fillna("")
        d["e2_source_patient"] = [parse_slice_key(k)[0] if k else "" for k in d["e2_source"]]
        d["e2_strongest_patient"] = [parse_slice_key(k)[0] if isinstance(k, str) and k else "" for k in d["e2_strongest_train"]]
        d["e1_group"] = group_ids.get((set_name, "e1_copy"), pd.Series(dtype=str)).reindex(d.index).fillna("")
        d["e2_group"] = group_ids.get((set_name, "e2_flagged"), pd.Series(dtype=str)).reindex(d.index).fillna("")
        parts.append(d)
    out = pd.concat(parts) if parts else qt.iloc[:0].copy()
    tau_l = pd.Series(calibration.frozen_threshold(e3, "lesion", "tau_L", out["n_informative"].to_numpy()),
                      index=out["key"].to_numpy())
    src = pd.Series(out["e2_source"].to_numpy(), index=out["key"].to_numpy())
    q_area = e2_main.lesions.set_index("query_lesion")["area"]
    share = copied_area_share(e2_main.lesion_pairs, src[src != ""], tau_l, q_area, t_areas)
    out["copied_area_share"] = out["key"].map(share)
    out = out.join(strong)
    cols = ["set", "key", "mask_class", "scheme", "cluster", "n_lesions", "stratum_e1", "n_informative", "stratum_e2",
            "l1_area", "e1_score", "e1_copy", "e1_nn", "e1_nn_patient", "e1_group", "e2_class", "e2_C", "e2_nn",
            "e2_nn_patient", "e2_strongest_s", "e2_strongest_train", "e2_strongest_patient", "e2_source",
            "e2_source_patient", "e2_group", "e2_n_kept", "e2_all_copied", "copied_area_share", "near_exact",
            "beyond_heldout_e2"]
    return out[cols].reset_index(drop=True)


def outcome(rates: pd.DataFrame, near: pd.DataFrame, audit: dict) -> dict:
    """The experiment plan's outcome reading, applied mechanically (decision M): beyond chance if
    a primary cell is (Holm); rare but real if none is but a near-exact copy exists in the main
    set; otherwise a validated negative. With the E8 exit count."""
    primary = rates[rates["primary"]]
    beyond = primary[primary["beyond_chance"]]
    main = near[near["set"] == "query main"].iloc[0]
    if len(beyond):
        reading = "memorization beyond chance"
    elif main["near_exact"] >= 1:
        reading = "rare but real memorization"
    else:
        reading = "validated negative"
    return {"reading": reading, "primary_beyond": [f"{r['class']} {r['stratum']}" for _, r in beyond.iterrows()],
            "near_exact_main": int(main["near_exact"]), "e8_strong_groups": int(main["e8_strong_groups"]),
            "e8_eligible": bool(main["e8_eligible"]), "e8_min": int(audit["e8_min_strong"])}


# --- figures ----------------------------------------------------------------------------------

def score_figure(qt: pd.DataFrame, ht: pd.DataFrame, e3: dict, label: str, path: Path, title: str):
    """Per score and stratum, the query set's and H-eval's distributions (density) with the frozen
    threshold."""
    fig = Figure(figsize=(14, 6.5))
    axes = fig.subplots(2, 3)
    bins = np.linspace(0, 1, 41)
    for j, (col, stratum_col, (det, thr), name) in enumerate(SCORES):
        for i, s in enumerate(STRATA):
            ax = axes[i, j]
            for t, lab, color in ((qt, label, "tab:red"), (ht, "H-eval", "tab:green")):
                v = t.loc[t[stratum_col] == s, col].fillna(0.0)
                if len(v):
                    ax.hist(v, bins=bins, density=True, histtype="step", color=color, label=f"{lab} (n={len(v)})")
            ax.axvline(e3["thresholds"][det][thr][s], ls="--", color="k", lw=0.8)
            ax.set_title(f"{name}, {s}", fontsize=9)
            ax.legend(fontsize=6)
    return save(fig, path, title)


def forest_figure(rates: pd.DataFrame, path: Path, title: str):
    """Per class: the excess (query set - H-eval) with its interval for every set, variant and
    stratum; primary cells in bold red."""
    fig = Figure(figsize=(15, 9))
    for ax, cls in zip(fig.subplots(1, 4), CLASSES):
        d = rates[(rates["class"] == cls) & (rates["stratum"] != "all")].reset_index(drop=True)
        y = np.arange(len(d))[::-1]
        for primary, color in ((False, "tab:gray"), (True, "tab:red")):
            sel = (d["primary"] == primary).to_numpy()
            e = d[sel]
            ax.errorbar(e["excess"], y[sel], xerr=[e["excess"] - e["excess_low"], e["excess_high"] - e["excess"]],
                        fmt="o", ms=3, color=color, elinewidth=1)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y, [f"{r.set} | {r.variant} | {r.stratum}" for r in d.itertuples()], fontsize=6)
        ax.set_title(cls, fontsize=9)
        ax.set_xlabel("excess rate (query set - H-eval)", fontsize=7)
    return save(fig, path, title)


def copied_area_figure(flagged: pd.DataFrame, path: Path, title: str):
    """Copied-area share of the E2-flagged queries by class and stratum."""
    fig = Figure(figsize=(10, 3.5))
    for ax, s in zip(fig.subplots(1, 2), STRATA):
        for cls, color in (("full", "tab:red"), ("partial", "tab:blue")):
            v = flagged.loc[(flagged["e2_class"] == cls) & (flagged["stratum_e2"] == s), "copied_area_share"].dropna()
            if len(v):
                ax.hist(v, bins=np.linspace(0, 1, 21), histtype="step", color=color, label=f"{cls} (n={len(v)})")
        ax.set_title(f"{s}-lesion", fontsize=9)
        ax.set_xlabel("copied-area share", fontsize=8)
        ax.legend(fontsize=7)
    return save(fig, path, title)


def examples_figure(rows: pd.DataFrame, source_col: str, labels_of: Callable, images_of: Callable,
                    path: Path, title: str):
    """Query masks next to their source training slice on a shared zoom window, with scores (up to
    MAX_EXAMPLES pairs, as given)."""
    rows = rows.head(MAX_EXAMPLES)
    keys, srcs = rows["key"].tolist(), rows[source_col].tolist()
    labels, images = labels_of(keys + srcs), images_of(keys + srcs)
    fig, axes = tile_figure(int(np.ceil(max(len(rows), 1) / 3)), 6, tile=1.9)
    tiles = axes.ravel()
    for k, r in enumerate(rows.itertuples(index=False)):
        q, t = labels[r.key], labels[getattr(r, source_col)]
        win = zoom_window([q > 0, t > 0])
        draw_labels(tiles[2 * k], images[r.key][win], q[win], lesion_colors(q),
                    f"{r.key}\nE1 {r.e1_score:.2f}, C {r.e2_C:.2f}")
        draw_labels(tiles[2 * k + 1], images[getattr(r, source_col)][win], t[win], lesion_colors(t),
                    f"T {getattr(r, source_col)}")
    hide_unused(axes, 2 * len(rows))
    return save(fig, path, title)


def draw_figures(q: Scored, h: Scored, tables: dict[str, pd.DataFrame], e3: dict, prefix: str, labels_of: Callable,
                 images_of: Callable, out: Path) -> None:
    """Every figure of an analysis (moved from `rehearse` on 2026-11-15, so S gets the figures that
    were rehearsed): score distributions, the excess forest plot, the copied-area share, near-exact
    copies and the largest E2 groups next to their source slices (the last three only when there
    is something to show). `prefix` starts each title."""
    qt, ht = q.table("main"), h.table("main")
    score_figure(qt, ht, e3, q.name, out / "score_distributions.png",
                 f"{prefix}: scores of {q.name} and of H-eval by stratum; dashed = frozen threshold")
    forest_figure(tables["rates"], out / "rates_forest.png",
                  f"{prefix}: excess rate over H-eval with 95% intervals per set, variant and stratum "
                  f"(red = primary family)")
    flagged = tables["flagged"]
    main = flagged[flagged["set"] == "main"]
    if main["e2_class"].isin(["full", "partial"]).any():
        copied_area_figure(main, out / "copied_area.png", f"{prefix}: copied-area share of the E2-flagged masks (main set)")
    near = main[main["near_exact"]].sort_values("e1_score", ascending=False)
    if len(near):
        examples_figure(near, "e1_nn", labels_of, images_of, out / "near_exact.png",
                        f"{prefix}: near-exact copies (decision Q, highest E1 score first) next to their source slice")
    e2 = main[main["e2_group"] != ""]
    if len(e2):
        top = e2["e2_group"].value_counts().head(MAX_EXAMPLES).index
        rows = e2[e2["e2_group"].isin(top)].sort_values("e2_C", ascending=False).drop_duplicates("e2_group")
        examples_figure(rows, "e2_source", labels_of, images_of, out / "groups_top.png",
                        f"{prefix}: one mask per largest E2 group (decision N; highest C) next to its source slice")


# --- review text ------------------------------------------------------------------------------

def review_text(header: str, q: Scored, tables: dict[str, pd.DataFrame], out: dict, log: pd.DataFrame | None) -> str:
    """The readable summary of an analysis, ending with the outcome reading."""
    rates = tables["rates"]
    shown = rates[(rates["variant"] == "main") & (rates["stratum"] != "all")].copy()
    shown["S"] = [f"{r:.3f} [{lo:.3f}, {hi:.3f}] ({k}/{n})" for r, lo, hi, k, n in
                  zip(shown["rate_s"], shown["rate_s_low"], shown["rate_s_high"], shown["flagged_s"], shown["n_s"])]
    shown["H-eval"] = [f"{r:.3f} ({k}/{n})" for r, k, n in zip(shown["rate_h"], shown["flagged_h"], shown["n_h"])]
    shown["excess"] = [f"{e:+.3f} [{lo:+.3f}, {hi:+.3f}]" for e, lo, hi in
                       zip(shown["excess"], shown["excess_low"], shown["excess_high"])]
    primary = rates[rates["primary"]][["class", "stratum", "excess", "excess_low", "excess_high", "p", "p_holm",
                                       "beyond_chance"]]
    sens = rates[(rates["set"] == "main") & (rates["stratum"] != "all")].pivot_table(
        index=["class", "stratum"], columns="variant", values="excess", sort=False)
    std = tables["rates_by_size"]
    std = std[std["bin"] == "standardised"][["class", "stratum", "rate_s", "rate_h", "excess", "excess_low", "excess_high"]]
    counts = q.meta["mask_class"].value_counts().to_dict()
    return (
        f"{header}\n\nQuery set: {len(q.meta)} slices by mask class {counts}; schemes "
        f"{q.meta['scheme'].value_counts().to_dict()}.\n\n"
        + (f"Scoring log:\n{log.to_string(index=False)}\n\n" if log is not None else "")
        + f"Rates, main variant (query set by its clusters, H-eval by patient; 95% intervals):\n"
        f"{shown[['set', 'class', 'stratum', 'S', 'H-eval', 'excess']].to_string(index=False)}\n\n"
        f"Primary family (decision M; Holm over {len(primary)} cells):\n{primary.round(4).to_string(index=False)}\n\n"
        f"Excess per sensitivity variant (main set; decision O):\n{sens.round(3).to_string()}\n\n"
        f"Size-confound check (H-eval standardised to the query set's stratum x size mix):\n"
        f"{std.round(3).to_string(index=False)}\n\n"
        f"AUC (P(query score > H-eval score); Mann-Whitney p treats slices as independent):\n"
        f"{tables['auc'].round(4).to_string(index=False)}\n\n"
        f"RQ2 cross-tabulation (informative subset):\n{tables['rq2_crosstab'].round(3).to_string(index=False)}\n\n"
        f"H2 (decision P; holds if the interval of E2 - E1 excess is above 0):\n"
        f"{tables['h2'].round(4).to_string(index=False)}\n\n"
        f"Adjacent-slice groups (decision N, chained):\n{tables['groups'].to_string(index=False)}\n\n"
        f"Near-exact copies and the beyond-held-out tier (decision Q):\n{tables['near_exact'].round(4).to_string(index=False)}\n\n"
        f"Flagged queries in flagged.csv: {len(tables['flagged'])}.\n\n"
        f"OUTCOME (mechanical reading): {out['reading']}"
        f"{' (beyond chance: ' + ', '.join(out['primary_beyond']) + ')' if out['primary_beyond'] else ''}; "
        f"near-exact copies in the main set: {out['near_exact_main']}. E8 exit count: {out['e8_strong_groups']} "
        f"strong groups (eligible at >= {out['e8_min']}: {'yes' if out['e8_eligible'] else 'no'}).\n")


def write_tables(tables: dict[str, pd.DataFrame], out: Path) -> None:
    """Every table as `<name>.csv` in `out` (the outcome reading goes into the review text)."""
    out.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
