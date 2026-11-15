"""E5 building blocks: comparing a query set with the held-out baseline (`docs/e5-plan.md`).

- Per-slice classes of both detectors under the frozen thresholds (`class_table`, `class_flags`),
  shared with E4's false-positive table. Moved from `memtk.e4.validate` on 2026-11-15.
- Cluster bootstrap of rates (`resample_sums`): held-out slices are clustered by patient; synthetic
  masks are independent, i.e. clusters of one. Resampling clusters with replacement is the same as
  drawing how often each cluster is chosen from a multinomial; clusters with identical sums are
  pooled into one type first, so 20,000 independent masks cost no more than 26 patients and the
  bootstrap is exact. Excess, intervals and one-sided p (decision M), Holm's adjustment.
- AUC of two score samples with a two-level bootstrap (decision M, secondary).
- Chained adjacent-slice groups (decision N), the copied-area share (E5 step 7), standardised rates
  (size-confound check) and held-out maxima per stratum and size bin (decision Q).
- Scored query sets (`Scored`, `query_table`, `load_scored`, `heldout`) and the source slice and
  chained group of each flag (`sources`, decision J; `chained`, decision N). Moved from
  `memtk.e5.analyze` / `memtk.e5.score` on 2026-11-15, because E6 and E7 read the same scored sets.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from memtk import calibration, config
from memtk.ids import parse_lesion_id, parse_slice_key
from memtk.io import read_table
from memtk.lesionmatch import LesionMatch
from memtk.sets import SliceSet
from memtk.stages import require

CLASSES = ("e1_copy", "e2_full", "e2_partial", "e2_flagged")
CHUNK = 100   # bootstrap draws per block (bounds memory for large samples)


# --- classes ----------------------------------------------------------------------------------

def class_table(s: SliceSet, e1: pd.DataFrame, e2: LesionMatch, e2c: pd.DataFrame) -> pd.DataFrame:
    """One row per slice of a set: patient, E1 strata, score, nearest neighbour and copy flag; E2
    stratum (by informative lesions; "none" without one), class, C, nearest neighbour and
    strongest pair. Moved from `memtk.e4.validate.heval_classes` (E4 writes it for H-eval as
    `heval_classes.csv`)."""
    out = pd.DataFrame({"key": s.keys, "patient_id": s.patients})
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


def class_flags(table: pd.DataFrame, cls: str) -> tuple[pd.Series, pd.Series]:
    """The slices of a class table that a class is counted on (E1: a lesion of at least the
    minimum area; E2: an informative lesion), and which of them it flags. Moved from
    `memtk.e4.validate.fp_flags`."""
    if cls == "e1_copy":
        return table["stratum_e1"].notna(), table["e1_copy"].eq(True)
    on = table["stratum_e2"] != "none"
    wanted = {"e2_full": ["full"], "e2_partial": ["partial"], "e2_flagged": ["full", "partial"]}[cls]
    return on, table["e2_class"].isin(wanted)


def share_row(flags: pd.Series, patients: pd.Series, draws: int, rng: np.random.Generator) -> dict:
    """Count, share and bootstrap-by-patient interval of a boolean column. Moved from
    `memtk.e4.validate`."""
    low, high = calibration.bootstrap_share(flags.to_numpy(), patients.to_numpy(), draws, rng)
    return {"count": int(flags.sum()), "share": float(flags.mean()) if len(flags) else np.nan,
            "ci_low": low, "ci_high": high}


# --- cluster bootstrap ------------------------------------------------------------------------

def resample_sums(values: np.ndarray, clusters: np.ndarray | Sequence[str], draws: int,
                  rng: np.random.Generator) -> np.ndarray:
    """Column sums of `values` (n rows, or n x m) over `draws` cluster-bootstrap resamples: the
    clusters are drawn with replacement, as many as there are, and every row of a drawn cluster
    counts. Returns (draws, m). Pass one cluster per row for independent rows."""
    values = np.asarray(values, dtype=float)
    values = values[:, None] if values.ndim == 1 else values
    _, codes = np.unique(np.asarray(clusters), return_inverse=True)
    sums = np.zeros((codes.max() + 1 if len(codes) else 0, values.shape[1]))
    np.add.at(sums, codes, values)
    types, counts = np.unique(sums, axis=0, return_counts=True)
    n = int(counts.sum())
    out = np.empty((draws, values.shape[1]))
    for start in range(0, draws, CHUNK):
        k = min(CHUNK, draws - start)
        out[start:start + k] = rng.multinomial(n, counts / n, size=k) @ types
    return out


def interval(boot: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of bootstrap values (NaN draws ignored)."""
    low, high = np.nanpercentile(boot, [50 * (1 - level), 50 * (1 + level)])
    return float(low), float(high)


def one_sided_p(boot: np.ndarray) -> float:
    """Bootstrap p for "the statistic is above 0": (1 + draws at or below 0) / (draws + 1)."""
    boot = np.asarray(boot)[~np.isnan(boot)]
    return float((1 + np.sum(boot <= 0)) / (len(boot) + 1))


def excess_test(flags_s: np.ndarray, clusters_s, flags_h: np.ndarray, clusters_h, draws: int,
                rng: np.random.Generator, level: float = 0.95) -> dict:
    """Rate of a query set S and of the held-out set H, and the excess S - H with its interval and
    one-sided p; each set resampled by its own clusters, independently (decision M)."""
    flags_s, flags_h = np.asarray(flags_s, dtype=bool), np.asarray(flags_h, dtype=bool)
    bs = resample_sums(np.c_[flags_s, np.ones(len(flags_s))], clusters_s, draws, rng)
    bh = resample_sums(np.c_[flags_h, np.ones(len(flags_h))], clusters_h, draws, rng)
    rate_s, rate_h = bs[:, 0] / bs[:, 1], bh[:, 0] / bh[:, 1]
    boot = rate_s - rate_h
    point_s, point_h = float(flags_s.mean()), float(flags_h.mean())
    return {"n_s": len(flags_s), "flagged_s": int(flags_s.sum()), "rate_s": point_s,
            "rate_s_low": interval(rate_s, level)[0], "rate_s_high": interval(rate_s, level)[1],
            "n_h": len(flags_h), "flagged_h": int(flags_h.sum()), "rate_h": point_h,
            "rate_h_low": interval(rate_h, level)[0], "rate_h_high": interval(rate_h, level)[1],
            "excess": point_s - point_h, "excess_low": interval(boot, level)[0],
            "excess_high": interval(boot, level)[1], "p": one_sided_p(boot)}


def holm(p: np.ndarray | Sequence[float]) -> np.ndarray:
    """Holm's step-down adjustment of a family of p-values (adjusted values, capped at 1, in the
    input order)."""
    values = np.asarray(p, dtype=float)
    order = np.argsort(values)
    adjusted = np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order])
    out = np.empty(len(values))
    out[order] = np.minimum(adjusted, 1.0)
    return out


def standardized_rate(k: np.ndarray, n: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rate standardised to a reference mix: sum over groups of weight x (k / n), with the weights
    normalised over the groups that have slices (last axis = groups; works for one row of counts
    or a bootstrap array of them). NaN where no group with weight has a slice."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    w = np.broadcast_to(np.asarray(weights, dtype=float), n.shape) * (n > 0)
    rate = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    total = w.sum(axis=-1)
    return np.divide((w * rate).sum(axis=-1), total, out=np.full(total.shape, np.nan), where=total > 0)


def cell_rates(flags: np.ndarray, cells: np.ndarray, clusters, levels: Sequence, weights: np.ndarray,
               draws: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Rates of one or more flag columns (n x m) standardised to `weights` over the cells
    `levels` (e.g. stratum x size bin), as a point estimate (m,) and over `draws` cluster-bootstrap
    resamples (draws x m). With the set's own cell shares as weights this is its crude rate; with
    another set's shares, that set's mix. Cells absent from a resample drop out of its weights."""
    flags = np.asarray(flags, dtype=bool)
    flags = flags[:, None] if flags.ndim == 1 else flags
    cells = np.asarray(cells)
    in_cell = np.stack([cells == c for c in levels], axis=1)                       # n x C
    values = np.concatenate([(flags[:, :, None] & in_cell[:, None, :]).reshape(len(flags), -1), in_cell], axis=1)
    m, c = flags.shape[1], len(levels)
    point = values.sum(axis=0)
    boot = resample_sums(values, clusters, draws, rng)

    def rates(sums: np.ndarray) -> np.ndarray:
        k = sums[..., :m * c].reshape(*sums.shape[:-1], m, c)
        n = sums[..., m * c:][..., None, :]
        return standardized_rate(k, np.broadcast_to(n, k.shape), np.asarray(weights, dtype=float))

    return rates(point), rates(boot)


# --- AUC --------------------------------------------------------------------------------------

def auc(scores_s: np.ndarray | Sequence[float], scores_h: np.ndarray | Sequence[float]) -> float:
    """P(a score of S > a score of H), ties counting one half (= Mann-Whitney U / (n_s n_h))."""
    s, h = np.sort(np.asarray(scores_s, dtype=float)), np.sort(np.asarray(scores_h, dtype=float))
    below = np.searchsorted(h, s, side="left")
    ties = np.searchsorted(h, s, side="right") - below
    return float((below + 0.5 * ties).sum() / (len(s) * len(h)))


def auc_bootstrap(scores_s: np.ndarray, clusters_s, scores_h: np.ndarray, clusters_h, draws: int,
                  rng: np.random.Generator) -> np.ndarray:
    """AUC over `draws` two-level bootstrap resamples: S and H each resampled by their own
    clusters (weights from a multinomial over clusters), independently."""
    s, h = np.asarray(scores_s, dtype=float), np.asarray(scores_h, dtype=float)
    _, cs = np.unique(np.asarray(clusters_s), return_inverse=True)
    _, ch = np.unique(np.asarray(clusters_h), return_inverse=True)
    # wins[i, c]: comparisons of S score i won against cluster c of H (ties one half)
    wins = np.zeros((len(s), ch.max() + 1))
    for c in range(ch.max() + 1):
        hc = np.sort(h[ch == c])
        below = np.searchsorted(hc, s, side="left")
        wins[:, c] = below + 0.5 * (np.searchsorted(hc, s, side="right") - below)
    wins_by_cluster_s = np.zeros((cs.max() + 1, wins.shape[1]))
    np.add.at(wins_by_cluster_s, cs, wins)
    size_s, size_h = np.bincount(cs), np.bincount(ch)
    out = np.empty(draws)
    for start in range(0, draws, CHUNK):
        k = min(CHUNK, draws - start)
        ws = rng.multinomial(len(size_s), np.full(len(size_s), 1 / len(size_s)), size=k)
        wh = rng.multinomial(len(size_h), np.full(len(size_h), 1 / len(size_h)), size=k)
        won = ((ws @ wins_by_cluster_s) * wh).sum(axis=1)   # (draws x S clusters) @ (S clusters x H clusters)
        out[start:start + k] = won / ((ws @ size_s) * (wh @ size_h))
    return out


# --- groups, copied area, held-out maxima -----------------------------------------------------

def chain_groups(patients: Sequence[str], z: Sequence[int], max_gap: int) -> np.ndarray:
    """Group number of each flag (decision N): per source patient, the source slices sorted by z
    form one group as long as consecutive z differ by at most `max_gap`. Groups are numbered from
    0 in order of patient, then first z; flags on the same slice share a group."""
    table = pd.DataFrame({"patient": np.asarray(patients), "z": np.asarray(z, dtype=int)})
    ordered = table.sort_values(["patient", "z"], kind="stable")
    new = (ordered["patient"] != ordered["patient"].shift()) | (ordered["z"].diff() > max_gap)
    groups = pd.Series(np.cumsum(new.to_numpy()) - 1, index=ordered.index)
    return groups.reindex(table.index).to_numpy()


def copied_area_share(pairs: pd.DataFrame, sources: pd.Series, tau_L: pd.Series,
                      query_area: pd.Series, train_area: pd.Series) -> pd.Series:
    """Per query slice, the share of the informative lesion area of the query and its source
    slice that lies in lesion pairs with s >= tau_L (E5 step 7): (area of the kept query lesions +
    area of their training partners) / (all informative lesion area of both slices).

    `pairs`: Hungarian lesion pairs (`query`, `train`, `query_lesion`, `train_lesion`, `s`);
    `sources`: source slice by query; `tau_L`: threshold by query; `query_area`, `train_area`:
    informative lesion area by lesion ID. Queries without a source get NaN."""
    def slice_total(area: pd.Series) -> pd.Series:
        return area.groupby([parse_lesion_id(i)[0] for i in area.index]).sum()

    on_source = pairs[pairs["train"].to_numpy() == sources.reindex(pairs["query"]).to_numpy()]
    kept = on_source[on_source["s"] >= on_source["query"].map(tau_L)]
    copied = (kept["query_lesion"].map(query_area) + kept["train_lesion"].map(train_area)).groupby(kept["query"]).sum()
    q_total, t_total = slice_total(query_area), slice_total(train_area)
    src = sources[sources.astype(bool)]
    total = q_total.reindex(src.index).to_numpy(dtype=float) + t_total.reindex(src.to_numpy()).to_numpy(dtype=float)
    share = pd.Series(copied.reindex(src.index).fillna(0).to_numpy() / total, index=src.index)
    return share.reindex(sources.index)


def cell_maxima(scores: np.ndarray, strata: np.ndarray, sizes: np.ndarray, edges: Sequence[int]) -> pd.DataFrame:
    """Highest score per stratum and size bin (lower edge of the bin of `sizes`; the last bin
    open), with the number of slices; also one row per stratum with `lower` = -1 for the whole
    stratum. Scores of slices below the first edge are left out."""
    table = pd.DataFrame({"stratum": np.asarray(strata), "lower": calibration.area_bins(np.asarray(sizes), edges),
                          "score": np.asarray(scores, dtype=float)}).dropna()
    cells = table.groupby(["stratum", "lower"])["score"].agg(["max", "size"]).reset_index()
    whole = table.groupby("stratum")["score"].agg(["max", "size"]).reset_index().assign(lower=-1)
    out = pd.concat([cells, whole], ignore_index=True).rename(columns={"size": "slices"})
    out["lower"] = out["lower"].astype(int)
    return out[["stratum", "lower", "max", "slices"]]


def beyond_max(scores: np.ndarray, strata: np.ndarray, sizes: np.ndarray, maxima: pd.DataFrame,
               edges: Sequence[int]) -> np.ndarray:
    """Whether each score is strictly above the held-out maximum of its stratum and size bin
    (`cell_maxima`); a cell without held-out slices uses the whole stratum's maximum, and a
    stratum without any uses none (False)."""
    lower = calibration.area_bins(np.asarray(sizes), edges).to_numpy()
    by_cell = dict(zip(zip(maxima["stratum"].tolist(), maxima["lower"].tolist()), maxima["max"].tolist()))

    def limit(stratum: str, lo: float) -> float:
        whole = by_cell.get((stratum, -1), np.inf)
        return whole if np.isnan(lo) else by_cell.get((stratum, int(lo)), whole)

    limits = np.array([limit(st, lo) for st, lo in zip(np.asarray(strata), lower)], dtype=float)
    return np.asarray(scores, dtype=float) > limits


# --- scored sets ------------------------------------------------------------------------------

@dataclass
class Scored:
    """A scored set: `meta` (one row per slice: key, cluster, mask_class, scheme, n_lesions,
    l1_area) and the class table of each variant (`main` and E5 decision O)."""
    name: str
    meta: pd.DataFrame
    classes: dict[str, pd.DataFrame]

    def table(self, variant: str) -> pd.DataFrame:
        """The class table of a variant joined with `meta` (one row per slice)."""
        return self.classes[variant].drop(columns=["patient_id"]).merge(
            self.meta.drop(columns=["n_lesions"]), on="key", how="left", validate="one_to_one")


def query_table(q: SliceSet, clusters: np.ndarray | Sequence[str], mask_class: Sequence[str],
                scheme: Sequence[str]) -> pd.DataFrame:
    """One row per query slice: `key`, `cluster` (the bootstrap unit: the sample itself for S,
    the patient for real slices), `mask_class`, `scheme` (naming scheme or source set),
    `n_lesions` (lesions >= the minimum area) and `l1_area` (area of the largest lesion, strays
    included: the size used for near-exact copies and the size-confound check)."""
    flat = q.labels.reshape(len(q.labels), -1)
    return pd.DataFrame({"key": q.keys, "cluster": list(clusters), "mask_class": list(mask_class),
                         "scheme": list(scheme), "n_lesions": flat.max(axis=1).astype(int),
                         "l1_area": (flat == 1).sum(axis=1).astype(int)})


def load_scored(name: str, score_dir: Path, variants: list[str]) -> Scored:
    """A query set scored by E5's `score_query` (its `queries.csv` and class tables)."""
    return Scored(name, read_table(score_dir / "queries.csv"),
                  {v: read_table(score_dir / f"classes_{v}.csv") for v in ["main", *variants]})


def heldout(cfg: dict, dataset: str, heval: SliceSet, variants: list[str]) -> Scored:
    """H-eval as a scored set (the baseline): its slices clustered by patient, the main class table
    from E4 and one per variant from E5 `prepare` (pass no variants for the main table only)."""
    meta = query_table(heval, heval.patients, ["clean"] * len(heval.keys), ["H-eval"] * len(heval.keys))
    e4_dir, e5_dir = (config.output_dir(cfg, dataset) / e for e in ("e4", "e5"))
    require(e4_dir / "heval_classes.csv", "e4_validate.py validate")
    classes = {"main": read_table(e4_dir / "heval_classes.csv")}
    for v in variants:
        require(e5_dir / f"heval_classes_{v}.csv", "e5_audit.py prepare")
        classes[v] = read_table(e5_dir / f"heval_classes_{v}.csv")
    return Scored("H-eval", meta, classes)


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
