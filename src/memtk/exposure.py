"""E6 building blocks (`docs/e6-e7-plan.md`, decisions R, U, V, W): patient exposure, copying vs
placement, rarity (H3) and duplication (H1). No knowledge of stages or file locations.

- **Exposure (R):** observed flags per source patient against the expected share pi_p, the share of
  held-out masks whose nearest neighbour is p, smoothed toward p's share w_p of the T lesion slices:
  pi_p = (n_p + m w_p) / (N + m). Ratio r_p = o_p / (pi_p O) with a bootstrap interval (flags
  resampled as a multinomial, the held-out set by patient). Concentration: the dispersion
  D = sum (o_p - e_p)^2 / e_p, the top-k share and the patients above their null band, against flags
  drawn from pi redrawn in every draw (the uncertainty of pi included).
- **Margins (U):** E2 margin = C of the nearest neighbour minus the best C of any slice of another
  patient (the E1 margin is stored by `wholemask.match`).
- **Rarity (V):** slice rarity = 1 - the best score against slices of other patients, on each
  detector's scale; patient rarity = mean distance to the k nearest other patients on z-scored
  features; Spearman with a permutation p and a bootstrap interval.
- **Duplication (W, H1):** lesion credits (query masks with a lesion pair >= tau_L to a 2D instance
  of a 3D lesion, over every stored slice pair), their bootstrap, and Spearman(span, credits)
  within size bins for a query set against the held-out reference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from memtk import lesionmatch, wholemask
from memtk.audit import CHUNK, interval, resample_sums
from memtk.sets import SliceSet

NULL_BAND = (2.5, 97.5)   # percentiles of the per-patient null band of the exposure ratio


# --- expected shares and exposure (decision R) ------------------------------------------------

def slice_shares(patients: np.ndarray | Sequence[str]) -> pd.Series:
    """w_p: each patient's share of the given slices (one entry per slice), sorted by patient."""
    counts = pd.Series(np.asarray(patients)).value_counts().sort_index()
    return counts / counts.sum()


def expected_shares(nn_counts: pd.Series | np.ndarray, weights: pd.Series, m: float) -> pd.Series:
    """pi_p = (n_p + m w_p) / (N + m) over the patients of `weights` (w_p, summing to 1);
    `nn_counts` are the held-out nearest-neighbour counts per patient (a Series by patient, missing
    patients count 0, or an array in the order of `weights`)."""
    if isinstance(nn_counts, pd.Series):
        unknown = set(nn_counts.index) - set(weights.index)
        if unknown:
            raise ValueError(f"Nearest-neighbour patients without a weight: {sorted(unknown)[:5]}")
        n = nn_counts.reindex(weights.index, fill_value=0).to_numpy(float)
    else:
        n = np.asarray(nn_counts, dtype=float)
    return pd.Series((n + m * weights.to_numpy()) / (n.sum() + m), index=weights.index)


def one_hot(values: np.ndarray | Sequence[str], levels: pd.Index) -> np.ndarray:
    """(len(values), len(levels)) indicator matrix of each value's level (all levels must exist)."""
    codes = levels.get_indexer(pd.Index(values))
    if (codes < 0).any():
        raise ValueError("Values outside the given levels")
    out = np.zeros((len(codes), len(levels)))
    out[np.arange(len(codes)), codes] = 1.0
    return out


def exposure_table(sources: np.ndarray | Sequence[str], source_clusters: np.ndarray | Sequence[str],
                   nn_patients: np.ndarray | Sequence[str], nn_clusters: np.ndarray | Sequence[str], weights: pd.Series, m: float, draws: int, rng: np.random.Generator,
                   level: float = 0.95) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Decision R for one cell. `sources`, `source_clusters`: the source patient and the cluster of
    each flag of the query set (a synthetic mask is its own cluster; mock sets have patients);
    `nn_patients`, `nn_clusters`: the nearest-neighbour patient and the patient (cluster) of each
    scorable held-out mask; `weights`: w_p over every T patient. Both sides are resampled by their
    clusters, independently. Returns the table (one row per T patient: observed, held-out NN count,
    pi, expected, ratio with interval), the ratio draws and the pi draws (draws x patients, in the
    order of `weights`)."""
    patients = weights.index
    o = pd.Series(np.asarray(sources)).value_counts().reindex(patients, fill_value=0).to_numpy(float)
    n_nn = pd.Series(np.asarray(nn_patients)).value_counts().reindex(patients, fill_value=0).to_numpy(float)
    pi = expected_shares(n_nn, weights, m).to_numpy()
    total = o.sum()
    n_draws = resample_sums(one_hot(nn_patients, patients), nn_clusters, draws, rng)
    pi_draws = (n_draws + m * weights.to_numpy()) / (n_draws.sum(axis=1, keepdims=True) + m)
    if total > 0:
        o_draws = resample_sums(one_hot(sources, patients), source_clusters, draws, rng)
        ratio_draws = o_draws / (pi_draws * o_draws.sum(axis=1, keepdims=True))
        ratio = o / (pi * total)
        low, high = np.percentile(ratio_draws, [50 * (1 - level), 50 * (1 + level)], axis=0)
    else:
        ratio_draws = np.full((draws, len(patients)), np.nan)
        ratio = low = high = np.full(len(patients), np.nan)
    table = pd.DataFrame({"patient_id": patients, "observed": o.astype(int), "heldout_nn": n_nn.astype(int),
                          "pi": pi, "expected": pi * total, "ratio": ratio, "ratio_low": low, "ratio_high": high})
    return table, ratio_draws, pi_draws


def concentration_test(observed: np.ndarray, pi: np.ndarray, pi_draws: np.ndarray, top_k: int,
                       rng: np.random.Generator) -> tuple[dict, pd.DataFrame]:
    """Concentration of the observed flags over patients against a null of the same number of
    flags drawn from pi, with pi redrawn in every draw (`pi_draws`, draws x patients): the
    dispersion D = sum (o - e)^2 / e (e = pi O), the top-k share, and the number of patients above
    the upper end of their null band (`NULL_BAND`) with its null expectation. p = (1 + null draws at
    or above the observed value) / (draws + 1). Also returns the null band per patient (ratio units)."""
    o = np.asarray(observed, dtype=float)
    pi = np.asarray(pi, dtype=float)
    total = o.sum()
    e = pi * total
    null = np.vstack([rng.multinomial(int(total), p) for p in pi_draws]).astype(float)

    def dispersion(x: np.ndarray) -> np.ndarray:
        return (np.square(x - e) / e).sum(axis=-1)

    def top_share(x: np.ndarray) -> np.ndarray:
        return np.sort(x, axis=-1)[..., -top_k:].sum(axis=-1) / total

    low, high = np.percentile(null, NULL_BAND, axis=0)
    d_obs, d_null = float(dispersion(o)), dispersion(null)
    t_obs, t_null = float(top_share(o)), top_share(null)
    out = {"flags": int(total), "patients": len(o), "patients_flagged": int((o > 0).sum()),
           "dispersion": d_obs, "dispersion_null_median": float(np.median(d_null)),
           "dispersion_null_p95": float(np.percentile(d_null, 95)),
           "dispersion_p": float((1 + np.sum(d_null >= d_obs)) / (len(d_null) + 1)),
           "dispersion_null": d_null,
           f"top{top_k}_share": t_obs, f"top{top_k}_null_median": float(np.median(t_null)),
           f"top{top_k}_null_p95": float(np.percentile(t_null, 95)),
           f"top{top_k}_p": float((1 + np.sum(t_null >= t_obs)) / (len(t_null) + 1)),
           "above_null_band": int((o > high).sum()),
           "above_null_band_expected": float((null > high).sum(axis=1).mean())}
    band = pd.DataFrame({"null_low": low / np.where(e > 0, e, np.nan), "null_high": high / np.where(e > 0, e, np.nan)})
    return out, band


# --- margins (decision U) ---------------------------------------------------------------------

def patient_of_keys(keys: pd.Series) -> pd.Series:
    """The patient part of real slice keys (`<patient>/z<z>`), vectorised."""
    return keys.str.rsplit("/z", n=1).str[0]


def e2_margins(e2: lesionmatch.LesionMatch) -> pd.Series:
    """Per query slice with a candidate: C of its best slice minus the best C of any slice of
    another patient (0 when only one patient has a candidate slice). Indexed by query."""
    sc = e2.slice_scores(e2.candidate_min_s)
    sc = sc.assign(patient=patient_of_keys(sc["train"]))
    best = sc.groupby(["query", "patient"], sort=False)["C"].max().reset_index()
    best = best.sort_values(["query", "C"], ascending=[True, False])
    rank = best.groupby("query").cumcount()
    first = best[rank == 0].set_index("query")["C"]
    second = best[rank == 1].set_index("query")["C"].reindex(first.index, fill_value=0.0)
    return (first - second).rename("e2_margin")


# --- rarity (decision V) ----------------------------------------------------------------------

def slice_rarity(t: SliceSet, wm: dict, les: dict) -> pd.DataFrame:
    """For every T slice, its best score against the T slices of other patients on each detector's
    scale, and rarity = 1 - that score: E1 (`wholemask.match`, best shifted Dice) and E2 (best
    configuration score C; NaN for a slice without an informative lesion, 0 when no other slice is a
    candidate). `wm`, `les`: keywords from `calibration.frozen_settings`. One run per patient."""
    rows = []
    for p in np.unique(t.patients):
        own = t.patients == p
        q_keys = [k for k, s in zip(t.keys, own) if s]
        r_keys = [k for k, s in zip(t.keys, own) if not s]
        e1 = wholemask.match(t.labels[own], t.labels[~own], t.patients[~own], q_keys, r_keys, **wm)
        e2 = lesionmatch.match(t.labels[own], q_keys, t.labels[~own], r_keys, **les)
        best_c = e2.slice_scores(e2.candidate_min_s).groupby("query")["C"].max().reindex(q_keys, fill_value=0.0)
        n_inf = e2.queries.set_index("query")["n_informative"].reindex(q_keys)
        e1 = e1.set_index("query").reindex(q_keys)
        rows.append(pd.DataFrame({"key": q_keys, "patient_id": p, "n_informative": n_inf.to_numpy(int),
                                  "e1_best_other": e1["score"].to_numpy(float), "e1_best_other_slice": e1["nn"].to_numpy(),
                                  "e2_best_other": np.where(n_inf.to_numpy() > 0, best_c.to_numpy(float), np.nan)}))
    out = pd.concat(rows, ignore_index=True)
    out["rarity_e1"] = 1 - out["e1_best_other"]
    out["rarity_e2"] = 1 - out["e2_best_other"]
    return out


def patient_features(lesions: pd.DataFrame) -> pd.DataFrame:
    """Decision V features per patient from its 2D lesions (one row per lesion: `patient_id`, `key`,
    `area`, `centroid_row_brain`, `centroid_col_brain`, `lesion3d_id`): `n_lesions3d`, `volume_px`
    (lesion pixels over all slices), `median_area_px`, `max_lesions_per_slice`, `position_spread`
    (sqrt of the summed variances of the brain-relative centroid; 0 for one lesion)."""
    g = lesions.groupby("patient_id")
    out = pd.DataFrame({
        "n_lesions3d": g["lesion3d_id"].nunique(),
        "volume_px": g["area"].sum(),
        "median_area_px": g["area"].median(),
        "max_lesions_per_slice": lesions.groupby(["patient_id", "key"]).size().groupby("patient_id").max(),
        "position_spread": np.sqrt(g["centroid_row_brain"].var(ddof=0) + g["centroid_col_brain"].var(ddof=0)),
    })
    return out


def knn_rarity(features: pd.DataFrame, columns: Sequence[str], log1p: Sequence[str], k: int) -> pd.Series:
    """Patient rarity: the mean Euclidean distance to the k nearest other patients, on the given
    feature columns, `log1p` columns log-transformed, all z-scored (a constant column contributes 0;
    a missing value is set to the column mean)."""
    x = features[list(columns)].astype(float).copy()
    for c in log1p:
        x[c] = np.log1p(x[c])
    sd = x.std(ddof=0).replace(0, 1.0)
    z = ((x - x.mean()) / sd).fillna(0.0).to_numpy()
    dist = np.sqrt(np.square(z[:, None, :] - z[None, :, :]).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    nearest = np.sort(dist, axis=1)[:, :k]
    return pd.Series(nearest.mean(axis=1), index=features.index, name="rarity")


# --- Spearman ---------------------------------------------------------------------------------

def rowwise_spearman(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of x with the same row of y (2D arrays, average ranks for
    ties); NaN for a row where either side is constant."""
    rx, ry = rankdata(x, axis=1), rankdata(y, axis=1)
    rx = rx - rx.mean(axis=1, keepdims=True)
    ry = ry - ry.mean(axis=1, keepdims=True)
    den = np.sqrt(np.square(rx).sum(axis=1) * np.square(ry).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry).sum(axis=1) / den, np.nan)


def spearman_test(x: np.ndarray, y: np.ndarray, permutations: int, draws: int, rng: np.random.Generator,
                  y_draws: np.ndarray | None = None, level: float = 0.95) -> dict:
    """Spearman rho of x and y, a two-sided permutation p ((1 + permutations with |rho| at least the
    observed) / (permutations + 1)) and a percentile interval from resampling the pairs (with draw
    d of `y_draws`, draws x n, in place of y when given: the uncertainty of y itself)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    rho = float(rowwise_spearman(x[None, :], y[None, :])[0])
    perm = rowwise_spearman(np.broadcast_to(x, (permutations, len(x))),
                            rng.permuted(np.broadcast_to(y, (permutations, len(y))), axis=1))
    idx = rng.integers(0, len(x), size=(draws, len(x)))
    yy = (y_draws if y_draws is not None else np.broadcast_to(y, (draws, len(y))))
    boot = rowwise_spearman(x[idx], np.take_along_axis(yy, idx, axis=1))
    low, high = interval(boot, level)
    return {"n": len(x), "rho": rho, "rho_low": low, "rho_high": high,
            "p_permutation": float((1 + np.sum(np.abs(perm) >= abs(rho))) / (permutations + 1))}


# --- duplication (decision W, H1) -------------------------------------------------------------

def lesion_credits(lesion_pairs: pd.DataFrame, tau_L: pd.Series, lesion3d_of: pd.Series) -> pd.DataFrame:
    """The distinct (query, lesion3d_id) pairs where a query lesion pairs with a 2D instance of the 3D
    lesion at s >= tau_L of the query (`tau_L` by query), over every stored slice pair.
    `lesion3d_of`: 3D lesion ID by training lesion ID."""
    kept = lesion_pairs[lesion_pairs["s"].to_numpy() >= lesion_pairs["query"].map(tau_L).to_numpy()]
    out = pd.DataFrame({"query": kept["query"].to_numpy(), "lesion3d_id": kept["train_lesion"].map(lesion3d_of).to_numpy()})
    return out.dropna().drop_duplicates().reset_index(drop=True)


def credit_counts(credits: pd.DataFrame, lesions: pd.Index) -> np.ndarray:
    """Credited queries per 3D lesion, in the order of `lesions`."""
    return credits["lesion3d_id"].value_counts().reindex(lesions, fill_value=0).to_numpy(float)


def credit_draws(credits: pd.DataFrame, lesions: pd.Index, draws: int, rng: np.random.Generator,
                 n_queries: int | None = None, clusters: pd.Series | None = None) -> np.ndarray:
    """Bootstrap draws (draws x lesions) of the credit counts. Independent queries: pass `n_queries`
    (every scorable query, credited or not), resampled as a multinomial. Clustered queries: pass
    `clusters` (cluster by query, every scorable query), resampled by cluster."""
    query, lesion = pd.Index(credits["query"]), pd.Index(credits["lesion3d_id"])
    if clusters is not None:
        m = np.zeros((len(clusters), len(lesions)))
        m[clusters.index.get_indexer(query), lesions.get_indexer(lesion)] = 1.0
        return resample_sums(m, clusters.to_numpy(), draws, rng)
    if n_queries is None:
        raise ValueError("Pass n_queries (independent queries) or clusters")
    credited = query.unique()
    m = np.zeros((len(credited), len(lesions)))
    m[credited.get_indexer(query), lesions.get_indexer(lesion)] = 1.0
    probs = np.r_[np.ones(len(credited)), n_queries - len(credited)] / n_queries
    out = np.empty((draws, len(lesions)))
    for start in range(0, draws, CHUNK):
        k = min(CHUNK, draws - start)
        out[start:start + k] = rng.multinomial(n_queries, probs, size=k)[:, :-1] @ m
    return out


def span_rates(span: np.ndarray, bins: np.ndarray, counts: np.ndarray, count_draws: np.ndarray, n_queries: int,
               groups: Sequence[tuple[int, int | None]], order: Sequence[str], level: float = 0.95) -> pd.DataFrame:
    """H1 shown as rates (descriptive, added 2026-11-15 after the S run): per size bin (in `order`) and
    span group (lowest, highest span; None = open), the 3D lesions, their mean credits per 1,000
    scorable queries with an interval from `count_draws`, and that rate relative to the size bin's mean
    rate (1 = the bin's average; the shape H1 is about) with its interval."""
    span = np.asarray(span)
    rows = []
    for b in order:
        in_bin = bins == b
        if not in_bin.any():
            continue
        bin_rate = counts[in_bin].mean()
        bin_draws = count_draws[:, in_bin].mean(axis=1)
        for lo, hi in groups:
            sel = in_bin & (span >= lo) & ((span <= hi) if hi is not None else True)
            if not sel.any():
                continue
            rate, draws_ = counts[sel].mean(), count_draws[:, sel].mean(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel_draws = draws_ / bin_draws
            low, high = interval(draws_ * 1000 / n_queries, level)
            rel_low, rel_high = interval(rel_draws, level)
            rows.append({"bin": b, "span": f"{lo}-{hi}" if hi is not None else f">= {lo}", "lesions3d": int(sel.sum()),
                         "rate_per_1000": rate * 1000 / n_queries, "rate_low": low, "rate_high": high,
                         "relative": rate / bin_rate if bin_rate > 0 else np.nan, "relative_low": rel_low,
                         "relative_high": rel_high})
    return pd.DataFrame(rows)


def h1_table(span: np.ndarray, bins: np.ndarray, c_s: np.ndarray, c_h: np.ndarray, c_s_draws: np.ndarray,
             c_h_draws: np.ndarray, level: float = 0.95, order: Sequence[str] | None = None) -> pd.DataFrame:
    """Decision W: per size bin, Spearman(span, credits) of the query set (rho_s) and the held-out
    set (rho_h) and their difference with an interval (both sets' draws); a last row "weighted" with
    the mean over bins weighted by their 3D lesions (bins whose rho is undefined left out). Each
    array is in the order of the 3D lesions; `bins` holds each lesion's bin label; `order` lists the
    bins in the order of the rows (default: as they first appear)."""
    rows, diffs, weights = [], [], []
    for b in (order if order is not None else pd.unique(bins)):
        if not (bins == b).any():
            continue
        sel = bins == b
        x = np.asarray(span, dtype=float)[sel]
        rho_s = rowwise_spearman(x[None, :], c_s[None, sel])[0]
        rho_h = rowwise_spearman(x[None, :], c_h[None, sel])[0]
        xd = np.broadcast_to(x, (len(c_s_draws), len(x)))
        diff = rowwise_spearman(xd, c_s_draws[:, sel]) - rowwise_spearman(xd, c_h_draws[:, sel])
        low, high = interval(diff, level) if np.isfinite(diff).any() else (np.nan, np.nan)
        rows.append({"bin": b, "lesions3d": int(sel.sum()), "credited_s": int((c_s[sel] > 0).sum()),
                     "credited_h": int((c_h[sel] > 0).sum()), "rho_s": rho_s, "rho_h": rho_h,
                     "difference": rho_s - rho_h, "difference_low": low, "difference_high": high})
        if np.isfinite(rho_s - rho_h):
            diffs.append(diff)
            weights.append(sel.sum())
    if diffs:
        w = np.asarray(weights, dtype=float)
        stacked = np.vstack(diffs)
        # a draw where a bin's rho is undefined uses the other bins' weights
        ok = np.isfinite(stacked)
        boot = (np.where(ok, stacked, 0) * w[:, None]).sum(axis=0) / (ok * w[:, None]).sum(axis=0)
        point = sum(r["difference"] * r["lesions3d"] for r in rows if np.isfinite(r["difference"])) / w.sum()
        low, high = interval(boot, level)
        rows.append({"bin": "weighted", "lesions3d": int(w.sum()), "difference": point,
                     "difference_low": low, "difference_high": high})
    return pd.DataFrame(rows)
