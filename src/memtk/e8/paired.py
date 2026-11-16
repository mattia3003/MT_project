"""E8 stage `paired` (`docs/e8-plan.md`, step 4) and the measurement and analysis it shares with
`rehearse` (decisions AC, AD), so the code that reads S's MRIs is exactly the rehearsed code.

- **Stage `paired`:** reads `select`'s pairs, S's MRIs (the only E8 stage that does) and the H-eval
  reference measures stored by `rehearse`; refuses unless E5's lock is unchanged, `select` and
  `rehearse` ran with the current `paired:` and the rehearsal readings were as required. Writes
  `results/<pool>/e8/paired/`.

- **Measure (AC):** for each pair of a pair table, the query image (synthetic MRI, held-out or mock
  BRAVO) and its brain are moved by the pair's shift (zero fill) and z-scored by the moved brain; the
  comparison slice's BRAVO is z-scored by its own brain; the Gaussian SSIM map is averaged over the
  lesion neighbourhood (the query's informative lesions grown, inside both brains), the whole brain
  (inside both) and the ring (the neighbourhood without the lesion pixels of either mask as saved).
- **Analyse (AD):** on rows of kind `t`, the AUC of cases (A) against controls (B) per region with the
  two-level bootstrap by cluster and the one-sided Mann-Whitney p; against a reference set (H-eval)
  the difference of AUCs from independent draws, and the mechanical reading on the primary region.
  Secondary: per reason, per half of query brain area. Within A (S only): t against the runner-up
  slice (paired Wilcoxon) and the mean SSIM per z offset (bootstrap by cluster).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import mannwhitneyu, wilcoxon

from memtk import config
from memtk.audit import auc, auc_bootstrap, interval, resample_sums
from memtk.calibration import load_frozen
from memtk.e5.common import lock_unchanged
from memtk.e8.common import paired_config, real_slices, synthetic_samples
from memtk.io import LabelStack, read_table
from memtk.lesions import brain_mask
from memtk.paired import align, lesion_region, region_means, regions, ssim_map, zscore_in_brain
from memtk.plotting import save
from memtk.stages import refuse_overwrite, require

# (image, mask as saved, label map) of a query key
Query = Callable[[str], tuple[np.ndarray, np.ndarray, np.ndarray]]
READINGS = ("supported (observational)", "explained by mask resemblance", "not supported")
Z_DISPLAY = (-3.5, 6.0)   # display range of z-scored images in the example figure
# E7 reports always drawn in the S example figure (plan, figures): report 50, where the synthetic MRI
# shows a lesion inside the mask's outline (`docs/experiment-plan.md`, E8 "Carried over from E6 + E7")
EXAMPLE_REPORTS = ("50",)


# --- stages -----------------------------------------------------------------------------------

def run_paired(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """Decision AD on S: measure every pair of `select` (group A, matched group B, within-A rows) on
    S's MRIs, compare A with B against the stored H-eval reference, write the reading. Needs every
    `paired:` value, E5's lock unchanged, `select` and `rehearse` made with the current `paired:` and
    the rehearsal readings as required. Writes `results/<pool>/e8/paired/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "e8") / "paired"
    refuse_overwrite([out], overwrite)
    p = paired_config(cfg)
    lock = lock_unchanged(cfg, name, pool_name)
    select_dir = config.results_dir(cfg, pool_name, "e8") / "select"
    rehearse_dir = config.results_dir(cfg, name, "e8") / "rehearse"
    for d, stage in ((select_dir, "e8_paired.py select"), (rehearse_dir, "e8_paired.py rehearse")):
        require(d / "run_info.json", stage)
        made_with = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        if made_with["config"].get("paired") != cfg["paired"]:
            raise SystemExit(f"`paired:` changed since '{stage}' ran; run it again first.")
        if stage.endswith("rehearse") and not made_with.get("as_required"):
            raise SystemExit("The rehearsal readings were not as required (docs/e8-plan.md, decision AE).")
    e3, e3_hash = load_frozen(cfg, name)
    s = settings(cfg, p, e3)
    draws = int(cfg["calibration"]["bootstrap_draws"])
    rng = np.random.default_rng(cfg["seed"])

    pairs = read_table(select_dir / "pairs.csv")
    reference = read_table(rehearse_dir / "heval" / "measures.csv")
    m, query, real = measure_s(cfg, name, pool_name, pairs, s)
    tables, read = analyze(m, reference, p, s, draws, rng)
    primary = f"ssim_{p['measure']['primary']}"

    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "measures.csv", index=False)
    for tname, tab in tables.items():
        tab.to_csv(out / f"{tname}.csv", index=False)
    draw_groups(m, reference, tables["tests"], ("S group A", "S group B"), out / "ssim_groups.png",
                "E8 paired: SSIM of synthetic MRIs with their comparison slices, S group A vs B and H-eval flagged vs novel")
    draw_within(tables["within_a"], p["measure"]["primary"], out / "within_a.png",
                "E8 paired, within group A: SSIM by z offset from the source slice t; t minus the runner-up slice")
    reports = report_numbers(cfg, pool_name)
    keys = example_keys(m, primary, reports)
    draw_examples(m, keys, query, real, s, primary, out / "examples.png",
                  "E8 paired: group A examples (median per reason, the strongest, E7 report 50), z-scored, lesion "
                  "neighbourhood outlined", notes={k: f"E7 report {reports[k]}" for k in keys if k in reports})
    text = review_text(pool_name, m, tables, read, primary, lock)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "paired", "inputs": ["S MRIs", "T", "H-eval reference"],
                                                            "frozen_sha256": e3_hash, "e5_lock": lock, "reading": read})
    print(f"\n{text}\nWrote {out}")


def measure_s(cfg: dict, dataset: str, pool: str, pairs: pd.DataFrame,
              s: Settings) -> tuple[pd.DataFrame, Query, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Every pair of `select` measured on S's MRIs (masks as saved, label maps of the pool) against
    the comparison slices' BRAVO; also the query loader and the real slices. Shared by `run_paired`
    and `check`."""
    print(f"Reading {pairs['query'].nunique()} synthetic samples and {pairs['comparison'].nunique()} real slices ...")
    samples = synthetic_samples(cfg, pool, pairs["query"])
    real = real_slices(cfg, dataset, pairs["comparison"])
    stack = LabelStack(config.output_dir(cfg, pool) / "labels.npz")
    labels = {k: stack[k] for k in samples}

    def query(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, saved = samples[key]
        return image, saved, labels[key]

    print(f"Measuring {len(pairs)} pairs ...")
    return measure_pairs(pairs, query, real, s), query, real


def report_numbers(cfg: dict, pool: str) -> dict[str, str]:
    """E7's report number of each S sample that has a report (`report_index.csv`)."""
    index = read_table(config.results_dir(cfg, pool, "e7") / "reports" / "report_index.csv")
    index = index[index["set"] == "S"]
    return {str(k): str(r).split("_")[0] for k, r in zip(index["key"], index["report"])}


def example_keys(m: pd.DataFrame, primary: str, reports: dict[str, str]) -> list[str]:
    """The S example cases: one per reason (median), the group-A case with the highest primary SSIM,
    and the cases of `EXAMPLE_REPORTS`; each once."""
    a = m[(m["kind"] == "t") & (m["group"] == "A") & m[primary].notna()]
    keys = example_cases(m, primary)
    keys.append(str(a["query"].to_numpy()[np.argmax(a[primary].to_numpy())]))
    keys += [k for k, n in reports.items() if n in EXAMPLE_REPORTS and k in set(a["query"])]
    return list(dict.fromkeys(keys))


@dataclass(frozen=True)
class Settings:
    """Decision AC's measurement settings (brain from the `brain:` config, E0b decision A)."""
    brain_method: str
    low_threshold: float
    bound: int            # informative-lesion bound (frozen E3 file)
    grow_px: int
    data_range: float
    sigma: float
    regions: tuple[str, ...]


def settings(cfg: dict, p: dict, e3: dict) -> Settings:
    """The measurement settings from the config (`brain:`, `paired.measure`) and the frozen E3 file."""
    m = p["measure"]
    return Settings(cfg["brain"]["method"], float(cfg["brain"]["low_threshold"]), int(e3["informative_bound_px"]),
                    int(m["region_grow_px"]), float(m["data_range"]), float(m["ssim_sigma"]), tuple(m["regions"]))


# --- measurement ------------------------------------------------------------------------------

@dataclass
class PairImages:
    """One aligned pair: both z-scored images, the SSIM map and the regions (None if a brain is
    empty or constant)."""
    query: np.ndarray
    comparison: np.ndarray
    smap: np.ndarray
    masks: dict[str, np.ndarray]


def pair_images(image: np.ndarray, saved: np.ndarray, labels: np.ndarray, t_image: np.ndarray,
                t_saved: np.ndarray, dr: int, dc: int, s: Settings) -> tuple[PairImages | None, int, int]:
    """Decision AC for one pair: the query moved by (dr, dc) with its brain, both z-scored by their
    own brains, the SSIM map and the regions; also both brain areas (px, before the move)."""
    q_brain = brain_mask(image, s.brain_method, s.low_threshold)
    t_brain = brain_mask(t_image, s.brain_method, s.low_threshold)
    q_moved, b_moved = align(image, dr, dc), align(q_brain, dr, dc)
    try:
        zq, zt = zscore_in_brain(q_moved, b_moved), zscore_in_brain(t_image, t_brain)
    except ValueError:
        return None, int(q_brain.sum()), int(t_brain.sum())
    region = align(lesion_region(labels, s.bound, s.grow_px), dr, dc)
    masks = regions(region, align(saved, dr, dc), t_saved, b_moved, t_brain)
    masks = {k: v for k, v in masks.items() if k in s.regions}
    return PairImages(zq, zt, ssim_map(zq, zt, s.data_range, s.sigma), masks), int(q_brain.sum()), int(t_brain.sum())


def measure_pairs(pairs: pd.DataFrame, query: Query, real: dict[str, tuple[np.ndarray, np.ndarray]],
                  s: Settings) -> pd.DataFrame:
    """The pair table with, per row, `ssim_<region>` and `px_<region>` (NaN and 0 for an empty
    region or brain), `brain_q_px`, `brain_t_px`. `real` holds (BRAVO, mask as saved) of every
    comparison slice."""
    rows = []
    for q, c, dr, dc in zip(pairs["query"], pairs["comparison"], pairs["shift_row"], pairs["shift_col"]):
        image, saved, labels = query(q)
        t_image, t_saved = real[c]
        imgs, bq, bt = pair_images(image, saved, labels, t_image, t_saved, int(dr), int(dc), s)
        row: dict[str, float] = {"brain_q_px": bq, "brain_t_px": bt}
        if imgs is None:
            row.update({**{f"ssim_{r}": np.nan for r in s.regions}, **{f"px_{r}": 0 for r in s.regions}})
        else:
            row.update(region_means(imgs.smap, imgs.masks))
        rows.append(row)
    return pd.concat([pairs.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


# --- analysis ---------------------------------------------------------------------------------

def auc_draws(m: pd.DataFrame, col: str, draws: int, rng: np.random.Generator) -> tuple[float, np.ndarray, dict]:
    """AUC of group A against group B on a column (rows of kind `t` with a value), its two-level
    bootstrap draws (each group by its clusters) and counts / medians / Mann-Whitney p."""
    t = m[(m["kind"] == "t") & m[col].notna()]
    a, b = t[t["group"] == "A"], t[t["group"] == "B"]
    info = {"n_a": len(a), "n_b": len(b), "excluded": int(((m["kind"] == "t") & m[col].isna()).sum()),
            "median_a": float(a[col].median()), "median_b": float(b[col].median())}
    if not len(a) or not len(b):
        return np.nan, np.full(draws, np.nan), {**info, "mw_p": np.nan}
    info["mw_p"] = float(mannwhitneyu(a[col], b[col], alternative="greater").pvalue)
    boot = auc_bootstrap(a[col].to_numpy(), a["cluster"], b[col].to_numpy(), b["cluster"], draws, rng)
    return auc(a[col].to_numpy(), b[col].to_numpy()), boot, info


def reading(row: dict | pd.Series, auc_null: float) -> str:
    """Decision AD's mechanical reading from one test row (AUC and difference intervals)."""
    if not row["auc_low"] > auc_null:
        return READINGS[2]
    return READINGS[0] if row["delta_low"] > 0 else READINGS[1]


def auc_tests(m: pd.DataFrame, reference: pd.DataFrame | None, s: Settings, draws: int,
               rng: np.random.Generator) -> pd.DataFrame:
    """Per region: AUC of A vs B with interval and Mann-Whitney p; with a reference set also its AUC
    (flagged vs novel), the difference and their intervals."""
    rows = []
    for r in s.regions:
        col = f"ssim_{r}"
        a_auc, boot, info = auc_draws(m, col, draws, rng)
        row = {"region": r, "auc": a_auc, "auc_low": interval(boot)[0], "auc_high": interval(boot)[1], **info}
        if reference is not None:
            h_auc, h_boot, _ = auc_draws(reference, col, draws, rng)
            delta = boot - h_boot
            row.update({"auc_ref": h_auc, "auc_ref_low": interval(h_boot)[0], "auc_ref_high": interval(h_boot)[1],
                        "delta": a_auc - h_auc, "delta_low": interval(delta)[0], "delta_high": interval(delta)[1]})
        rows.append(row)
    return pd.DataFrame(rows)


def secondary_table(m: pd.DataFrame, col: str, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """AUC of A vs B on the primary column within each reason of the cases (with their own
    controls) and within each half of query brain area (split at the median of all `t` rows)."""
    t = m[m["kind"] == "t"]
    split = float(t["brain_q_px"].median())
    subsets = [("reason", r, t[t["reason"] == r]) for r in sorted(t["reason"].unique())]
    subsets += [("brain half", f"< {split:.0f} px", t[t["brain_q_px"] < split]),
                ("brain half", f">= {split:.0f} px", t[t["brain_q_px"] >= split])]
    rows = []
    for split_name, level, d in subsets:
        a_auc, boot, info = auc_draws(d, col, draws, rng)
        rows.append({"split": split_name, "level": level, "auc": a_auc, "auc_low": interval(boot)[0],
                     "auc_high": interval(boot)[1], **info})
    return pd.DataFrame(rows)


def cluster_mean(values: np.ndarray, clusters: np.ndarray, draws: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Mean of values with a percentile interval from a bootstrap over clusters."""
    sums = resample_sums(np.column_stack([values, np.ones(len(values))]), clusters, draws, rng)
    low, high = interval(sums[:, 0] / sums[:, 1])
    return float(np.mean(values)), low, high


def within_table(m: pd.DataFrame, s: Settings, draws: int, rng: np.random.Generator) -> pd.DataFrame:
    """Within A (decision AD), per region: SSIM to t minus SSIM to the runner-up slice (mean with a
    bootstrap interval by cluster, one-sided paired Wilcoxon), and the mean SSIM per z offset
    (0 = t). Empty when the table has no within-A rows."""
    a = m[m["group"] == "A"]
    if not (a["kind"] != "t").any():
        return pd.DataFrame()
    rows = []
    for r in s.regions:
        col = f"ssim_{r}"
        t = a[a["kind"] == "t"].set_index("case")
        ru = a[a["kind"] == "runner_up"].set_index("case")
        both = t[[col, "cluster"]].join(ru[[col]], rsuffix="_ru").dropna()
        diff = (both[col] - both[f"{col}_ru"]).to_numpy()
        mean, low, high = cluster_mean(diff, both["cluster"].to_numpy(), draws, rng)
        p = float(wilcoxon(diff, alternative="greater").pvalue) if len(diff) and np.any(diff != 0) else np.nan  # pyright: ignore[reportAttributeAccessIssue]  # scipy stub lacks the result's pvalue
        rows.append({"region": r, "measure": "t minus runner-up", "dz": 0, "n": len(diff), "mean": mean,
                     "low": low, "high": high, "wilcoxon_p": p})
        profile = a[a["kind"].isin(["t", "dz"]) & a[col].notna()]
        for dz in sorted(profile["dz"].astype(int).unique()):
            d = profile[profile["dz"] == dz]
            mean, low, high = cluster_mean(d[col].to_numpy(), d["cluster"].to_numpy(), draws, rng)
            rows.append({"region": r, "measure": "mean SSIM by z offset", "dz": int(dz), "n": len(d), "mean": mean,
                         "low": low, "high": high, "wilcoxon_p": np.nan})
    return pd.DataFrame(rows)


def analyze(m: pd.DataFrame, reference: pd.DataFrame | None, p: dict, s: Settings, draws: int,
            rng: np.random.Generator) -> tuple[dict[str, pd.DataFrame], str | None]:
    """Every E8 table of a measured set and the reading (None without a reference set)."""
    primary = f"ssim_{p['measure']['primary']}"
    tests = auc_tests(m, reference, s, draws, rng)
    tables = {"tests": tests, "secondary": secondary_table(m, primary, draws, rng),
              "within_a": within_table(m, s, draws, rng)}
    if reference is None:
        return tables, None
    row = tests[tests["region"] == p["measure"]["primary"]].iloc[0]
    return tables, reading(row, float(p["test"]["auc_null"]))


# --- figures ----------------------------------------------------------------------------------

def draw_groups(m: pd.DataFrame, reference: pd.DataFrame | None, tests: pd.DataFrame, labels: tuple[str, str],
                path: Path, title: str) -> Path:
    """Per region, SSIM of cases and controls (points and box) next to the reference set's flagged
    and novel slices, with the AUCs in the panel title."""
    regions_ = tests["region"].tolist()
    fig = Figure(figsize=(3.6 * len(regions_), 3.4))
    axes = fig.subplots(1, len(regions_), squeeze=False)[0]
    rng = np.random.default_rng(0)                       # point jitter only
    groups = [(m, "A", labels[0]), (m, "B", labels[1])]
    if reference is not None:
        groups += [(reference, "A", "H-eval flagged"), (reference, "B", "H-eval novel")]
    for ax, r in zip(axes, regions_):
        col = f"ssim_{r}"
        data = [d.loc[(d["kind"] == "t") & (d["group"] == g), col].dropna().to_numpy() for d, g, _ in groups]
        ax.boxplot(data, widths=0.5, showfliers=False)
        for i, v in enumerate(data, start=1):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=4, alpha=0.4,
                       color="tab:red" if i % 2 else "tab:gray")
        ax.set_xticks(range(1, len(groups) + 1), [g[2] for g in groups], rotation=20, fontsize=7)
        row = tests[tests["region"] == r].iloc[0]
        text = f"{r}: AUC {row['auc']:.2f} [{row['auc_low']:.2f}, {row['auc_high']:.2f}]"
        if "auc_ref" in row:
            text += f"\nH-eval {row['auc_ref']:.2f}; difference {row['delta']:+.2f} [{row['delta_low']:+.2f}, {row['delta_high']:+.2f}]"
        ax.set_title(text, fontsize=7)
        ax.set_ylabel("mean SSIM in region", fontsize=7)
        ax.tick_params(labelsize=7)
    return save(fig, path, title)


def draw_within(within: pd.DataFrame, primary: str, path: Path, title: str) -> Path:
    """Within A: the mean SSIM per z offset with intervals (primary region and whole brain) and the
    t minus runner-up difference per region."""
    fig = Figure(figsize=(9, 3.2))
    ax1, ax2 = fig.subplots(1, 2)
    prof = within[within["measure"] == "mean SSIM by z offset"]
    for r, d in prof.groupby("region"):
        ax1.errorbar(d["dz"], d["mean"], yerr=[d["mean"] - d["low"], d["high"] - d["mean"]], marker="o", ms=3,
                     capsize=2, label=str(r), linewidth=1.5 if r == primary else 0.8)
    ax1.set_xlabel("z offset from the source slice t", fontsize=8)
    ax1.set_ylabel("mean SSIM", fontsize=8)
    ax1.legend(fontsize=7)
    ru = within[within["measure"] == "t minus runner-up"]
    ax2.errorbar(range(len(ru)), ru["mean"], yerr=[ru["mean"] - ru["low"], ru["high"] - ru["mean"]], fmt="o", capsize=3)
    ax2.axhline(0, color="gray", linewidth=0.7)
    ax2.set_xticks(range(len(ru)), [f"{r}\np {p:.3g}" for r, p in zip(ru["region"], ru["wilcoxon_p"])], fontsize=7)
    ax2.set_ylabel("SSIM to t minus to runner-up", fontsize=8)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)
    return save(fig, path, title)


def example_cases(m: pd.DataFrame, primary: str) -> list[str]:
    """One case per reason: the one whose primary SSIM is closest to its reason's median."""
    a = m[(m["kind"] == "t") & (m["group"] == "A") & m[primary].notna()]
    out = []
    for _, d in a.groupby("reason", sort=True):
        distance = (d[primary] - d[primary].median()).abs().to_numpy()
        out.append(str(d["query"].to_numpy()[np.argmin(distance)]))
    return out


def draw_examples(m: pd.DataFrame, keys: list[str], query: Query, real: dict, s: Settings, primary: str,
                  path: Path, title: str, notes: dict[str, str] | None = None) -> Path:
    """Per example case: the aligned query image and the comparison BRAVO (z-scored), the SSIM map,
    the lesion neighbourhood outlined on all three; `notes` adds text to a case's caption (e.g. its
    E7 report)."""
    fig = Figure(figsize=(7.2, 2.5 * max(1, len(keys))))
    axes = fig.subplots(max(1, len(keys)), 3, squeeze=False)
    for row_axes, key in zip(axes, keys):
        r = m[(m["kind"] == "t") & (m["query"] == key)].iloc[0]
        image, saved, labels = query(key)
        t_image, t_saved = real[r["comparison"]]
        imgs, _, _ = pair_images(image, saved, labels, t_image, t_saved, int(r["shift_row"]), int(r["shift_col"]), s)
        if imgs is None:
            continue
        region = imgs.masks[primary.removeprefix("ssim_")]
        source = r.get("image", key)          # only the positive mock's image is another slice than the query
        caption = f"query {key} ({r['reason']})" + (f"\nimage: BRAVO of {source}" if source != key else "")
        if notes and key in notes:
            caption += f"\n{notes[key]}"
        panels = [(imgs.query, caption, "gray", Z_DISPLAY),
                  (imgs.comparison, f"comparison {r['comparison']}", "gray", Z_DISPLAY),
                  (imgs.smap, f"SSIM map; {primary} {r[primary]:.2f}", "viridis", (-0.2, 1.0))]
        for ax, (img, text, cmap, (lo, hi)) in zip(row_axes, panels):
            ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
            if region.any():
                ax.contour(region.astype(float), levels=[0.5], colors="orange", linewidths=0.7)
            ax.set_title(text, fontsize=7)
            ax.axis("off")
    return save(fig, path, title)


# --- review -----------------------------------------------------------------------------------

def review_lines(name: str, m: pd.DataFrame, tables: dict[str, pd.DataFrame], read: str | None, primary: str) -> list[str]:
    """The text block of one measured set: counts, the tests, secondary splits, within A, the reading."""
    t = m[m["kind"] == "t"]
    per_case = t[t["group"] == "B"].groupby("case").size().reindex(t.loc[t["group"] == "A", "query"], fill_value=0)
    lines = [f"--- {name} ---",
             f"Pairs of kind t: {int((t['group'] == 'A').sum())} cases, {int((t['group'] == 'B').sum())} controls "
             f"(cases with fewer than {per_case.max()} controls: {int((per_case < per_case.max()).sum())}, with none: "
             f"{int((per_case == 0).sum())}; every case counts in the AUC); "
             f"query brain median {t['brain_q_px'].median():.0f} px, comparison brain median {t['brain_t_px'].median():.0f} px",
             "Tests (AUC = P(case SSIM > control SSIM); intervals by cluster bootstrap):"]
    for _, r in tables["tests"].iterrows():
        line = (f"  {r['region']}: median SSIM {r['median_a']:.3f} vs {r['median_b']:.3f}; AUC {r['auc']:.3f} "
                f"[{r['auc_low']:.3f}, {r['auc_high']:.3f}], Mann-Whitney p {r['mw_p']:.3g}; n {r['n_a']} / {r['n_b']}, "
                f"left out (empty region) {r['excluded']}")
        if "auc_ref" in r:
            line += (f"; H-eval AUC {r['auc_ref']:.3f} [{r['auc_ref_low']:.3f}, {r['auc_ref_high']:.3f}], difference "
                     f"{r['delta']:+.3f} [{r['delta_low']:+.3f}, {r['delta_high']:+.3f}]")
        lines.append(line)
    lines.append(f"Secondary ({primary}):")
    lines += [f"  {r['split']} {r['level']}: AUC {r['auc']:.3f} [{r['auc_low']:.3f}, {r['auc_high']:.3f}] "
              f"(n {r['n_a']} / {r['n_b']})" for _, r in tables["secondary"].iterrows()]
    w = tables["within_a"]
    if len(w):
        lines.append("Within A:")
        for _, r in w[w["measure"] == "t minus runner-up"].iterrows():
            lines.append(f"  {r['region']}: SSIM to t minus to runner-up {r['mean']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}], "
                         f"Wilcoxon p {r['wilcoxon_p']:.3g} (n {r['n']})")
        prof = w[(w["measure"] == "mean SSIM by z offset") & (w["region"] == primary.removeprefix("ssim_"))]
        lines.append("  z offset profile (" + primary + "): "
                     + ", ".join(f"{int(r['dz']):+d}: {r['mean']:.3f}" for _, r in prof.iterrows()))
    if read is not None:
        lines.append(f"READING ({primary}, against H-eval): {read}")
    return lines


def review_text(pool: str, m: pd.DataFrame, tables: dict[str, pd.DataFrame], read: str | None, primary: str,
                lock: dict) -> str:
    """`review.txt` of `paired`: what was measured, the rehearsed text block on S, the reading and
    what it can and cannot say."""
    lines = [f"E8 paired ({pool}): synthetic MRIs of group A (strong resemblances, E7) and matched group B (novel "
             f"masks) against the BRAVO of their comparison slices, judged against H-eval (decision AD). S's MRIs "
             f"were read by this stage only; E5's lock unchanged ({', '.join(f'{k} {v[:12]}' for k, v in lock.items())}).",
             "", *review_lines("S", m, tables, read, primary), "",
             "Observational only: a 'supported' reading would suggest that stage 2 draws tissue closer to the matched "
             "training slice for these masks (mask as key), but cannot separate that from both stages learning the same "
             "patient, or from common configurations giving similar tissue; only RQ-M1 (master thesis) can."]
    return "\n".join(lines) + "\n"
