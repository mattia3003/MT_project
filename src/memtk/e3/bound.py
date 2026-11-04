"""E3 stage `bound-review` (evidence for decision G: provisional tau_L, chance-rate cut,
informative-lesion bound).

Runs the E2 detector with every lesion (>= the minimum lesion area) of H-cal against T, takes
each H-cal lesion's best lesion score, and shows by lesion size how often such a chance match
reaches tau_L. Reads the E0b lesion tables (T, H-cal) and the E0 bound-impact table (what each
bound costs each set, S included, without any detector on S). Writes
`results/<dataset>/e3/bound_review/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from memtk import calibration, config, lesionmatch
from memtk.e3.common import lesion_settings
from memtk.ids import parse_lesion_id
from memtk.plotting import HIGHLIGHT_COLOR, LESION_COLOR, hide_unused, save, show_slice, tile_figure
from memtk.sets import SliceSet, load_set
from memtk.stages import decided, pick, refuse_overwrite, require

# Lower edges of the area bins: the minimum lesion area, then the candidate bounds of the E0
# bound-impact table, so every proposed bound has its cost in that table
EDGES = (4, 5, 8, 10, 12, 15, 20, 25, 30, 40)
TAU_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# The provisional tau_L comes from lesions of at least this area: the lower end of the expected
# bound (10-20 px), so that blobs too small to carry shape do not set it
PROVISIONAL_MIN_AREA = 10
CUTS = (0.05, 0.10)
CROP = 16  # half-size of the example crops, px


# --- stages -----------------------------------------------------------------------------------

def run_bound_review(cfg: dict, dataset: str | None = None, pool: str | None = None,
                     overwrite: bool = False) -> None:
    """Evidence for decision G (`calibration.provisional_tau_L`, `chance_rate_max`,
    `informative_bound_px`): chance-match rates of H-cal lesions by area bin for a grid of
    tau_L, the proposal by the experiment plan's one-round rule, the cost of each bound (E0),
    and example matches. Needs the E0b `lesions` and `stats` stages and decision F. Writes
    `results/<dataset>/e3/bound_review/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, name, "e3") / "bound_review"
    refuse_overwrite([out], overwrite)
    impact_path = config.results_dir(cfg, name, "e0b") / "stats" / f"bound_impact_{pool_name}.csv"
    require(impact_path, "stats")
    min_area = int(decided(cfg, "lesions.min_lesion_area", "e0b_audit.py lesions-review"))
    percentile = float(cfg["calibration"]["percentile"])
    settings = lesion_settings(cfg)
    rng = np.random.default_rng(cfg["seed"])

    t, hcal = load_set(cfg, name, "T"), load_set(cfg, name, "H-cal")
    res = lesionmatch.match(hcal.labels, hcal.keys, t.labels, t.keys, min_area, **settings)
    best = res.lesions.copy()
    best["patient_id"] = best["query"].map(dict(zip(hcal.keys, hcal.patients)))
    best["lower"] = calibration.area_bins(best["area"].to_numpy(), EDGES)

    big = best["area"] >= PROVISIONAL_MIN_AREA
    tau0 = round(calibration.percentile_threshold(best.loc[big, "best_s"].to_numpy(), percentile), 2)
    taus = sorted({*TAU_GRID, tau0})
    rates = calibration.chance_rates(best["best_s"].to_numpy(), best["area"].to_numpy(),
                                     best["patient_id"].to_numpy(), EDGES, taus)
    at_tau0 = rates[rates["tau_L"] == tau0]
    bounds = {cut: calibration.propose_bound(at_tau0, cut) for cut in CUTS}
    impact = pd.read_csv(impact_path)
    grid = rates.pivot(index="bin", columns="tau_L", values="rate").reindex(at_tau0["bin"])
    grid.insert(0, "patients", at_tau0.set_index("bin")["patients"])
    grid.insert(0, "lesions", at_tau0.set_index("bin")["lesions"])

    rates_figure(rates, tau0, out / "chance_rates.png",
                 f"{name}: chance-match rate of H-cal lesions by area (share whose best lesion score "
                 f"against any T lesion reaches tau_L); bold: provisional tau_L = {tau0}; dashed: the 5% and 10% cuts")
    example_sheet(best, hcal, t, rng, out / "examples.png",
                  f"{name}: H-cal lesions (red, left) and their best T match (green, right), 3 random "
                  f"matched lesions per area bin; s = shape x position x size; other lesions of the slice in grey")
    best.drop(columns="lower").to_csv(out / "best_matches_hcal.csv", index=False)
    rates.to_csv(out / "chance_rates.csv", index=False)
    text = review_text(name, pool_name, t, hcal, best, settings, min_area, percentile, tau0, grid, bounds, impact)
    (out / "review.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={
        "stage": "bound-review", "pool": pool_name, "inputs": ["T", "H-cal"], "settings_px": settings})
    print(f"\n{text}\nWrote {out}")


# --- review text ------------------------------------------------------------------------------

def review_text(name: str, pool_name: str, t: SliceSet, hcal: SliceSet, best: pd.DataFrame, settings: dict,
                min_area: int, percentile: float, tau0: float, grid: pd.DataFrame, bounds: dict,
                impact: pd.DataFrame) -> str:
    """The readable summary with the proposal for decision G."""
    matched = best["best_s"].notna()
    big = best["area"] >= PROVISIONAL_MIN_AREA
    candidates = sorted({b for b in bounds.values() if b is not None} | {10, 15, 20})
    cost = (impact[impact["bound"].isin(candidates)]
            .pivot(index="bound", columns="set", values="share_slices_without")
            .reindex(columns=["T", "H-cal", "H-eval", "S main"]))
    patients = impact[(impact["set"] == "H-cal") & impact["bound"].isin(candidates)].set_index("bound")
    cost["H-cal patients kept"] = patients["patients_with_informative"].astype(int)
    bound_lines = "\n".join(
        f"  cut {cut:.0%}: bound = {b} px" if b is not None else f"  cut {cut:.0%}: no bound (even the largest bin reaches it)"
        for cut, b in bounds.items())
    return (
        f"E3 bound-review: dataset '{name}' (S cost from the E0 table of pool '{pool_name}'; no detector on S)\n\n"
        f"Sets: T {len(t.keys)} slices ({left_out_text(t)}); H-cal {len(hcal.keys)} slices, "
        f"{len(set(hcal.patients))} patients ({left_out_text(hcal)}).\n"
        f"E2 settings: every lesion >= {min_area} px; sigma {settings['sigma_px']:.1f} px; prefilter "
        f"{settings['prefilter_px']:.1f} px; candidate cut s >= {settings['candidate_min_s']}.\n"
        f"H-cal lesions: {len(best)}; with a best match s >= {settings['candidate_min_s']}: {int(matched.sum())} "
        f"(the others count as no match at every tau_L).\n\n"
        f"Chance-match rate by area bin (share of H-cal lesions whose best score reaches tau_L):\n"
        f"{grid.round(3).to_string()}\n\n"
        f"Proposal for decision G (the experiment plan's one-round rule):\n"
        f"  calibration.provisional_tau_L = {tau0}: the {percentile:g}th percentile of the best scores of the "
        f"{int(big.sum())} H-cal lesions >= {PROVISIONAL_MIN_AREA} px (no match counts as 0).\n"
        f"  calibration.informative_bound_px = the smallest bin edge from which every larger bin stays below the cut:\n"
        f"{bound_lines}\n"
        f"  calibration.chance_rate_max: 0.05 or 0.10 (the experiment plan allows 5-10%); it picks the bound above.\n"
        f"  The final tau_L is computed by 'calibrate' from the informative lesions only.\n\n"
        f"What each bound costs (share of slices left without an informative lesion; E0 stats, "
        f"bound_impact_{pool_name}.csv):\n{cost.round(3).to_string()}\n"
        f"If the bound is 20 px or more, about half of the main audit leaves the lesion-level analysis "
        f"(still scored by E1): weigh that cost when choosing, and record the reason with the decision.\n\n"
        f"Look at chance_rates.png and examples.png (what a chance match looks like at each size), then set "
        f"calibration.provisional_tau_L, calibration.chance_rate_max and calibration.informative_bound_px in "
        f"configs/default.yaml (dated comment) and run 'e3_calibrate.py calibrate'.\n")


def left_out_text(s: SliceSet) -> str:
    """The slices left out of a set, by reason, e.g. "0 removed by decision E, 3 no lesion ..."."""
    return ", ".join(f"{n} {reason}" for reason, n in s.left_out.items())


# --- figures ----------------------------------------------------------------------------------

def rates_figure(rates: pd.DataFrame, tau0: float, path, title: str):
    """Chance-match rate by area bin, one line per tau_L (the provisional one bold), with the cuts."""
    fig = Figure(figsize=(9, 4.5))
    ax = fig.subplots()
    bins = rates.drop_duplicates("lower").sort_values("lower")
    x = np.arange(len(bins))
    for tau, d in rates.groupby("tau_L"):
        d = d.sort_values("lower")
        bold = tau == tau0
        ax.plot(x, d["rate"], marker="o", lw=2.5 if bold else 1, color="black" if bold else None,
                label=f"tau_L = {tau:g}" + (" (provisional)" if bold else ""), zorder=3 if bold else 2)
    for cut in CUTS:
        ax.axhline(cut, ls="--", color="grey", lw=0.8)
    ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(bins["bin"], bins["lesions"])], fontsize=7)
    ax.set_xlabel("H-cal lesion area")
    ax.set_ylabel("chance-match rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7)
    return save(fig, path, title)


def lesion_crop(labels: np.ndarray, number: int) -> tuple[np.ndarray, np.ndarray]:
    """A (2 * CROP)^2 window of the cleaned mask around lesion `number`, and that lesion's mask in it."""
    rows, cols = np.nonzero(labels == number)
    r, c = int(np.rint(rows.mean())) + CROP, int(np.rint(cols.mean())) + CROP
    padded = np.pad(labels, CROP)
    window = padded[r - CROP:r + CROP, c - CROP:c + CROP]
    return (window > 0).astype(float), window == number


def example_sheet(best: pd.DataFrame, hcal: SliceSet, t: SliceSet, rng: np.random.Generator, path, title: str):
    """Per area bin, up to 3 random matched H-cal lesions next to their best T lesion."""
    maps = {**dict(zip(t.keys, t.labels)), **dict(zip(hcal.keys, hcal.labels))}
    matched = best[best["best_s"].notna()]
    rows = [pick(rng, matched[matched["lower"] == lower], 3) for lower in EDGES]
    fig, axes = tile_figure(len(EDGES), 6, tile=1.7)
    for row_axes, picks in zip(axes, rows):
        for k, rec in enumerate(picks.itertuples(index=False)):
            for j, (lid, colour, caption) in enumerate((
                    (rec.query_lesion, LESION_COLOR, f"{rec.query_lesion}\n{rec.area} px"),
                    (rec.best_train_lesion, HIGHLIGHT_COLOR,
                     f"{rec.best_train_lesion}\ns {rec.best_s:.2f} = {rec.best_shape:.2f} x "
                     f"{rec.best_position:.2f} x {rec.best_size:.2f}"))):
                key, number = parse_lesion_id(str(lid))
                window, lesion = lesion_crop(maps[key], number)
                show_slice(row_axes[2 * k + j], window * 0.4, [(lesion, colour)], caption)
        hide_unused(row_axes, 2 * len(picks))
    return save(fig, path, title)
