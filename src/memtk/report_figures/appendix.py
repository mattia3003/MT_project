"""Report-figures stage `appendix`: the figures the report shows only in its appendix
(`docs/report-synthesis.md`, section 3 and the appendix plan), at print resolution.

Most are the experiments' own figures, drawn by their own functions from the tables they stored and
saved at 300 dpi (`plotting.resolution`), without a suptitle (the report's caption replaces it). Their
example tiles are picked with this stage's seeded generator, so they can differ from the original
sheets while showing the same kind of case. A few are redrawn compactly from stored counts. Nothing is
re-scored, and no experiment stage is re-run (that would rewrite run logs that E5's lock checks).

Appendix B (E0, E3: data, preprocessing, calibration), C (E4: validation), D (E5, E6: audit and
exposure), E (E7, E8: match reports, paired MRI). S is read only through E5's stored tables (E5's lock
is checked) and, for the E8 examples, its MRIs. Writes `results/<pool>/report/appendix/` (PNG per
figure, `figures.csv`, run log).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from memtk import calibration, config, lesionmatch
from memtk.calibration import load_frozen
from memtk.e0b.common import Slices
from memtk.e0b.brain import method_grid
from memtk.e0b.stats import adjacency_figure, adjacency_pairs, area_ratio_figure, brain_height_figure
from memtk.e3.bound import EDGES, example_sheet
from memtk.e3.calibrate import score_figure
from memtk.e3.common import load_hcal_tables
from memtk.e4.plant import area_figure
from memtk.e4.validate import criterion_caught, criterion_variants, factors_figure, flag_examples, flags_figure, misses_figure
from memtk.e5.analyze import MAX_EXAMPLES, examples_figure, forest_figure
from memtk.e5.common import audit_config, lock_unchanged
from memtk.e6.common import exposure_config, load_heval, load_query, load_reference
from memtk.e6.exposure import exposure_figure, h1_figure, margins_figure, rarity_figure
from memtk.e7.common import chance_rates, class_rate, patient_name
from memtk.e7.reports import PROFILE_EXAMPLES, Context, group_of, profile_data
from memtk.e8.common import paired_config, real_slices, synthetic_samples
from memtk.e8.paired import draw_examples, draw_within, example_keys, report_numbers, settings
from memtk.io import LabelStack, load_real, read_table, slice_images
from memtk.lesions import attach_or_drop, brain_mask
from memtk.plotting import lesion_sheet, resolution
from memtk.qc import QC_CLASSES
from memtk.report_figures.common import COLORS, DPI, figure, index_row, save_both, style, write_index
from memtk.reports import draw_patient_profile, draw_report, report_data, verdict
from memtk.sets import load_planted, load_set
from memtk.stages import decided, pick, refuse_overwrite
from memtk.synthetic import load_sample

BRAIN_EXAMPLES = (4, 2)            # T slices where plain Otsu leaves a lesion mostly outside, and random ones
BRAIN_METHODS_SHOWN = ("otsu", "otsu_all", "low_threshold_all")
QC_PER_CLASS = 4
SPECK_GAPS = (("gap 1", 1, 1), ("gap 2", 2, 1), ("gap 3", 3, 1), ("gap 4-5", 4.5, 2), ("gap 6-10", 8, 5))
EXTRA_REPORTS = ("02_near-exact.png", "57_random_S_flag.png", "67_H-eval_chance_flag.png")   # E7 reports redrawn
PROFILE_PATIENT = "Mets_058"       # the full patient profile in appendix E (17 metastases)
REHEARSAL_MARGINS = ("null", "positive", "placement_compact")


# --- stage ------------------------------------------------------------------------------------

def run_appendix(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The appendix figures at 300 dpi. Needs E0b (brain, lesions, qc, stats, lesions-review), E3
    bound-review and calibrate, E4 plant and validate, E5 analyze (lock unchanged), E6 rehearse and
    exposure, E7 reports, E8 paired. Writes `results/<pool>/report/appendix/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "report") / "appendix"
    refuse_overwrite([out], overwrite)
    hashes = lock_unchanged(cfg, name, pool_name)
    rows = []
    with style(), resolution(DPI, for_print=True):
        rows += e0_figures(cfg, name, pool_name, pool_cfg, out)
        rows += e3_figures(cfg, name, out)
        rows += e4_figures(cfg, name, out)
        s = SLoader(cfg, name, pool_name, pool_cfg)
        rows += e5_figures(cfg, pool_name, s, out)
        rows += e6_figures(cfg, name, pool_name, out)
        rows += e7_figures(cfg, name, pool_name, s, out)
        rows += e8_figures(cfg, name, pool_name, s, out)
    write_index(rows, out)
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "appendix", "pool": pool_name, **hashes})
    print(pd.DataFrame(rows)[["figure", "section"]].to_string(index=False))
    print(f"Wrote {out}")


def rng_for(cfg: dict, figure_name: str) -> np.random.Generator:
    """A generator seeded by the project seed and the figure's name, so each figure's picks stay the
    same when other figures are added or removed."""
    return np.random.default_rng([int(cfg["seed"]), *figure_name.encode()])


# --- appendix B: data and preprocessing (E0) --------------------------------------------------------

def e0_figures(cfg: dict, name: str, pool_name: str, pool_cfg: dict, out: Path) -> list[dict]:
    """B1 native vs model-space lesion area, B2 brain area by slice height, B3 component sizes, B4 specks
    by gap, B5 brain-mask methods, B6 QC classes, B7 Dice between neighbouring slices."""
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    stats = config.results_dir(cfg, name, "e0b") / "stats"
    review = config.results_dir(cfg, name, "e0b") / "lesions_review"
    min_area = int(decided(cfg, "lesions.min_lesion_area", "e0b_audit.py lesions-review"))
    gap = int(decided(cfg, "lesions.attach_gap", "e0b_audit.py lesions-review"))
    rows = []

    path = stats / f"native_vs_model_{name}.csv"
    area_ratio_figure(read_table(path), out / "B1_native_vs_model.png", "")
    rows.append(index_row("B1_native_vs_model", "App. B (2.3)", "lesion area in model space against plain "
                          "downscaling; enlargement factor by native lesion size", [path]))

    path = stats / f"brain_by_height_{name}.csv"
    table = pd.read_csv(path, index_col="position")
    real_px = read_table(real_dir / "brain.csv")["brain_px"].astype(float)
    syn = read_table(syn_dir / "brain.csv").merge(read_table(syn_dir / "qc.csv")[["sample_id", "mask_class"]], on="sample_id")
    syn = syn[syn["mask_class"].isin(["clean", "minor"]) & ~syn["mri_failed"].astype(str).str.lower().eq("true")]
    brain_height_figure(table, real_px, syn["brain_px"].astype(float), out / "B2_brain_by_height.png", "")
    rows.append(index_row("B2_brain_by_height", "App. B (3.1)", "real brain area by position in the head against "
                          "synthetic brains; brain-area distributions", [path, real_dir / "brain.csv", syn_dir / "brain.csv"]))

    path = review / f"component_sizes_{pool_name}.csv"
    rows.append(component_figure(read_table(path), min_area, path, out))
    path = review / f"speck_gaps_{pool_name}.csv"
    rows.append(speck_gap_figure(read_table(path), gap, path, out))

    rows.append(brain_methods_figure(cfg, real_dir, out))

    qc = read_table(syn_dir / "qc.csv")
    rng = rng_for(cfg, "B6")
    src = Slices(syn_dir, pool_cfg)
    items = []
    for cls in QC_CLASSES:
        for key in pick(rng, qc[qc["qc_class"] == cls], QC_PER_CLASS)["key"]:
            image, mask, brain, box = src[str(key)]
            res = attach_or_drop(mask, min_area, gap)
            items.append((f"{cls.replace('_', ' ')}: {key}\n{res.n_lesions} lesions, {res.n_specks} specks", image, res,
                          brain, box))
    lesion_sheet(items, out / "B6_qc_classes.png", "", n_cols=QC_PER_CLASS)
    rows.append(index_row("B6_qc_classes", "App. B (2.3)", f"{QC_PER_CLASS} random samples per QC class (rows: "
                          f"{', '.join(QC_CLASSES)})", [syn_dir / "qc.csv", syn_dir / "masks_raw.npz"]))

    pairs = adjacency_pairs(read_table(real_dir / "slices.csv"), LabelStack(real_dir / "labels.npz"))
    adjacency_figure(pairs, out / "B7_adjacency_dice.png", "")
    rows.append(index_row("B7_adjacency_dice", "App. B (2.8)", "Dice between consecutive lesion slices of a patient",
                          [real_dir / "slices.csv", real_dir / "labels.npz"]))
    return rows


def component_figure(table: pd.DataFrame, min_area: int, path: Path, out: Path) -> dict:
    """Share of 8-connected mask components by size (1-10 px, larger) per set, log scale, with the
    minimum lesion area (decision B) marked."""
    sizes = [c for c in table.columns if c not in ("group", "components")]
    fig = figure(6.0)
    ax = fig.subplots()
    groups = list(table["group"])
    width = 0.8 / len(groups)
    colors = {"T": COLORS["T"], "H-cal": "0.65", "H-eval": "0.35", "S": COLORS["S"]}
    for k, g in enumerate(groups):
        r = table[table["group"] == g].iloc[0]
        share = r[sizes].astype(float).to_numpy() / float(r["components"])
        ax.bar(np.arange(len(sizes)) + (k - (len(groups) - 1) / 2) * width, share, width=width,
               color=colors.get(g, "0.5"), label=f"{g} ({int(r['components']):,} components)")
    ax.axvline(sizes.index(str(min_area)) - 0.5, color="k", ls="--", lw=0.8)
    ax.text(sizes.index(str(min_area)) - 0.4, 0.5, f"lesion ≥ {min_area} px", fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.set_yscale("log")
    ax.set_xticks(range(len(sizes)), sizes)
    ax.set_xlabel("component size (px)")
    ax.set_ylabel("share of components (log)")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", ncols=2)
    ax.set_ylim(top=3)
    save_both(fig, out, "B3_component_sizes")
    return index_row("B3_component_sizes", "App. B (2.3)", "component sizes per set (decision B)", [path])


def speck_gap_figure(table: pd.DataFrame, gap: int, path: Path, out: Path) -> dict:
    """Synthetic specks (components below the minimum lesion area) by their gap to the nearest lesion,
    per 1 px of gap (wider bins divided by their width): the halo at 1-2 px against the flat noise level
    (decision C)."""
    r = table[table["group"] == "S"].iloc[0]
    x = np.array([g[1] for g in SPECK_GAPS])
    y = np.array([float(r[col]) / width for col, _, width in SPECK_GAPS])
    fig = figure(5.5, width_in=10 / 2.54)
    ax = fig.subplots()
    ax.bar(x, y, width=[w * 0.85 for _, _, w in SPECK_GAPS], color=COLORS["S"])
    ax.axvline(gap + 0.5, color="k", ls="--", lw=0.8)
    ax.text(gap + 0.6, 0.92, f"attached ≤ {gap} px", fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.set_xlabel("gap to the nearest lesion (px)")
    ax.set_ylabel("synthetic specks per 1 px of gap")
    ax.set_title(f"S: {int(r['gap >=11']):,} more specks at ≥ 11 px", fontsize=7)
    save_both(fig, out, "B4_specks_by_gap")
    return index_row("B4_specks_by_gap", "App. B (2.3)", "synthetic specks by gap to the nearest lesion (decision C)",
                     [path])


def brain_methods_figure(cfg: dict, real_dir: Path, out: Path) -> dict:
    """T slices where plain Otsu leaves a lesion more than half outside the brain, and random ones,
    under plain Otsu, Otsu keeping every region and the chosen fixed threshold (decision A)."""
    low = float(decided(cfg, "brain.low_threshold", "e0b_audit.py brain-review"))
    slices = read_table(real_dir / "slices.csv")
    labels = LabelStack(real_dir / "labels.npz")
    t = slices[slices["set"] == "T"]
    rng = rng_for(cfg, "B5")
    failing, examples = [], []
    for pid in rng.permutation(t["patient_id"].unique()):
        bravo, _ = load_real(real_dir, str(pid))
        for key, z in zip(t.loc[t["patient_id"] == pid, "key"], t.loc[t["patient_id"] == pid, "z"]):
            image, lab = bravo[:, :, int(z)], labels[str(key)]
            brain = brain_mask(image, "otsu", low)
            outside = [((lab == n) & ~brain).sum() > 0.5 * (lab == n).sum() for n in range(1, int(lab.max()) + 1)]
            if any(outside) and len(failing) < BRAIN_EXAMPLES[0]:
                failing.append((str(key), image, lab))
        if len(failing) >= BRAIN_EXAMPLES[0]:
            break
    for key in pick(rng, t[~t["key"].isin([f[0] for f in failing])], BRAIN_EXAMPLES[1])["key"]:
        pid, z = str(key).split("/z")
        examples.append((str(key), load_real(real_dir, pid)[0][:, :, int(z)], labels[str(key)]))
    method_grid(failing + examples, out / "B5_brain_methods.png", "", low, methods=BRAIN_METHODS_SHOWN)
    return index_row("B5_brain_methods", "App. B (2.3)", f"brain masks by {', '.join(BRAIN_METHODS_SHOWN)} on "
                     f"{BRAIN_EXAMPLES[0]} slices where plain Otsu fails and {BRAIN_EXAMPLES[1]} random T slices "
                     "(decision A)", [real_dir / "volumes", real_dir / "labels.npz"])


# --- appendix B: calibration (E3) -----------------------------------------------------------------

def e3_figures(cfg: dict, name: str, out: Path) -> list[dict]:
    """B8 H-cal score histograms with the thresholds, B9 example chance lesion matches by area bin."""
    e3_res = config.results_dir(cfg, name, "e3")
    e1, e2, patients = load_hcal_tables(cfg, name)
    table = read_table(e3_res / "calibrate" / "thresholds.csv")
    les = calibration.lesion_slices(e2, patients)
    score_figure({"tau_D": (e1, "score", "E1 score (best Dice over shifts)"),
                  "tau_L": (les, "strongest_s", "E2 strongest single lesion pair s"),
                  "tau_C": (les, "C", "E2 configuration score C of the nearest neighbour")},
                 table, out / "B8_score_histograms.png", "")
    best = read_table(e3_res / "bound_review" / "best_matches_hcal.csv")
    best["lower"] = calibration.area_bins(best["area"].to_numpy(), EDGES)
    t, hcal = load_set(cfg, name, "T"), load_set(cfg, name, "H-cal")
    example_sheet(best, hcal, t, rng_for(cfg, "B9"), out / "B9_chance_lesion_matches.png", "")
    return [index_row("B8_score_histograms", "App. B (2.6, 3.2)", "H-cal scores against T by stratum with the "
                      "thresholds (pooled and per stratum)", [e3_res / "calibrate" / "thresholds.csv", "e3 H-cal tables"]),
            index_row("B9_chance_lesion_matches", "App. B (3.2)", "H-cal lesions and their best T match, three per area "
                      "bin", [e3_res / "bound_review" / "best_matches_hcal.csv"])]


# --- appendix C: validation (E4) --------------------------------------------------------------------

def e4_figures(cfg: dict, name: str, out: Path) -> list[dict]:
    """C1 lesion area before and after P4 and P5, C2 the three factors per variant, C3 the gated copies
    missed by their criterion's detector, C4 H-eval chance flags next to their training slice."""
    e4_res, e4_dir = config.results_dir(cfg, name, "e4"), config.output_dir(cfg, name) / "e4"
    e3, _ = load_frozen(cfg, name)
    criteria = config.value(cfg, "validation.criteria")
    fate_path = e4_res / "plant" / "lesion_fate.csv"
    area_figure(read_table(fate_path), int(e3["informative_bound_px"]), out / "C1_area_change.png", "")
    outcomes = read_table(e4_res / "validate" / "planted_outcomes.csv")
    for col in ("gated",):
        outcomes[col] = outcomes[col].astype(str).str.lower().eq("true")
    e2_p = lesionmatch.load_tables(e4_dir, "planted", float(e3["detectors"]["lesion"]["candidate_min_s"]))
    factors_figure(e2_p, outcomes, out / "C2_factors.png", "")

    t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
    planted, planted_labels = load_planted(cfg, name)
    labels = {**dict(zip(t.keys, t.labels)), **dict(zip(heval.keys, heval.labels)), **dict(zip(planted["key"], planted_labels))}
    hc = read_table(e4_dir / "heval_classes.csv")
    misses = outcomes[outcomes["gated"] & outcomes["variant"].isin(criterion_variants(criteria))
                      & ~criterion_caught(outcomes, criteria)]
    flags = flag_examples(hc, rng_for(cfg, "C4"))
    images = slice_images(cfg, name, [*outcomes["base_key"], *misses["e2_nn"].replace("", np.nan).dropna(),
                                      *misses["e1_nn"].dropna(), *flags["key"], *flags["nn"]])
    misses_figure(misses, labels, images, rng_for(cfg, "C3"), out / "C3_misses.png", "")
    flags_figure(flags, labels, images, out / "C4_heval_chance_flags.png", "")
    return [index_row("C1_area_change", "App. C (2.7)", "area of every informative base lesion before and after P4 and "
                      "P5 (decision K)", [fate_path]),
            index_row("C2_factors", "App. C (3.3)", "shape, position and size factors and s of each planted lesion "
                      "against its own base lesion, per variant", [e4_dir / "e2_planted_lesion_pairs.parquet"]),
            index_row("C3_misses", "App. C (3.3)", f"the {len(misses)} gated planted masks missed by their criterion's "
                      "detector, with the nearest neighbour found", [e4_res / "validate" / "planted_outcomes.csv"]),
            index_row("C4_heval_chance_flags", "App. C (3.2, 3.3)", "H-eval slices flagged at the frozen thresholds, "
                      "random per class, next to the training slice that flagged them", [e4_dir / "heval_classes.csv"])]


# --- S and real slices for the sheets of appendices D and E -----------------------------------------

class SLoader:
    """Label maps and images of T, H-eval and S keys, as E5 and E7 load them for their sheets (S
    samples read from the pool files, real slices from the model-space volumes)."""

    def __init__(self, cfg: dict, name: str, pool_name: str, pool_cfg: dict):
        self.cfg, self.name, self.pool_cfg = cfg, name, pool_cfg
        t, heval = load_set(cfg, name, "T"), load_set(cfg, name, "H-eval")
        self.real = {**dict(zip(t.keys, t.labels)), **dict(zip(heval.keys, heval.labels))}
        self.pool = LabelStack(config.output_dir(cfg, pool_name) / "labels.npz")
        index = read_table(config.output_dir(cfg, pool_name) / "synthetic_index.csv")
        self.files = dict(zip(index["sample_id"], index["filename"]))
        self.pool_dir = config.resolve_path(pool_cfg["dir"])

    def labels_of(self, keys) -> dict:
        """Label map per key (real slice or S sample)."""
        return {k: self.real[k] if k in self.real else self.pool[k] for k in keys}

    def images_of(self, keys) -> dict:
        """Image per key: model-space BRAVO of a real slice, the synthetic MRI of an S sample."""
        real = slice_images(self.cfg, self.name, [k for k in keys if k in self.real])
        return {k: real[k] if k in real else load_sample(self.pool_dir / self.files[k], self.pool_cfg)[0] for k in keys}


# --- appendix D: audit and exposure (E5, E6) -------------------------------------------------------

def e5_figures(cfg: dict, pool_name: str, s: SLoader, out: Path) -> list[dict]:
    """D1 the excess with intervals for every set, variant and stratum (sensitivity), D2 near-exact copies
    next to their source slice, D3 one mask per largest E2 group next to its source slice."""
    a = config.results_dir(cfg, pool_name, "e5") / "analyze"
    forest_figure(read_table(a / "rates.csv"), out / "D1_sensitivity_forest.png", "")
    flagged = read_table(a / "flagged.csv")
    main = flagged[flagged["set"] == "main"]
    near = main[main["near_exact"].astype(str).str.lower().eq("true")].sort_values("e1_score", ascending=False)
    examples_figure(near, "e1_nn", s.labels_of, s.images_of, out / "D2_near_exact.png", "")
    e2 = main[main["e2_group"].fillna("") != ""]
    top = e2["e2_group"].value_counts().head(MAX_EXAMPLES).index
    rows = e2[e2["e2_group"].isin(top)].sort_values("e2_C", ascending=False).drop_duplicates("e2_group")
    examples_figure(rows, "e2_source", s.labels_of, s.images_of, out / "D3_groups_top.png", "")
    return [index_row("D1_sensitivity_forest", "App. D (3.4, 4.6)", "excess over H-eval with 95% intervals per set, "
                      "variant and stratum (red = primary family)", [a / "rates.csv"]),
            index_row("D2_near_exact", "App. D (3.4)", f"near-exact masks (decision Q; top {MAX_EXAMPLES} by E1 score) "
                      "next to their source slice", [a / "flagged.csv"]),
            index_row("D3_groups_top", "App. D (3.4)", "one mask per largest E2 group next to its source slice",
                      [a / "flagged.csv"])]


def e6_figures(cfg: dict, name: str, pool_name: str, out: Path) -> list[dict]:
    """D4 observed vs expected flags per patient in both primary cells, D5 flags by patient-rarity group
    (H3), D6 Spearman of span and lesion credits per size bin (H1), D7-D9 margins and source rarity of the
    rehearsal mocks (null, positive, compact placement) against H-eval chance flags."""
    exp = exposure_config(cfg)
    m = float(exp["expected"]["smoothing_m"])
    d = config.results_dir(cfg, pool_name, "e6") / "exposure"
    margins = read_table(d / "margins.csv")
    cells = margins.loc[margins["primary"].astype(str).str.lower().eq("true"), "cell"].tolist()
    exposure = read_table(d / "exposure.csv")
    exposure_figure(exposure, cells, m, out / "D4_exposure_sorted.png", "", {})
    rarity_figure(exposure, read_table(d / "h3.csv"), load_reference(cfg, name), cells, m,
                  out / "D5_rarity_exposure.png", "")
    h1_figure(read_table(d / "h1.csv"), out / "D6_h1_span.png", "")
    rows = [index_row("D4_exposure_sorted", "App. D (3.5)", "observed vs expected flags per patient with chance "
                      "bands, both primary cells", [d / "exposure.csv"]),
            index_row("D5_rarity_exposure", "App. D (3.5)", "flags by patient-rarity group, observed / expected (H3)",
                      [d / "exposure.csv", d / "h3.csv"]),
            index_row("D6_h1_span", "App. D (3.5)", "Spearman(span, lesion credits) per size bin, S vs H-eval (H1)",
                      [d / "h1.csv"])]
    rehearse = config.results_dir(cfg, name, "e6") / "rehearse"
    for i, mock in enumerate(REHEARSAL_MARGINS):
        r = rehearse / mock
        fig_name = f"D{7 + i}_rehearsal_margins_{mock}"
        mr = read_table(r / "margins.csv")
        mock_cells = mr.loc[mr["primary"].astype(str).str.lower().eq("true"), "cell"].tolist()
        margins_figure(read_table(r / "flag_details.csv"), mr, mock_cells, out / f"{fig_name}.png", "")
        rows.append(index_row(fig_name, "App. D (2.10)", f"rehearsal mock '{mock}': margins and source rarity of its "
                              "flags against H-eval chance flags (decision U)", [r / "margins.csv", r / "flag_details.csv"]))
    return rows


# --- appendix E: match reports and the paired MRI (E7, E8) ----------------------------------------

def e7_figures(cfg: dict, name: str, pool_name: str, s: SLoader, out: Path) -> list[dict]:
    """E1-E3 three more match reports (the 1 px rounding limit of E2 caught by E1, a random S flag, an
    H-eval chance flag), redrawn with E7's own report data and context; E4 the full profile of
    `PROFILE_PATIENT`."""
    audit, exp = audit_config(cfg), exposure_config(cfg)
    e3, _ = load_frozen(cfg, name)
    cut = float(e3["detectors"]["lesion"]["candidate_min_s"])
    q = load_query("S", config.output_dir(cfg, pool_name) / "e5", audit["sets"]["main"], cut)
    h = load_heval(cfg, name, cut)
    flagged = read_table(config.results_dir(cfg, pool_name, "e5") / "analyze" / "flagged.csv")
    flagged = flagged[flagged["set"] == "main"].reset_index(drop=True)
    flagged = flagged.assign(group=flagged.apply(group_of, axis=1))
    e6_dir = config.results_dir(cfg, pool_name, "e6") / "exposure"
    e6 = {k: read_table(e6_dir / f"{k}.csv") for k in ("exposure", "top_patients", "flag_details")}
    ref = load_reference(cfg, name)
    context_of = Context(flagged, q, h, e6, ref, exp, e3, patient_name)
    chance = chance_rates(cfg, name)
    index_path = config.results_dir(cfg, pool_name, "e7") / "reports" / "report_index.csv"
    index = read_table(index_path)
    rows = []
    for i, report in enumerate(EXTRA_REPORTS):
        r = index[index["report"] == report].iloc[0].to_dict()
        from_s = r["set"] == "S"
        table, e1, e2 = (flagged, q.e1, q.e2) if from_s else (h.table, h.e1, h.e2)
        row = table[table["key"] == r["key"]].iloc[0]
        d = report_data(row, e1[e1["query"] == r["key"]].iloc[0], e2, e3)
        line = verdict(d, patient_name, class_rate(chance, d.e2_class, d.stratum))
        fig_name = f"E{i + 1}_report_{report.split('_')[0]}"
        draw_report(d, s.labels_of, s.images_of, patient_name, context_of(row, d, from_s), line,
                    out / f"{fig_name}.png", "")
        rows.append(index_row(fig_name, "App. E (3.6)", f"E7 match report {report} ({r['set']} {r['key']})",
                              [index_path]))
    data = profile_data(flagged, q, e6["top_patients"], ref, exp, e3)
    if PROFILE_PATIENT in data:
        slices, lesions = data[PROFILE_PATIENT]
        method = exp["lesions3d"]["main"]
        l3 = ref.lesions3d[(ref.lesions3d["method"] == method) & (ref.lesions3d["patient_id"] == PROFILE_PATIENT)
                           & (ref.lesions3d["n_2d_lesions"] > 0)]
        totals = (slices["e1_flags"] + slices["e2_flags"]).to_numpy()
        slice_key = f"{PROFILE_PATIENT}/z{int(slices['z'].to_numpy()[totals.argmax()])}"
        named = flagged[(flagged["e1_nn"] == slice_key) | (flagged["e2_source"] == slice_key)]
        examples = [(k, slice_key) for k in named.nlargest(PROFILE_EXAMPLES, "e1_score")["key"]]
        draw_patient_profile(PROFILE_PATIENT, l3, slices, lesions, examples, s.labels_of, s.images_of,
                             out / f"E4_profile_{PROFILE_PATIENT}.png", "")
        rows.append(index_row(f"E4_profile_{PROFILE_PATIENT}", "App. E (3.6)", f"patient profile of {PROFILE_PATIENT}: "
                              "S matches over its 3D lesions, flags per slice, the most-matched slice",
                              [config.results_dir(cfg, pool_name, "e7") / "reports" / "patient_profiles.csv"]))
    return rows


def e8_figures(cfg: dict, name: str, pool_name: str, s: SLoader, out: Path) -> list[dict]:
    """E5 the E8 example pairs (the median case per reason, the strongest, E7 report 50) with their SSIM
    maps, E6 within group A: SSIM by z offset and the runner-up difference."""
    p = paired_config(cfg)
    e3, _ = load_frozen(cfg, name)
    st = settings(cfg, p, e3)
    primary = f"ssim_{p['measure']['primary']}"
    d = config.results_dir(cfg, pool_name, "e8") / "paired"
    m = read_table(d / "measures.csv")
    reports = report_numbers(cfg, pool_name)
    keys = example_keys(m, primary, reports)
    samples = synthetic_samples(cfg, pool_name, keys)
    real = real_slices(cfg, name, m.loc[(m["kind"] == "t") & m["query"].isin(keys), "comparison"])

    def query(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        image, saved = samples[key]
        return image, saved, s.pool[key]

    draw_examples(m, keys, query, real, st, primary, out / "E5_e8_examples.png", "",
                  notes={k: f"E7 report {reports[k]}" for k in keys if k in reports})
    draw_within(read_table(d / "within_a.csv"), p["measure"]["primary"], out / "E6_e8_within_a.png", "")
    return [index_row("E5_e8_examples", "App. E (3.7)", "E8 group-A examples (median per reason, the strongest, E7 "
                      "report 50): synthetic MRI, comparison BRAVO, SSIM map, lesion region", [d / "measures.csv"]),
            index_row("E6_e8_within_a", "App. E (3.7)", "within group A: SSIM by z offset from the source; source "
                      "minus runner-up per region", [d / "within_a.csv"])]
