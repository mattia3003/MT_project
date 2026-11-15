"""E7 building blocks: match reports and their evaluation (`docs/e6-e7-plan.md`, decisions X, Y, Z).
No knowledge of stages or file locations.

- **Report data** (`ReportData`, `report_data`): for one flagged query, everything a report shows,
  from the stored detector tables: both detectors' nearest neighbours, the source slice of E5's
  decision J (E2 full -> nearest neighbour, partial -> strongest-pair slice), the Hungarian lesion
  pairs with the source slice and their shape, position and size factors, each pair's contribution
  to C, and which pairs pass tau_L.
- **Verdict line** (`verdict`, decision X): names the decision-J source; adds the strongest-pair
  patient when it differs; says "configuration match", not "copy", when no lesion pair reaches
  tau_L; states the chance rate of the class and stratum.
- **Report figure** (`draw_report`): Panel A (query, source, overlay coloured by overlap, query-only
  and source-only, with lines between paired lesions; the E1 nearest neighbour and the strongest-pair
  slice when they differ), Panels B and C as text (pairs, factors, contributions; context), verdict.
- **Correctness** on planted copies (`correctness_row`, decision Z): source patient, source slice
  within a window, and for P6 the lesions.
- **Deletion test** (`contributors`, `deletion_effects`, `deletion_summary`, decision Y): remove the
  top or the smallest contributor and measure (i) C against the source slice, (ii) whether a full
  re-match still flags the mask with the same source patient, (iii) the whole-mask score against the
  source slice.
- **Taxonomy** (`taxonomy`, optional): exact, shifted, deformed, partial, from the factors of the pairs.
"""

from __future__ import annotations

import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.figure import Figure
from scipy.stats import binomtest, wilcoxon

from memtk import calibration, lesionmatch, wholemask
from memtk.ids import parse_lesion_id, parse_slice_key
from memtk.plotting import draw_labels, fill, lesion_colors, save, show_slice, zoom_window
from memtk.sets import SliceSet

# Taxonomy (decision Z): factor levels that count as "the same" (fixed on the planted variants)
SAME_SHAPE, SAME_SIZE, SAME_POSITION = 0.95, 0.95, 0.99
FLAGGED = ("full", "partial")


# --- report data ------------------------------------------------------------------------------

@dataclass
class ReportData:
    """Everything one report shows about query `key` (a row of a class table plus its lesion pairs)."""
    key: str
    stratum: str
    e2_class: str
    C: float
    nn: str                   # E2 nearest neighbour (highest C)
    source: str               # decision J's source slice
    strongest_s: float
    strongest_train: str
    e1_score: float
    e1_nn: str
    e1_shift: tuple[int, int]
    tau_L: float
    tau_C: float
    n_query: int              # informative lesions of the query
    n_train: int              # informative lesions of the source slice
    pairs: pd.DataFrame       # Hungarian pairs with the source: lesion IDs, factors, s, contribution, kept

    @property
    def source_patient(self) -> str:
        return parse_slice_key(self.source)[0] if self.source else ""

    @property
    def strongest_patient(self) -> str:
        return parse_slice_key(self.strongest_train)[0] if self.strongest_train else ""

    @property
    def n_kept(self) -> int:
        return int(self.pairs["kept"].sum())

    @property
    def all_copied(self) -> bool:
        return self.n_kept == self.n_query == self.n_train


def report_data(row: pd.Series, e1_row: pd.Series | None, e2: lesionmatch.LesionMatch, frozen: dict) -> ReportData:
    """The report data of one query: `row` from a class table (`key`, `n_informative`, `stratum_e2`,
    `e2_class`, `e2_C`, `e2_nn`, `e2_strongest_s`, `e2_strongest_train`, `e1_score`, `e1_nn`),
    `e1_row` from its E1 table (`shift_row`, `shift_col`), the stored E2 result and the frozen file
    (thresholds of the query's stratum)."""
    key, n_inf = row["key"], int(row["n_informative"])
    tau_L = float(calibration.frozen_threshold(frozen, "lesion", "tau_L", np.array([n_inf]))[0])
    tau_C = float(calibration.frozen_threshold(frozen, "lesion", "tau_C", np.array([n_inf]))[0])
    nn = row["e2_nn"] if isinstance(row["e2_nn"], str) else ""
    strongest = row["e2_strongest_train"] if isinstance(row["e2_strongest_train"], str) else ""
    source = strongest if row["e2_class"] == "partial" else nn
    sp = e2.slice_pairs[(e2.slice_pairs["query"] == key) & (e2.slice_pairs["train"] == source)]
    pairs = e2.lesion_pairs[(e2.lesion_pairs["query"] == key) & (e2.lesion_pairs["train"] == source)].copy()
    n_train = int(sp["n_train"].iloc[0]) if len(sp) else 0
    counted = pairs["s"] >= e2.candidate_min_s
    pairs["contribution"] = np.where(counted, pairs["s"] / max(n_inf, n_train, 1), 0.0)
    pairs["kept"] = pairs["s"] >= tau_L
    pairs = pairs.sort_values("s", ascending=False).reset_index(drop=True)
    shift = (0, 0) if e1_row is None else (int(e1_row["shift_row"]), int(e1_row["shift_col"]))
    return ReportData(key, str(row["stratum_e2"]), str(row["e2_class"]), float(row["e2_C"]), nn, source,
                      float(row["e2_strongest_s"]) if pd.notna(row["e2_strongest_s"]) else 0.0, strongest,
                      float(row["e1_score"]) if pd.notna(row["e1_score"]) else np.nan,
                      row["e1_nn"] if isinstance(row["e1_nn"], str) else "", shift, tau_L, tau_C, n_inf, n_train,
                      pairs)


# --- verdict ----------------------------------------------------------------------------------

def slice_name(key: str, name_of: Callable[[str], str]) -> str:
    """`<pseudonym>, slice <z>` for a real slice key."""
    patient, z = parse_slice_key(key)
    return f"{name_of(patient)}, slice {z}"


def verdict(d: ReportData, name_of: Callable[[str], str], chance_rate: float | None) -> str:
    """Decision X: the rule-based verdict line of a report."""
    if d.e2_class not in FLAGGED or not d.source:
        text = "No lesion-level match closer than chance (E2: novel)"
    elif d.n_kept:
        unpaired = d.n_query - int((d.pairs["contribution"] > 0).sum())
        weak = d.n_query - d.n_kept - unpaired
        text = (f"{d.n_kept} of {d.n_query} informative lesion{'s' if d.n_query > 1 else ''} match "
                f"{slice_name(d.source, name_of)} closer than chance")
        text += f"; {weak} paired but not closer than chance" if weak > 0 else ""
        text += f"; {unpaired} without a partner (novel)" if unpaired > 0 else ""
        if d.all_copied:
            text += " -> full copy"
        elif d.e2_class == "full":
            text += f" -> configuration copy (C {d.C:.2f} >= {d.tau_C:.2f}), not every lesion"
        else:
            text += " -> partial copy"
    else:
        text = (f"Configuration match with {slice_name(d.source, name_of)} (C {d.C:.2f} >= {d.tau_C:.2f}); "
                f"no single lesion closer than chance")
    if d.strongest_train and d.strongest_patient != d.source_patient:
        text += f"; the strongest single lesion points to {slice_name(d.strongest_train, name_of)} (s {d.strongest_s:.2f})"
    if chance_rate is not None and d.e2_class in FLAGGED:
        text += f". H-eval chance rate for this class and stratum: {chance_rate:.1%}"
    return text


# --- report figure ----------------------------------------------------------------------------

def draw_report(d: ReportData, labels_of: Callable[[list[str]], dict], images_of: Callable[[list[str]], dict],
                name_of: Callable[[str], str], context: Sequence[str], verdict_line: str, path: Path,
                title: str) -> Path:
    """One match report as a figure: the query, its source slice, their overlay (green both, red
    query only, blue source only; lines join paired lesions, labelled with s), the E1 nearest
    neighbour and the strongest-pair slice when they differ from the source; below, the lesion
    pairs (Panel B), the context lines (Panel C) and the verdict."""
    extra = [k for k in dict.fromkeys([d.e1_nn, d.strongest_train]) if k and k != d.source]
    keys = [d.key, *([d.source] if d.source else []), *extra]
    labels, images = labels_of(keys), images_of(keys)
    n_img = 1 + (2 if d.source else 0) + len(extra)
    width = max(2.8 * n_img, 12.0)
    fig = Figure(figsize=(width, 6.0))
    grid = fig.add_gridspec(2, n_img, height_ratios=[1, 0.75])
    window = zoom_window([labels[k] > 0 for k in keys])

    def tile(i: int, key: str, caption: str) -> None:
        ax = fig.add_subplot(grid[0, i])
        draw_labels(ax, images[key][window], labels[key][window], lesion_colors(labels[key]), caption)

    tile(0, d.key, f"A  query\n{d.key}")
    i = 1
    if d.source:
        tile(i, d.source, f"source (named)\n{slice_name(d.source, name_of)}")
        i += 1
        ax = fig.add_subplot(grid[0, i])
        q, s = labels[d.key][window] > 0, labels[d.source][window] > 0
        show_slice(ax, images[d.key][window], [], "overlay: green both,\nred query only, blue source only")
        fill(ax, [(q & s, "lime"), (q & ~s, "red"), (s & ~q, "royalblue")])
        draw_pair_lines(ax, d.pairs, labels[d.key], labels[d.source], window)
        i += 1
    for key in extra:
        role = "E1 nearest neighbour" if key == d.e1_nn else "strongest single pair"
        tile(i, key, f"{role}\n{slice_name(key, name_of)}")
        i += 1
    ax = fig.add_subplot(grid[1, :])
    ax.axis("off")
    chars = int(width * 72 / (0.6 * 7)) - 4   # monospace 7 pt: about 0.6 x 7 pt per character
    ax.text(0, 1, panel_text(d, name_of, context, verdict_line, chars), va="top", ha="left", family="monospace",
            fontsize=7, transform=ax.transAxes)
    return save(fig, path, title)


def draw_pair_lines(ax, pairs: pd.DataFrame, q_labels: np.ndarray, t_labels: np.ndarray,
                    window: tuple[slice, slice]) -> None:
    """Lines between the centroids of paired lesions (solid if the pair passes tau_L), labelled s."""
    for p in pairs.itertuples():
        qa = np.argwhere(q_labels == parse_lesion_id(str(p.query_lesion))[1]).mean(axis=0)
        ta = np.argwhere(t_labels == parse_lesion_id(str(p.train_lesion))[1]).mean(axis=0)
        (r0, c0), (r1, c1) = qa - [window[0].start, window[1].start], ta - [window[0].start, window[1].start]
        ax.plot([c0, c1], [r0, r1], color="yellow", lw=0.8, ls="-" if p.kept else ":")
        ax.text((c0 + c1) / 2, (r0 + r1) / 2, f"{p.s:.2f}", color="yellow", fontsize=6,
                path_effects=[patheffects.withStroke(linewidth=1.5, foreground="black")])


def panel_text(d: ReportData, name_of: Callable[[str], str], context: Sequence[str], verdict_line: str,
               width: int = 200) -> str:
    """Panels B (lesion pairs) and C (context) and the verdict as text; the verdict wrapped to `width`
    characters."""
    lines = [f"B  E2 class {d.e2_class} (stratum {d.stratum}): C {d.C:.2f} (tau_C {d.tau_C:.2f}); "
             f"strongest pair s {d.strongest_s:.2f} (tau_L {d.tau_L:.2f}); E1 score {d.e1_score:.2f} "
             f"(shift {d.e1_shift}) against {slice_name(d.e1_nn, name_of) if d.e1_nn else '-'}",
             ]
    wq = max([len("query lesion"), *(len(str(v)) for v in d.pairs["query_lesion"])]) + 2
    wt = max([len("source lesion"), *(len(str(v)) for v in d.pairs["train_lesion"])]) + 2
    lines.append(f"   {'query lesion':<{wq}}{'source lesion':<{wt}}{'shape':>7}{'position':>10}{'size':>7}{'s':>7}"
                 f"{'to C':>7}  closer than chance")
    for p in d.pairs.itertuples():
        lines.append(f"   {str(p.query_lesion):<{wq}}{str(p.train_lesion):<{wt}}{p.shape:>7.2f}{p.position:>10.2f}"
                     f"{p.size:>7.2f}{p.s:>7.2f}{p.contribution:>7.2f}  {'yes' if p.kept else 'no'}")
    unpaired = d.n_query - len(d.pairs)
    if unpaired > 0:
        lines.append(f"   {unpaired} informative query lesion(s) without a partner on the source slice")
    lines += ["C  " + context[0] if context else "C", *[f"   {c}" for c in context[1:]], "",
              textwrap.fill(f"Verdict: {verdict_line}", width, subsequent_indent="   ")]
    return "\n".join(lines)


# --- correctness on planted copies (decision Z) -----------------------------------------------

def correctness_row(d: ReportData, base_key: str, base_patient: str, sources: dict[str, str], window: int) -> dict:
    """Whether a planted mask's report names the truth: the decision-J source patient is the base
    patient; the source slice is within `window` slices of the base slice (same patient); either
    the source or the strongest-pair slice names the base patient; and the lesions: every informative
    planted lesion whose source is a base lesion is paired with that very lesion at s >= tau_L on the
    reported slice, and every other informative planted lesion (P6-add's added one) is not.
    `sources`: source lesion ID by planted lesion ID (`planted_lesions.csv`)."""
    patient_ok = d.source_patient == base_patient
    slice_ok = patient_ok and abs(parse_slice_key(d.source)[1] - parse_slice_key(base_key)[1]) <= window
    kept = d.pairs[d.pairs["kept"]].set_index("query_lesion")["train_lesion"]
    informative = [lid for lid in sources if parse_lesion_id(lid)[0] == d.key]
    lesions_ok = True
    for lid in informative:
        src = sources[lid]
        from_base = src.startswith(f"{base_key}/")
        if from_base and kept.get(lid) != src:
            lesions_ok = False
        if not from_base and lid in kept.index:
            lesions_ok = False
    return {"key": d.key, "source": d.source, "patient_ok": patient_ok, "slice_ok": slice_ok,
            "either_ok": patient_ok or d.strongest_patient == base_patient, "lesions_ok": lesions_ok}


# --- deletion test (decision Y) ---------------------------------------------------------------

def contributors(pairs: pd.DataFrame, min_paired: int) -> tuple[int, int] | None:
    """The lesion numbers of the top and the smallest contributor (largest and smallest s among the
    pairs that count in C), or None with fewer than `min_paired` such pairs."""
    counted = pairs[pairs["contribution"] > 0].sort_values("s", ascending=False)
    if len(counted) < min_paired:
        return None
    return (parse_lesion_id(str(counted["query_lesion"].iloc[0]))[1],
            parse_lesion_id(str(counted["query_lesion"].iloc[-1]))[1])


def without(labels: np.ndarray, number: int) -> np.ndarray:
    """A label map with lesion `number` removed."""
    out = labels.copy()
    out[out == number] = 0
    return out


def deletion_effects(keys: Sequence[str], labels: np.ndarray, top: Sequence[int], smallest: Sequence[int],
                     sources: Sequence[str], source_patients: Sequence[str], t: SliceSet, frozen: dict) -> pd.DataFrame:
    """Decision Y for each query (label map, top and smallest contributor, decision-J source slice
    and patient): C against the source slice, the full E2 re-match (still flagged with the same
    source patient?) and the E1 score against the source slice, for the mask as it is and with either
    lesion removed. One row per query."""
    wm, les = calibration.frozen_settings(frozen)
    t_index = {k: i for i, k in enumerate(t.keys)}
    versions = {"before": labels,
                "top": np.stack([without(m, n) for m, n in zip(labels, top)]),
                "small": np.stack([without(m, n) for m, n in zip(labels, smallest)])}
    out = pd.DataFrame({"key": list(keys), "source": list(sources), "top_lesion": list(top),
                        "small_lesion": list(smallest)})
    for name, stack in versions.items():
        c, e1 = [], []
        for k, m, src in zip(keys, stack, sources):
            ref = t.labels[t_index[src]]
            one = lesionmatch.match(m[None], [k], ref[None], [src], **les)
            sc = one.slice_scores(one.candidate_min_s)
            c.append(float(sc["C"].iloc[0]) if len(sc) else 0.0)
            e1.append(float(wholemask.shifted_dice(m > 0, (ref > 0)[None], wm["max_shift"]).max()) if (m > 0).any()
                      else 0.0)
        out[f"C_{name}"], out[f"e1_{name}"] = c, e1
        e2 = lesionmatch.match(stack, list(keys), t.labels, t.keys, **les)
        cls = calibration.e2_classes(e2, frozen).set_index("query")
        cls = cls.reindex(list(keys))
        patient = np.where(cls["class"] == "partial", cls["strongest_patient"], cls["nn_patient"])
        out[f"flag_{name}"] = cls["class"].isin(FLAGGED).to_numpy() & (patient == np.asarray(source_patients))
    return out


def deletion_summary(d: pd.DataFrame) -> dict:
    """Decision Y's statistics: mean drops in C and in the E1 score after removing the top vs the
    smallest contributor with one-sided paired Wilcoxon p (top drops more), and the flips (no longer
    flagged with the same source patient) with an exact McNemar p on the discordant pairs."""
    out: dict = {"masks": len(d), "still_flagged_before": int(d["flag_before"].sum())}
    if not len(d):
        return out
    for score in ("C", "e1"):
        drop_top, drop_small = d[f"{score}_before"] - d[f"{score}_top"], d[f"{score}_before"] - d[f"{score}_small"]
        out[f"{score}_drop_top"], out[f"{score}_drop_small"] = float(drop_top.mean()), float(drop_small.mean())
        diff = (drop_top - drop_small).to_numpy()
        test = wilcoxon(diff, alternative="greater") if np.any(diff != 0) else None
        out[f"{score}_wilcoxon_p"] = float(test.pvalue) if test is not None else np.nan  # pyright: ignore[reportAttributeAccessIssue]  # scipy stub lacks the result's pvalue
    flip_top, flip_small = ~d["flag_top"], ~d["flag_small"]
    b, c = int((flip_top & ~flip_small).sum()), int((flip_small & ~flip_top).sum())
    out.update({"flips_top": int(flip_top.sum()), "flips_small": int(flip_small.sum()),
                "discordant_top_only": b, "discordant_small_only": c,
                "mcnemar_p": float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c else np.nan})
    return out


# --- taxonomy (optional) ----------------------------------------------------------------------

def taxonomy(d: ReportData) -> str:
    """Rule-based category of a flagged report: partial (an informative lesion of either slice has no
    partner among the pairs that count in C), otherwise from the weakest counted pair: exact (shape,
    size and position all the same), shifted (shape and size the same, position not), deformed (shape
    or size changed). First version used "not every pair passes tau_L" for partial, which called 5 px
    shifts (P3) partial; changed on the planted variants (2026-11-15)."""
    if d.e2_class not in FLAGGED:
        return "not flagged"
    counted = d.pairs[d.pairs["contribution"] > 0]
    if len(counted) < max(d.n_query, d.n_train):
        return "partial"
    if counted["shape"].min() >= SAME_SHAPE and counted["size"].min() >= SAME_SIZE:
        return "exact" if counted["position"].min() >= SAME_POSITION else "shifted"
    return "deformed"
