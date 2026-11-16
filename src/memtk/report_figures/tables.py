"""Report-figures stage `tables`: the report's tables as LaTeX code (booktabs), written straight from
the tables the experiments stored, so no number is copied by hand (`docs/report-synthesis.md`,
section 3, table register; appendix A, D and F).

Each `.tex` file holds one `table` environment with a placeholder caption and a label, ready for
`\\input{...}` in Overleaf (needs `\\usepackage{booktabs}`). Rates are percentages with one decimal,
intervals are 95% bootstrap intervals in brackets. Nothing is recomputed. Writes
`results/<pool>/report/tables/` (`.tex` per table, `tables.csv`, run log).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from memtk import config
from memtk.e5.common import lock_unchanged
from memtk.io import read_table
from memtk.report_figures.common import index_row, write_index
from memtk.stages import refuse_overwrite

STRATA = ("single", "multi")
CLASS_NAMES = {"e1_copy": "E1 copy", "e2_full": "E2 full", "e2_partial": "E2 partial", "e2_flagged": "E2 flagged"}
VARIANTS = ("P1", "P2", "P3", "P4", "P5-erode", "P5-dilate", "P6-remove", "P6-add")
CHECKS = (("E0a", "stanford_brainmets", "e0a/check_report_amibekk_v1.txt"),
          ("E0b", "stanford_brainmets", "e0b/check_report_amibekk_v1.txt"),
          ("E3", "stanford_brainmets", "e3/check_report.txt"), ("E4", "stanford_brainmets", "e4/check_report.txt"),
          ("E5", "amibekk_v1", "e5/check_report.txt"), ("E6", "amibekk_v1", "e6/check_report.txt"),
          ("E7", "amibekk_v1", "e7/check_report.txt"), ("E8", "amibekk_v1", "e8/check_report.txt"),
          ("Report figures", "amibekk_v1", "report/check_report.txt"))


# --- stage ------------------------------------------------------------------------------------

def run_tables(cfg: dict, dataset: str | None = None, pool: str | None = None, overwrite: bool = False) -> None:
    """The report's tables as LaTeX. Needs every experiment's analysis outputs (E5's lock unchanged).
    Writes `results/<pool>/report/tables/`."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "report") / "tables"
    refuse_overwrite([out], overwrite)
    hashes = lock_unchanged(cfg, name, pool_name)
    real, syn = (config.resolve_path(cfg["results_root"]) / n for n in (name, pool_name))
    out.mkdir(parents=True, exist_ok=True)
    rows = [data_sets(real, out), description(real, pool_name, out), thresholds(real, out), chance_rates(real, out),
            validation(real, out), gate(real, out), main_audit(syn, out), exposure(syn, out), h1_h3(syn, out),
            explanations(real, syn, out), paired(syn, out), decisions(out), h2(syn, out), rates_by_size(syn, out),
            rehearsals(real, out), check_results(cfg, out), software(syn, out)]
    write_index(rows, out, "tables.csv")
    appendix_tex(out.parent)
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "tables", "pool": pool_name, **hashes})
    print(pd.DataFrame(rows)[["figure", "section"]].to_string(index=False))
    print(f"Wrote {out}")


# --- LaTeX ------------------------------------------------------------------------------------

def tex(text: object) -> str:
    """Text escaped for LaTeX (underscores, percent, ampersand, hash; >= and <= as symbols)."""
    s = re.sub(r"τ_([A-Za-z])", "\x00\\1", str(text))          # tau with a subscript, restored below
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 (">=", r"$\geq$"), ("<=", r"$\leq$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("×", r"$\times$"),
                 ("τ", r"$\tau$"), ("ρ", r"$\rho$"), ("Δ", r"$\Delta$"), ("–", "--"), ("→", r"$\to$")):
        s = s.replace(a, b)
    s = re.sub(r"(?<!\$)([<>])(?!\$)", r"$\1$", s)
    return re.sub("\x00([A-Za-z])", r"$\\tau_\1$", s)


def pct(x: Any, digits: int = 1) -> str:
    """A share as a percentage (`7.4`), or an en dash for a missing value."""
    return "--" if pd.isna(x) else f"{100 * float(x):.{digits}f}"


def num(x: Any, digits: int = 2, sign: bool = False) -> str:
    """A number with fixed decimals (and a sign), or an en dash."""
    if pd.isna(x):
        return "--"
    return f"{float(x):+.{digits}f}" if sign else f"{float(x):.{digits}f}"


def ci(low: Any, high: Any, f=num, **kw) -> str:
    """An interval `[low, high]` formatted with `f`."""
    return f"[{f(low, **kw)}, {f(high, **kw)}]"


def write_tex(out: Path, name: str, header: list[str], body: list[list[str]], caption: str, label: str,
              align: str | None = None, size: str = r"\small") -> Path:
    """One `table` environment with a booktabs tabular and the caption below it (the author's choice,
    2026-11-16); cells are already LaTeX (header escaped here)."""
    align = align or "l" + "r" * (len(header) - 1)
    lines = [r"\begin{table}[htbp]", r"\centering", size,
             rf"\begin{{tabular}}{{{align}}}", r"\toprule", " & ".join(tex(h) for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{tab:{label}}}", r"\end{table}", ""]
    path = out / f"{name}.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# --- main text --------------------------------------------------------------------------------

def data_sets(real: Path, out: Path) -> dict:
    """Table 1 (2.1): patients and lesion slices (or masks) per set, with each set's role."""
    path = real / "e0b" / "stats" / "description_amibekk_v1.csv"
    d: dict[Any, Any] = pd.read_csv(path, index_col=0).to_dict()
    roles = {"T": "training patients: what could be memorized (reference)", "H-cal": "held out: sets the thresholds",
             "H-eval": "held out: false-positive check and baseline", "S main": "synthetic, QC clean or minor: audited",
             "S all": "every synthetic sample, noisy masks included"}
    body = [[tex(s), "--" if pd.isna(d[s]["patients"]) else f"{int(d[s]['patients'])}",
             f"{int(d[s]['slices']):,}", tex(roles[s])] for s in roles]
    write_tex(out, "T1_data_sets", ["Set", "Patients", "Lesion slices / masks", "Role"], body,
              "TODO: the data sets.", "data_sets", "lrrl")
    return index_row("T1_data_sets", "2.1", "patients and lesion slices per set, with roles", [path])


def description(real: Path, pool_name: str, out: Path) -> dict:
    """Table 2 (3.1): real lesion slices against the synthetic main audit."""
    path = real / "e0b" / "stats" / f"description_{pool_name}.csv"
    d: dict[Any, Any] = pd.read_csv(path, index_col=0).to_dict()
    sets = ["T", "H-cal", "H-eval", "S main"]
    spec = [("Lesions per slice (mean)", lambda s: num(d[s]["lesions_per_slice"])),
            ("Most lesions in one slice", lambda s: f"{int(d[s]['max_lesions_per_slice'])}"),
            ("Single-lesion slices (\\%)", lambda s: pct(d[s]["share_single_lesion"])),
            ("Multi-lesion slices (\\%)", lambda s: pct(d[s]["share_multi_lesion"])),
            ("Lesion area, median (IQR), px", lambda s: f"{d[s]['area_median']:.0f} ({d[s]['area_p25']:.0f}--"
                                                        f"{d[s]['area_p75']:.0f})"),
            ("Slices with a lesion < 40 px (\\%)", lambda s: pct(d[s]["share_with_lesion_lt40"])),
            ("Centroid SD row / col, px", lambda s: f"{d[s]['centroid_row_sd_px']:.1f} / {d[s]['centroid_col_sd_px']:.1f}"),
            ("Centroid SD row / col, brain box", lambda s: f"{d[s]['centroid_row_sd_brain']:.2f} / "
                                                           f"{d[s]['centroid_col_sd_brain']:.2f}")]
    body = [[label, *[f(s) for s in sets]] for label, f in spec]
    write_tex(out, "T2_description", ["", *sets], body, "TODO: real lesion slices against the synthetic main audit.",
              "description")
    return index_row("T2_description", "3.1", "lesion counts, areas and positions, real sets vs S main", [path])


def thresholds(real: Path, out: Path) -> dict:
    """Table 3a (3.2): the frozen thresholds with bootstrap-by-patient intervals, per stratum."""
    path = real / "e3" / "calibrate" / "thresholds.csv"
    t = read_table(path)
    names = {"tau_D": "τ_D (E1 whole-mask score)", "tau_L": "τ_L (E2 strongest lesion pair)",
             "tau_C": "τ_C (E2 configuration score C)"}
    body = []
    for thr, label in names.items():
        cells = []
        for s in STRATA:
            r = t[(t["threshold"] == thr) & (t["stratum"] == s)].iloc[0]
            cells.append(f"{float(r['value']):.3f} {ci(r['ci_low'], r['ci_high'], digits=3)}")
        body.append([tex(label), *cells])
    write_tex(out, "T3a_thresholds", ["Threshold", "Single-lesion", "Multi-lesion"], body,
              "TODO: frozen thresholds (95th percentile of H-cal) with bootstrap-by-patient intervals.", "thresholds")
    return index_row("T3a_thresholds", "3.2", "frozen thresholds with intervals", [path])


def chance_rates(real: Path, out: Path) -> dict:
    """Table 3b (3.2, 3.3): chance rates at the frozen thresholds on H-cal and H-eval, with the E4 limit."""
    path = real / "e4" / "validate" / "fp_heval.csv"
    f = read_table(path)
    body = [[tex(CLASS_NAMES[str(r["class"])]), str(r["stratum"]), pct(r["hcal_rate"]),
             f"{pct(r['rate'])} {ci(r['ci_low'], r['ci_high'], f=pct)}", pct(r["limit"], 0),
             "yes" if str(r["pass"]).lower() == "true" else "no"] for _, r in f.iterrows()]
    write_tex(out, "T3b_chance_rates", ["Class", "Stratum", "H-cal (%)", "H-eval (%) [95% CI]", "Limit (%)", "Pass"],
              body, "TODO: share flagged by chance at the frozen thresholds; limits of the E4 gate.", "chance_rates",
              "llrrrl")
    return index_row("T3b_chance_rates", "3.2, 3.3", "chance rates on H-cal and H-eval with the gate limits", [path])


def validation(real: Path, out: Path) -> dict:
    """Table 4 (3.3): recall of the gated planted copies per variant and detector."""
    path = real / "e4" / "validate" / "validation.csv"
    v = read_table(path)
    v = v[v["subset"] == "gated"]
    body = []
    for var in VARIANTS:
        d = v[v["variant"] == var].set_index("detector")
        cells = [f"{pct(d.at[k, 'recall'], 0)} {ci(d.at[k, 'ci_low'], d.at[k, 'ci_high'], f=pct, digits=0)}"
                 if k in d.index else "--" for k in ("e1", "e2", "any")]
        body.append([tex(var), f"{int(d['masks'].iloc[0])}", *cells])
    write_tex(out, "T4_validation", ["Variant", "Masks", "E1 (%)", "E2 (%)", "Either (%)"], body,
              "TODO: planted copies caught (flagged with the right source patient), gated masks, 95% CI by patient.",
              "validation")
    return index_row("T4_validation", "3.3", "recall per variant and detector", [path])


def gate(real: Path, out: Path) -> dict:
    """Table 4b (3.3): the go/no-go criteria of E4 with their values and the accepted deviation."""
    path = real / "e4" / "validate" / "gate.csv"
    g = read_table(path)
    body = [[tex(r["criterion"]), tex(r["detail"]), num(r["value"], 3), tex(r["limit"]),
             "pass" if str(r["pass"]).lower() == "true" else
             ("accepted" if str(r["accepted"]).lower() == "true" else "fail")] for _, r in g.iterrows()]
    write_tex(out, "T4b_gate", ["Criterion", "Detail", "Value", "Limit", "Result"], body,
              "TODO: go/no-go criteria of the gate (decision K).", "gate", "llrll", r"\footnotesize")
    return index_row("T4b_gate", "3.3", "E4 gate criteria and results", [path])


def main_audit(syn: Path, out: Path) -> dict:
    """Table 5 (3.4): S main against H-eval per class and stratum; primary family with Holm."""
    path = syn / "e5" / "analyze" / "rates.csv"
    r = read_table(path)
    r = r[(r["set"] == "main") & (r["variant"] == "main")]
    body = []
    for cls in CLASS_NAMES:
        for s in STRATA:
            x = r[(r["class"] == cls) & (r["stratum"] == s)].iloc[0]
            primary = str(x["primary"]).lower() == "true"
            body.append([tex(CLASS_NAMES[cls]), s, f"{int(x['n_s']):,}", pct(x["rate_s"]), pct(x["rate_h"]),
                         f"{num(100 * float(x['excess']), 1, sign=True)} "
                         f"{ci(100 * float(x['excess_low']), 100 * float(x['excess_high']), digits=1, sign=True)}",
                         num(x["p_holm"], 3) if primary else "--"])
    write_tex(out, "T5_main_audit", ["Class", "Stratum", "S masks", "S (%)", "H-eval (%)", "Excess (points) [95% CI]",
                                     "Holm p"], body,
              "TODO: share of S main and H-eval flagged per class and stratum; primary family (E1 copy, E2 full) with "
              "Holm-adjusted p.", "main_audit", "llrrrrr")
    return index_row("T5_main_audit", "3.4", "main audit: rates, excess, Holm p", [path])


def exposure(syn: Path, out: Path) -> dict:
    """Table 6 (3.5): concentration, margin and source rarity against chance, and the reading, per primary
    cell (decision U)."""
    d = syn / "e6" / "exposure"
    conc, marg = read_table(d / "concentration.csv"), read_table(d / "margins.csv")
    conc = conc[pd.to_numeric(conc["m"]) == 52.5].set_index("cell")
    marg = marg.set_index("cell")
    readings = dict(re.findall(r"^\s+(\S+ \S+): ([a-z ]+) \(", (d / "review.txt").read_text(encoding="utf-8"), re.M))
    body = []
    for cell in marg.index[marg["primary"].astype(str).str.lower() == "true"]:
        c, m = conc.loc[cell], marg.loc[cell]
        body.append([tex(cell.replace("e1_copy", "E1 copy").replace("e2_full", "E2 full")), f"{int(c['flags']):,}",
                     f"{float(c['dispersion']):,.0f} ({float(c['dispersion_null_p95']):,.0f})",
                     f"{int(c['above_null_band'])} ({float(c['above_null_band_expected']):.1f})",
                     f"{num(m['margin_auc'])} {ci(m['margin_auc_low'], m['margin_auc_high'])}",
                     f"{num(m['rarity_auc'])} {ci(m['rarity_auc_low'], m['rarity_auc_high'])}",
                     tex(readings.get(cell, "--"))])
    write_tex(out, "T6_exposure", ["Cell", "Flags", "Dispersion (null 95th)", "Above band (expected)",
                                   "Margin AUC", "Rarity AUC", "Reading"], body,
              "TODO: concentration of S flags over patients, and margin and source rarity against H-eval chance flags.",
              "exposure", "lrrrrrl", r"\footnotesize")
    return index_row("T6_exposure", "3.5", "E6 readings per primary cell", [d / "concentration.csv", d / "margins.csv"])


def h1_h3(syn: Path, out: Path) -> dict:
    """Table 6b (3.5): H1 per size bin (slice-to-slice linking) with the weighted difference, and H3."""
    d = syn / "e6" / "exposure"
    h1 = read_table(d / "h1.csv")
    h1 = h1[(h1["method"] == "slice_linking") & h1["difference"].notna()]
    body = [[tex(r["bin"]), f"{int(r['lesions3d']):,}", num(r["rho_s"]), num(r["rho_h"]),
             f"{num(r['difference'], 3, sign=True)} {ci(r['difference_low'], r['difference_high'], digits=3, sign=True)}"]
            for _, r in h1.iterrows()]
    h3 = read_table(d / "h3.csv")
    h3 = h3[h3["primary"].astype(str).str.lower() == "true"]
    body += [[tex(f"H3 {c}"), f"{int(n)}", rf"$\rho$ {num(rho)} {ci(lo, hi)}", f"p = {num(p, 2)}", ""]
             for c, n, rho, lo, hi, p in zip(h3["cell"], h3["n"], h3["rho"], h3["rho_low"], h3["rho_high"],
                                             h3["p_permutation"])]
    write_tex(out, "T6b_h1_h3", ["Size bin / test", "3D lesions / patients", "ρ S", "ρ H-eval", "ρ S − ρ H-eval [95% CI]"],
              body, "TODO: H1 per size bin (weighted: the pre-stated test) and H3.", "h1_h3")
    return index_row("T6b_h1_h3", "3.5", "H1 per size bin and weighted; H3 Spearman", [d / "h1.csv", d / "h3.csv"])


def explanations(real: Path, syn: Path, out: Path) -> dict:
    """Table 7 (3.6): correctness criteria on planted copies, and the deletion test on planted copies, S
    and H-eval chance flags."""
    crit = read_table(real / "e7" / "planted" / "criteria.csv")
    body = [[tex(r["criterion"]), pct(r["value"]), pct(r["min"], 0), "yes" if str(r["pass"]).lower() == "true" else "no"]
            for _, r in crit.iterrows()]
    write_tex(out, "T7a_correctness", ["Criterion", "Value (%)", "Minimum (%)", "Pass"], body,
              "TODO: correctness of the match reports on planted copies.", "correctness", "lrrl")
    dels = pd.concat([read_table(real / "e7" / "planted" / "deletion_summary.csv"),
                      read_table(syn / "e7" / "reports" / "deletion_summary.csv")])
    body = [[tex(r["set"]), f"{int(r['masks'])}", f"{num(r['C_drop_top'])} / {num(r['C_drop_small'])}",
             f"{float(r['C_wilcoxon_p']):.2g}", f"{num(r['e1_drop_top'])} / {num(r['e1_drop_small'])}",
             f"{float(r['e1_wilcoxon_p']):.2g}", f"{int(r['flips_top'])} / {int(r['flips_small'])}",
             f"{float(r['mcnemar_p']):.2g}"] for _, r in dels.iterrows()]
    write_tex(out, "T7b_deletion", ["Set", "Masks", "ΔC top / small", "p", "ΔE1 top / small", "p",
                                    "Lost flag top / small", "p"], body,
              "TODO: deletion test, top against smallest contributor (paired Wilcoxon; McNemar for lost flags).",
              "deletion", "lrrrrrrr", r"\footnotesize")
    return index_row("T7a_correctness; T7b_deletion", "3.6", "correctness criteria; deletion test",
                     [real / "e7" / "planted" / "criteria.csv", syn / "e7" / "reports" / "deletion_summary.csv"])


def paired(syn: Path, out: Path) -> dict:
    """Table 8 (3.7): A vs B per region against H-eval, and within A against the runner-up."""
    d = syn / "e8" / "paired"
    t = read_table(d / "tests.csv")
    w = read_table(d / "within_a.csv")
    w = w[w["measure"] == "t minus runner-up"].set_index("region")
    body = [[tex(r["region"]), f"{num(r['auc'])} {ci(r['auc_low'], r['auc_high'])}",
             f"{num(r['auc_ref'])} {ci(r['auc_ref_low'], r['auc_ref_high'])}",
             f"{num(r['delta'], sign=True)} {ci(r['delta_low'], r['delta_high'], sign=True)}",
             f"{num(w.at[r['region'], 'mean'], 3, sign=True)} {ci(w.at[r['region'], 'low'], w.at[r['region'], 'high'], digits=3, sign=True)}"]
            for _, r in t.iterrows()]
    reading = json.loads((d / "run_info.json").read_text(encoding="utf-8")).get("reading", "")
    write_tex(out, "T8_paired", ["Region", "AUC A vs B", "AUC H-eval", "Difference", "Source − runner-up"], body,
              f"TODO: paired-MRI check (reading: {tex(reading)}).", "paired", "lrrrr", r"\footnotesize")
    return index_row("T8_paired", "3.7", "E8 tests per region and within A", [d / "tests.csv", d / "within_a.csv"])


# --- appendix ---------------------------------------------------------------------------------

def decisions(out: Path) -> dict:
    """Appendix A: decisions A-AE from the decision register in `docs/report-synthesis.md` (section 5)."""
    path = config.REPO_ROOT / "docs" / "report-synthesis.md"
    text = path.read_text(encoding="utf-8")
    section = text[text.index("## 5. Decision register"):text.index("## 6.")]
    rows = [line for line in section.split("\n") if re.match(r"^\| (A[A-E]?|[B-Z]) \| ", line)]
    body = []
    for line in rows:
        cells = [c.strip() for c in line.strip("|").split("|")]
        body.append([tex(cells[0]), tex(cells[1]), tex(cells[2].replace("`", "")), tex(cells[3].replace("`", "")),
                     tex(cells[4])])
    write_tex(out, "A_decisions", ["", "Fixes", "Value", "Config key", "Section"], body,
              "TODO: the analysis decisions, fixed before the data they concern were read.", "decisions",
              r"lp{3.2cm}p{4.6cm}p{3.2cm}l", r"\scriptsize")
    return index_row("A_decisions", "App. A", "decisions A-AE: what each fixes, its value, config key, section", [path])


def h2(syn: Path, out: Path) -> dict:
    """Appendix D: H2 at equal false-positive rate."""
    path = syn / "e5" / "analyze" / "h2.csv"
    h = read_table(path)
    body = [[tex(r["comparison"]), str(r["stratum"]), f"{num(100 * float(r['excess_e2']), 1, sign=True)}",
             f"{num(100 * float(r['excess_e1']), 1, sign=True)}",
             f"{num(100 * float(r['difference']), 1, sign=True)} "
             f"{ci(100 * float(r['difference_low']), 100 * float(r['difference_high']), digits=1, sign=True)}"]
            for _, r in h.iterrows()]
    write_tex(out, "D_h2", ["Comparison", "Stratum", "E2 excess", "E1 excess", "E2 − E1 [95% CI]"], body,
              "TODO: H2, excess over H-eval in points at equal false-positive rate.", "h2", "llrrr", r"\footnotesize")
    return index_row("D_h2", "App. D (4.2)", "H2: excesses and their difference", [path])


def rates_by_size(syn: Path, out: Path) -> dict:
    """Appendix D: rates of the primary classes by size of the largest lesion (size-confound check)."""
    path = syn / "e5" / "analyze" / "rates_by_size.csv"
    r = read_table(path)
    r = r[r["class"].isin(["e1_copy", "e2_full"])]
    body = [[tex(CLASS_NAMES[str(x["class"])]), str(x["stratum"]), tex(x["bin"]), f"{int(x['n_s']):,}", pct(x["rate_s"]),
             f"{int(x['n_h'])}", pct(x["rate_h"]),
             f"{num(100 * float(x['excess']), 1, sign=True)} "
             f"{ci(100 * float(x['excess_low']), 100 * float(x['excess_high']), digits=1, sign=True)}"]
            for _, x in r.iterrows()]
    write_tex(out, "D_rates_by_size", ["Class", "Stratum", "Largest lesion", "S", "S (%)", "H-eval", "H-eval (%)",
                                       "Excess [95% CI]"], body,
              "TODO: rates by size of the largest lesion.", "rates_by_size", "lllrrrrr", r"\scriptsize")
    return index_row("D_rates_by_size", "App. D (3.4)", "primary-class rates by largest-lesion size", [path])


def rehearsals(real: Path, out: Path) -> dict:
    """Appendix F: every mock set's reading against the reading it had to give."""
    e5 = real / "e5" / "rehearse"
    e6 = real / "e6" / "rehearse"
    e8 = json.loads((real / "e8" / "rehearse" / "run_info.json").read_text(encoding="utf-8")).get("readings", {})

    def e5_reading(case: str) -> str:
        m = re.search(r"OUTCOME \(mechanical reading\): ([^;(]+)", (e5 / case / "review.txt").read_text(encoding="utf-8"))
        return m.group(1).strip() if m else "--"

    def e6_reading(case: str) -> str:
        m = re.search(r"Reading \(decision U, primary cells\): (.+)", (e6 / case / "review.txt").read_text(encoding="utf-8"))
        return m.group(1).strip().lower() if m else "--"

    spec = [("E5", "null (H-cal)", "not beyond chance", e5_reading("null")),
            ("E5", "positive (H-cal + planted copies)", "memorization beyond chance", e5_reading("positive")),
            ("E6", "null (H-cal)", "not patient copying", e6_reading("null")),
            ("E6", "positive (planted copies)", "patient copying", e6_reading("positive")),
            ("E6", "placement (random T positions)", "not patient copying", e6_reading("placement")),
            ("E6", "placement, compact", "not patient copying", e6_reading("placement_compact")),
            ("E8", "null (H-cal)", "not supported", str(e8.get("null", "--"))),
            ("E8", "positive (planted, BRAVO z + 2)", "supported", str(e8.get("positive", "--")))]
    body = [[a, tex(b), tex(c), tex(d)] for a, b, c, d in spec]
    write_tex(out, "F_rehearsals", ["Experiment", "Mock set", "Required", "Obtained"], body,
              "TODO: the analyses rehearsed on mock sets with a known answer before S was read.", "rehearsals", "llll")
    return index_row("F_rehearsals", "App. F (2.8)", "mock sets: required vs obtained readings",
                     [e5 / "null" / "review.txt", e6 / "null" / "review.txt", real / "e8" / "rehearse" / "run_info.json"])


def check_results(cfg: dict, out: Path) -> dict:
    """Appendix F: the consistency checks of every experiment."""
    root = config.resolve_path(cfg["results_root"])
    body, sources = [], []
    for exp, where, rel in CHECKS:
        path = root / where / rel
        sources.append(path)
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        m = re.search(r"(\d+) passed, (\d+) failed", text)
        body.append([tex(exp), m.group(1) if m else "--", m.group(2) if m else "--"])
    write_tex(out, "F_checks", ["Stage package", "Checks passed", "Failed"], body,
              "TODO: consistency checks (each reproduces the stored outputs from their inputs).", "checks", "lrr")
    return index_row("F_checks", "App. F (2.13)", "check results per experiment", sources)


def software(syn: Path, out: Path) -> dict:
    """Appendix F: the package versions recorded by the run logs (E8 paired) and Python."""
    path = syn / "e8" / "paired" / "run_info.json"
    info = json.loads(path.read_text(encoding="utf-8"))
    body = [[tex(k), tex(v)] for k, v in {"python": str(info.get("python", "3.11.9")).split()[0],
                                          **info.get("packages", {})}.items()]
    write_tex(out, "F_software", ["Package", "Version"], body, "TODO: software versions (pinned in requirements-lock.txt).",
              "software", "ll")
    return index_row("F_software", "App. F (2.13)", "software versions from the run logs", [path])


# --- the appendix as one LaTeX file -----------------------------------------------------------------

APPENDIX_LAYOUT = (
    ("Settings and decisions", "app:decisions", [(None, ["table:A_decisions"])]),
    ("Data, preprocessing and calibration", "app:data",
     [("Model space and lesions", ["B1", "B3", "B4"]), ("Brain masks and quality control", ["B2", "B5", "B6"]),
      ("Neighbouring slices and calibration", ["B7", "B8", "B9"])]),
    ("Validation with planted copies", "app:validation", [(None, ["C1", "C2", "C3", "C4"])]),
    ("Main audit and exposure", "app:audit",
     [("Main audit", ["D1", "D2", "D3", "table:D_h2", "table:D_rates_by_size"]),
      ("Exposure and duplication", ["D4", "D5", "D6"]), ("Rehearsal of the placement reading", ["D7", "D8", "D9"])]),
    ("Match reports and the paired-MRI check", "app:reports",
     [("Match reports", ["E1", "E2", "E3", "E4"]), ("Paired-MRI check", ["E5", "E6"])]),
    ("Reproducibility", "app:repro", [(None, ["table:F_rehearsals", "table:F_checks", "table:F_software"])]),
)
TALL_SHEETS = ("B5", "B6", "B9", "C3", "C4", "D2", "D3", "E4", "E5")   # given a page of their own ([p])
REPORT_DIR = "figures"   # where the report folder (results/<pool>/report/) is uploaded in the LaTeX project


def appendix_tex(report: Path) -> Path:
    r"""`report/appendix.tex`: the whole appendix (sections A-F, subsections, every appendix figure and
    table) with placeholder captions from the figures' descriptions, for `\input{appendix}` after
    `\appendix`. Paths assume `report/` is uploaded as `REPORT_DIR/` next to the main file."""
    index = read_table(report / "appendix" / "figures.csv")
    figures = {str(f).split("_")[0]: (str(f), str(c)) for f, c in zip(index["figure"], index["content"])}
    used = {code for _, _, subs in APPENDIX_LAYOUT for _, items in subs for code in items}
    missing = sorted(set(figures) - used)
    if missing:
        raise SystemExit(f"Appendix figures without a place in APPENDIX_LAYOUT: {missing}")
    lines = ["% The report's appendix, generated by `scripts/r_report_figures.py tables` (do not edit the paths by hand;",
             f"% captions are placeholders). Assumes results/<pool>/report/ is uploaded as `{REPORT_DIR}/` next to the",
             r"% main file, and in the preamble: \usepackage{graphicx, booktabs, placeins}.",
             r"% Use after \appendix:  \input{appendix}", "",
             r"\counterwithin{figure}{section}", r"\counterwithin{table}{section}", ""]
    for title, label, subsections in APPENDIX_LAYOUT:
        lines += [rf"\section{{{title}}}\label{{{label}}}", "% TODO: one or two sentences on what this appendix shows.", ""]
        for subtitle, items in subsections:
            if subtitle:
                lines += [rf"\subsection{{{subtitle}}}", ""]
            for item in items:
                if item.startswith("table:"):
                    lines += [rf"\input{{{REPORT_DIR}/tables/{item.removeprefix('table:')}}}", ""]
                    continue
                name, content = figures[item]
                size = r"height=0.85\textheight,width=\textwidth,keepaspectratio" if item in TALL_SHEETS else r"width=\textwidth"
                lines += [rf"\begin{{figure}}[{'p' if item in TALL_SHEETS else 'htbp'}]", r"  \centering",
                          rf"  \includegraphics[{size}]{{{REPORT_DIR}/appendix/{name}.pdf}}",
                          rf"  \caption{{TODO: {tex(content)}.}}", rf"  \label{{fig:app_{item}}}", r"\end{figure}", ""]
        lines += [r"\FloatBarrier", ""]
    path = report / "appendix.tex"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
