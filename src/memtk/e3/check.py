"""E3 stage `check`: consistency checks of all E3 outputs.

Decisions set and outputs made with them (run logs, in order, with only T and H-cal as inputs);
the frozen file reproduced from the stored H-cal tables and matching the current settings; the
conditions that keep the E2 shortcuts exact; the shortcuts against brute force on random H-cal
slices; a self-match sanity check on T (code only, not E4); the unit tests. Writes
`results/<dataset>/e3/check_report.txt`; exit code 1 on any failure.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from memtk import checks, config, lesionmatch, wholemask
from memtk.calibration import read_frozen
from memtk.e3.calibrate import thresholds
from memtk.e3.common import lesion_settings, load_hcal_tables
from memtk.e3.freeze import compare_tables, frozen_thresholds, shortcut_problems
from memtk.io import read_table
from memtk.sets import SliceSet, load_set
from memtk.stages import decided

DECISION_KEYS = ["detectors.lesion.position_frame", "calibration.provisional_tau_L", "calibration.chance_rate_max",
                 "calibration.informative_bound_px", "calibration.strata.whole_mask", "calibration.strata.lesion"]
DETECTOR_KEYS = ["lesions.min_lesion_area", "detectors.whole_mask", "detectors.lesion", "calibration.percentile"]
CALIBRATE_KEYS = [*DETECTOR_KEYS, "calibration.informative_bound_px", "calibration.bootstrap_draws"]
FREEZE_KEYS = [*CALIBRATE_KEYS, "calibration.strata"]


# --- stage ------------------------------------------------------------------------------------

def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None, n_brute: int = 20,
              n_self: int = 50) -> None:
    """Consistency checks of all E3 outputs (`n_brute` random H-cal slices for the brute-force
    comparison, `n_self` random T slices for the self-match). Writes
    `results/<dataset>/e3/check_report.txt`; exit code 1 on any failure."""
    name, _ = config.real_dataset(cfg, dataset)
    out = config.results_dir(cfg, name, "e3")
    frozen_path = config.frozen_file(cfg, name)
    rng = np.random.default_rng(cfg["seed"])
    report = checks.CheckReport()

    # decisions, run logs and inputs
    unset = [k for k in DECISION_KEYS if config.value(cfg, k) is None]
    report.add("all E1-E3 decisions (F, G, H) are set in the config", not unset, ", ".join(unset))
    logs = {"bound-review": (out / "bound_review" / "run_info.json", DETECTOR_KEYS),
            "calibrate": (config.output_dir(cfg, name) / "e3" / "calibrate.run_info.json", CALIBRATE_KEYS),
            "freeze": (frozen_path.parent / f"e3_{name}.run_info.json", FREEZE_KEYS)}
    checks.check_run_logs(report, cfg, logs)
    for stage, (path, _) in logs.items():
        inputs = json.loads(path.read_text(encoding="utf-8")).get("inputs") if path.exists() else None
        report.add(f"'{stage}' read only T and H-cal (S never scored before the freeze)", inputs == ["T", "H-cal"],
                   str(inputs))

    # the frozen file against the stored H-cal tables and the current settings
    settings = lesion_settings(cfg)
    if not frozen_path.exists():
        report.add("frozen threshold file exists", False, str(frozen_path))
    else:
        frozen = read_frozen(frozen_path)
        e1, e2, patients = load_hcal_tables(cfg, name)
        recomputed = thresholds(e1, e2, patients, float(cfg["calibration"]["percentile"]), 0, rng)
        stored = read_table(out / "calibrate" / "thresholds.csv")
        differs = compare_tables(recomputed, stored)
        report.add("thresholds.csv reproduced from the stored H-cal tables", not differs, "; ".join(differs[:3]))
        strata = {d: cfg["calibration"]["strata"][d] for d in ("whole_mask", "lesion")}
        expected, _ = frozen_thresholds(stored, strata)
        report.add("frozen thresholds == thresholds.csv under decision H", frozen["thresholds"] == expected,
                   _flat(frozen["thresholds"]))
        same = {"strata": frozen["strata"] == strata,
                "bound": frozen["informative_bound_px"] == cfg["calibration"]["informative_bound_px"],
                "percentile": frozen["percentile"] == float(cfg["calibration"]["percentile"]),
                "whole_mask settings": frozen["detectors"]["whole_mask"] == dict(cfg["detectors"]["whole_mask"]),
                "lesion settings": all(frozen["detectors"]["lesion"][k] == v for k, v in cfg["detectors"]["lesion"].items())}
        report.add("frozen file settings == current config", all(same.values()),
                   ", ".join(k for k, ok in same.items() if not ok))
        problems = shortcut_problems(frozen["thresholds"], settings)
        report.add("E2 shortcuts exact at the frozen thresholds (tau_L >= cut and > score at the prefilter; tau_C > 0)",
                   not problems, "; ".join(problems))

    # the shortcuts against brute force on random H-cal slices; self-match on T
    t, hcal = load_set(cfg, name, "T"), load_set(cfg, name, "H-cal")
    bound = int(decided(cfg, "calibration.informative_bound_px", "e3_calibrate.py bound-review"))
    pick = np.sort(rng.choice(len(hcal.keys), size=min(n_brute, len(hcal.keys)), replace=False))
    n_pairs, bad = brute_force_differences(hcal, t, pick, bound, settings)
    report.add("E2 shortcuts == brute force (candidate slices and C)", not bad,
               f"{n_pairs} candidate slice pairs of {len(pick)} random H-cal slices" + (f"; {bad[:3]}" if bad else ""))
    self_e1, self_e2 = self_match(t, rng, n_self, bound, settings, cfg["detectors"]["whole_mask"])
    report.add("self-match: random T slices find an identical slice (E1 score 1, E2 C 1)", not (self_e1 or self_e2),
               f"{n_self} slices; E1 failing {self_e1[:3]}, E2 failing {self_e2[:3]}" if (self_e1 or self_e2)
               else f"{n_self} slices")

    # the unit tests
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = f"E3 check: dataset '{name}', seed {cfg['seed']}\n\n{report.to_text()}\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check"}, filename="check.run_info.json")
    print(f"\n{text}\nWrote {out / 'check_report.txt'}")
    if not report.all_passed:
        raise SystemExit(1)


# --- checks -----------------------------------------------------------------------------------

def _flat(values: dict) -> str:
    """Thresholds as one readable line, e.g. "tau_D 0.905/0.729, ..."."""
    return ", ".join(f"{name} {v['single']:.3f}/{v['multi']:.3f}" for d in values.values() for name, v in d.items())


def brute_force_differences(hcal: SliceSet, t: SliceSet, pick: np.ndarray, bound: int,
                            settings: dict) -> tuple[int, list[str]]:
    """Scores every lesion pair of the picked H-cal slices with every T lesion (no prefilter, no
    candidate shortcut) and compares the candidate slices and C with `lesionmatch.match`."""
    keys = [hcal.keys[i] for i in pick]
    res = lesionmatch.match(hcal.labels[pick], keys, t.labels, t.keys, bound, **settings)
    found = res.slice_scores(settings["candidate_min_s"]).set_index(["query", "train"])["C"]
    q = lesionmatch.lesion_records(hcal.labels[pick], keys, bound)
    r = lesionmatch.lesion_records(t.labels, t.keys, bound)
    half = max(q.half_size(), r.half_size())
    qc, rc = lesionmatch.packed_crops(q, half), lesionmatch.packed_crops(r, half)
    qi, ri = (a.ravel() for a in np.meshgrid(np.arange(len(q.area)), np.arange(len(r.area)), indexing="ij"))
    distance = np.hypot(*(q.centroid[qi] - r.centroid[ri]).T)
    s = (lesionmatch.position_factor(distance, settings["sigma_px"]) * lesionmatch.size_factor(q.area[qi], r.area[ri])
         * lesionmatch.shape_factor(qc[qi], rc[ri], q.area[qi], r.area[ri])).reshape(len(q.area), len(r.area))
    cut, n_pairs, bad = settings["candidate_min_s"], 0, []
    for a, key in enumerate(keys):
        rows = q.slice == a
        if not rows.any():
            continue
        for b in np.unique(r.slice[s[rows].max(axis=0) >= cut]):
            sub = s[np.ix_(rows, r.slice == b)]
            n_pairs += 1
            pr, pc = lesionmatch.assign_pairs(sub)
            c = sub[pr, pc][sub[pr, pc] >= cut].sum() / max(sub.shape)
            if abs(found.get((key, t.keys[b]), -1.0) - c) > 1e-9:
                bad.append(f"{key} vs {t.keys[b]}")
    if len(found) != n_pairs:
        bad.append(f"{len(found)} candidate slice pairs found, {n_pairs} by brute force")
    return n_pairs, bad


def self_match(t: SliceSet, rng: np.random.Generator, n: int, bound: int, settings: dict,
               wm: dict) -> tuple[list[str], list[str]]:
    """Random T slices queried against T: each must find a slice with E1 score 1 and, when it has
    an informative lesion, E2 C = 1 (itself or an identical neighbour). A check of the code only."""
    pick = np.sort(rng.choice(len(t.keys), size=min(n, len(t.keys)), replace=False))
    keys = [t.keys[i] for i in pick]
    e1 = wholemask.match(t.labels[pick], t.labels, t.patients, keys, t.keys, wm["top_k_dice"],
                         wm["top_k_centroid"], wm["max_shift_px"])
    e1_bad = e1.loc[~np.isclose(e1["score"], 1.0), "query"].tolist()
    e2 = lesionmatch.match(t.labels[pick], keys, t.labels, t.keys, bound, **settings)
    best = e2.slice_scores(settings["candidate_min_s"]).groupby("query")["C"].max()
    informative = e2.queries.loc[e2.queries["n_informative"] > 0, "query"]
    e2_bad = [k for k in informative if not np.isclose(best.get(k, 0.0), 1.0)]
    return e1_bad, e2_bad
