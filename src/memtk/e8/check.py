"""E8 stage `check` (`docs/e8-plan.md`, step 5): consistency checks of all E8 outputs.

Decisions AA-AE set; the run logs of `select`, `rehearse` and `paired` made with the current settings,
in that order, with S's MRIs read by `paired` only; E5's lock unchanged and as recorded by `select` and
`paired`; the rehearsal readings as required; `select`'s pairs, every rehearsal table and reading, and
`paired`'s measures, tables and reading reproduced with the seed; the brain masks equal E0b's stored
ones; every figure present; the unit tests. Reproducing `paired`'s measures reads S's MRIs again: it
verifies the stored values and cannot change a result. Writes `results/<pool>/e8/check_report.txt`;
exit code 1 on any failure.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

from memtk import checks, config
from memtk.calibration import load_frozen
from memtk.e5.common import lock_unchanged
from memtk.e8.common import paired_config
from memtk.e8.paired import READINGS, Query, analyze, measure_s, settings
from memtk.e8.rehearse import rehearse_cases
from memtk.e8.select import select_pairs
from memtk.io import MaskStack, read_table
from memtk.lesions import brain_mask

SELECT_KEYS = ["seed", "paired"]
ANALYSIS_KEYS = ["seed", "paired", "calibration.bootstrap_draws", "brain"]
TOL = 1e-9
FIGURES = {"rehearse": ["ssim_groups.png", "examples.png"], "paired": ["ssim_groups.png", "within_a.png", "examples.png"]}


# --- stage ------------------------------------------------------------------------------------

def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None) -> None:
    """Consistency checks of all E8 outputs. Writes `results/<pool>/e8/check_report.txt`; exit code
    1 on any failure."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "e8")
    dirs = {"select": out / "select", "rehearse": config.results_dir(cfg, name, "e8") / "rehearse", "paired": out / "paired"}
    report = checks.CheckReport()

    try:
        p = paired_config(cfg)
        report.add("all E8 decisions (AA-AE) are set and implemented (paired:)", True, "")
    except SystemExit as e:
        report.add("all E8 decisions (AA-AE) are set and implemented (paired:)", False, str(e))
        p = cfg["paired"]
    logs = {"select": (dirs["select"] / "run_info.json", SELECT_KEYS),
            "rehearse": (dirs["rehearse"] / "run_info.json", ANALYSIS_KEYS),
            "paired": (dirs["paired"] / "run_info.json", ANALYSIS_KEYS)}
    checks.check_run_logs(report, cfg, logs)
    info = {s: json.loads(path.read_text(encoding="utf-8")) if path.exists() else {} for s, (path, _) in logs.items()}
    report.add("S's MRIs read by 'paired' only ('select' read S's tables, 'rehearse' no S data)",
               "S MRIs" in info["paired"].get("inputs", [])
               and not any("MRI" in i for i in info["select"].get("inputs", []))
               and not any(i.startswith("S") for i in info["rehearse"].get("inputs", [])),
               "; ".join(f"{s}: {info[s].get('inputs')}" for s in info))
    try:
        now = lock_unchanged(cfg, name, pool_name)
        report.add("E5's lock unchanged since S was scored, and as recorded by 'select' and 'paired'",
                   all(info[s].get("e5_lock") == now for s in ("select", "paired")), "")
    except SystemExit as e:
        report.add("E5's lock unchanged since S was scored, and as recorded by 'select' and 'paired'", False, str(e))
    report.add("the rehearsal was as required when 'paired' ran", bool(info["rehearse"].get("as_required")),
               str(info["rehearse"].get("readings")))

    e3, _ = load_frozen(cfg, name)
    s = settings(cfg, p, e3)
    draws = int(cfg["calibration"]["bootstrap_draws"])

    # select
    pairs, _, _ = select_pairs(cfg, name, pool_name, p, e3)
    differ = checks.differing_tables({"pairs": pairs}, dirs["select"], TOL)
    report.add("'select' pairs reproduced (same seed)", not differ, ", ".join(differ) or f"{len(pairs)} pairs")

    # rehearse
    results, real_h, _ = rehearse_cases(cfg, name, p, e3, s)
    readings = {case: r.reading for case, r in results.items()}
    report.add("rehearsal readings reproduced and as required (null not supported, positive supported)",
               readings == info["rehearse"].get("readings") and readings["null"] != READINGS[0]
               and readings["positive"] == READINGS[0], str(readings))
    differ = []
    for case, r in results.items():
        tables = {"measures": r.measures, **{k: v for k, v in r.tables.items() if len(v)}}
        differ += [f"{case}/{d}" for d in checks.differing_tables(tables, dirs["rehearse"] / case, TOL)]
    report.add("every rehearsal table reproduced", not differ, ", ".join(differ) or "3 cases")

    # paired (re-reads S's MRIs to reproduce the stored measures)
    stored_pairs = read_table(dirs["select"] / "pairs.csv")
    reference = read_table(dirs["rehearse"] / "heval" / "measures.csv")
    m, query, real_s = measure_s(cfg, name, pool_name, stored_pairs, s)
    tables, read = analyze(m, reference, p, s, draws, np.random.default_rng(cfg["seed"]))
    differ = checks.differing_tables({"measures": m, **tables}, dirs["paired"], TOL)
    report.add("'paired' measures and tables reproduced (same seed)", not differ, ", ".join(differ) or "4 tables")
    report.add("'paired' reading reproduced", read == info["paired"].get("reading"), f"{read}")

    # brain masks and figures
    bad, n = brain_mismatches(cfg, name, pool_name, real_h | real_s, query, stored_pairs["query"].unique(), s)
    report.add("brain masks equal E0b's stored ones (every stored real slice and S sample used)", not bad,
               ", ".join(bad[:5]) or f"{n} images")
    missing = [f"rehearse/{c}/{f}" for c in results for f in FIGURES["rehearse"] if not (dirs["rehearse"] / c / f).exists()]
    missing += [f"paired/{f}" for f in FIGURES["paired"] if not (dirs["paired"] / f).exists()]
    report.add("every figure exists", not missing, ", ".join(missing) or f"{3 * len(FIGURES['rehearse']) + 3} figures")

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = (f"E8 check: pool '{pool_name}' against '{name}', seed {cfg['seed']}\n"
            f"(reproducing 'paired' re-reads S's MRIs; it verifies the stored values and cannot change a result)\n\n"
            f"{report.to_text()}\n")
    (out / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check"}, filename="check.run_info.json")
    print(f"\n{text}\nWrote {out / 'check_report.txt'}")
    if not report.all_passed:
        raise SystemExit(1)


# --- brain masks ------------------------------------------------------------------------------

def brain_mismatches(cfg: dict, dataset: str, pool: str, real: dict[str, tuple[np.ndarray, np.ndarray]], query: Query,
                     samples, s) -> tuple[list[str], int]:
    """Keys whose brain mask (`lesions.brain_mask` with the `brain:` config) differs from E0b's
    stored one, over the real slices and S samples used that E0b stored; also the number compared."""
    stored_real = MaskStack(config.output_dir(cfg, dataset) / "brain_masks.npz")
    stored_s = MaskStack(config.output_dir(cfg, pool) / "brain_masks.npz")
    images = [(k, img, stored_real) for k, (img, _) in real.items() if k in stored_real]
    images += [(str(k), query(str(k))[0], stored_s) for k in samples if str(k) in stored_s]
    bad = [k for k, img, stack in images if not np.array_equal(brain_mask(img, s.brain_method, s.low_threshold), stack[k])]
    return bad, len(images)
