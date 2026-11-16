"""Report-figures stage `check`: consistency checks of every report figure (`method`, `results`,
`appendix`). Draws nothing and changes no result. Writes `results/<pool>/report/check_report.txt` and
`check.run_info.json`; exit code 1 on any failure.

Checks: each stage's run log exists and was made with the current config; each stage ran after the
experiment stages whose outputs it draws; E5's lock is unchanged and equals what `results` and
`appendix` recorded; every figure listed in a stage's `figures.csv` exists (PNG, and PDF where the
stage writes one) and no unlisted figure lies in its folder; every repository path named as a source
exists; the unit tests pass.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from memtk import checks, config
from memtk.e5.common import lock_unchanged
from memtk.io import read_table

STAGES = ("method", "results", "appendix")
CONFIG_KEYS = ["seed", "brain", "lesions", "detectors", "calibration", "audit", "exposure", "reports", "paired"]
WITH_PDF = {"method": "all", "results": "all", "appendix": "all"}
LOCK_KEYS = ("audit_sha256", "frozen_sha256", "e5_sha256")


def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None) -> None:
    """Consistency checks of all report figures. Writes `results/<pool>/report/check_report.txt`; exit
    code 1 on any failure."""
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, _ = config.synthetic_pool(cfg, pool)
    out = config.results_dir(cfg, pool_name, "report")
    report = checks.CheckReport()

    infos = {}
    for stage in STAGES:
        checks.check_run_logs(report, cfg, {stage: (out / stage / "run_info.json", CONFIG_KEYS)})
        path = out / stage / "run_info.json"
        infos[stage] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    inputs = input_logs(cfg, name, pool_name)
    for stage in STAGES:
        stamp = str(infos[stage].get("timestamp", ""))
        later = {k: str(v) for k, v in inputs.items() if v > stamp}
        report.add(f"'{stage}' drawn after every experiment stage it reads", bool(stamp) and not later,
                   ", ".join(f"{k} ({v})" for k, v in later.items()) or f"{len(inputs)} run logs older")

    try:
        now = lock_unchanged(cfg, name, pool_name)
        recorded = {s: {k: infos[s].get(k) for k in LOCK_KEYS} for s in ("results", "appendix")}
        report.add("E5's lock unchanged and as recorded by 'results' and 'appendix'",
                   all(r == now for r in recorded.values()), "")
    except SystemExit as e:
        report.add("E5's lock unchanged and as recorded by 'results' and 'appendix'", False, str(e))

    for stage in STAGES:
        missing, stray, sources = figure_problems(out / stage, stage)
        report.add(f"'{stage}': every listed figure exists (PNG{', PDF' if WITH_PDF[stage] == 'all' else ''})",
                   not missing, ", ".join(missing) or "")
        report.add(f"'{stage}': no unlisted figure in the folder", not stray, ", ".join(stray) or "")
        report.add(f"'{stage}': every source path named exists", not sources, ", ".join(sources) or "")

    tables = out / "tables"
    checks.check_run_logs(report, cfg, {"tables": (tables / "run_info.json", CONFIG_KEYS)})
    info = json.loads((tables / "run_info.json").read_text(encoding="utf-8")) if (tables / "run_info.json").exists() else {}
    report.add("'tables' made after every experiment stage it reads",
               bool(info) and all(v <= str(info.get("timestamp", "")) for v in inputs.values()), "")
    listed = [n.strip() for cell in (read_table(tables / "tables.csv")["figure"] if (tables / "tables.csv").exists() else [])
              for n in str(cell).split(";")]
    bad = [n for n in listed if not (tables / f"{n}.tex").exists()
           or r"\begin{tabular}" not in (tables / f"{n}.tex").read_text(encoding="utf-8")]
    report.add("'tables': every listed table exists as a LaTeX tabular", bool(listed) and not bad,
               ", ".join(bad) or f"{len(listed)} tables")

    tex_path = out / "appendix.tex"
    tex = tex_path.read_text(encoding="utf-8") if tex_path.exists() else ""
    absent = [f.stem for f in (out / "appendix").glob("*.pdf") if f"appendix/{f.name}" not in tex]
    absent += [f.stem for f in (out / "tables").glob("[ADF]_*.tex") if f"tables/{f.stem}}}" not in tex]
    report.add("'appendix.tex' includes every appendix figure and appendix table", bool(tex) and not absent,
               ", ".join(absent) or f"{tex.count('includegraphics')} figures")

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = f"Report figures check: pool '{pool_name}', dataset '{name}'\n\n{report.to_text()}"
    (out / "check_report.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check"}, filename="check.run_info.json")
    print(f"\n{text}\nWrote {out / 'check_report.txt'}")
    if not report.all_passed:
        raise SystemExit(1)


def input_logs(cfg: dict, name: str, pool_name: str) -> dict[str, str]:
    """Timestamps of the run logs of the experiment stages whose outputs the figures draw."""
    paths = {
        "E4 validate": config.results_dir(cfg, name, "e4") / "validate" / "run_info.json",
        "E5 analyze": config.results_dir(cfg, pool_name, "e5") / "analyze" / "run_info.json",
        "E6 exposure": config.results_dir(cfg, pool_name, "e6") / "exposure" / "run_info.json",
        "E7 planted": config.results_dir(cfg, name, "e7") / "planted" / "run_info.json",
        "E7 reports": config.results_dir(cfg, pool_name, "e7") / "reports" / "run_info.json",
        "E8 rehearse": config.results_dir(cfg, name, "e8") / "rehearse" / "run_info.json",
        "E8 paired": config.results_dir(cfg, pool_name, "e8") / "paired" / "run_info.json",
    }
    return {k: str(json.loads(p.read_text(encoding="utf-8"))["timestamp"]) for k, p in paths.items() if p.exists()}


def figure_problems(folder: Path, stage: str) -> tuple[list[str], list[str], list[str]]:
    """For one stage folder: listed figures without their files, figure files not listed, and named
    repository source paths that do not exist (descriptions and paths with placeholders are skipped)."""
    if not (folder / "figures.csv").exists():
        return ["figures.csv"], [], []
    index = read_table(folder / "figures.csv")
    listed = set(index["figure"])
    pdf = listed if WITH_PDF[stage] == "all" else set(WITH_PDF[stage])
    missing = [f"{f}.png" for f in sorted(listed) if not (folder / f"{f}.png").exists()]
    missing += [f"{f}.pdf" for f in sorted(pdf) if not (folder / f"{f}.pdf").exists()]
    stray = sorted(p.name for p in folder.glob("*.p*") if p.suffix in (".png", ".pdf") and p.stem not in listed)
    sources = sorted({s.strip() for cell in index["sources"].fillna("") for s in str(cell).split(";")
                      if is_repo_path(s.strip()) and not (config.REPO_ROOT / s.strip()).exists()})
    return missing, stray, sources


def is_repo_path(text: str) -> bool:
    """Whether a source entry names a repository path (relative, under results/, processed_data/ or
    configs/, without placeholders)."""
    return text.startswith(("results/", "processed_data/", "configs/")) and "<" not in text
