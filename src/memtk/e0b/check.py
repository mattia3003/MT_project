"""E0b stage `check`: consistency checks of all E0b outputs.

`run_check` runs the check functions of `memtk.checks` on every output and the unit tests, and
writes a PASS/FAIL report. The constants below list which config values each production stage
depends on (for the stale-output check) and which values are decisions.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from memtk import checks, config
from memtk.io import LabelStack, MaskStack, load_real, read_table
from memtk.lesions import brain_mask
from memtk.stages import decided
from memtk.synthetic import load_sample


# --- stage ------------------------------------------------------------------------------------

STAGE_KEYS = {
    "brain": ["brain.method", "brain.low_threshold", "brain.min_tissue_px"],
    "lesions": ["lesions.connectivity", "lesions.min_lesion_area", "lesions.attach_gap", "qc.small_lesion_px"],
    "qc": ["qc.max_isolated_specks", "qc.edge_outside_frac"],
}


DECISION_KEYS = ["brain.low_threshold", "brain.method", "brain.min_tissue_px", "lesions.min_lesion_area",
                 "lesions.attach_gap", "qc.max_isolated_specks", "filters.real_sets"]


def run_check(cfg: dict, dataset: str | None = None, pool: str | None = None,
              n_synthetic: int = 300) -> None:
    """Consistency checks of all E0b outputs (functions in `memtk.checks`): decisions set and outputs
    made with them (run logs), lesion tables against label maps, label maps against the masks as
    saved, caches against the source files (`n_synthetic` random samples), QC classes, the unit
    tests. Writes `results/<dataset>/e0b/check_report_<pool>.txt`; exit code 1 on any failure."""
    min_area = int(decided(cfg, "lesions.min_lesion_area", "e0b_audit.py lesions-review"))
    gap = int(decided(cfg, "lesions.attach_gap", "e0b_audit.py lesions-review"))
    max_specks = int(decided(cfg, "qc.max_isolated_specks", "e0b_audit.py qc-review"))
    name, _ = config.real_dataset(cfg, dataset)
    pool_name, pool_cfg = config.synthetic_pool(cfg, pool)
    real_dir, syn_dir = config.output_dir(cfg, name), config.output_dir(cfg, pool_name)
    out = config.results_dir(cfg, name, "e0b")
    rng = np.random.default_rng(cfg["seed"])
    report = checks.CheckReport()

    # decisions and run logs: every decision set; production outputs made with today's values, in order
    unset = [k for k in DECISION_KEYS if config.value(cfg, k) is None]
    report.add("all E0b decisions are set in the config", not unset, ", ".join(unset))
    for d, stages in ((real_dir, ("brain", "lesions")), (syn_dir, ("brain", "lesions", "qc"))):
        checks.check_run_logs(report, cfg, {f"{s} ({d.name})": (d / f"{s}.run_info.json", STAGE_KEYS[s])
                                            for s in stages})

    # real: tables against label maps, label maps against the masks as saved, label volumes
    real_sl, real_les = read_table(real_dir / "slices.csv"), read_table(real_dir / "lesions.csv")
    real_labels = LabelStack(real_dir / "labels.npz")
    checks.check_stack_keys(report, f"{name} brain_masks.npz", MaskStack(real_dir / "brain_masks.npz"), real_sl["key"])
    report.add(f"{name}: brain.csv rows == lesion slices",
               set(read_table(real_dir / "brain.csv")["key"]) == set(real_sl["key"]))
    checks.check_lesion_tables(report, name, real_les, real_sl, real_labels)
    real_masks = {}
    for pid, group in real_sl.groupby("patient_id"):
        _, seg = load_real(real_dir, str(pid))
        real_masks.update({key: seg[:, :, int(z)] for key, z in zip(group["key"], group["z"])})
    checks.check_labels_vs_masks(report, name, real_masks, real_labels, real_sl["key"], min_area, gap)
    checks.check_label_volumes(report, real_dir, real_sl, real_labels)

    # synthetic: the same, plus the caches against the source files and the QC table
    index = read_table(syn_dir / "synthetic_index.csv")
    samples = index.loc[index["read_ok"].astype(bool), "sample_id"]
    syn_sl, syn_les = read_table(syn_dir / "slices.csv"), read_table(syn_dir / "lesions.csv")
    raw, syn_brains = MaskStack(syn_dir / "masks_raw.npz"), MaskStack(syn_dir / "brain_masks.npz")
    syn_labels = LabelStack(syn_dir / "labels.npz")
    for stack_name, stack in (("masks_raw.npz", raw), ("brain_masks.npz", syn_brains)):
        checks.check_stack_keys(report, f"{pool_name} {stack_name}", stack, samples)
    report.add(f"{pool_name}: slices.csv and brain.csv rows == readable samples",
               set(syn_sl["key"]) == set(samples) and set(read_table(syn_dir / "brain.csv")["sample_id"]) == set(samples))
    checks.check_lesion_tables(report, pool_name, syn_les, syn_sl, syn_labels)
    checks.check_labels_vs_masks(report, pool_name, raw, syn_labels, syn_sl["key"], min_area, gap)
    files = dict(zip(index["sample_id"], index["filename"]))
    method, low = decided(cfg, "brain.method", "e0b_audit.py brain-review"), float(cfg["brain"]["low_threshold"])
    pool_dir, differs = config.resolve_path(pool_cfg["dir"]), []
    picks = rng.choice(samples.to_numpy(), size=min(n_synthetic, len(samples)), replace=False)
    for sid in picks:
        image, mask = load_sample(pool_dir / files[sid], pool_cfg)
        if not (np.array_equal(raw[sid], mask > 0) and np.array_equal(syn_brains[sid], brain_mask(image, method, low))):
            differs.append(sid)
    report.add(f"{pool_name}: cached masks and brain masks == recomputed from the files", not differs,
               f"{len(differs)} failing, e.g. {differs[:5]}" if differs else f"{len(picks)} random samples")
    checks.check_qc_table(report, read_table(syn_dir / "qc.csv"), samples, max_specks)

    # the unit tests
    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.REPO_ROOT, capture_output=True, text=True)
    report.add("unit tests (pytest) pass", tests.returncode == 0, (tests.stdout.strip().splitlines() or [""])[-1])

    text = f"E0b check: dataset '{name}', pool '{pool_name}', seed {cfg['seed']}\n\n{report.to_text()}\n"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"check_report_{pool_name}.txt").write_text(text, encoding="utf-8")
    config.write_run_info(out, cfg, script=__file__, extra={"stage": "check", "pool": pool_name},
                          filename=f"check_{pool_name}.run_info.json")
    print(f"\n{text}\nWrote {out / f'check_report_{pool_name}.txt'}")
    if not report.all_passed:
        raise SystemExit(1)
