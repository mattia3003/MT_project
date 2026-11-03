"""E0b: data audit and mask quality control.

Review stages only produce evidence and a proposed config value. Production stages apply your
decision and refuse to run while the value they need is still null in the config. The stages
themselves live in `src/memtk/e0b/` (one module per topic); this script is only the command line.

Stages, run in this order. Each one can be run and checked on its own:
  brain-review   evidence for decision A (brain-mask method, failed-MRI tissue cut-off), using T,
                 a random sample of S for the methods and all of S for tissue area
                 -> results/<dataset>/e0b/brain_review/
  brain          (needs decision A) brain mask, box and tissue area of every real lesion slice
                 (T, H-cal, H-eval) and every synthetic sample, the failed-MRI flag, and the
                 synthetic masks as saved
                 -> <output_root>/<dataset>/brain.csv + brain_masks.npz,
                    <output_root>/<pool>/brain.csv + brain_masks.npz + masks_raw.npz,
                    results/<dataset|pool>/e0b/brain/ (summary and outline sheets)
  lesions-review evidence for decisions B and C (minimum lesion area, attach gap): component
                 sizes of all real lesion slices and synthetic masks, attach-or-drop counts for
                 nearby settings, specks by gap; proposal from T only
                 -> results/<dataset>/e0b/lesions_review/
  lesions        (needs decisions B, C) attach-or-drop of every real lesion slice and every
                 synthetic mask as saved: one row per lesion (ID, features in three position
                 frames), one row per slice, label maps; a label volume per real patient
                 -> <output_root>/<dataset|pool>/lesions.csv + slices.csv + labels.npz,
                    <output_root>/<dataset>/volumes/<pid>/labels.nii.gz,
                    results/<dataset|pool>/e0b/lesions/summary.txt + overview.png
  view           (helper, after lesions) one slice at full size with its lesion rows; optionally
                 a synthetic sample's image and labels as NIfTI for ITK-SNAP
                 -> results/<dataset|pool>/e0b/views/<key>.png
  qc-review      evidence for decision D (isolated-speck cut-off between minor and noisy masks):
                 masks by number of isolated specks, sheets per stratum, clean masks with a
                 lesion outside the brain -> results/<pool>/e0b/qc_review/
  qc             (needs decision D) one row per synthetic sample: mask class (from the mask alone),
                 failed-MRI flag, combined QC class, edge and small-lesion flags; summary, old QC
                 list cross-table, the edge check on the real sets, example sheet
                 -> <output_root>/<pool>/qc.csv, results/<pool>/e0b/qc/
  stats          (after qc) E0 description of real sets and the synthetic main audit: description
                 table, lesions per slice, lesion area, centroid density in pixels and brain-
                 relative, impact of candidate informative-lesion bounds (input for E3)
                 -> results/<dataset>/e0b/stats/
  filters-review evidence for decision E (which of the sampling script's filters to apply to H-cal
                 and H-eval): each filter on the masks as saved and cleaned, for every set, and
                 what applying it would cost -> results/<dataset>/e0b/filters_review/
  check          consistency checks of all E0b outputs: decisions set and outputs made with them
                 (run logs), lesion tables against label maps, label maps against the masks as
                 saved, caches against the source files, QC classes, unit tests; exit code 1 on
                 any failure -> results/<dataset>/e0b/check_report_<pool>.txt

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py brain-review
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py view --sample file_2_1033
  .venv\\Scripts\\python.exe scripts\\e0b_audit.py view --patient Mets_120 --z 40
"""

import argparse

from memtk import config
from memtk.e0b import brain, check, filters, lesions, qc, stats


def add_stage(stages, name: str, help: str, run, dataset: bool = True, overwrite: bool = True):
    """A stage subcommand with the options every stage shares: --config, --dataset (real
    dataset), --pool (synthetic pool) and --overwrite. Returns its parser for extra options;
    each option reaches the stage function as the keyword of the same name."""
    p = stages.add_parser(name, help=help)
    config.add_config_arg(p)
    if dataset:
        p.add_argument("--dataset", help="real dataset name in the config (default: default_real_dataset)")
    p.add_argument("--pool", help="synthetic pool name in the config (default: default_synthetic_pool)")
    if overwrite:
        p.add_argument("--overwrite", action="store_true", help="regenerate existing output")
    p.set_defaults(run=run)
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    p = add_stage(stages, "brain-review", "evidence for decision A: brain-mask method, failed-MRI cut-off",
                  brain.run_brain_review)
    p.add_argument("--n-synthetic", type=int, default=2000, help="random synthetic samples to evaluate (default: 2000)")
    p.add_argument("--min-area", type=int, default=4,
                   help="components of at least this many px count as lesions here (provisional; default: 4)")
    add_stage(stages, "brain", "brain masks, boxes and failed-MRI flags for all data (needs decision A)",
              brain.run_brain)
    add_stage(stages, "lesions-review", "evidence for decisions B, C: minimum lesion area, attach gap",
              lesions.run_lesions_review)
    add_stage(stages, "lesions", "lesion table, slice table and label maps for all data (needs B, C)",
              lesions.run_lesions)
    add_stage(stages, "qc-review", "evidence for decision D: isolated-speck cut-off (minor vs noisy)",
              qc.run_qc_review, dataset=False)
    add_stage(stages, "qc", "QC class per synthetic sample (needs D)", qc.run_qc)
    add_stage(stages, "stats", "E0 description: tables, distribution figures, bound impact (after 'qc')",
              stats.run_stats)
    add_stage(stages, "filters-review", "evidence for decision E: sampling filters for H-cal/H-eval",
              filters.run_filters_review)
    p = add_stage(stages, "check", "consistency checks of all E0b outputs (exit code 1 on any failure)",
                  check.run_check, overwrite=False)
    p.add_argument("--n-synthetic", type=int, default=300,
                   help="random synthetic samples whose cached masks are recomputed from the files (default: 300)")
    p = add_stage(stages, "view", "one slice at full size with its lesion rows (helper, after 'lesions')",
                  lesions.run_view)
    p.add_argument("--sample", help="synthetic sample ID (full file stem, e.g. file_2_1033)")
    p.add_argument("--patient", help="real patient ID (with --z)")
    p.add_argument("--z", type=int, help="0-based slice index of a real patient (ITK-SNAP shows z + 1)")
    p.add_argument("--nifti", action="store_true", help="synthetic: also write the image and its labels as NIfTI")

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
