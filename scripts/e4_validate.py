"""E4: planted-copy validation, the go/no-go gate before the main audit (E5).

Production stages apply your decisions and refuse to run while a value they need is still null
in the config. The stages live in `src/memtk/e4/` (one module per topic); this script is only the
command line. E4 reads T, H-eval, the planted set and the frozen E3 file: S is never read, and the
detector settings and thresholds come only from the frozen file.

Stages, run in this order (see docs/e4-plan.md):
  plant      (needs decision I) base masks from T and their variants P1-P6 (P6-add with H-eval
             lesions), labelled like real masks, each planted lesion traced to its source; no
             detector runs -> evidence for decision K (the go/no-go criteria)
             -> <output_root>/<dataset>/e4/ (planted set), results/<dataset>/e4/plant/
  validate   (needs decision K) both detectors on the planted set and H-eval with the frozen
             thresholds; recall per variant (gated masks), false positives per stratum, the gate
             verdict (PASS / FAIL) against validation.criteria, figures
             -> <output_root>/<dataset>/e4/ (detector tables), results/<dataset>/e4/validate/
  check      consistency checks of all E4 outputs: decisions and run logs (only T, H-eval and the
             planted set read), the frozen file unchanged, the planted set reproduced from the seed,
             the gate reproduced from the stored tables, the unit tests; exit code 1 on any failure
             -> results/<dataset>/e4/check_report.txt

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e4_validate.py plant
  .venv\\Scripts\\python.exe scripts\\e4_validate.py validate
  .venv\\Scripts\\python.exe scripts\\e4_validate.py check
"""

import argparse

from memtk import config
from memtk.e4 import check, plant, validate
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "plant", "planted copies P1-P6 of T base masks, provenance, review figures (needs I)",
              plant.run_plant)
    add_stage(stages, "validate", "both detectors on the planted set and H-eval; gate verdict (needs K)",
              validate.run_validate)
    add_stage(stages, "check", "consistency checks of all E4 outputs (exit code 1 on any failure)",
              check.run_check, overwrite=False)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
