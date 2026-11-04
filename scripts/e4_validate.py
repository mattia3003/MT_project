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
  validate   (needs decision K; step 5, not yet implemented) both detectors on the planted set and
             H-eval with the frozen thresholds; validation table; gate verdict
  check      (step 6, not yet implemented) consistency checks of all E4 outputs

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e4_validate.py plant
"""

import argparse

from memtk import config
from memtk.e4 import plant
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "plant", "planted copies P1-P6 of T base masks, provenance, review figures (needs I)",
              plant.run_plant)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
