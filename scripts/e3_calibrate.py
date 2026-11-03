"""E3: chance calibration on H-cal (runs the E1 and E2 detectors from `memtk`).

Review stages only produce evidence and a proposed config value. Production stages apply your
decision and refuse to run while the value they need is still null in the config. The stages
live in `src/memtk/e3/` (one module per topic); this script is only the command line. E3 reads
T and H-cal only: S is never scored before the thresholds are frozen.

Stages, run in this order (see docs/e1-e3-plan.md):
  bound-review   evidence for decision G (provisional tau_L, chance-rate cut, informative-lesion
                 bound): every H-cal lesion's best E2 match in T, chance-match rate by lesion area,
                 the proposal, the cost of each bound (E0), example matches
                 -> results/<dataset>/e3/bound_review/
  (planned: calibrate, freeze, check)

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e3_calibrate.py bound-review
"""

import argparse

from memtk import config
from memtk.e3 import bound
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "bound-review", "evidence for decision G: chance-match rates by lesion size, bound proposal",
              bound.run_bound_review)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
