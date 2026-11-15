"""E7: match reports and their evaluation (docs/e6-e7-plan.md, decisions X, Y, Z).

The report generator is built and checked on E4's planted copies (known source) before any report
on S is drawn, so no report rule is tuned on S. Every choice is in the config section `reports:`.
The stages live in `src/memtk/e7/`; this script is only the command line.

Stages, run in this order (see docs/e6-e7-plan.md):
  planted    (needs E4 validate, decisions X, Y, Z) reports for every planted mask, correctness per
             variant against the criteria, the deletion test on planted controls and H-eval chance
             flags, the taxonomy's confusion table, one example report per variant (no S)
             -> results/<dataset>/e7/planted/
  (planned: reports, check)

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e7_reports.py planted
"""

import argparse

from memtk import config
from memtk.e7 import planted
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "planted", "reports, correctness, deletion test and taxonomy on E4's planted copies (no S)",
              planted.run_planted)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
