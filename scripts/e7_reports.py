"""E7: match reports and their evaluation (docs/e6-e7-plan.md, decisions X, Y, Z).

The report generator is built and checked on E4's planted copies (known source) before any report
on S is drawn, so no report rule is tuned on S. Every choice is in the config section `reports:`.
The stages live in `src/memtk/e7/`; this script is only the command line.

Stages, run in this order (see docs/e6-e7-plan.md):
  planted    (needs E4 validate, decisions X, Y, Z) reports for every planted mask, correctness per
             variant against the criteria, the deletion test on planted controls and H-eval chance
             flags, the taxonomy's confusion table, one example report per variant (no S)
             -> results/<dataset>/e7/planted/
  reports    (needs E5 score + analyze with the lock unchanged, E6 exposure, planted with every criterion
             passed) match reports on S (strong tiers, most exposed patients, random S and H-eval contrasts),
             the deletion test on S, the taxonomy of S's flags against H-eval's
             -> results/<pool>/e7/reports/
  check      consistency checks of all E7 outputs: decisions, run logs in order (S read by reports only),
             E5's lock, the planted criteria, every planted table, the report selection, verdicts, deletion
             test, taxonomy and patient profiles on S reproduced, the unit tests; exit code 1 on any failure
             -> results/<pool>/e7/check_report.txt

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e7_reports.py planted
  .venv\\Scripts\\python.exe scripts\\e7_reports.py reports
  .venv\\Scripts\\python.exe scripts\\e7_reports.py check
"""

import argparse

from memtk import config
from memtk.e7 import check, planted, reports
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "planted", "reports, correctness, deletion test and taxonomy on E4's planted copies (no S)",
              planted.run_planted)
    add_stage(stages, "reports", "match reports, deletion test and taxonomy on S (needs E6 exposure and the planted "
                                 "criteria passed)", reports.run_reports)
    add_stage(stages, "check", "consistency checks of all E7 outputs (exit code 1 on any failure)", check.run_check,
              overwrite=False)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
