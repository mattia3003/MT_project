"""E8: the paired synthetic MRI check (docs/e8-plan.md, decisions AA-AE; RQ4, H4, observational).

Is the paired synthetic MRI of a strongly matched mask closer to the matched training slice's BRAVO
than the paired MRI of a novel mask is to its nearest training slice, beyond what similar masks give
by chance (H-eval)? Every choice is in the config section `paired:`. The stages live in
`src/memtk/e8/`; this script is only the command line.

Stages, run in this order (see docs/e8-plan.md):
  select     (needs E5 analyze with the lock unchanged, E7 reports, decisions AA, AB, AD) group A, matched
             group B, comparison slices, alignment shifts, the within-A rows; reads no MRI
             -> results/<pool>/e8/select/
  rehearse   (needs E4 validate, E5 rehearse, decisions AB-AE) the measurement and analysis `paired` will
             run on S, on the H-eval reference, the H-cal null and the planted positive mock; no S read
             -> results/<dataset>/e8/rehearse/
  paired     (needs select and rehearse with the current `paired:`, the rehearsal as required, E5's lock
             unchanged) measures every pair on S's MRIs, A vs B against the H-eval reference, the reading;
             the only stage that reads S's MRIs
             -> results/<pool>/e8/paired/
  check      consistency checks of all E8 outputs: decisions, run logs in order (S's MRIs read by paired only),
             E5's lock, the rehearsal readings, every table and reading reproduced with the seed, brain masks
             equal E0b's, every figure present, the unit tests; exit code 1 on any failure
             -> results/<pool>/e8/check_report.txt

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e8_paired.py select
  .venv\\Scripts\\python.exe scripts\\e8_paired.py rehearse
  .venv\\Scripts\\python.exe scripts\\e8_paired.py paired
  .venv\\Scripts\\python.exe scripts\\e8_paired.py check
"""

import argparse

from memtk import config
from memtk.e8 import check, paired, rehearse, select
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "select", "group A, matched group B, comparison slices and shifts of S (reads no MRI)",
              select.run_select)
    add_stage(stages, "rehearse", "the analysis on the H-eval reference, H-cal null and planted positive mock (no S)",
              rehearse.run_rehearse)
    add_stage(stages, "paired", "measure S's pairs on the synthetic MRIs, A vs B against H-eval, the reading (one run)",
              paired.run_paired)
    add_stage(stages, "check", "consistency checks of all E8 outputs (exit code 1 on any failure)", check.run_check,
              overwrite=False)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
