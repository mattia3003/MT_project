"""E5: the main audit. S is scored once, with every analysis choice fixed before (docs/e5-plan.md).

The analysis is fixed in the config section `audit:` (decisions L-Q) and in the E5 file that
`prepare` writes; `score` hashes both together with the frozen E3 file, and later stages refuse to
run if they changed (the lock). The stages live in `src/memtk/e5/`; this script is only the
command line. Detector settings and thresholds come only from the frozen files.

Stages, run in this order (see docs/e5-plan.md):
  prepare    (needs decisions O, P, Q) from T, H-cal and H-eval only: drop-only real label maps (the
             held-out ones must not change), the sensitivity thresholds re-derived on H-cal (equal to
             E3's), tau_D* for H2, the held-out maxima for near-exact copies, H-eval under every
             variant; refuses once S has been scored
             -> configs/frozen/e5_<dataset>.yaml, <output_root>/<dataset>/e5/, results/<dataset>/e5/prepare/
  rehearse   (needs prepare) the score and analyze code that will run on S, run on mock query sets (no S):
             null case = H-cal (no excess expected); positive case = H-cal + the planted copies P1, P2,
             P4 of E4 (excess, right source patients, near-exact copies expected)
             -> <output_root>/<dataset>/e5/rehearse/<case>/, results/<dataset>/e5/rehearse/<case>/
  score      (needs the E4 gate PASS/ACCEPTED, prepare, rehearse) S scored once: the main and noisy
             masks, main and every sensitivity variant; raw tables only, no rate shown; writes a
             completion marker, after which score and prepare refuse to run (the lock)
             -> <output_root>/<pool>/e5/
  analyze    (needs score) every table and figure of decisions L-Q on S against H-eval, flagged.csv,
             the mechanical outcome reading and the E8 exit count; refuses unless audit: and both
             frozen files are those score recorded (the lock)
             -> results/<pool>/e5/analyze/
  check      consistency checks of all E5 outputs: decisions, run logs in order (S read by score only),
             the lock (hashes unchanged since score; one complete score run), prepare values and every
             analyze table reproduced (same seed), the unit tests; exit code 1 on any failure
             -> results/<pool>/e5/check_report.txt

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e5_audit.py prepare
  .venv\\Scripts\\python.exe scripts\\e5_audit.py rehearse
  .venv\\Scripts\\python.exe scripts\\e5_audit.py score
  .venv\\Scripts\\python.exe scripts\\e5_audit.py analyze
  .venv\\Scripts\\python.exe scripts\\e5_audit.py check
"""

import argparse

from memtk import config
from memtk.e5 import analyze, check, prepare, rehearse, score
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "prepare", "values fixed before S: sensitivity thresholds, tau_D*, held-out maxima, "
                                 "drop-only check, H-eval under every variant (no S)", prepare.run_prepare)
    add_stage(stages, "rehearse", "the scoring and analysis code run on mock query sets: H-cal (null) and H-cal + "
                                  "planted copies (positive); no S", rehearse.run_rehearse)
    add_stage(stages, "score", "S scored once (main + noisy, every variant); raw tables only; locks the analysis",
              score.run_score)
    add_stage(stages, "analyze", "every table and figure on S against H-eval, flagged.csv, the outcome reading "
                                 "(needs score; checks the lock)", analyze.run_analyze)
    add_stage(stages, "check", "consistency checks of all E5 outputs (exit code 1 on any failure)", check.run_check,
              overwrite=False)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
