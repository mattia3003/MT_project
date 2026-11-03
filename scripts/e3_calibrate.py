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
  calibrate      (needs decision G) E1 and E2 on H-cal against T; tau_D, tau_L, tau_C at the
                 percentile of H-cal, pooled and per stratum, with bootstrap-by-patient intervals;
                 sensitivity table; exclusion counts -> evidence for decision H
                 -> <output_root>/<dataset>/e3/ (H-cal detector tables), results/<dataset>/e3/calibrate/
  freeze         (needs decision H) the frozen threshold file E4 and E5 read: tau_D, tau_L, tau_C per
                 stratum (or pooled) with intervals, the class rule, the bound, the detector settings,
                 the chance rates on H-cal; checks the conditions that keep the E2 shortcuts exact
                 -> configs/frozen/e3_<dataset>.yaml (+ run log)
  check          consistency checks of all E3 outputs: decisions and run logs (only T and H-cal read),
                 the frozen file reproduced, the shortcuts against brute force, a self-match on T, the
                 unit tests; exit code 1 on any failure -> results/<dataset>/e3/check_report.txt

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e3_calibrate.py bound-review
  .venv\\Scripts\\python.exe scripts\\e3_calibrate.py freeze
"""

import argparse

from memtk import config
from memtk.e3 import bound, calibrate, check, freeze
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "bound-review", "evidence for decision G: chance-match rates by lesion size, bound proposal",
              bound.run_bound_review)
    add_stage(stages, "calibrate", "E1 + E2 on H-cal, thresholds with bootstrap intervals (needs G; evidence for H)",
              calibrate.run_calibrate)
    add_stage(stages, "freeze", "write the frozen threshold file for E4/E5 (needs H)", freeze.run_freeze)
    p = add_stage(stages, "check", "consistency checks of all E3 outputs (exit code 1 on any failure)",
                  check.run_check, overwrite=False)
    p.add_argument("--n-brute", type=int, default=20,
                   help="random H-cal slices compared with brute force (default: 20)")
    p.add_argument("--n-self", type=int, default=50, help="random T slices for the self-match (default: 50)")

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
