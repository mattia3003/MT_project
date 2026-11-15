"""E6: patient exposure, copying vs placement, rarity (H3) and duplication (H1) (docs/e6-e7-plan.md).

Every analysis choice is in the config section `exposure:` (decisions R, U, V, W), set before the
stage that reads S's tables runs. S comes only from E5's stored tables; E6 never scores S again.
The stages live in `src/memtk/e6/`; this script is only the command line.

Stages, run in this order (see docs/e6-e7-plan.md):
  reference  (needs decisions R, V, W) from T and the stored H-eval tables only: 3D lesions of T under
             both methods of decision W, slice rarity on both detectors' scales, patient features and
             rarity, T slice shares, H-eval nearest-neighbour patients
             -> <output_root>/<dataset>/e6/, results/<dataset>/e6/reference/
  rehearse   (needs reference, E5 rehearse) the E6 analysis that will run on S, run on E5's mock sets (no S):
             null case = H-cal (ratios near 1, AUC about 0.5); positive case = H-cal + planted copies (base
             patients over-exposed, reading "patient copying")
             -> results/<dataset>/e6/rehearse/<case>/
  exposure   (needs E5 score with its lock unchanged, reference, rehearse with the current exposure:)
             the same analysis on S (E5's stored tables, set main): exposure, concentration, margins,
             source rarity, H3, H1, top patients, the reading of decision U
             -> results/<pool>/e6/exposure/
  check      consistency checks of all E6 outputs: decisions, run logs in order (S read by exposure only),
             E5's lock, every reference and exposure table reproduced (same seed), the rehearsal readings,
             the unit tests; exit code 1 on any failure
             -> results/<pool>/e6/check_report.txt

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e6_exposure.py reference
  .venv\\Scripts\\python.exe scripts\\e6_exposure.py rehearse
  .venv\\Scripts\\python.exe scripts\\e6_exposure.py exposure
  .venv\\Scripts\\python.exe scripts\\e6_exposure.py check
"""

import argparse

from memtk import config
from memtk.e6 import check, exposure, reference, rehearse
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "reference", "3D lesions, slice and patient rarity, T slice shares, H-eval nearest neighbours "
                                   "(no S)", reference.run_reference)
    add_stage(stages, "rehearse", "the E6 analysis run on E5's mock sets: H-cal (null) and H-cal + planted copies "
                                  "(positive); no S", rehearse.run_rehearse)
    add_stage(stages, "exposure", "the rehearsed E6 analysis on S (E5's stored tables, set main); needs E5's lock "
                                  "unchanged", exposure.run_exposure)
    add_stage(stages, "check", "consistency checks of all E6 outputs (exit code 1 on any failure)", check.run_check,
              overwrite=False)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
