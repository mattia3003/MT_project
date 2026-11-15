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
  (planned: exposure, check)

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\e6_exposure.py reference
  .venv\\Scripts\\python.exe scripts\\e6_exposure.py rehearse
"""

import argparse

from memtk import config
from memtk.e6 import reference, rehearse
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "reference", "3D lesions, slice and patient rarity, T slice shares, H-eval nearest neighbours "
                                   "(no S)", reference.run_reference)
    add_stage(stages, "rehearse", "the E6 analysis run on E5's mock sets: H-cal (null) and H-cal + planted copies "
                                  "(positive); no S", rehearse.run_rehearse)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
