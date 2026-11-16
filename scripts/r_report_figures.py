"""Report figures: compact, print-sized versions of the project's figures for the written report
(`docs/report-synthesis.md`, section 3, figure register).

Every figure is redrawn from tables the experiments stored; nothing is recomputed or re-scored, and S
is read only through E5's stored tables (E5's lock is checked). Figures are 16 cm wide, saved as PNG
(300 dpi) and PDF. The stages live in `src/memtk/report_figures/`; this script is only the command line.

Stages (the name sorts after e8_, as this runs last):
  method     (needs E0b brain, lesions, qc; E4 plant, validate; E7 planted) Method figures: preprocessing and
             QC, E1's shift search, E2's lesion pairs, the planted variants, the match report
             -> results/<pool>/report/method/
  appendix   (needs the stages of E0b-E8) the appendix figures at 300 dpi, mostly the experiments' own figures
             redrawn from their stored tables (E5's lock checked)
             -> results/<pool>/report/appendix/
  tables     (needs every experiment's analysis outputs) the report's tables as LaTeX (booktabs), straight from
             the stored tables, with placeholder captions
             -> results/<pool>/report/tables/
  check      consistency checks: run logs current and later than their inputs, E5's lock, every listed figure
             present and nothing unlisted, every named source path present, the unit tests; exit code 1 on failure
             -> results/<pool>/report/check_report.txt
  results    (needs E0b stats, E3 bound-review, E4 validate, E5 analyze, E6 exposure, E7 reports, E8 paired)
             the figures for Results 3.1-3.7
             -> results/<pool>/report/results/

Examples (from the repository root):
  .venv\\Scripts\\python.exe scripts\\r_report_figures.py method
  .venv\\Scripts\\python.exe scripts\\r_report_figures.py results
  .venv\\Scripts\\python.exe scripts\\r_report_figures.py appendix
  .venv\\Scripts\\python.exe scripts\\r_report_figures.py check
"""

import argparse

from memtk import config
from memtk.report_figures import appendix, check, method, results, tables
from memtk.stages import add_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)

    add_stage(stages, "method", "Method figures: preprocessing and QC, detectors and report, planted variants",
              method.run_method)
    add_stage(stages, "results", "the figures for the report's Results sections 3.1-3.7", results.run_results)
    add_stage(stages, "appendix", "the appendix figures at 300 dpi (the experiments' own figures, redrawn)",
              appendix.run_appendix)
    add_stage(stages, "tables", "the report's tables as LaTeX (booktabs) from the stored tables", tables.run_tables)
    add_stage(stages, "check","consistency checks of every report figure (exit code 1 on any failure)", check.run_check,
              overwrite=False)

    args = parser.parse_args()
    options = {k: v for k, v in vars(args).items() if k not in ("stage", "run", "config")}
    args.run(config.load_config(args.config), **options)


if __name__ == "__main__":
    main()
