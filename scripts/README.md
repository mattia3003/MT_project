# Pipeline scripts

One entry script per experiment, named by experiment ID, so alphabetical order is run order.
Building blocks (transforms, detectors, IO) live in `src/memtk/` and have no scripts of their own.
A script with several steps has stages (subcommands) that can be run and checked one at a time.
The script itself is only the command line: the stages live in a package per experiment
(`src/memtk/e0b/`, one module per topic; e.g. `e0b_audit.py qc` runs `memtk.e0b.qc.run_qc`), so
they can also be called from a notebook: `run_qc(load_config(), overwrite=True)`.

Run from the repository root. Every stage takes `--config` (default `configs/default.yaml`).
Dataset-specific stages also take `--dataset` (real data) or `--pool` (synthetic data), defaulting
to `default_real_dataset` / `default_synthetic_pool` in the config. Each stage writes a run log
(time, command, seed, full config, package versions) next to its outputs.

All commands, ready to copy, are in [Commands](#commands) below.

Status: ✅ implemented · ⏳ planned

| Script / stage | Status | Inputs | Outputs |
| --- | --- | --- | --- |
| `e0a_prepare_data.py split` | ✅ | raw real dataset | `splits/<dataset>_seed<seed>.csv` (patient → T / H-cal / H-eval), `.run_info.json` |
| `e0a_prepare_data.py real` | ✅ | raw real dataset, split CSV | `<output_root>/<dataset>/volumes/<pid>/{bravo,seg_soft,seg}.nii.gz` (128×128×D), `patients.csv`, `lesion_slices.csv`, `run_info.json` |
| `e0a_prepare_data.py synthetic` | ✅ | raw synthetic pool (read in place) | `<output_root>/<pool>/synthetic_index.csv` (one row per sample: ID, naming scheme, read/shape/binary checks, mask px, components, image range, legacy QC flag), `run_info.json` |
| `e0a_prepare_data.py check` | ✅ | outputs of the stages above, raw data | `results/<dataset>/e0a/check_report_<pool>.txt` (PASS/FAIL per check; exit code 1 on any failure), `check_grid_<pool>.png`, `check_<pool>.run_info.json` |
| `e0b_audit.py brain-review` | ✅ | E0a outputs (T only), synthetic pool | `results/<dataset>/e0b/brain_review/`: method comparison, example figures, near-blank sheet, tissue counts, proposal (`review.txt`) → **decision A** in the config |
| `e0b_audit.py brain` | ✅ | decision A, E0a outputs, synthetic pool | `<output_root>/<dataset>/brain.csv` + `brain_masks.npz` (all lesion slices); `<output_root>/<pool>/brain.csv` + `brain_masks.npz` + `masks_raw.npz` (all samples, failed-MRI flag); `results/<dataset\|pool>/e0b/brain/` (summary, outline sheet) |
| `e0b_audit.py lesions-review` | ✅ | E0a outputs, `masks_raw.npz` from `brain` | `results/<dataset>/e0b/lesions_review/`: component sizes, attach-or-drop counts for nearby settings, specks by gap, crop sheets, proposal (`review.txt`) → **decisions B, C** |
| `e0b_audit.py lesions` | ✅ | decisions B, C, `brain` outputs | `<output_root>/<dataset\|pool>/lesions.csv` (one row per lesion, ID + features), `slices.csv`, `labels.npz`; `volumes/<pid>/labels.nii.gz` (real); `results/<dataset\|pool>/e0b/lesions/summary.txt` + `overview.png` |
| `e0b_audit.py view` | ✅ | lesion outputs | helper: `view --sample <id>` or `view --patient <id> --z <z>` → `results/<dataset\|pool>/e0b/views/<key>.png` (slice at full size + its lesion rows); `--nifti` also writes a synthetic sample's image and labels for ITK-SNAP |
| `e0b_audit.py qc-review` | ✅ | `lesions` and `brain` outputs, synthetic index | `results/<pool>/e0b/qc_review/`: masks by number of isolated specks, exact counts, sheets per stratum, clean masks with a lesion outside the brain, proposal (`review.txt`) → **decision D** |
| `e0b_audit.py qc` | ✅ | decision D, `lesions` and `brain` outputs | `<output_root>/<pool>/qc.csv` (per sample: mask class, failed-MRI flag, combined QC class, edge and small-lesion flags); `results/<pool>/e0b/qc/` (summary, old-QC cross-table, edge check on the real sets, example sheet) |
| `e0b_audit.py stats` | ✅ | `lesions`, `qc`, `brain` outputs, `patients.csv`, real volumes | `results/<dataset>/e0b/stats/`: description table, native sizes, lesions per slice, lesion area, centroid density (pixels and brain-relative), bound-impact table (input for E3), adjacent-slice Dice, native vs model lesion area, brain size by slice height |
| `e0b_audit.py filters-review` | ✅ | `lesions`, `brain`, `qc` outputs, real `seg` volumes | `results/<dataset>/e0b/filters_review/`: each sampling filter on the masks as saved and cleaned, per set; the cost of applying it to the real sets; proposal (`review.txt`) → **decision E** |
| `e0b_audit.py check` | ✅ | all E0b outputs, synthetic pool | `results/<dataset>/e0b/check_report_<pool>.txt` (PASS/FAIL per check: decisions set and outputs made with them, lesion tables vs label maps, label maps vs masks as saved, caches vs source files, QC classes, unit tests; exit code 1 on any failure), `check_<pool>.run_info.json` |
| `e3_calibrate.py bound-review` | ✅ | E0b lesion tables (T, H-cal only), decision F, E0 `bound_impact_<pool>.csv` | `results/<dataset>/e3/bound_review/`: every H-cal lesion's best E2 match in T, chance-match rate by lesion area and tau_L, proposal (`review.txt`), cost of each bound, example matches → **decision G** |
| `e3_calibrate.py calibrate` | ✅ | decision G, E0b lesion tables (T, H-cal), E0 `bound_impact_<pool>.csv` | `<output_root>/<dataset>/e3/` (E1 and E2 tables of H-cal); `results/<dataset>/e3/calibrate/`: tau_D, tau_L, tau_C pooled and per stratum with bootstrap-by-patient intervals, E2 classes on H-cal, sensitivity, exclusions, histograms → **decision H** |
| `e3_calibrate.py freeze` | ✅ | decision H, the `calibrate` tables and `thresholds.csv` | `configs/frozen/e3_<dataset>.yaml`: tau_D, tau_L, tau_C per stratum with intervals, class rule, bound, detector settings, H-cal chance rates (read by E4 and E5); refuses if the stored tables and `thresholds.csv` disagree or the E2 shortcuts would not be exact |
| `e3_calibrate.py check` | ✅ | all E3 outputs, T and H-cal | `results/<dataset>/e3/check_report.txt` (PASS/FAIL: decisions, run logs reading only T and H-cal, frozen file reproduced, shortcuts against brute force, self-match on T, unit tests; exit code 1 on any failure) |
| `e4_validate.py plant` | ✅ | decision I (`validation.bases`, `validation.variants`), decision C, the frozen file (bound, minimum area), E0b lesion tables (T, H-eval), brain masks, real BRAVO volumes (figures) | `<output_root>/<dataset>/e4/` (planted label maps, `planted.csv`, `planted_lesions.csv`); `results/<dataset>/e4/plant/`: bases, variant summary, lesion fate and area change by size, variants grid, one sheet per variant, area-change figure, proposal (`review.txt`) → **decision K**; no detector runs |
| `e4_validate.py validate` | ✅ | decisions I (gate rule), J, K, `validation.accepted_deviation`, frozen file, T, H-eval, planted set | `<output_root>/<dataset>/e4/` (E1 and E2 tables of the planted set and H-eval, `heval_classes.csv`: E5's baseline); `results/<dataset>/e4/validate/`: per-mask outcomes, recall per variant with intervals, false positives per stratum, RQ2 counts, `gate.csv`, figures, `review.txt` ending in PASS / ACCEPTED / FAIL (E5 runs on PASS or ACCEPTED) |
| `e4_validate.py check` | ✅ | all E4 outputs, T and H-eval, the frozen file | `results/<dataset>/e4/check_report.txt` (PASS/FAIL: decisions, run logs reading only T, H-eval and the planted set, frozen file unchanged, planted set reproduced from the seed, P1 = bases, bases from T and added lesions from H-eval, gate and verdict reproduced from the stored tables, unit tests; exit code 1 on any failure) |
| `e5_audit.py prepare` | ✅ | decisions O, P, Q (`audit:`), frozen file, E3 H-cal tables and `sensitivity.csv`, E4 H-eval tables, T / H-cal / H-eval label maps and `seg` volumes; never S | `configs/frozen/e5_<dataset>.yaml` (sensitivity thresholds re-derived on H-cal and equal to E3's, tau_D* for H2, held-out maxima for near-exact copies, drop-only check; read by `score` and `analyze`); `<output_root>/<dataset>/e5/` (drop-only T labels, H-eval under every variant); `results/<dataset>/e5/prepare/` (`review.txt`, thresholds, maxima, H-eval rates per variant); refuses once S has been scored |
| `e5_audit.py rehearse` | ✅ | `prepare` outputs (same `audit:`), frozen files, T, H-cal, H-eval, E4 planted set; never S | the scoring and analysis code that will run on S, on mock query sets: `null` (H-cal: no excess expected) and `positive` (H-cal + planted P1, P2, P4: excess, right source patients, near-exact copies expected); `<output_root>/<dataset>/e5/rehearse/<case>/` (raw tables), `results/<dataset>/e5/rehearse/<case>/` (every table and figure `analyze` will write, `review.txt` with the outcome reading) |
| `e5_audit.py score` | ✅ | the E4 gate (PASS / ACCEPTED), `prepare` and `rehearse` made with the current `audit:` and frozen files, T, S (pool `qc.csv`, `labels.npz`, `masks_raw.npz`; main and noisy masks) | `<output_root>/<pool>/e5/`: raw tables of S (main and every variant), `score_log.csv`, `score.complete.json`; no rate shown; runs once (the marker locks `prepare`, `audit:` and itself) |
| `e5_audit.py analyze` | ✅ | `score` outputs and marker (the `audit:` section and both frozen files must equal those `score` recorded), H-eval baseline, T, pool labels and sample files (figures) | `results/<pool>/e5/analyze/`: every table and figure of the rehearsal, on S; `flagged.csv` (read by E6, E7, E8); `review.txt` ending in the mechanical outcome reading and the E8 exit count |
| `e5_audit.py check` | ✅ | all E5 outputs, the frozen files, `audit:` | `results/<pool>/e5/check_report.txt` (PASS/FAIL: decisions, run logs in order with S read by `score` only, the lock, one complete score run, `prepare` values and every `analyze` table reproduced with the same seed, unit tests; exit code 1 on any failure) |
| `e6_exposure.py reference` | ✅ | decisions R, V, W (`exposure:`), the frozen E3 file, T label maps, `lesions.csv`, model-space `seg` volumes, E4 `heval_classes.csv`; never S | `<output_root>/<dataset>/e6/`: 3D lesions of T (both methods) and the 2D → 3D map, slice rarity (E1, E2), patient features and rarity, T slice shares, H-eval nearest-neighbour counts; `results/<dataset>/e6/reference/` (`review.txt`, `reference.png`); ~3 min |
| `e6_exposure.py rehearse` | ✅ | `reference` outputs, E5's mock sets (null: H-cal; positive: H-cal + planted P1, P2, P4), H-cal and T (placement mocks: lesions moved to T lesion positions, built and scored here), E4 H-eval tables, the frozen E3 file; never S | `<output_root>/<dataset>/e6/rehearse/<case>/` (placement mocks); `results/<dataset>/e6/rehearse/<case>/`: exposure, concentration, margins, H3, H1, flag details, top patients, figures, `review.txt` ending in the reading of decision U (placement mocks also E5's rates, exploratory); ~45 s |
| `e6_exposure.py exposure` | ✅ | E5's S tables (set main; refuses if E5's lock changed), E6 reference tables, `rehearse` made with the current `exposure:` | `results/<pool>/e6/exposure/`: the rehearsed analysis on S, `review.txt` ending in the reading; ~15 s |
| `e6_exposure.py check` | ✅ | all E6 outputs, T, H-eval tables, E5's S tables | `results/<pool>/e6/check_report.txt` (PASS/FAIL; reproduces every reference and exposure table; exit code 1 on any failure); ~4 min |
| `e7_reports.py planted` | ✅ | decisions X, Y, Z (`reports:`), E4's planted set and tables, E4's H-eval tables and `fp_heval.csv`, T and H-eval label maps, the frozen E3 file; never S | `results/<dataset>/e7/planted/`: report table with verdicts, correctness per variant and the criteria, deletion test (planted controls, H-eval chance flags), taxonomy confusion, one example report per variant; ~30 s |
| `e7_reports.py reports` | ✅ | E5's S tables and `flagged.csv` (lock unchanged), E6 `exposure`, the planted criteria passed, E4 H-eval tables; config `reports:` | `results/<pool>/e7/reports/`: 76 match reports (`reports/`), `report_index.csv`, patient profiles of the most exposed patients (`patients/`, `patient_profiles.csv`), deletion test on S, taxonomy S vs H-eval, `review.txt`; ~7 min |
| `e7_reports.py check` | ✅ | all E7 outputs, the planted set, T, H-eval, E5's S tables | `results/<pool>/e7/check_report.txt` (PASS/FAIL; reproduces every planted table, the selection, verdicts, deletion test, taxonomy and profiles on S; exit code 1 on any failure); ~8 min |
| `e8_paired.py select` | ✅ | decisions AA, AB, AD (`paired:`), E7 `report_index.csv`, E5 `flagged.csv` and S tables (set main; refuses if E5's lock changed), pool `brain.csv`, `qc.csv`, `labels.npz`, T label maps, `patients.csv`; reads no MRI | `results/<pool>/e8/select/`: `pairs.csv` (group A, matched group B, comparison slices, alignment shifts, runner-up and Δz rows), `review.txt`; seconds |
| `e8_paired.py rehearse` | ✅ | decisions AB–AE (`paired:`), the frozen E3 file, T / H-eval / H-cal label maps, real `brain.csv`, `patients.csv`, model-space BRAVO and `seg` volumes, E4 `heval_classes.csv`, `planted.csv`, `planted_labels.npz`, `planted_outcomes.csv`, E5's H-cal mock tables (`e5/rehearse/null/`); never S | `results/<dataset>/e8/rehearse/{heval,null,positive}/`: `measures.csv` (the pairs with SSIM per region), `tests.csv`, `secondary.csv`, `ssim_groups.png`, `examples.png`; `review.txt` (readings; null must not read *supported*, positive must), `run_info.json`; ~20 s |
| `e8_paired.py paired` | ✅ | `select`'s `pairs.csv` and `rehearse`'s H-eval `measures.csv` (both made with the current `paired:`, rehearsal as required), E5's lock unchanged, S's MRIs and masks as saved (pool files via `synthetic_index.csv`), pool `labels.npz`, model-space BRAVO and `seg` volumes, E7 `report_index.csv` (figure captions), the frozen E3 file; the only stage that reads S's MRIs | `results/<pool>/e8/paired/`: `measures.csv`, `tests.csv`, `secondary.csv`, `within_a.csv`, `ssim_groups.png`, `within_a.png`, `examples.png`, `review.txt` ending in the reading; ~15 s |
| `e8_paired.py check` | ✅ | all E8 outputs, their inputs (re-reads S's MRIs to reproduce `paired`'s measures), E0b's brain masks | `results/<pool>/e8/check_report.txt` (PASS/FAIL; reproduces every `select`, `rehearse` and `paired` table and reading; exit code 1 on any failure); ~1 min |
| `r_report_figures.py method` | ✅ | E0b outputs, one patient's native volume, E4 planted set and tables, E7 planted reports, the frozen E3 file | `results/<pool>/report/method/`: figures 2-3, 2-4, 2-5, 2-7, 2-11 (PNG 300 dpi + PDF), `figures.csv`, `run_info.json`; ~20 s |
| `r_report_figures.py appendix` | ✅ | the stored tables of E0b-E8 (E5's lock checked), the S samples and T slices shown | `results/<pool>/report/appendix/`: B1-B9, C1-C4, D1-D9, E1-E6 at 300 dpi (PNG + PDF, cropped), `figures.csv`, `run_info.json`; ~50 s |
| `r_report_figures.py tables` | ✅ | the experiments' stored tables and check reports, the synthesis's decision register | `results/<pool>/report/tables/`: 18 LaTeX tables (booktabs, caption below), `tables.csv`, `run_info.json`; and `results/<pool>/report/appendix.tex` (the whole appendix); seconds |
| `r_report_figures.py check` | ✅ | the figure and table folders and run logs, the experiment stages' run logs | `results/<pool>/report/check_report.txt` (PASS/FAIL, 21 checks; exit code 1 on any failure); ~20 s |
| `r_report_figures.py results` | ✅ | the stored tables of E0b–E8 (E5's lock checked; nothing re-scored), one S sample and one T slice for figure 3.6 | `results/<pool>/report/results/`: figures 3-1 … 3-7 as PNG (300 dpi) and PDF, 16 cm wide; `figures.csv` (section, content, sources), `run_info.json`; ~1 min |

## Commands

Copy-paste from the repository root (PowerShell). Run order top to bottom; each stage refuses to
overwrite its outputs unless `--overwrite` is given. `--help` on any script or stage lists its
options, e.g. `.venv\Scripts\python.exe scripts\e0b_audit.py view --help`.

**Setup** (once; a fresh venv with the pinned versions that produced E0, see the header of
`requirements-lock.txt` for why `--no-deps`)
```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --no-deps -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
```

**E0a: prepare the data** (`--dataset`, `--pool` default to the config's defaults)
```powershell
.venv\Scripts\python.exe scripts\e0a_prepare_data.py split       # patient split (fixed for the project)
.venv\Scripts\python.exe scripts\e0a_prepare_data.py real        # real volumes -> model space (128x128xD)
.venv\Scripts\python.exe scripts\e0a_prepare_data.py synthetic   # index the synthetic pool
.venv\Scripts\python.exe scripts\e0a_prepare_data.py check       # consistency checks (exit code 1 on failure)
```

**E0b: audit and QC** (review stages give evidence; set the decision in the config, then run
the production stage)
```powershell
.venv\Scripts\python.exe scripts\e0b_audit.py brain-review       # evidence for decision A
.venv\Scripts\python.exe scripts\e0b_audit.py brain              # needs A: brain masks, boxes, failed-MRI flag
.venv\Scripts\python.exe scripts\e0b_audit.py lesions-review     # evidence for decisions B, C
.venv\Scripts\python.exe scripts\e0b_audit.py lesions            # needs B, C: lesion table, label maps, overview
.venv\Scripts\python.exe scripts\e0b_audit.py qc-review          # evidence for decision D (speck cut-off)
.venv\Scripts\python.exe scripts\e0b_audit.py qc                 # needs D: QC class per synthetic sample
.venv\Scripts\python.exe scripts\e0b_audit.py stats              # E0 description, figures, bound impact (a few min)
.venv\Scripts\python.exe scripts\e0b_audit.py filters-review     # evidence for decision E (filters for H-cal/H-eval)
.venv\Scripts\python.exe scripts\e0b_audit.py check              # consistency checks of all E0b outputs (exit code 1 on failure)
```
`check` also takes `--n-synthetic 300` (random samples whose cached masks are recomputed from the files).
`brain-review` also takes `--n-synthetic 2000` (samples for the method comparison) and
`--min-area 4` (provisional lesion size for that comparison).

**E3: calibration on H-cal** (T and H-cal only; S is never scored before the freeze)
```powershell
.venv\Scripts\python.exe scripts\e3_calibrate.py bound-review     # evidence for decision G (provisional tau_L, cut, bound)
.venv\Scripts\python.exe scripts\e3_calibrate.py calibrate        # needs G: E1 + E2 on H-cal, thresholds -> evidence for decision H
.venv\Scripts\python.exe scripts\e3_calibrate.py freeze           # needs H: write configs\frozen\e3_<dataset>.yaml for E4/E5
.venv\Scripts\python.exe scripts\e3_calibrate.py check            # consistency checks of all E3 outputs (exit code 1 on failure)
```
`check` also takes `--n-brute 20` (random H-cal slices compared with brute force) and `--n-self 50`
(random T slices for the self-match).

**E4: planted-copy validation, the gate** (T, H-eval and the frozen file only; never S)
```powershell
.venv\Scripts\python.exe scripts\e4_validate.py plant            # needs I: planted copies P1-P6, provenance, figures -> evidence for decision K
.venv\Scripts\python.exe scripts\e4_validate.py validate         # needs K: both detectors on planted set + H-eval, gate verdict
.venv\Scripts\python.exe scripts\e4_validate.py check            # consistency checks of all E4 outputs (exit code 1 on failure)
```

**E5: main audit, S scored once** (every analysis choice fixed first: `audit:` in the config,
decisions L–Q, and the E5 file from `prepare`; see `docs/e5-plan.md`)
```powershell
.venv\Scripts\python.exe scripts\e5_audit.py prepare             # no S: sensitivity thresholds, tau_D*, held-out maxima, H-eval per variant
.venv\Scripts\python.exe scripts\e5_audit.py rehearse            # no S: the S code on mock sets (H-cal; H-cal + planted copies), a few min
```
```powershell
.venv\Scripts\python.exe scripts\e5_audit.py score               # S once (main + noisy, every variant), ~1 h; raw tables only
.venv\Scripts\python.exe scripts\e5_audit.py analyze             # needs score: every table and figure, flagged.csv, the outcome reading
.venv\Scripts\python.exe scripts\e5_audit.py check               # consistency checks of all E5 outputs (exit code 1 on failure; re-runs the analysis, a few min)
```
`prepare` and `score` refuse to run once S has been scored, even with `--overwrite`. `score
--overwrite` only replaces the tables of a run that stopped before writing `score.complete.json`;
record such a re-run in `docs/process-log.md`.

**E6: patient exposure, copying vs placement, rarity, duplication** (every choice in `exposure:`,
decisions R, U, V, W; S only from E5's stored tables; see `docs/e6-e7-plan.md`)
```powershell
.venv\Scripts\python.exe scripts\e6_exposure.py reference         # no S: 3D lesions, slice/patient rarity, H-eval NN counts, ~3 min
.venv\Scripts\python.exe scripts\e6_exposure.py rehearse          # no S: the E6 analysis on mock sets (null, positive, 2 placement mocks), ~45 s
.venv\Scripts\python.exe scripts\e6_exposure.py exposure          # the rehearsed analysis on S (E5's stored tables; E5's lock checked), ~15 s
.venv\Scripts\python.exe scripts\e6_exposure.py check             # consistency checks of all E6 outputs (exit code 1 on failure), ~4 min
```

**E7: match reports** (every choice in `reports:`, decisions X, Y, Z; built on planted copies before S)
```powershell
.venv\Scripts\python.exe scripts\e7_reports.py planted           # no S: reports, correctness, deletion test, taxonomy on planted copies, ~30 s
.venv\Scripts\python.exe scripts\e7_reports.py reports           # match reports, deletion test, taxonomy on S (needs E6 exposure, planted passed), ~7 min
.venv\Scripts\python.exe scripts\e7_reports.py check             # consistency checks of all E7 outputs (exit code 1 on failure), ~8 min
```

**E8: paired synthetic MRI check** (every choice in `paired:`, decisions AA–AE; rehearsed before S's MRIs are read)
```powershell
.venv\Scripts\python.exe scripts\e8_paired.py select              # no MRI: group A, matched B, comparison slices, shifts, within-A rows
.venv\Scripts\python.exe scripts\e8_paired.py rehearse            # no S: the analysis on the H-eval reference, H-cal null, planted positive mock, ~20 s
.venv\Scripts\python.exe scripts\e8_paired.py paired              # the rehearsed analysis on S's MRIs (one run; E5's lock checked), ~15 s
.venv\Scripts\python.exe scripts\e8_paired.py check               # consistency checks of all E8 outputs (exit code 1 on failure), ~1 min
```

**Report figures** (compact redraws for the written report; register in `docs/report-synthesis.md`, section 3)
```powershell
.venv\Scripts\python.exe scripts\r_report_figures.py method                  # Method figures 2.3, 2.4 (E1), 2.5 (E2), 2.7, 2.11 (report), ~20 s
.venv\Scripts\python.exe scripts\r_report_figures.py results                 # Results figures 3.1-3.7, PNG + PDF, ~1 min
.venv\Scripts\python.exe scripts\r_report_figures.py appendix --overwrite    # the 14 appendix figures (B-E) at print size, PNG + PDF, ~40 s
.venv\Scripts\python.exe scripts\r_report_figures.py check                   # consistency checks of every report figure and table (exit code 1 on failure), ~20 s
.venv\Scripts\python.exe scripts\r_report_figures.py tables                  # the report's tables as LaTeX (booktabs), seconds
.venv\Scripts\python.exe scripts\r_report_figures.py results --overwrite     # redraw after a style change
```

**Look at one slice** (after `lesions`): full-size picture with its lesion table, written to
`results/<dataset|pool>/e0b/views/<key>.png`
```powershell
.venv\Scripts\python.exe scripts\e0b_audit.py view --sample file_2_1033            # synthetic sample
.venv\Scripts\python.exe scripts\e0b_audit.py view --sample file_2_1033 --nifti    # + image and labels for ITK-SNAP
.venv\Scripts\python.exe scripts\e0b_audit.py view --patient Mets_120 --z 40       # real slice (z 0-based)
```
For a real slice in ITK-SNAP, open `processed_data/model_space/<dataset>/volumes/<pid>/bravo.nii.gz`
with `labels.nii.gz` (or `seg.nii.gz`) as segmentation; ITK-SNAP slice = z + 1.

**Tests**
```powershell
.venv\Scripts\python.exe -m pytest           # all tests
.venv\Scripts\python.exe -m pytest -q tests\test_lesions.py   # one file
```

## Notes

**Review → decision → production.** E0b's review stages only produce evidence and a proposed
value. You set the value by hand in `configs/default.yaml` (with a dated comment); the production
stage that needs it refuses to run while it is still `null`.

The split is fixed for the whole project: `split` refuses to replace an existing split file
unless `--overwrite` is given (re-running with the same seed gives an identical file).

**A new synthetic pool** (e.g. from the master-thesis models) is a new entry under
`synthetic_pools` in the config, run with `--pool <name>`. The E0b decisions in the config are
global, and some depend on the data: `brain.low_threshold` (0.02) assumes skull-stripped images
with a near-zero background, and `qc.max_isolated_specks` (9) was read from this pool's speck
counts. For a new pool, re-run the review stages on it (`brain-review`, `qc-review`) before the
production stages. `brain-review`'s threshold-sensitivity table shows whether the threshold
still sits in a gap. If the evidence differs, add per-pool overrides under
`synthetic_pools.<name>` rather than changing the global values; the code reads only the global
values today, so that needs a small change where the stages call `decided(...)`.

Conventions:
- Slice index `z` is always the 0-based array index along the slice axis. ITK-SNAP numbers
  slices from 1: its slice 83 is our `z82`.
- Synthetic `sample_id` is the full file stem (e.g. `file_1033`, `1033`: different samples), never an int.
  Read pipeline tables with `memtk.io.read_table`, which keeps ID columns as text (plain
  `pd.read_csv` would turn an ID like `1033` into a number).
- Lesion IDs follow `memtk.ids`: `<slice key>/L<n>`, e.g. `Mets_005/z127/L1`, `file_2_1033/L2`.
- Real data is compared only in model space (128×128, per-volume min–max, area resize, mask > 0.01).
