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
| `e4_validate.py check` | ⏳ (plan: `docs/e4-plan.md`) | all E4 outputs | consistency checks |
| `e5_audit.py` | ⏳ | validated config, S | main audit tables |
| `e6_exposure.py` | ⏳ | E5 outputs | patient exposure, rarity, duplication |
| `e7_reports.py` | ⏳ | E5, E6 outputs, planted copies | match reports, deletion test, correctness |
| `e8_paired_mri.py` | ⏳ (optional) | E5 outputs, paired MRIs | paired-MRI similarity |

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
```
`check` follows in step 6 (`docs/e4-plan.md`).

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
