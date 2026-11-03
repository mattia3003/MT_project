# Pipeline scripts

One entry script per experiment, named by experiment ID, so alphabetical order is run order.
Building blocks (transforms, detectors, IO) live in `src/memtk/` and have no scripts of their own.
A script with several steps has stages (subcommands) that can be run and checked one at a time.

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
| `e0b_audit.py check` | ⏳ | all E0b outputs | consistency checks |
| `e3_calibrate.py` | ⏳ | E0 outputs, H-cal | frozen thresholds (runs the E1 + E2 detectors from `memtk`) |
| `e4_validate.py` | ⏳ | thresholds, T, H-eval | planted-copy validation table (gate) |
| `e5_audit.py` | ⏳ | validated config, S | main audit tables |
| `e6_exposure.py` | ⏳ | E5 outputs | patient exposure, rarity, duplication |
| `e7_reports.py` | ⏳ | E5, E6 outputs, planted copies | match reports, deletion test, correctness |
| `e8_paired_mri.py` | ⏳ (optional) | E5 outputs, paired MRIs | paired-MRI similarity |

## Commands

Copy-paste from the repository root (PowerShell). Run order top to bottom; each stage refuses to
overwrite its outputs unless `--overwrite` is given. `--help` on any script or stage lists its
options, e.g. `.venv\Scripts\python.exe scripts\e0b_audit.py view --help`.

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
```
`brain-review` also takes `--n-synthetic 2000` (samples for the method comparison) and
`--min-area 4` (provisional lesion size for that comparison).

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

Conventions:
- Slice index `z` is always the 0-based array index along the slice axis. ITK-SNAP numbers
  slices from 1: its slice 83 is our `z82`.
- Synthetic `sample_id` is the full file stem (e.g. `file_1033`, `1033`: different samples), never an int.
  Read pipeline tables with `memtk.io.read_table`, which keeps ID columns as text (plain
  `pd.read_csv` would turn an ID like `1033` into a number).
- Lesion IDs follow `memtk.ids`: `<slice key>/L<n>`, e.g. `Mets_005/z127/L1`, `file_2_1033/L2`.
- Real data is compared only in model space (128×128, per-volume min–max, area resize, mask > 0.01).
