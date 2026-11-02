# Pipeline scripts

One entry script per experiment, named by experiment ID, so alphabetical order is run order.
Building blocks (transforms, detectors, IO) live in `src/memtk/` and have no scripts of their own.
A script with several steps has stages (subcommands) that can be run and checked one at a time.

Run from the repository root. Every stage takes `--config` (default `configs/default.yaml`).
Dataset-specific stages also take `--dataset` (real data) or `--pool` (synthetic data), defaulting
to `default_real_dataset` / `default_synthetic_pool` in the config. Each stage writes a run log
(time, command, seed, full config, package versions) next to its outputs.

```powershell
.venv\Scripts\python.exe scripts\e0a_prepare_data.py split
```

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
| `e0b_audit.py lesions` / `view` | ⏳ | decisions B, C | lesion table, label maps, overview sheets; `view` shows one slice |
| `e0b_audit.py qc-review` / `qc` | ⏳ | lesion table | evidence for **decision D** (speck cut-off), then QC classes |
| `e0b_audit.py stats` / `check` | ⏳ | all E0b outputs | E0 statistics and figures; consistency checks |
| `e3_calibrate.py` | ⏳ | E0 outputs, H-cal | frozen thresholds (runs the E1 + E2 detectors from `memtk`) |
| `e4_validate.py` | ⏳ | thresholds, T, H-eval | planted-copy validation table (gate) |
| `e5_audit.py` | ⏳ | validated config, S | main audit tables |
| `e6_exposure.py` | ⏳ | E5 outputs | patient exposure, rarity, duplication |
| `e7_reports.py` | ⏳ | E5, E6 outputs, planted copies | match reports, deletion test, correctness |
| `e8_paired_mri.py` | ⏳ (optional) | E5 outputs, paired MRIs | paired-MRI similarity |

**Review → decision → production.** E0b's review stages only produce evidence and a proposed
value. You set the value by hand in `configs/default.yaml` (with a dated comment); the production
stage that needs it refuses to run while it is still `null`.

**Unit tests:** `.venv\Scripts\python.exe -m pytest` (in `tests/`).

The split is fixed for the whole project: `split` refuses to replace an existing split file
unless `--overwrite` is given (re-running with the same seed gives an identical file).

Conventions:
- Slice index `z` is always the 0-based array index along the slice axis. ITK-SNAP numbers
  slices from 1: its slice 83 is our `z82`.
- Synthetic `sample_id` is the full file stem (e.g. `file_1033`, `1033`: different samples), never an int.
- Real data is compared only in model space (128×128, per-volume min–max, area resize, mask > 0.01).
