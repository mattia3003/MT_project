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
| `e0a_prepare_data.py synthetic` | ⏳ | raw synthetic pool | `<output_root>/<pool>/synthetic_index.csv` |
| `e0a_prepare_data.py check` | ⏳ | outputs of the stages above | assertion report, `results/<dataset>/E0/preprocessing_grid.png` |
| `e0b_audit.py` | ⏳ | E0a outputs | lesion table, mask QC classes, E0 statistics |
| `e3_calibrate.py` | ⏳ | E0 outputs, H-cal | frozen thresholds (runs the E1 + E2 detectors from `memtk`) |
| `e4_validate.py` | ⏳ | thresholds, T, H-eval | planted-copy validation table (gate) |
| `e5_audit.py` | ⏳ | validated config, S | main audit tables |
| `e6_exposure.py` | ⏳ | E5 outputs | patient exposure, rarity, duplication |
| `e7_reports.py` | ⏳ | E5, E6 outputs, planted copies | match reports, deletion test, correctness |
| `e8_paired_mri.py` | ⏳ (optional) | E5 outputs, paired MRIs | paired-MRI similarity |

The split is fixed for the whole project: `split` refuses to replace an existing split file
unless `--overwrite` is given (re-running with the same seed gives an identical file).

Conventions:
- Slice index `z` is always the 0-based array index along the slice axis.
- Synthetic `sample_id` is the full file stem (e.g. `file_1033`, `1033`: different samples), never an int.
- Real data is compared only in model space (128×128, per-volume min–max, area resize, mask > 0.01).
