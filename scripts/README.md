# Pipeline scripts

Run from the repository root, in numerical order. Every script takes `--config`
(default `configs/default.yaml`); dataset-specific scripts also take `--dataset` (real data) or
`--pool` (synthetic data), defaulting to `default_real_dataset` / `default_synthetic_pool` in the
config. Each script writes a `run_info.json` (time, command, seed, full config, package versions)
next to its outputs.

```powershell
.venv\Scripts\python.exe scripts\01_make_split.py
```

Status legend: ✅ implemented · ⏳ planned

| # | Script | Status | Inputs | Outputs |
| --- | --- | --- | --- | --- |
| 01 | `01_make_split.py` | ⏳ | raw real dataset (`real_datasets.<name>.root`) | `splits/<dataset>_seed<seed>.csv` (patient → T / H-cal / H-eval) |
| 02 | `02_preprocess_real.py` | ⏳ | raw real dataset, split CSV | `<output_root>/<dataset>/volumes/<pid>/{bravo,seg_soft,seg}.nii.gz` (128×128×D model space), `lesion_slices.csv` |
| 03 | `03_index_synthetic.py` | ⏳ | raw synthetic pool (`synthetic_pools.<name>.dir`) | `<output_root>/<pool>/synthetic_index.csv` |
| 04 | `04_check_preprocessing.py` | ⏳ | outputs of 01–03 | assertion report, `results/<dataset>/E0/preprocessing_grid.png` |

Conventions:
- Slice index `z` is always the 0-based array index along the slice axis.
- Synthetic `sample_id` is the full file stem (e.g. `file_1033`, `1033`: different samples), never an int.
- Real data is compared only in model space (128×128, per-volume min–max, area resize, mask > 0.01).
