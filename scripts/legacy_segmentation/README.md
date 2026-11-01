# Legacy: segmentation experiments (not part of the memorization project)

These scripts are from the start of the project, when the focus was training and evaluating
2D segmentation models (UNet / DynUNet) on real and synthetic data. They are kept for reference
only and are **not** used by the memorization pipeline (see `scripts/README.md`).

Path note: the scripts were written to be run from `scripts/`. Hardcoded relative defaults such as
`../processed_data/...` (in `preprocess_3d_to_2d.py`, `new_convert_synthetic_to_2d.py`,
`corrupt_data_detection.py`) now need one more `../` if run from this folder. The commands below
pass paths explicitly and are run from the repository root.

Known issues found during the 2026-11-01 review (why they are not reused):
- `preprocess_3d_to_2d.py` saves slices at native resolution (no scaling/resize to the 128x128
  model space), as float64, with 1-based slice numbers in the filenames (`slice001` = array index 0).
- `new_convert_synthetic_to_2d.py` only duplicates the raw synthetic files (they are already 2D),
  and silently excludes nothing if the exclusion file path is wrong.

## Command-line scripts

Run the commands below from the repository root. Paths may be absolute or relative to the current directory. The training and evaluation scripts expect NIfTI files under matching `bravo/` and `seg/` directories.

### `scripts/legacy_segmentation/unified_train_2d_unet.py`

Trains a 2D UNet, splitting validated BRAVO/SEG pairs into train and validation groups and saving the best checkpoint as `best_<dataset_name>_model.pth`.

| Command | Required CLI arguments | Optional CLI arguments |
| --- | --- | --- |
| Train UNet | `--dataset_name NAME --data_dir DATA_DIR` | `--epochs` (default: `40`), `--batch_size` (default: `64`), `--target_size HEIGHT WIDTH` (default: `256 256`), `--resize_mode {pad_crop,resize}` (default: `pad_crop`) |

```powershell
python scripts/legacy_segmentation/unified_train_2d_unet.py --dataset_name real --data_dir processed_data/StanfordSkullStripped_1mm_all/train
python scripts/legacy_segmentation/unified_train_2d_unet.py --dataset_name synthetic --data_dir processed_data/Synthetic_data_2d_all --epochs 40 --batch_size 64 --target_size 256 256 --resize_mode pad_crop
```

### `scripts/legacy_segmentation/unified_train_2d_dynunet.py`

Trains a 2D DynUNet with deep supervision, saving the best checkpoint as `best_<dataset_name>_model.pth`.

| Command | Required CLI arguments | Optional CLI arguments |
| --- | --- | --- |
| Train DynUNet | `--dataset_name NAME --data_dir DATA_DIR` | `--epochs` (default: `40`), `--batch_size` (default: `32`), `--target_size HEIGHT WIDTH` (default: `256 256`), `--resize_mode {pad_crop,resize}` (default: `pad_crop`) |

```powershell
python scripts/legacy_segmentation/unified_train_2d_dynunet.py --dataset_name real --data_dir processed_data/StanfordSkullStripped_1mm_all/train
python scripts/legacy_segmentation/unified_train_2d_dynunet.py --dataset_name synthetic --data_dir processed_data/Synthetic_data_2d_10_0 --epochs 40 --batch_size 32 --target_size 256 256 --resize_mode pad_crop
```

### `scripts/legacy_segmentation/evaluate_2d_unet.py`

Evaluates a trained 2D UNet on paired test data. The required input is `--data_dir`; this directory must contain matching `bravo/` and `seg/` files.

| Command | Required CLI arguments | Optional CLI arguments |
| --- | --- | --- |
| Evaluate UNet without saving predictions | `--data_dir DATA_DIR` | `--model_path MODEL.pth` (default: `best_synthetic_model_TS256_40epochs.pth`), `--threshold` (default: `0.5`), `--target_size HEIGHT WIDTH` (default: `256 256`), `--resize_mode {pad_crop,resize}` (default: `pad_crop`) |
| Evaluate UNet and save predictions | `--data_dir DATA_DIR --save_preds` | Same options as above |

```powershell
python scripts/legacy_segmentation/evaluate_2d_unet.py --data_dir processed_data/StanfordSkullStripped_1mm_all/test --model_path models/best_real_model_TS256_40epochs_unet.pth
python scripts/legacy_segmentation/evaluate_2d_unet.py --data_dir processed_data/StanfordSkullStripped_1mm_all/test --model_path models/best_real_model_TS256_40epochs_unet.pth --threshold 0.5 --save_preds
```

### `scripts/legacy_segmentation/evaluate_2d_dynunet.py`

Evaluates a trained 2D DynUNet on paired test data. The required input is `--data_dir`; this directory must contain matching `bravo/` and `seg/` files.

| Command | Required CLI arguments | Optional CLI arguments |
| --- | --- | --- |
| Evaluate DynUNet without saving predictions | `--data_dir DATA_DIR` | `--model_path MODEL.pth` (default: `best_synthetic_model_TS256_40epochs_dynunet_10_0.pth`), `--threshold` (default: `0.5`), `--target_size HEIGHT WIDTH` (default: `256 256`), `--resize_mode {pad_crop,resize}` (default: `pad_crop`) |
| Evaluate DynUNet and save predictions | `--data_dir DATA_DIR --save_preds` | Same options as above |

```powershell
python scripts/legacy_segmentation/evaluate_2d_dynunet.py --data_dir processed_data/StanfordSkullStripped_1mm_all/test --model_path models/best_real_model_TS256_40epochs_dynunet.pth
python scripts/legacy_segmentation/evaluate_2d_dynunet.py --data_dir processed_data/Synthetic_data_2d_10_0 --model_path scripts/best_synthetic_model_TS256_40epochs_dynunet_10_0.pth --threshold 0.5 --save_preds
```

### `scripts/legacy_segmentation/new_metrics.py`

This script evaluates image-quality and distribution metrics. `--real_dir` is required for every command. `--syn_dir` is required for comparisons between real and synthetic data. The metric flags can be combined, and `--all` selects every available metric.

| Command / metric | Required CLI arguments | Optional CLI arguments |
| --- | --- | --- |
| Inception Score | `--real_dir REAL_DIR --syn_dir SYN_DIR --is_score` | `--modality` (default: `bravo`), `--target_size HEIGHT WIDTH` (default: `256 256`), `--resize_mode {pad_crop,resize}` (default: `pad_crop`), `--exclusion_file` (default: `corrupted_files.txt`), `--batch_size` (default: `32`) |
| Inception Score for real data only | `--real_dir REAL_DIR --is_score` | Same as above; `--syn_dir` is optional and adds the synthetic score |
| Standard FID | `--real_dir REAL_DIR --syn_dir SYN_DIR --fid` | `--fid_samples`, `--seed` (default: `42`), plus the common options above |
| Medical FID (RadImageNet ResNet-50) | `--real_dir REAL_DIR --syn_dir SYN_DIR --med_fid` | `--med_fid_weights WEIGHTS`, `--fid_samples`, `--seed`, plus the common options above |
| CMMD (CLIP-MMD) | `--real_dir REAL_DIR --syn_dir SYN_DIR --cmmd` | `--cmmd_model` (default: `openai/clip-vit-base-patch32`), `--cmmd_sigma`, `--cmmd_block_size` (default: `256`), `--cmmd_samples`, `--seed`, plus the common options above |
| RadImageNet feature-space MMD | `--real_dir REAL_DIR --syn_dir SYN_DIR --rad_mmd` | `--med_fid_weights WEIGHTS`, `--rad_mmd_sigma`, `--rad_mmd_samples`, `--seed`, plus the common options above |
| SSIM and MS-SSIM | `--real_dir REAL_DIR --syn_dir SYN_DIR --ssim --ssim_paired` | Common options above |
| 1-Nearest Neighbor feature accuracy | `--real_dir REAL_DIR --syn_dir SYN_DIR --one_nn` | `--med_fid_weights WEIGHTS`, `--one_nn_samples`, `--seed`, plus the common options above |
| All metrics | `--real_dir REAL_DIR --syn_dir SYN_DIR --all` | All metric-specific options above; add `--ssim_paired` to enable paired SSIM/MS-SSIM |

Examples:

```powershell
python scripts/legacy_segmentation/new_metrics.py --real_dir processed_data/StanfordSkullStripped_1mm_all/test --syn_dir processed_data/Synthetic_data_2d_all --is_score
python scripts/legacy_segmentation/new_metrics.py --real_dir processed_data/StanfordSkullStripped_1mm_all/test --syn_dir processed_data/Synthetic_data_2d_all --fid --fid_samples 400
python scripts/legacy_segmentation/new_metrics.py --real_dir processed_data/StanfordSkullStripped_1mm_all/test --syn_dir processed_data/Synthetic_data_2d_all --all --ssim_paired --med_fid_weights models/radimagenet/ResNet50.pt
```

Common metric CLI arguments:

| CLI | Description |
| --- | --- |
| `--real_dir REAL_DIR` | Required real-data root; the script searches recursively for `<modality>/*.nii*`. |
| `--syn_dir SYN_DIR` | Optional synthetic-data root; required by comparative metrics. |
| `--modality MODALITY` | Modality folder to evaluate (default: `bravo`). |
| `--target_size HEIGHT WIDTH` | Spatial size used before calculating metrics (default: `256 256`). |
| `--resize_mode {pad_crop,resize}` | Use padding/cropping or interpolation (default: `pad_crop`). |
| `--exclusion_file FILE` | Corrupted-file list (default: `corrupted_files.txt`). |
| `--batch_size N` | Metric loader batch size (default: `32`). |
| `--seed N` | Reproducible subsampling seed (default: `42`). |
