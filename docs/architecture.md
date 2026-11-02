# Architecture

How the project fits together: the experiment pipeline, the code modules, and every file the
pipeline reads or writes. Implemented parts are drawn solid; planned parts dashed and grey.

The diagrams are written in [Mermaid](https://mermaid.js.org/). GitHub renders them directly.
In VS Code, install the extension *Markdown Preview Mermaid Support* (`bierner.markdown-mermaid`)
and open the Markdown preview.

**Keeping it current:** updating this file is part of finishing each experiment, together with
the status table in `scripts/README.md` and the entries in `docs/report-notes.md`.

*Status: E0a implemented; E0b in progress, `brain-review` and `brain` implemented (2026-11-02).*

---

## 1. Pipeline

One script per experiment; its steps are subcommands (stages). Every stage reads
`configs/default.yaml` (datasets, pools, seed 2026, parameters) and writes a run log next to its
outputs. Cylinders are data products, the diamond is the go/no-go gate. Dotted arrows inside E0b
are decisions you make by hand in the config, after viewing the evidence of a review stage.

```mermaid
flowchart TD
    RAW["Real dataset (data/)<br/>105 train + 51 test volumes"]
    POOL["Synthetic pool (data/)<br/>37,624 samples, 128×128×2"]

    subgraph E0A["E0a · e0a_prepare_data.py ✅"]
        SPLIT["split"]
        REAL["real"]
        SYN["synthetic"]
        CHECK["check"]
    end

    SPLITCSV[("Patient split<br/>T · H-cal · H-eval")]
    VOLS[("Model-space volumes 128×128×D<br/>patients.csv · lesion_slices.csv")]
    INDEX[("synthetic_index.csv")]
    REPORT[("Check report + sample grid")]

    RAW --> SPLIT --> SPLITCSV
    RAW --> REAL
    SPLITCSV --> REAL --> VOLS
    POOL --> SYN --> INDEX
    SPLITCSV & VOLS & INDEX --> CHECK --> REPORT

    subgraph E0B["E0b · e0b_audit.py (in progress)"]
        BREV["brain-review ✅"]
        BRAIN["brain ✅"]
        LREV["lesions-review"]:::planned
        LES["lesions · view"]:::planned
        QREV["qc-review"]:::planned
        QC["qc"]:::planned
        STATS["stats · check"]:::planned
    end
    BREV -.->|"decision A"| BRAIN
    BRAINOUT[("brain.csv · brain_masks.npz<br/>masks_raw.npz (synthetic)")]
    BRAIN --> BRAINOUT --> LREV
    LREV -.->|"decisions B, C"| LES
    LES --> QREV
    QREV -.->|"decision D"| QC
    QC --> STATS

    E3["E3 · e3_calibrate.py<br/>E1 + E2 detectors on H-cal"]:::planned
    FROZEN[("FREEZE: thresholds + bound<br/>S is never scored before here")]:::planned
    E4{"E4 · e4_validate.py<br/>planted copies: gate"}:::planned
    E5["E5 · e5_audit.py<br/>score S once"]:::planned
    E6["E6 · e6_exposure.py<br/>exposure · rarity · duplication"]:::planned
    E7["E7 · e7_reports.py<br/>match reports · deletion test"]:::planned
    E8["E8 · e8_paired_mri.py<br/>optional"]:::planned

    VOLS & INDEX --> BREV
    STATS --> E3 --> FROZEN --> E4
    E4 -->|pass| E5
    E4 -.->|"fail: adjust on H-cal, re-freeze"| E3
    E5 --> E6 & E7
    E6 --> E7
    E5 -.-> E8

    classDef planned stroke-dasharray: 5 5,color:#888
```

| Script | Stage | Status | Reads | Writes |
| --- | --- | --- | --- | --- |
| `e0a_prepare_data.py` | `split` | ✅ | raw real masks | `splits/<dataset>_seed<seed>.csv` |
| | `real` | ✅ | raw real volumes, split CSV | `volumes/`, `patients.csv`, `lesion_slices.csv` |
| | `synthetic` | ✅ | raw synthetic pool, legacy QC list | `synthetic_index.csv` |
| | `check` | ✅ | all of the above, raw data | `check_report_<pool>.txt`, `check_grid_<pool>.png` |
| `e0b_audit.py` | `brain-review` | ✅ | E0a outputs (T only), synthetic pool | `results/<dataset>/e0b/brain_review/` (evidence for decision A) |
| | `brain` | ✅ | decision A, E0a outputs, synthetic pool | `brain.csv`, `brain_masks.npz` (dataset and pool), `masks_raw.npz` (pool), `results/<dataset\|pool>/e0b/brain/` |
| | `lesions-review` | planned | T | evidence for decisions B, C |
| | `lesions`, `view` | planned | decisions B, C | lesion table, label maps, overview sheets |
| | `qc-review`, `qc` | planned | lesion table | evidence for decision D, QC classes |
| | `stats`, `check` | planned | all E0b outputs | E0 statistics; consistency checks |
| `e3_calibrate.py` | *tbd* | planned | E0b outputs, H-cal | frozen thresholds |
| `e4_validate.py` | *tbd* | planned | frozen thresholds, T, H-eval | validation table, planted copies |
| `e5_audit.py` | — | planned | frozen thresholds, S | main audit tables |
| `e6_exposure.py` | *tbd* | planned | E5 outputs | exposure, rarity, duplication |
| `e7_reports.py` | *tbd* | planned | E5, E6 outputs, planted copies | match reports, deletion test |
| `e8_paired_mri.py` | — | optional | E5 outputs, paired MRIs | paired-MRI similarity |

E1 (whole-mask detector) and E2 (lesion-level detector) have no scripts: they are `memtk`
modules, run inside E3 (on H-cal), E4 (on planted copies) and E5 (on S).

---

## 2. Code modules

Arrows mean "imports". `config` sits at the bottom: it is the only module that knows about the
YAML config and the repository root.

```mermaid
flowchart LR
    S_E0A["scripts/e0a_prepare_data.py"]
    S_E0B["scripts/e0b_audit.py"]

    subgraph MEMTK["src/memtk"]
        M_CHECKS["checks"]
        M_AUDIT["audit"]
        M_SPLIT["split"]
        M_SYN["synthetic"]
        M_PLOT["plotting"]
        M_LESIONS["lesions"]
        M_TRANSFORMS["transforms"]
        M_IO["io"]
        M_CONFIG["config"]
    end

    S_E0A --> M_CHECKS & M_SPLIT & M_SYN & M_TRANSFORMS & M_IO & M_CONFIG
    S_E0B --> M_AUDIT & M_LESIONS & M_SYN & M_IO & M_CONFIG
    M_CHECKS --> M_LESIONS & M_PLOT & M_SYN & M_TRANSFORMS & M_IO & M_CONFIG
    M_AUDIT --> M_LESIONS & M_PLOT
    M_SPLIT --> M_TRANSFORMS & M_IO
    M_SYN --> M_LESIONS & M_IO & M_CONFIG
    M_IO --> M_CONFIG
```

| Module | Purpose | Main functions | External libraries |
| --- | --- | --- | --- |
| `config` | Load the YAML config, look up datasets and pools, resolve paths from the repo root, write run logs | `load_config`, `real_dataset`, `synthetic_pool`, `output_dir`, `results_dir`, `split_file`, `write_run_info` | PyYAML |
| `io` | Read and write NIfTI; locate patients and their files; mask stacks (many 2D boolean masks in one bit-packed `.npz`, looked up by key) | `load_image`, `save_nifti`, `list_patients`, `patient_files`, `MaskStackWriter`, `MaskStack` | nibabel |
| `transforms` | The original training transform (model space) and slice counts | `model_space_transform`, `convert_volume_pair`, `binarize_mask`, `lesion_px_per_slice` | MONAI, PyTorch |
| `split` | Seeded matched-pair split into T / H-cal / H-eval | `make_split`, `paired_split`, `summarize_split` | pandas |
| `synthetic` | Read synthetic samples in place; sample IDs; index a pool; parse the legacy QC list | `index_pool`, `load_sample`, `sample_id`, `name_scheme`, `read_legacy_exclusions` | — |
| `lesions` | 8-connected labelling; brain masks (six candidate methods), tissue area, brain box. Attach-or-drop and lesion features follow in the E0b `lesions` stage. | `label_components`, `count_components`, `brain_mask`, `tissue_px`, `brain_box`, `fraction_outside` | SciPy, scikit-image |
| `checks` | E0a consistency checks, independent reference transform, sample grid | `CheckReport`, `check_split`, `check_real_volumes`, `check_recompute`, `check_synthetic_index`, `sample_grid` | PyTorch, matplotlib |
| `audit` | Evidence for the E0b review stages (candidate comparisons, figures, proposals) and overviews of the production stages | `evaluate_brain_methods`, `summarize_brain_methods`, `propose_brain_method`, `method_grid`, `threshold_sensitivity`, `summarize_threshold_sensitivity`, `sorted_sheet`, `tissue_counts`, `tissue_histogram`, `summarize_brains`, `brain_sheet` | pandas, matplotlib |
| `plotting` | Shared figure helpers (no pyplot, so importing memtk never changes the plotting backend). `save` wraps the figure title to the figure width and uses the constrained layout, so the title never covers tile titles or runs off the image | `tile_figure`, `show_slice`, `hide_unused`, `save` | matplotlib |

Tests (pytest, on small hand-built inputs): `tests/test_lesions.py` covers labelling, brain
masks, tissue area and the brain box; `tests/test_io.py` covers the mask-stack round trip.
Run with `.venv\Scripts\python.exe -m pytest`.

**Planned modules** (from the experiment plan; names may change when each experiment is planned):

| Module | For | Purpose |
| --- | --- | --- |
| `qc` | E0b | Synthetic mask quality classes (clean / minor / noisy / no_lesion / mri_failed) |
| `detectors/whole_mask` | E1 | Dice, top-k candidates, shift search |
| `detectors/lesion_level` | E2 | Lesion score s(a, b), Hungarian C(q, t), prefilters, the three output tables |
| `calibration` | E3 | Percentile thresholds, informative-lesion bound |
| `planted` | E4 | Planted-copy variants P1–P6 |
| `stats` | E5, E6 | Bootstrap CIs, Mann-Whitney / AUC, permutation tests |
| `exposure` | E6 | Exposure ratio, rarity, 3D lesion span |
| `reports` | E7 | Match-report figures and verdict lines |

---

## 3. Data dictionary

Placeholders: `<dataset>` = a `real_datasets` entry (now `stanford_brainmets`), `<pool>` = a
`synthetic_pools` entry (now `amibekk_v1`), `<output_root>` = `processed_data/model_space`.
"In git" says whether the file is version-controlled; files that are not can be regenerated
with the stage that writes them.

### Config

| File | Written by | Read by | In git |
| --- | --- | --- | --- |
| `configs/default.yaml` | hand-edited | every stage | yes |

Keys: `seed`; `output_root`, `results_root`, `splits_dir`; `default_real_dataset`,
`default_synthetic_pool`; `preprocessing` (`spatial_size`, `interpolation`, `mask_threshold`,
`slice_axis`); `brain.low_threshold` (data-dependent, checked by `brain-review`); E0b
decisions, `null` until decided: `brain` (`method`, `min_tissue_px`),
`lesions` (`connectivity`, `min_lesion_area`, `attach_gap`), `qc` (`max_isolated_specks`,
`edge_outside_frac`, `small_lesion_px`); `real_datasets.<name>` (`root`, `image_file`, `mask_file`, `train_split`,
`heldout_split`, `heldout_sets`); `synthetic_pools.<name>` (`dir`, `glob`, `image_channel`,
`mask_channel`, `trained_on`, `legacy_exclusion_file`).

### Patient split: `splits/<dataset>_seed<seed>.csv`

Written by `split`, read by `real` and `check` (and every later experiment). In git. One row
per patient.

| Column | Meaning |
| --- | --- |
| `patient_id` | Patient folder name, e.g. `Mets_005` |
| `orig_split` | Source folder: `train` or `test` |
| `set` | `T` (training reference), `H-cal` or `H-eval` |
| `n_lesion_slices` | Slices with any lesion voxel, at native resolution |

### Model-space volumes: `<output_root>/<dataset>/volumes/<patient_id>/`

Written by `real`, read by `check` (later: E0b onwards). Not in git. Shape (128, 128, D), with
D = the native number of slices. Identity affine: the array index (i, j, z) is the only
coordinate system, and `z` is 0-based.

| File | dtype | Content |
| --- | --- | --- |
| `bravo.nii.gz` | float32 | MRI after per-volume min-max scaling and area resize, in [0, 1] |
| `seg_soft.nii.gz` | float32 | Mask after the same transform, before thresholding, in [0, 1] |
| `seg.nii.gz` | uint8 | Binary mask: `seg_soft > 0.01` |

### `<output_root>/<dataset>/patients.csv`

Written by `real`, read by `check`. Not in git. One row per patient.

| Column | Meaning |
| --- | --- |
| `patient_id`, `orig_split`, `set` | As in the split CSV (`set` is a copy, checked by `check`) |
| `native_dim0`, `native_dim1` | Native in-plane size (sets how much the slices were squeezed) |
| `n_slices` | Native number of slices = model-space depth D |
| `n_lesion_slices_native`, `n_lesion_slices_model` | Lesion slices before and after conversion |
| `lesion_px_native`, `lesion_px_model` | Total lesion pixels before and after conversion |

### `<output_root>/<dataset>/lesion_slices.csv`

Written by `real`, read by `check` (later: E0b onwards). Not in git. One row per slice with a
lesion in either space.

| Column | Meaning |
| --- | --- |
| `dataset`, `patient_id`, `orig_split`, `set` | Identity of the slice's patient |
| `z` | 0-based slice index = array index in the volume |
| `lesion_px_native`, `lesion_px_model` | Lesion pixels in this slice before and after conversion |

### `<output_root>/<pool>/synthetic_index.csv`

Written by `synthetic`, read by `check` (later: E0b onwards). Not in git. One row per synthetic
file. The files themselves are read in place from the pool folder, never copied.

| Column | Meaning |
| --- | --- |
| `pool` | Pool name from the config |
| `sample_id` | Full file stem; never parse it to a number |
| `filename` | File name in the pool folder |
| `name_scheme` | Naming pattern, e.g. `file_2_<n>` (most likely the generation round) |
| `read_ok` | False if the file could not be read as (H, W, channels) |
| `shape`, `shape_ok` | In-plane shape, and whether it equals `spatial_size` |
| `mask_binary` | Mask values only 0 and 1 |
| `mask_px` | Mask pixels (−1 if unreadable) |
| `n_components_8conn` | 8-connected mask components, before any QC (−1 if unreadable) |
| `img_min`, `img_max`, `img_mean` | Intensity range of the synthetic MRI (NaN if unreadable) |
| `legacy_excluded`, `legacy_reason` | Flag and reason from the old QC list; for comparison only, never a filter |
| `read_error` | Error message if `read_ok` is False; empty otherwise |

### Brain masks: `<output_root>/<dataset>/brain.csv` and `<output_root>/<pool>/brain.csv`

Written by `brain` with the decided `brain.method`, read by later E0b stages. Not in git. One row
per real lesion slice (all sets) or per readable synthetic sample.

| Column | Meaning |
| --- | --- |
| `key` (real) | Slice key `<patient_id>/z<z>`, e.g. `Mets_005/z127`; the key in `brain_masks.npz` |
| `patient_id`, `set`, `z` (real) | Identity of the slice |
| `sample_id` (synthetic) | Full file stem; the key in `brain_masks.npz` and `masks_raw.npz` |
| `brain_px` | Area of the brain mask |
| `tissue_px` | Brain tissue area (`memtk.lesions.tissue_px`; equals `brain_px` with `low_threshold_all`) |
| `box_r0`, `box_r1`, `box_c0`, `box_c1` | Inclusive brain box (rows, columns); −1 if the slice has no brain |
| `mri_failed` (synthetic) | `tissue_px < brain.min_tissue_px`: a near-blank MRI without brain |

### Mask stacks: `brain_masks.npz`, `masks_raw.npz`

Written by `brain`, read with `memtk.io.MaskStack` (`stack[key]` gives a boolean 128×128 mask).
Not in git. Arrays `keys`, `packed` (1 bit per pixel) and `shape`.

| File | Keys | Content |
| --- | --- | --- |
| `<output_root>/<dataset>/brain_masks.npz` | `<patient_id>/z<z>` | Brain mask of every lesion slice |
| `<output_root>/<pool>/brain_masks.npz` | `sample_id` | Brain mask of every synthetic sample |
| `<output_root>/<pool>/masks_raw.npz` | `sample_id` | The synthetic mask as saved (binary, specks included); never modified |

Run logs next to them: `brain.run_info.json`.

### Results: `results/<dataset>/e0b/brain/` and `results/<pool>/e0b/brain/`

Written by `brain`. In git.

| File | Content |
| --- | --- |
| `summary.txt` | Per set (real) or per failed/ok group (synthetic): slices, slices without brain, brain-area 1st/5th percentile and median, median box size; failed-MRI count |
| `outlines.png` | Brain outline (cyan), box (dashed yellow) and mask (red). Real: random lesion slices per set. Synthetic: random samples, the smallest non-failed brains, failed MRIs |
| `run_info.json` | Run log |

### Results: `results/<dataset>/e0a/`

Written by `check`. In git.

| File | Content |
| --- | --- |
| `check_report_<pool>.txt` | One PASS/FAIL line per check, with counts and failing examples |
| `check_grid_<pool>.png` | Random lesion slices of T, H-cal and H-eval, and random synthetic samples, with mask outlines |
| `check_<pool>.run_info.json` | Run log |

### Results: `results/<dataset>/e0b/brain_review/`

Written by `brain-review`. In git. Evidence for decision A; nothing here is used by later stages.

| File | Content |
| --- | --- |
| `review.txt` | Threshold sensitivity (does `brain.low_threshold` sit in a gap?), method comparison, proposed `brain.method` with its rule, tissue-cut counts, what to set next |
| `threshold_sensitivity_<pool>.csv` | Per group (T, S sample) and multiple of `brain.low_threshold` (0.25–3×): slices, median and 95th-percentile absolute change of the brain area |
| `brain_methods_<pool>.csv` | Per group (T, S sample) and method: lesions (≥ provisional min area), share more than half outside / partly outside the brain, brain-area 1st/5th percentile and median |
| `methods_real.png` | T slices where plain Otsu fails, plus random ones; one column per method; cyan = brain, red = lesion |
| `methods_synthetic_<pool>.png` | The same for small synthetic brains, synthetic lesions outside, and random samples |
| `near_blank_<pool>.png` | Synthetic MRIs with the least brain tissue, samples just above candidate cuts, and the T slices with the least tissue |
| `tissue_hist_<pool>.png` | Tissue area of all synthetic samples vs T lesion slices (log axes) |
| `tissue_counts_<pool>.csv` | For each candidate `min_tissue_px`: synthetic samples below it, and T lesion slices below it |
| `run_info.json` | Run log |

### Run logs: `run_info.json` and `*.run_info.json`

Written by every stage next to its outputs. Fields: `timestamp`, `script`, `argv`, `seed`,
`python`, `packages` (versions of memtk, numpy, scipy, scikit-image, nibabel, monai, torch,
pandas), `config` (the full config used), and stage-specific extras such as `stage` and `pool`.
