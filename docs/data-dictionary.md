# Data dictionary

Every file the pipeline reads or writes, in the order the pipeline runs: inputs, then E0a, then
E0b (processed data first, then results), then run logs. For each file: where it is, which stage
writes it, which stages read it, whether it is in git, and its columns. The short overview of
how everything fits together is `docs/architecture.md`; the code modules are described in
`docs/module-reference.md`.

Placeholders: `<dataset>` = a `real_datasets` entry (now `stanford_brainmets`), `<pool>` = a
`synthetic_pools` entry (now `amibekk_v1`), `<output_root>` = `processed_data/model_space`.
"In git" says whether the file is version-controlled; files that are not can be regenerated
with the stage that writes them. Read every pipeline CSV with `memtk.io.read_table`, which keeps
ID columns as text.

## Index: which stage reads and writes what

| Script | Stage | Status | Reads | Writes |
| --- | --- | --- | --- | --- |
| `e0a_prepare_data.py` | `split` | ✅ | raw real masks | `splits/<dataset>_seed<seed>.csv` |
| | `real` | ✅ | raw real volumes, split CSV | `volumes/`, `patients.csv`, `lesion_slices.csv` |
| | `synthetic` | ✅ | raw synthetic pool, legacy QC list | `synthetic_index.csv` |
| | `check` | ✅ | all of the above, raw data | `check_report_<pool>.txt`, `check_grid_<pool>.png` |
| `e0b_audit.py` | `brain-review` | ✅ | E0a outputs (T only), synthetic pool | `results/<dataset>/e0b/brain_review/` (evidence for decision A) |
| | `brain` | ✅ | decision A, E0a outputs, synthetic pool | `brain.csv`, `brain_masks.npz` (dataset and pool), `masks_raw.npz` (pool), `results/<dataset\|pool>/e0b/brain/` |
| | `lesions-review` | ✅ | E0a outputs, `masks_raw.npz` | `results/<dataset>/e0b/lesions_review/` (evidence for decisions B, C) |
| | `lesions` | ✅ | decisions B, C, `brain` outputs, real `seg` volumes | `lesions.csv`, `slices.csv`, `labels.npz` (dataset and pool), `volumes/<pid>/labels.nii.gz`, `results/<dataset\|pool>/e0b/lesions/` |
| | `view` (helper) | ✅ | lesion outputs | `results/<dataset\|pool>/e0b/views/<key>.png`; with `--nifti`, a synthetic sample's image and labels as NIfTI |
| | `qc-review` | ✅ | `lesions` and `brain` outputs, synthetic index | `results/<pool>/e0b/qc_review/` (evidence for decision D) |
| | `qc` | ✅ | decision D, `lesions` and `brain` outputs | `qc.csv` (pool), `results/<pool>/e0b/qc/` |
| | `stats` | ✅ | `lesions`, `qc`, `brain` outputs, `patients.csv`, real volumes (model space and native masks) | `results/<dataset>/e0b/stats/` |
| | `filters-review` | ✅ | `lesions`, `brain`, `qc` outputs, real `seg` volumes | `results/<dataset>/e0b/filters_review/` (evidence for decision E) |
| | `check` | ✅ | all E0b outputs, synthetic pool | `results/<dataset>/e0b/check_report_<pool>.txt` |
| `e3_calibrate.py` | `bound-review` | ✅ | E0b lesion tables (T, H-cal only), decision F, E0 `bound_impact_<pool>.csv` | `results/<dataset>/e3/bound_review/` (evidence for decision G) |
| | `calibrate` | ✅ | decision G, E0b lesion tables (T, H-cal), E0 `bound_impact_<pool>.csv` | `<output_root>/<dataset>/e3/` (E1 and E2 tables of H-cal), `results/<dataset>/e3/calibrate/` (evidence for decision H) |
| | `freeze`, `check` | planned (`docs/e1-e3-plan.md`) | decision H, the `calibrate` tables | `configs/frozen/e3_<dataset>.yaml`, check report |
| `e4_validate.py` | *tbd* | planned | frozen thresholds, T, H-eval | validation table, planted copies |
| `e5_audit.py` | — | planned | frozen thresholds, S | main audit tables |
| `e6_exposure.py` | *tbd* | planned | E5 outputs | exposure, rarity, duplication |
| `e7_reports.py` | *tbd* | planned | E5, E6 outputs, planted copies | match reports, deletion test |
| `e8_paired_mri.py` | — | optional | E5 outputs, paired MRIs | paired-MRI similarity |

E1 (whole-mask detector) and E2 (lesion-level detector) have no scripts: they are `memtk`
modules, run inside E3 (on H-cal), E4 (on planted copies) and E5 (on S).

---

## 1. Inputs

### Config: `configs/default.yaml`

Hand-edited, read by every stage. In git.

Keys: `seed`; `output_root`, `results_root`, `splits_dir`; `default_real_dataset`,
`default_synthetic_pool`; `preprocessing` (`spatial_size`, `interpolation`, `mask_threshold`,
`slice_axis`); `brain.low_threshold` (data-dependent, checked by `brain-review`); E0b
decisions, `null` until decided: `brain` (`method`, `min_tissue_px`),
`lesions` (`connectivity`, `min_lesion_area`, `attach_gap`), `qc` (`max_isolated_specks`,
`edge_outside_frac`, `small_lesion_px`); `filters` (`max_mask_px`, the sampling script's
pixel-sum limit; `real_sets`, decision E); E1–E3 (`docs/e1-e3-plan.md`): `detectors.whole_mask`
(`top_k_dice`, `top_k_centroid`, `max_shift_px`), `detectors.lesion` (`position_frame`,
decision F; `sigma`, `prefilter` as fractions of the 128-px field; `candidate_min_s`),
`calibration` (`percentile`, `bootstrap_draws`; decision G: `provisional_tau_L`,
`chance_rate_max`, `informative_bound_px`; decision H: `strata.whole_mask`, `strata.lesion`, the
latter for tau_L and tau_C);
`real_datasets.<name>` (`root`, `image_file`, `mask_file`, `train_split`,
`heldout_split`, `heldout_sets`); `synthetic_pools.<name>` (`dir`, `glob`, `image_channel`,
`mask_channel`, `trained_on`, `legacy_exclusion_file`).

### Raw data (read in place, never modified; not in git)

| Input | Where (config key) | Content |
| --- | --- | --- |
| Real dataset | `real_datasets.<dataset>.root`, folders `train_split` / `heldout_split`, one folder per patient | `image_file` (BRAVO) and `mask_file` (segmentation) per patient, native resolution |
| Synthetic pool | `synthetic_pools.<pool>.dir`, files matching `glob` | One NIfTI per sample, shape (128, 128, 2): channel `image_channel` = synthetic MRI, `mask_channel` = synthetic mask |
| Old QC list | `synthetic_pools.<pool>.legacy_exclusion_file` (in git, `docs/`) | Masks flagged by the earlier check, with a reason; comparison only |

---

## 2. E0a: prepared data (`e0a_prepare_data.py`)

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

Written by `real` (and `labels.nii.gz` by the E0b stage `lesions`), read by `check` and the E0b
stages. Not in git. Shape (128, 128, D), with D = the native number of slices. Identity affine:
the array index (i, j, z) is the only coordinate system, and `z` is 0-based.

| File | dtype | Content |
| --- | --- | --- |
| `bravo.nii.gz` | float32 | MRI after per-volume min-max scaling and area resize, in [0, 1] |
| `seg_soft.nii.gz` | float32 | Mask after the same transform, before thresholding, in [0, 1] |
| `seg.nii.gz` | uint8 | Binary mask: `seg_soft > 0.01` |
| `labels.nii.gz` | uint16 | Written by `lesions`: each slice's lesion numbers (L1 = 1, ...), to open as a segmentation next to `bravo.nii.gz` in ITK-SNAP |

### `<output_root>/<dataset>/patients.csv`

Written by `real`, read by `check` and `stats`. Not in git. One row per patient.

| Column | Meaning |
| --- | --- |
| `patient_id`, `orig_split`, `set` | As in the split CSV (`set` is a copy, checked by `check`) |
| `native_dim0`, `native_dim1` | Native in-plane size (sets how much the slices were squeezed) |
| `n_slices` | Native number of slices = model-space depth D |
| `n_lesion_slices_native`, `n_lesion_slices_model` | Lesion slices before and after conversion |
| `lesion_px_native`, `lesion_px_model` | Total lesion pixels before and after conversion |

### `<output_root>/<dataset>/lesion_slices.csv`

Written by `real`, read by `check` and the E0b stages. Not in git. One row per slice with a
lesion in either space.

| Column | Meaning |
| --- | --- |
| `dataset`, `patient_id`, `orig_split`, `set` | Identity of the slice's patient |
| `z` | 0-based slice index = array index in the volume |
| `lesion_px_native`, `lesion_px_model` | Lesion pixels in this slice before and after conversion |

### `<output_root>/<pool>/synthetic_index.csv`

Written by `synthetic`, read by `check` and the E0b stages. Not in git. One row per synthetic
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

### Results: `results/<dataset>/e0a/`

Written by `check`. In git.

| File | Content |
| --- | --- |
| `check_report_<pool>.txt` | One PASS/FAIL line per check, with counts and failing examples |
| `check_grid_<pool>.png` | Random lesion slices of T, H-cal and H-eval, and random synthetic samples, with mask outlines |
| `check_<pool>.run_info.json` | Run log |

---

## 3. E0b: processed data (`e0b_audit.py`), the caches every later experiment reads

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

### Lesion tables: `<output_root>/<dataset|pool>/lesions.csv`, `slices.csv`, `labels.npz`

Written by `lesions` with the decided `lesions.min_lesion_area` and `lesions.attach_gap`, read
by QC, `stats` and every later experiment. Not in git. Real data: all lesion slices of T, H-cal
and H-eval. Synthetic: every sample, from the mask as saved. IDs follow `memtk.ids`.

`lesions.csv`, one row per lesion:

| Column | Meaning |
| --- | --- |
| `source` | Dataset or pool name; `(source, lesion_id)` is unique across all data |
| `lesion_idx` | Integer row number, for compact joins inside one run (not stable across runs) |
| `lesion_id` | `<slice key>/L<n>`, e.g. `Mets_005/z127/L1`, `file_2_1033/L2` |
| `key` | Slice key (`Mets_005/z127` or the sample ID) |
| `patient_id`, `set`, `z` (real) / `sample_id` (synthetic) | Identity of the slice |
| `label` | Lesion number in the slice, 1 = largest (ties: top-left first) |
| `area`, `stray_px` | Lesion area including attached strays; how many of those pixels are strays |
| `centroid_row`, `centroid_col` | Centroid in model-space pixels (the frame E2 uses, decision F) |
| `centroid_row_img`, `centroid_col_img` | The same as a fraction of the image size |
| `centroid_row_brain`, `centroid_col_brain` | Brain-relative: 0 at the first, 1 at the last row/column of the brain box; outside [0, 1] beyond the box; NaN without a brain |
| `bbox_r0`, `bbox_r1`, `bbox_c0`, `bbox_c1` | Inclusive bounding box |
| `equiv_diameter`, `eccentricity`, `solidity` | Shape measures (scikit-image regionprops) |
| `frac_outside_brain` | Share of the lesion outside the brain mask (edge check: > `qc.edge_outside_frac`) |

`slices.csv`, one row per slice or sample (identity columns as above, plus):

| Column | Meaning |
| --- | --- |
| `n_components` | All 8-connected components of the mask as saved |
| `n_lesions`, `lesion_px`, `largest_lesion_px` | Lesions after attach-or-drop and their pixels |
| `has_lesion_lt40` | At least one lesion under `qc.small_lesion_px` (the sampling filter's rule) |
| `n_strays`, `stray_px` | Specks attached to a lesion |
| `n_specks`, `speck_px` | Isolated specks, dropped from the label map (counted by QC) |
| `specks_outside_brain` | Isolated specks with more than half their pixels outside the brain |

`labels.npz`: a label stack (`memtk.io.LabelStack`), key = slice key, value = the 128×128 label
map (0 = background or dropped speck, n = lesion `L<n>`, attached strays included). The cleaned
mask that the detectors use is `labels > 0`.

Real patients also get `volumes/<pid>/labels.nii.gz` (see the model-space volumes above).
Run logs: `lesions.run_info.json`.

### QC classes: `<output_root>/<pool>/qc.csv`

Written by `qc` with the decided `qc.max_isolated_specks`, read by `stats` and E5. Not in git.
One row per synthetic sample. Flags never exclude; the main audit (E1–E5) uses `mask_class`
clean or minor (decided at the `qc` stage, 2026-11-02).

| Column | Meaning |
| --- | --- |
| `source`, `key`, `sample_id` | Identity of the sample |
| `qc_class` | One class with the plan's precedence: `mri_failed` > `no_lesion` > `noisy` > `minor` > `clean` |
| `mask_class` | From the mask alone: `no_lesion` (no lesion), `noisy` (> cut-off isolated specks), `minor` (1..cut-off), `clean` (0). A failed MRI keeps its mask class |
| `mri_failed` | The paired MRI has no brain region (`brain.min_tissue_px`) |
| `edge`, `n_edge_lesions` | Lesions more than `qc.edge_outside_frac` outside the brain (flag only) |
| `has_lesion_lt40` | At least one lesion under `qc.small_lesion_px` (the sampling filter's rule) |
| `n_lesions`, `n_specks`, `n_strays`, `stray_px` | As in `slices.csv` |
| `legacy_excluded` | On the old QC list (comparison only) |

Run log: `qc.run_info.json`.

---

## 4. E0b: results (evidence and descriptions, in run order)

Review stages write evidence for a decision; nothing in a review folder is read by later stages.

### `results/<dataset>/e0b/brain_review/`

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

### `results/<dataset>/e0b/brain/` and `results/<pool>/e0b/brain/`

Written by `brain`. In git.

| File | Content |
| --- | --- |
| `summary.txt` | Per set (real) or per failed/ok group (synthetic): slices, slices without brain, brain-area 1st/5th percentile and median, median box size; failed-MRI count |
| `outlines.png` | Brain outline (cyan), box (dashed yellow) and mask (red). Real: random lesion slices per set. Synthetic: random samples, the smallest non-failed brains, failed MRIs |
| `run_info.json` | Run log |

### `results/<dataset>/e0b/lesions_review/`

Written by `lesions-review`. In git. Evidence for decisions B and C; nothing here is used by
later stages. Covers all real lesion slices (T, H-cal, H-eval; the proposal uses T only) and
all synthetic masks as saved.

| File | Content |
| --- | --- |
| `review.txt` | Component sizes, proposed `lesions.min_lesion_area` with its rule, attach-or-drop counts, specks by gap, what to set next |
| `component_sizes_<pool>.csv` / `.png` | Per group: components of each size 1–10 px and larger (table); share by size 1–30 px, log scale, with the proposed minimum marked (figure) |
| `attach_or_drop_<pool>.csv` | Per group, minimum area (proposal ± 1) and gap (0 = drop-only, 1, 2): slices, lesions, slices without lesion, strays and stray px attached, isolated specks, masks with an isolated speck |
| `speck_gaps_<pool>.csv` | Per group: specks at the proposed minimum area by gap to the nearest lesion (1, 2, 3, 4–5, 6–10, ≥ 11 px; or no lesion in the mask) |
| `small_components_T.png` | 32×32 crops of the smallest T components, one row per size 1 to proposal + 1 px; tile title gives slice, size and (row, col) |
| `small_components_T.csv` | Every T component up to proposal + 1 px: `patient_id`, `z`, `key`, `size`, `row`, `col` (model space, first pixel), `gap_to_lesion` (to the nearest other component ≥ proposal; −1 if none), `native_row`, `native_col` (approximate: (index + 0.5) × native size / 128 − 0.5; z is unchanged) |
| `specks_by_gap_<pool>.png` | 32×32 crops of synthetic specks at gap 1, 2 and 3 from a lesion, and T specks at gap 1–3 |
| `run_info.json` | Run log |

### `results/<dataset>/e0b/lesions/` and `results/<pool>/e0b/lesions/`

Written by `lesions`. In git.

| File | Content |
| --- | --- |
| `summary.txt` | Per set (real) or for the pool: slices, lesions, slices without a lesion, share with a lesion under 40 px, lesion area percentiles, strays, isolated specks (and the share outside the brain), lesions more than half outside the brain |
| `overview.png` | Lesions drawn on their slices: each lesion in its own colour with its L-number, attached strays lighter, dropped specks as orange ×, brain outline cyan, brain box dashed yellow. Real: random T, T with the most lesions, random H-cal and H-eval. Synthetic: random, the most lesions, masks with both strays and isolated specks |
| `run_info.json` | Run log |

### `results/<dataset|pool>/e0b/views/`

Written by the `view` helper, one file per request: `<key>.png` (slice key with `/` replaced by
`_`, e.g. `Mets_120_z40.png`): the slice at full size, drawn as in `overview.png`, with its
`lesions.csv` rows as a table. With `--nifti`, a synthetic sample also gets `<key>_image.nii.gz`
and `<key>_labels.nii.gz` for ITK-SNAP (real slices already have `volumes/<pid>/labels.nii.gz`).

### `results/<pool>/e0b/qc_review/`

Written by `qc-review`. In git. Evidence for decision D (`qc.max_isolated_specks`); nothing here
is used by later stages. Synthetic masks only (no detector has touched S yet).

| File | Content |
| --- | --- |
| `review.txt` | The tables below, proposed cut-off, the edge-check count, what to set next |
| `speck_strata_<pool>.csv` | Per number of isolated specks (0, 1, 2, 3, 4–5, 6–9, 10–19, ≥ 20): masks, share, cumulative share, failed MRIs, masks without a lesion, lesions per mask, share of specks outside the brain, share of lesions with a stray and grown > 25% by strays, share of lesions mostly outside the brain, share excluded by the old QC list |
| `speck_counts_<pool>.csv` | Masks with exactly 0..20 isolated specks, and more |
| `speck_strata_<pool>.png` | 12 random masks per stratum (failed MRIs left out), drawn as in the lesion overview |
| `edge_clean_<pool>.png` | 24 random clean masks with a lesion more than `qc.edge_outside_frac` outside the brain, the lesion named in the tile title |
| `run_info.json` | Run log |

### `results/<pool>/e0b/qc/`

Written by `qc`. In git.

| File | Content |
| --- | --- |
| `summary.txt` | Counts per QC class and per mask class, failed MRIs by mask class, the old-QC cross-table, the edge check on the real sets |
| `qc_classes_<pool>.csv` | Per QC class: masks, share, edge-flagged, without a lesion under 40 px, strays, on the old QC list |
| `old_qc_crosstab_<pool>.csv` | QC class against the old list's reason (noise fragments, pixels outside brain, not on the list) |
| `edge_check_real_<dataset>.csv` | The same edge rule on T, H-cal and H-eval: slices, lesions, edge lesions, edge slices |
| `qc_classes_<pool>.png` | 8 random samples per QC class, drawn as in the lesion overview |
| `run_info.json` | Run log |

### `results/<dataset>/e0b/stats/`

Written by `stats`. In git. Compares the real sets with a pool, so it lives under the dataset
with the pool name in the file names. Sets: T, H-cal, H-eval, S main (`mask_class` clean or
minor, the main audit) and S all.

| File | Content |
| --- | --- |
| `summary.txt` | The tables below in readable form |
| `description_<pool>.csv` | One column per set: patients, slices, lesions (per slice, max), share of slices with no / one / several lesions, lesion area (quartiles), share with a lesion under 40 px, centroid mean and SD in pixels and brain-relative, slices per patient (real) |
| `native_sizes_<dataset>.csv` | Patients per native in-plane size and set |
| `bound_impact_<pool>.csv` | Per set and candidate informative-lesion bound (5–40 px): lesions kept and their share, slices left without an informative lesion and their share, patients with an informative slice (real). Input for E3 |
| `lesions_per_slice_<pool>.png` | Share of slices by number of lesions, T, H-cal, H-eval, S main |
| `lesion_area_<pool>.png` | Cumulative share of lesions by area (log scale), with medians |
| `centroid_density_<pool>.png` | 2D density of lesion centroids for T, H (H-cal + H-eval) and S main; top row in pixels, bottom row relative to the brain box |
| `adjacency_dice_<dataset>.csv` / `.png` | Per real patient: pairs of consecutive lesion slices and their Dice (median, min, max) on the cleaned masks; histograms of all pairs and of per-patient medians per set |
| `native_vs_model_<dataset>.csv` / `.png` | Per slice with exactly one lesion in both spaces: native area, native area scaled to 128×128 (`expected_area`), model-space area, their `ratio`, native size bin; scatter and ratio by size |
| `brain_by_height_<dataset>.csv` | Brain area quartiles (`p25`, `median`, `p75`) of all real slices (decided brain method) per relative slice position (20 bins, indexed by bin centre; 0 = first, 1 = last slice with brain), with slices and lesion slices per bin |
| `brain_by_height_<pool>.png` | Left: real brain area by relative position (median, middle 50%) with the synthetic median as a red line. Right: brain-area distributions of real lesion slices and synthetic samples (main audit, failed MRIs left out) |
| `run_info.json` | Run log |

### `results/<dataset>/e0b/filters_review/`

Written by `filters-review`. In git. Evidence for decision E (`filters.real_sets`); nothing here
is used by later stages. The sampling script's filters: `max_mask_px` (pixel sum below
`filters.max_mask_px`), `small_lesion` (a component under `qc.small_lesion_px`), `inside_brain`
(every lesion pixel inside the brain; here with our brain mask, not the script's).

| File | Content |
| --- | --- |
| `review.txt` | The tables below, masks that pass `small_lesion` only because of a speck, the filters that hold in the pool, what to set next |
| `filter_pass_<pool>.csv` | Share of masks passing each filter, on the mask as saved (`raw_*`) and cleaned (`clean_*`), and all three, per set (T, H-cal, H-eval, S main, S all) |
| `filter_cost_<dataset>.csv` | Per real set and filter (cleaned masks): slices removed and their share, patients left |
| `run_info.json` | Run log |

### `results/<dataset>/e0b/check_report_<pool>.txt`

Written by `check`. In git. One PASS/FAIL line per check, with counts and failing examples:
every decision set; each production stage's run log has today's values of the settings it
depends on and ran after the stage before it; per source, lesion IDs unique and parseable,
lesion rows per slice, label maps holding labels 1..n with matching areas, centroids and
numbering, label maps reproduced by attach-or-drop from the masks as saved, real
`labels.nii.gz` volumes, stack keys, cached synthetic masks and brain masks recomputed from the
files (random sample), QC classes reproduced and counted; the unit tests. Exit code 1 on any
failure. Run log: `check_<pool>.run_info.json`.

---

## 5. Run logs: `run_info.json` and `*.run_info.json`

Written by every stage next to its outputs. Fields: `timestamp`, `script` (the module that wrote
the log, e.g. `src\memtk\e0b\stats.py`), `argv` (the command, with the entry script), `seed`,
`python`, `packages` (versions of memtk, numpy, scipy, scikit-image, nibabel, monai, torch,
pandas), `config` (the full config used), and stage-specific extras such as `stage` and `pool`.

---

## 6. E3: calibration on H-cal (`e3_calibrate.py`)

### `results/<dataset>/e3/bound_review/`

Written by `bound-review`. In git. Evidence for decision G (`calibration.provisional_tau_L`,
`chance_rate_max`, `informative_bound_px`); nothing here is read by later stages. E2 runs with
every H-cal lesion (≥ `lesions.min_lesion_area`) against every T lesion; S is not read (its
cost comes from the E0 bound-impact table).

| File | Content |
| --- | --- |
| `review.txt` | Sets and slices left out, E2 settings in pixels, the chance-match rate by area bin and tau_L, the proposal (provisional tau_L, bound at the 5% and 10% cuts), the cost of each candidate bound (E0), what to set next |
| `best_matches_hcal.csv` | One row per H-cal lesion: `query` (slice key), `query_lesion`, `area`, `best_train_lesion`, `best_s` and its factors `best_shape`, `best_position`, `best_size` (NaN when no T lesion reaches s ≥ `candidate_min_s`), `patient_id` |
| `chance_rates.csv` | Per area bin (`lower` edge, `bin` label) and `tau_L`: `lesions`, `patients`, `matched` (best score ≥ tau_L), `rate` |
| `chance_rates.png` | Chance-match rate by area bin, one line per tau_L (the provisional one bold), with the 5% and 10% cuts |
| `examples.png` | Per area bin, 3 random matched H-cal lesions (red) next to their best T lesion (green), 32×32 crops of the cleaned masks, with s and its factors |
| `run_info.json` | Run log (with `inputs`: T, H-cal, and the E2 settings in pixels) |

### Detector tables of H-cal: `<output_root>/<dataset>/e3/`

Written by `calibrate`, read by `freeze` and `check`. Not in git. H-cal slices against T, with
the decided bound. Read the CSVs with `memtk.io.read_table`, the Parquet files with
`pandas.read_parquet`; together the four E2 files rebuild a `memtk.lesionmatch.LesionMatch`.

| File | Content |
| --- | --- |
| `e1_hcal.csv` | E1, one row per H-cal slice: `query`, `nn`, `nn_patient`, `score`, `shift_row`, `shift_col`, `plain_dice`, `runner_up_patient`, `runner_up_score`, `margin`, `n_candidates` (see `memtk.wholemask.match`), plus `patient_id`, `n_lesions` (≥ min area) and `stratum` |
| `e2_hcal_queries.csv` | E2, one row per H-cal slice: `query`, `n_informative` (lesions ≥ the bound; 0 = left out of E2) |
| `e2_hcal_lesions.csv` | E2, one row per informative H-cal lesion: `query`, `query_lesion`, `area`, `best_train_lesion`, `best_s`, `best_shape`, `best_position`, `best_size` (NaN when no pair reaches the candidate cut) |
| `e2_hcal_slice_pairs.parquet` | E2, one row per candidate slice pair: `query`, `train`, `train_idx`, `n_query`, `n_train` (informative lesions), `n_matched` (Hungarian pairs with s > 0), `best_pair_s` |
| `e2_hcal_lesion_pairs.parquet` | E2, the Hungarian pairs of every candidate slice pair: `query`, `train`, `query_lesion`, `train_lesion`, `distance` (px), `shape`, `position`, `size`, `s` |
| `calibrate.run_info.json` | Run log (inputs T and H-cal, E2 settings in pixels, the bound) |

### `results/<dataset>/e3/calibrate/`

Written by `calibrate`. In git. Evidence for decision H (`calibration.strata.whole_mask`,
`calibration.strata.lesion`).

| File | Content |
| --- | --- |
| `review.txt` | Sets and slices left out, settings, the thresholds with intervals, the E2 classes on H-cal, the sensitivity table, exclusions by the bound, the proposal for decision H |
| `thresholds.csv` | One row per detector (`whole_mask`, `lesion`), threshold (`tau_D`, `tau_L`, `tau_C`) and stratum (`all` = pooled, `single`, `multi`): `value` (the percentile of H-cal), `ci_low`, `ci_high` (bootstrap by patient), `slices`, `patients`, `flagged_at_pooled` (share of the stratum's slices at or above the pooled threshold) |
| `classes_hcal.csv` | E2 classes of the H-cal slices under the `pooled` and the `stratified` thresholds (`thresholds` column; tau_L and tau_C), per stratum: `slices`, and the shares `flagged` (full or partial), `full` (C ≥ tau_C), `partial`, `all_lesions_copied` (every informative lesion paired with s ≥ tau_L; descriptive); all chance |
| `sensitivity.csv` | Threshold values per detector, threshold and stratum for the main setting and one alternative per setting: the 99th percentile, sigma 0.08, the bound one bin lower, the 5% cut's bound (25 px) |
| `exclusions.csv` | Per candidate bound: T and H-cal lesions kept and slices left without an informative lesion; S main share without (from the E0 table) |
| `score_histograms.png` | H-cal score histograms by stratum for tau_D (E1 score), tau_L (strongest lesion pair) and tau_C (configuration score C of the nearest neighbour), with the pooled and stratum thresholds |
| `run_info.json` | Run log |

## Planned files (E1–E3)

Specified in `docs/e1-e3-plan.md`; entries are added here as each stage is built: the frozen
thresholds `configs/frozen/e3_<dataset>.yaml` (in git, written by `freeze`) and the E3 check
report.
