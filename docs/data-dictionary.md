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
| | `freeze` | ✅ | decision H, the `calibrate` tables and `thresholds.csv` | `configs/frozen/e3_<dataset>.yaml` + `e3_<dataset>.run_info.json` |
| | `check` | ✅ | all E3 outputs, T and H-cal label maps | `results/<dataset>/e3/check_report.txt`, `check.run_info.json` |
| `e4_validate.py` | `plant` | ✅ | decision I (`validation.bases`, `validation.variants`), decision C, the frozen file (bound, minimum area), E0b lesion tables (T, H-eval), `brain_masks.npz`, real BRAVO volumes (figures) | `<output_root>/<dataset>/e4/` (planted set), `results/<dataset>/e4/plant/` (evidence for decision K) |
| | `validate` | ✅ | decisions I (gate rule), J, K and `validation.accepted_deviation`, the frozen file (settings, thresholds), T and H-eval label maps, the planted set, real BRAVO volumes (figures) | `<output_root>/<dataset>/e4/` (detector tables of the planted set and H-eval, `heval_classes.csv`), `results/<dataset>/e4/validate/` (gate verdict) |
| | `check` | ✅ | all E4 outputs, T and H-eval label maps, the frozen file | `results/<dataset>/e4/check_report.txt`, `check.run_info.json` |
| `e5_audit.py` | `prepare` | ✅ | decisions O, P, Q (`audit:`), the frozen file, E3 H-cal tables and `sensitivity.csv`, E4 H-eval tables, T / H-cal / H-eval label maps and model-space `seg` volumes; never S | `configs/frozen/e5_<dataset>.yaml` + run log, `<output_root>/<dataset>/e5/` (drop-only T labels, H-eval under every variant), `results/<dataset>/e5/prepare/` |
| | `rehearse` | ✅ | `prepare` outputs (same `audit:`), the frozen files, T, H-cal, H-eval, the E4 planted set; never S | `<output_root>/<dataset>/e5/rehearse/<case>/` (raw tables), `results/<dataset>/e5/rehearse/<case>/` (analysis) |
| | `score` | ✅ | the E4 gate, `prepare` and `rehearse` outputs (same hashes), the frozen files, T and its drop-only labels, S: pool `qc.csv`, `labels.npz`, `masks_raw.npz` (main and noisy masks) | `<output_root>/<pool>/e5/` (raw tables, `score_log.csv`, `score.complete.json`) |
| | `analyze` | ✅ | `score` outputs and marker (hashes must match), H-eval baseline (E4 and `prepare` class tables), T, pool `labels.npz`, `synthetic_index.csv` and sample files (figure backgrounds) | `results/<pool>/e5/analyze/` (layout as the rehearsal analysis above) |
| | `check` | ✅ | all E5 outputs, the frozen files, `audit:`, T, H-cal, H-eval, E3 and E4 tables | `results/<pool>/e5/check_report.txt`, `check.run_info.json` |
| `e6_exposure.py` | `reference` | ✅ | decisions R, V, W (`exposure:`), the frozen E3 file (settings), T label maps, `lesions.csv`, model-space `seg` volumes, E4 `heval_classes.csv`; never S | `<output_root>/<dataset>/e6/` (`lesions3d.csv`, `lesion_map.csv`, `rarity_slices.csv`, `patients.csv`, `heldout_nn.csv`), `results/<dataset>/e6/reference/` |
| | `rehearse` | ✅ | decisions R, U, V, W, `exposure.placement_mock`, the frozen E3 file (thresholds, settings), the E6 reference tables, E5's mock sets (`<output_root>/<dataset>/e5/rehearse/<case>/`), H-cal and T label maps and T `lesions.csv` (placement mocks), E4 H-eval tables, the planted table; never S | `<output_root>/<dataset>/e6/rehearse/<case>/` (placement mocks, E5's layout), `results/<dataset>/e6/rehearse/<case>/` (layout below) |
| | `exposure` | ✅ | E5's S tables (`<output_root>/<pool>/e5/`: `queries.csv`, `classes_main.csv`, `e1_main.csv`, `e2_main_*`; set main) with E5's lock unchanged, the E6 reference tables, E4 H-eval tables, the `rehearse` run log (same `exposure:`) | `results/<pool>/e6/exposure/` (same layout as the rehearsal, without `e5_rates.csv`, `e5_auc.csv`) |
| | `check` | ✅ | all E6 outputs, T, H-eval tables, E5's S tables | `results/<pool>/e6/check_report.txt`, `check.run_info.json` |
| `e7_reports.py` | `planted` | ✅ | decisions X, Y, Z (`reports:`), the frozen E3 file, E4's planted set and its tables (`planted_outcomes.csv`, `e1_planted.csv`, `e2_planted_*`, `planted_lesions.csv`), E4's H-eval tables and `fp_heval.csv`, T and H-eval label maps, real BRAVO volumes (figures); never S | `results/<dataset>/e7/planted/` |
| | `reports` | ✅ | decisions X, Y, Z, E5's S tables (set main; lock unchanged) and `flagged.csv`, E6 `exposure` tables, E6 reference tables, E7 `planted` criteria (must pass), E4 H-eval tables and `fp_heval.csv`, pool `labels.npz`, `synthetic_index.csv` and sample files, T and H-eval label maps, real BRAVO volumes | `results/<pool>/e7/reports/` |
| | `check` | ✅ | all E7 outputs, the planted set, T, H-eval, E5's S tables | `results/<pool>/e7/check_report.txt`, `check.run_info.json` |
| `e8_paired_mri.py` | — | optional | E5 outputs, paired MRIs | paired-MRI similarity |

E1 (whole-mask detector) and E2 (lesion-level detector) have no scripts: they are `memtk`
modules, run inside E3 (on H-cal), E4 (on planted copies) and E5 (on S).

---

## 1. Inputs

### Config: `configs/default.yaml`

Hand-edited, read by every stage. In git.

Keys: `seed`; `output_root`, `results_root`, `splits_dir`, `frozen_dir`; `default_real_dataset`,
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
latter for tau_L and tau_C); E4 (`docs/e4-plan.md`): `validation.bases` (decision I:
`min_informative_slices`, `patients_multi`, `min_multi_slices`, `multi_per_patient`,
`single_per_patient`, `patients_other`, `slices_per_patient`, `min_z_gap`),
`validation.variants` (decision I: `shift_small_px`, `shift_large_px`, `rotate_deg`,
`rotate_threshold`, `morph_px`, `add_min_gap_px`, `add_max_tries`), `validation.gate` (decision
I amended: the rule a planted mask must meet to count in the gate, `keeps_base_lesions`),
`validation.caught` (decision J), `validation.criteria` (decision K: `recall` per variant with
`detector` both / any / lesion, `min` and optional `variants` pooled; `reported_only`;
`false_positive_max` per class; read by `validate`, which refuses to run while it is `null`),
`validation.accepted_deviation` (`criteria`: the failed gate criteria accepted by the user, by
their name in `gate.csv`; `reason`; `null` = none; decided after a `validate` run and recorded
as such); E5 (`docs/e5-plan.md`, added 2026-11-15, every value fixed before `e5_audit.py score`,
which hashes the section): `audit.sets` (decision L: `main` and `noisy` as lists of `mask_class`
values), `audit.primary` (decision M: `classes`, `strata`, `adjust`, `alpha` of the primary
family), `audit.size_bins_px` (bins of the largest informative lesion for the size-confound
check), `audit.window` (decision N: `kind` chained, `max_gap`), `audit.sensitivity` (decision O:
the variants scored on S), `audit.h2` (decision P), `audit.near_exact` (decision Q: `min_score`,
`beyond`, `size_bins_px`), `audit.e8_min_strong` (the E8 exit rule); E6 + E7
(`docs/e6-e7-plan.md`, added 2026-11-15, every value fixed before the stage that reads S's tables,
which refuses while one is `null`): `exposure.classes`, `exposure.primary` (`classes`, `strata`),
`exposure.expected` (decision R: `baseline`, `smoothing_m`, `smoothing_m_sensitivity`),
`exposure.top_k`, `exposure.reading` (decision U: `dispersion_percentile`, `auc_null`),
`exposure.rarity` (decision V: `patient_k`, `features`, `log1p`, `permutations`),
`exposure.lesions3d` (decision W: `main`, `sensitivity`), `exposure.h1_size_bins_px`;
`reports.selection` (decision X: `strong_tiers`, `top_exposed_patients`, `random_s_multi`,
`random_heval`), `reports.deletion` (decision Y: `min_paired`, `planted_controls`),
`reports.correctness` (decision Z: `slice_window`, `patient_min`, `patient_variants`,
`lesions_min`, `lesion_variants`), `reports.taxonomy`. E4–E7
take detector settings and thresholds from the frozen file, not from these config sections;
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

### Frozen thresholds: `configs/frozen/e3_<dataset>.yaml`

Written by `freeze`, read by E4 and E5 (with `memtk.calibration.read_frozen` and
`frozen_threshold`) and by the E3 `check`. In git; do not edit by hand (changing it means
re-running E3 on H-cal, never tuning on S). Run log next to it: `e3_<dataset>.run_info.json`.

| Key | Content |
| --- | --- |
| `dataset`, `frozen` | Dataset name and the time of freezing |
| `calibration_set` | `H-cal`, its patients, slices scored by E1 and slices with an informative lesion (E2) |
| `percentile`, `strata` | The percentile (95) and decision H per detector (`stratified` or `pooled`) |
| `stratum_rule` | How a slice's stratum is counted per detector (lesions ≥ the minimum area for E1, informative lesions for E2; 1 = single, ≥ 2 = multi) |
| `thresholds` | `whole_mask.tau_D`, `lesion.tau_L`, `lesion.tau_C`, each with `single` and `multi` (the same value twice when pooled) |
| `intervals_95` | The bootstrap-by-patient 95% interval of each threshold, same layout |
| `class_rule` | E1: copy if score ≥ tau_D; E2: full if C(q, NN) ≥ tau_C, partial if not full but the strongest lesion pair ≥ tau_L, novel otherwise |
| `informative_bound_px`, `min_lesion_area` | Decision G's bound and the E0b minimum lesion area |
| `detectors` | The E1 and E2 settings (as in the config, plus `sigma_px`, `prefilter_px`) |
| `hcal_chance_rates` | Per stratum, the share of H-cal slices flagged at these thresholds: `e1_copy`, `e2_full`, `e2_partial`, `e2_flagged` (all chance) |
| `source` | Where the values came from (`thresholds.csv`, the H-cal tables) |

### `results/<dataset>/e3/check_report.txt`

Written by `check`. In git. One PASS/FAIL line per check: decisions F, G, H set; each E3 run log
made with today's settings, in order, reading only T and H-cal; `thresholds.csv` reproduced from
the stored H-cal tables; the frozen file equal to it under decision H and to the current settings;
the E2 shortcut conditions at the frozen thresholds; the shortcuts against brute force on random
H-cal slices; a self-match of random T slices (E1 score 1, E2 C 1); the unit tests. Exit code 1 on
any failure. Run log: `check.run_info.json`.

---

## 7. E4: planted-copy validation (`e4_validate.py`)

### Planted set: `<output_root>/<dataset>/e4/`

Written by `plant`; read by `validate` and `check` and by E7 (correctness of the match reports). Not in git; `plant` reproduces it from the seed. Planted keys are
`<variant>__<patient_id>_z<z>` (e.g. `P2__Mets_005_z127`), lesion IDs `<key>/L<n>`.

| File | Content |
| --- | --- |
| `planted_labels.npz` | Label stack (`memtk.io.LabelStack`), one label map per planted mask after attach-or-drop, by planted key |
| `planted.csv` | One row per planted mask: `key`, `variant` (P1, P2, P3, P4, P5-erode, P5-dilate, P6-remove, P6-add), `base_key`, `patient_id` (the base patient: the true source), `z`, `group` (`multi` or `other`, decision I), `base_n_informative`, `base_stratum`, `n_lesions` (≥ the minimum area), `n_informative` (≥ the bound), `stratum_e1`, `stratum_e2` (`none` = not scorable by that detector), `keeps_base_lesions` (every informative base lesion still a lesion ≥ the bound: the gate rule, `validation.gate`, for every variant but P6-remove), and the parameters `shift_row`, `shift_col` (P2, P3), `angle` (P4, degrees), `removed` (P6-remove: base lesion ID), `added` (P6-add: H-eval lesion ID), `tries` (P6-add: donors tried) |
| `planted_lesions.csv` | One row per planted lesion: `lesion_id`, `key`, `variant`, `base_key`, `number`, `area`, `source_lesion` (the base lesion it overlaps most after the same geometric change, or the added H-eval lesion; "" if none), `source_area` |
| `plant.run_info.json` | Run log (inputs T and H-eval, the frozen file's name and SHA-256, the random streams) |

### `results/<dataset>/e4/plant/`

Written by `plant`. In git. Evidence for decision K (`validation.criteria`); no detector runs.

| File | Content |
| --- | --- |
| `review.txt` | Bases by group, skipped patients, P6-add donors, the variant summary, variants not made, the area change by size bin, what to look at, and decision K as set in the config (or where to set it) |
| `bases.csv` | One row per base slice: `key`, `patient_id`, `z`, `n_informative`, `group`, `stratum` (by informative lesions) |
| `variant_summary.csv` | Per variant: `planted`, `skipped`, `scorable_e1`, `scorable_e2`, `e2_single`, `e2_multi`, `stratum_changed` (E2 stratum differs from the base's), `keeps_base_lesions`, `base_lesions`, `base_lesions_lost` (largest traced piece below the bound, gone or merged; P6-remove's removed lesion excluded), `median_area_ratio` |
| `lesion_fate.csv` | One row per planted mask and informative base lesion: `key`, `variant`, `base_key`, `base_lesion`, `base_area`, `planted_area` (all planted lesions traced to it), `largest_piece`, `kept` (largest piece ≥ the bound), `removed` (P6-remove's removed lesion) |
| `area_change.csv` | P4, P5-erode, P5-dilate by base lesion size bin (12–19, 20–39, 40–79, ≥ 80 px): `lesions`, `kept` (share), `median_ratio`, `min_ratio`, `max_ratio` (largest piece / base area) |
| `skipped_variants.csv` | Variants not made: `base_key`, `variant`, `reason` |
| `variants_grid.png` | 12 random bases (6 single-, 6 multi-lesion) × the original and every variant, zoomed on the BRAVO slice: base outline red, planted green; base lesions with no planted lesion traced to them (removed, eroded away) filled red, added H-eval lesions filled orange |
| `variant_<variant>.png` | One per variant, 20 random bases, three tiles each: the planted mask on the whole slice (zoom box dashed), the base lesions, and the planted lesions filled in the colour of the lesion they come from (orange = added H-eval lesion) over the base lesions dashed in their colour, with a dot and arrow where a lesion centre moved by ≥ 1 px |
| `area_change.png` | Largest planted piece against base lesion area for P4, P5-erode, P5-dilate (log axes), lost lesions in red, the bound dashed |
| `run_info.json` | Run log (as above) |

### Detector tables of the planted set and H-eval: `<output_root>/<dataset>/e4/`

Written by `validate` (settings and thresholds from the frozen file), next to the planted set.
Not in git. The H-eval tables are E5's baseline (E5 reuses them). `<set>` is `planted` or
`heval`; the four E2 files of a set rebuild a `memtk.lesionmatch.LesionMatch`, as for H-cal in E3.

| File | Content |
| --- | --- |
| `e1_<set>.csv` | E1, one row per query with a lesion ≥ the minimum area (unscorable planted masks left out): the columns of `memtk.wholemask.match` plus `n_lesions`, `stratum`, `tau_D`, `copy` (`memtk.calibration.e1_classes`) |
| `e2_<set>_queries.csv`, `e2_<set>_lesions.csv`, `e2_<set>_slice_pairs.parquet`, `e2_<set>_lesion_pairs.parquet` | E2, as the H-cal tables of E3 |
| `heval_classes.csv` | One row per H-eval slice: `key`, `patient_id`, `n_lesions`, `stratum_e1`, `e1_score`, `e1_nn`, `e1_nn_patient`, `e1_copy`, `n_informative`, `stratum_e2` (`none` without an informative lesion), `e2_class` (`full`, `partial`, `novel`, `none`), `e2_C`, `e2_nn`, `e2_nn_patient`, `e2_strongest_s`, `e2_strongest_train`, `e2_n_kept`, `e2_all_copied`; every flag is a false positive |
| `validate.run_info.json` | Run log (inputs T, H-eval, planted; the frozen file's SHA-256; the verdict) |

### `results/<dataset>/e4/validate/`

Written by `validate`. In git. The gate verdict, read by E5 through `memtk.e4.common.gate_status`.

| File | Content |
| --- | --- |
| `review.txt` | Sets, recall of the gated masks per variant and detector with intervals, flagged / source accuracy / NN = base, E2 classes, the reported-only variants by size, false positives with H-cal rates, RQ2 counts, the gate table, and the verdict (PASS, ACCEPTED with the deviation's reason, or FAIL with what to do) |
| `gate.csv` | One row per criterion of decision K: `criterion` (e.g. `recall P4`, `false positives e1_copy single`), `detail`, `value`, `limit`, `pass`, `accepted` (failed and listed in `validation.accepted_deviation`) |
| `planted_outcomes.csv` | One row per planted mask: `key`, `variant`, `base_key`, `patient_id`, `group`, `base_stratum`, `stratum_e1`, `stratum_e2`, `keeps_base_lesions`, `gated`, `base_l1_area`, `size_bin`; E1 `e1_score`, `e1_nn`, `e1_nn_patient`, `e1_copy`; E2 `e2_class`, `e2_C`, `e2_nn`, `e2_nn_patient`, `e2_strongest_s`, `e2_strongest_train`, `e2_strongest_patient`, `e2_n_kept`, `e2_all_copied`; `e1_caught`, `e2_caught`, `any_caught`, `both_caught` (decision J), `e1_nn_is_base`, `e2_nn_is_base` |
| `validation.csv` | Per `variant`, `subset` (`gated`; `all` = every planted mask, unscorable = missed) and `detector` (`e1`, `e2`, `any`, `both`): `masks`, `patients`, `caught`, `recall`, `ci_low`, `ci_high` (bootstrap by base patient), `patient_recall`; for e1 / e2 `flagged`, `source_accuracy`, `nn_is_base`; for e2 `full`, `partial`, `all_copied` |
| `recall_by_group.csv` | Per variant, `by` (`stratum` of the base, `size` of the base's largest lesion) and `group`: `masks`, recall for `e1`, `e2`, `any` (all masks) |
| `fp_heval.csv` | Per `class` (`e1_copy`, `e2_full`, `e2_partial`, `e2_flagged`) and `stratum` (`single`, `multi`, `all`): `slices`, `patients`, `flagged`, `rate`, `ci_low`, `ci_high` (bootstrap by patient), `hcal_rate` (frozen file), `limit`, `pass` |
| `rq2_planted.csv` | Per variant, gated masks: `masks`, `e2_only`, `e1_only`, `both`, `neither` |
| `score_distributions.png` | E1 score, E2 C(q, NN) and strongest pair per variant and for H-eval, by stratum, with the frozen thresholds |
| `recall_by_size.png` | Recall per variant (one panel each) by the size of the base's largest lesion, for E1, E2 and either |
| `factors.png` | Shape, position, size and s of the E2 pairs between each planted mask and its own base slice, per variant |
| `misses.png` | Gated planted masks missed by their criterion's detector: base, planted mask, nearest neighbour found, scores, and why ("not flagged" or "flagged, wrong patient"); written only when there are misses |
| `heval_flags.png` | Random H-eval false positives per class next to the training slice that flagged them |
| `run_info.json` | Run log (as above) |

### `results/<dataset>/e4/check_report.txt`

Written by `check`. In git. One PASS/FAIL line per check: decisions I, J, K set; the `plant` and
`validate` run logs made with today's settings, in order, reading only T, H-eval and the planted
set; the frozen file unchanged since both stages ran (SHA-256) and equal to the current settings;
the planted set and its provenance reproduced from the seed; every P1 map equal to its base; bases
from T and added lesions from H-eval only; the per-mask outcomes, `gate.csv` and the verdict
reproduced from the stored detector tables; the unit tests. Exit code 1 on any failure. Run log:
`check.run_info.json`.

---

## 8. E5: main audit (`e5_audit.py`; `docs/e5-plan.md`)

Every value E5 uses is fixed before `score` reads S: the config section `audit:` (decisions
L–Q) and the E5 file below. Variants (decision O): `percentile_99`, `sigma_0.08`, `bound_10`,
`bound_25`, `drop_only`; `main` = the frozen E3 settings and thresholds. Read every CSV here with
`memtk.io.read_table`: the ID columns of the detector tables (`query`, `nn`, `train`, lesion IDs,
...) are in `memtk.io.ID_COLUMNS` since 2026-11-15, because S sample IDs such as `1033` look
numeric.

### E5 file: `configs/frozen/e5_<dataset>.yaml`

Written by `prepare` from T, H-cal and H-eval only. In git. Read by `score` and `analyze` (through
`memtk.calibration.load_frozen(cfg, dataset, "e5")`), which record its SHA-256. Do not edit by
hand; `prepare` refuses to run once S has been scored. Run log next to it:
`e5_<dataset>.run_info.json`.

| Key | Content |
| --- | --- |
| `dataset`, `written`, `inputs` | Identity; inputs are always T, H-cal, H-eval |
| `e3_frozen_sha256`, `audit_sha256` | The frozen E3 file and the `audit:` section it was made with |
| `variants.<name>` | `kind` (percentile, sigma, bound, drop_only), `rerun` (detectors run again on S and H-eval: none for percentile; lesion for sigma and bound; both for drop-only), `settings` (`percentile`, `informative_bound_px`, `sigma_px`, `prefilter_px`, `candidate_min_s`, `attach_gap`: 0 for drop-only, null = as E0b), `thresholds` (tau_D, tau_L, tau_C per stratum, re-derived on H-cal: equal to E3 `sensitivity.csv`, drop-only equal to the frozen ones), `hcal_chance_rates` (per stratum: e1_copy, e2_full, e2_partial, e2_flagged) |
| `h2` | Decision P: `matched_rate` (E2 flagged's H-cal rate per stratum), `tau_D_star` (per stratum), `hcal_rate_at_tau_D_star` |
| `near_exact` | Decision Q: `min_score`, `size_bins_px`, `e1_maxima` (per stratum and bin lower edge, `lower` = −1 for the whole stratum: `max`, `slices`; H-cal and H-eval together), `e2_C_maxima` (per stratum) |
| `drop_only_labels` | Per real set: `slices`, `changed_at_gap_0` (H-cal and H-eval must be 0) |

### Pre-S tables: `<output_root>/<dataset>/e5/`

Written by `prepare`. Not in git (regenerated by `prepare` until S is scored).

| File | Content |
| --- | --- |
| `labels_drop_only_T.npz` | Label stack of T at attach gap 0 (drop-only), by slice key; the reference of the drop-only variant |
| `heval_classes_<variant>.csv` | H-eval per slice under a variant, columns as E4's `heval_classes.csv` (main = E4's file) |
| `e1_heval_drop_only.csv` | E1 of H-eval against the drop-only T, columns as E4's `e1_heval.csv` |
| `e2_heval_<variant>_*` | E2 of H-eval for the variants that re-run it (sigma, bound, drop-only): the four tables of `memtk.lesionmatch.save_tables` |
| `prepare.run_info.json` | Run log (inputs T, H-cal, H-eval; the frozen E3 file's and the `audit:` section's SHA-256) |

### `results/<dataset>/e5/prepare/`

Written by `prepare`. In git.

| File | Content |
| --- | --- |
| `review.txt` | The drop-only check, the sensitivity thresholds, H-cal chance rates per variant, tau_D*, the held-out maxima, H-eval rates per variant, what to look at |
| `thresholds.csv` | Per `variant`, `detector`, `threshold`, `stratum`: `value` (re-derived on H-cal), `e3_value` (E3 `sensitivity.csv`, or the frozen value for drop-only), `equal` |
| `heldout_maxima.csv` | Per `set` (H-cal, H-eval, held-out = both), `score` (`e1`, `e2_C`), `stratum`, `lower` (size-bin lower edge of the largest lesion; −1 = whole stratum): `max`, `slices` |
| `heval_rates.csv` | Per `variant` (main and decision O), `class`, `stratum` (single, multi, all): `slices`, `patients`, `flagged`, `rate`, `ci_low`, `ci_high` (bootstrap by patient), `hcal_rate` |
| `drop_only.csv` | Per real set: `slices`, `reproduces_e0b_at_gap` (the configured gap gives the E0b label maps), `changed_at_gap_0`, `changed_keys` |
| `run_info.json` | Run log (as above) |

### Raw tables of a scored query set: `<output_root>/<dataset>/e5/rehearse/<case>/` (and later `<output_root>/<pool>/e5/`)

Written by `memtk.e5.score.score_query`: by `rehearse` for the mock query sets (`null` = H-cal;
`positive` = H-cal + the gated planted copies P1, P2, P4) and, with the same layout, by `score`
for S. Not in git. No rates are computed here.

| File | Content |
| --- | --- |
| `queries.csv` | One row per query slice: `key`, `cluster` (bootstrap unit: the sample for S and planted masks, the patient for real slices), `mask_class`, `scheme` (naming scheme for S; `H-cal` / `planted` for mock sets), `n_lesions` (≥ the minimum area), `l1_area` (largest lesion, strays included) |
| `classes_<variant>.csv` | One row per query under `main` or a variant of decision O: the columns of E4's `heval_classes.csv` (`patient_id` = the cluster) |
| `e1_main.csv`, `e1_drop_only.csv` | E1 tables (columns as E4's `e1_<set>.csv`) |
| `e2_<variant>_*` | The four E2 tables (`lesionmatch.save_tables`) of `main` and of every variant that runs E2 again (sigma, bound, drop-only) |
| `rehearse.run_info.json` | Run log (inputs T, H-cal, H-eval, planted; SHA-256 of the frozen E3 file, the E5 file and `audit:`) |
| `score_log.csv` (S only) | Per run (main and every variant): `variant`, `seconds`, `e1_scored`, `e2_scored`, `lesion_pairs` |
| `score.run_info.json` (S only) | Run log of `score` (inputs T and S, the pool, the three SHA-256, the E4 verdict, masks left out) |
| `score.complete.json` (S only) | The completion marker: time, the SHA-256 of `audit:`, the frozen E3 file and the E5 file, queries, masks per mask class, variants, minutes. While it exists, `score` and `prepare` refuse to run (the lock), and `analyze` checks the hashes against it |

### Analysis of a scored query set: `results/<dataset>/e5/rehearse/<case>/` (and later `results/<pool>/e5/analyze/`)

Written by `memtk.e5.analyze` (via `rehearse`; later `analyze` for S). In git. "Query" = the
scored set (mock or S), always compared with H-eval (main: E4's tables; variants: `prepare`'s).
Excess intervals resample the query set by `cluster` and H-eval by patient, independently
(`calibration.bootstrap_draws` draws, seeded).

| File | Content |
| --- | --- |
| `review.txt` | Query set, scoring log, rates, the primary family with Holm, excess per variant, the size-confound check, AUC, RQ2, H2, groups, near-exact copies, and the mechanical OUTCOME line with the E8 exit count; `rehearse` adds what was expected and (positive case) the planted-copy recovery |
| `rates.csv` | Per `set` (main, noisy, main+noisy; decision L), `variant`, `class` (`e1_copy`, `e2_full`, `e2_partial`, `e2_flagged`), `stratum` (single, multi, all): `n_s`, `flagged_s`, `rate_s` with `rate_s_low` / `rate_s_high`, `n_h`, `flagged_h`, `rate_h` with interval, `excess`, `excess_low`, `excess_high`, `p` (one-sided bootstrap), `excess_count` (n_s × excess) with interval, `primary` (decision M family), `p_holm`, `beyond_chance` |
| `sensitivity.csv` | The rows of `rates.csv` for the main set, single and multi, every variant (decision O) |
| `auc.csv` | Per `score` (`e1_score`, `e2_C`, `e2_strongest_s`) and `stratum`: `n_s`, `n_h`, `auc`, `auc_low`, `auc_high` (two-level bootstrap), `mann_whitney_p` (one-sided; treats slices as independent) |
| `rates_by_size.csv` | Per `class`, `stratum` and `bin` (size range of the largest lesion, or `standardised`): the `rates.csv` measures per bin; the `standardised` row compares the query set with H-eval reweighted to the query set's (stratum ×) size mix: `rate_s`, `rate_h`, `excess` with interval, `p` |
| `rates_by_scheme.csv` | Per `scheme`, `class`, `stratum`: the `rates.csv` measures against all of H-eval |
| `rq2_crosstab.csv` | Per `set` (query, H-eval), on the informative subset: `e1_copy`, `e2_class`, `slices`, `share` |
| `h2.csv` | Per `comparison` ((i) E2 full vs E1 copy; (ii) E2 flagged vs E1 at tau_D*) and E2 `stratum` (all = H-eval standardised to the query set's mix): `n_s`, `n_h`, `rate_s_e2`, `rate_h_e2`, `excess_e2`, `rate_s_e1`, `rate_h_e1`, `excess_e1`, `difference` (E2 − E1 excess) with `difference_low` / `difference_high`, `p`, `h2_holds`, discordant counts `s_e2_only`, `s_e1_only`, `h_e2_only`, `h_e1_only` |
| `groups.csv` | Per `set` (query set by analysis set; H-eval) and `class`: `flags`, `source_slices`, `groups` (chained ±2, decision N), `patients` |
| `near_exact.csv` | Per `set`: `near_exact` and `near_exact_groups` (decision Q), `beyond_heldout_e2` and its groups, `e8_strong_groups` (groups of both together) and `e8_eligible`, the plain count and rate of E1 score > 0.9 per stratum |
| `flagged.csv` | One row per flagged query (E1 copy or E2 full / partial), main variant, per analysis set: `set`, `key`, `mask_class`, `scheme`, `cluster`, `n_lesions`, `stratum_e1`, `n_informative`, `stratum_e2`, `l1_area`, `e1_score`, `e1_copy`, `e1_nn`, `e1_nn_patient`, `e1_group`, `e2_class`, `e2_C`, `e2_nn`, `e2_nn_patient`, `e2_strongest_s`, `e2_strongest_train`, `e2_strongest_patient`, `e2_source` (decision J's slice), `e2_source_patient`, `e2_group`, `e2_n_kept`, `e2_all_copied`, `copied_area_share` (E2 flags), `near_exact`, `beyond_heldout_e2`. Groups are `<patient>:z<first z>` within the set. Read by E6, E7, E8 |
| `score_distributions.png` | E1 score, E2 C and strongest pair of the query set and H-eval per stratum, with the frozen thresholds |
| `rates_forest.png` | Excess with intervals per class, set, variant and stratum (primary cells red) |
| `copied_area.png` | Copied-area share of the E2-flagged queries by class and stratum (if any) |
| `near_exact.png` | Near-exact copies (highest E1 first, up to 12) next to their source slice (if any) |
| `groups_top.png` | One query per largest E2 group next to its source slice (if any) |
| `run_info.json` | Run log (as above; `analyze` also logs the three SHA-256 and the outcome reading) |

### `results/<pool>/e5/check_report.txt`

Written by `check`. In git. One PASS/FAIL line per check: decisions L–Q set and implemented; the
run logs of `prepare`, `rehearse`, `score` and `analyze` made with today's `audit:`, seed,
bootstrap draws and attach gap, in that order, with S read by `score` only; the lock (the `audit:`
section, the frozen E3 file and the E5 file unchanged since `score`, in the marker and both logs;
the E5 file made with them); one complete `score` run with every variant and one row per mask in
every class table; the sensitivity thresholds, tau_D*, held-out maxima and drop-only identity
reproduced; every `analyze` table and the outcome reading reproduced from the raw tables with the
same seed; no H-eval slice near-exact; the unit tests. Exit code 1 on any failure. Run log:
`check.run_info.json`.

## 9. E6: patient exposure (`e6_exposure.py`; `docs/e6-e7-plan.md`)

Every E6 choice is in the config section `exposure:` (decisions R, U, V, W). `reference` reads T and
stored H-eval tables only; the stages that read S's tables (planned) use E5's stored tables and
never score S again.

### Reference tables: `<output_root>/<dataset>/e6/`

Written by `reference`. Not in git. Read by the planned `rehearse`, `exposure` (E6) and `reports`
(E7, Panel C). Run log: `reference.run_info.json` (inputs T and the H-eval tables; SHA-256 of the
frozen E3 file).

| File | One row per | Columns |
| --- | --- | --- |
| `lesions3d.csv` | 3D lesion of a T volume, per labelling method (decision W) | `lesion3d_id` (`<patient_id>/D<n>`, numbered by first appearance within the method), `patient_id`, `number`, `z_first`, `z_last`, `span` (slices containing it, thin end slices included), `volume_px`, `largest_area_px` (on the mask as saved), `max_parts` (most separate pieces in one slice; > 1 only under 26-connectivity: joined neighbours), `method` (`slice_linking` or `connectivity_26`), `n_2d_lesions` (E0b lesions mapped to it), `largest_informative_px` (largest mapped E0b lesion of at least the bound; empty if none), `informative` |
| `lesion_map.csv` | E0b 2D lesion of a T slice (≥ 4 px; 6,101) | `lesion_id`, `key`, `patient_id`, `z`, `label`, `area`, `centroid_row_brain`, `centroid_col_brain` (from `lesions.csv`), `lesion3d_id_slice_linking`, `lesion3d_id_connectivity_26` (the 3D lesion holding most of its pixels), `informative` (area ≥ the bound) |
| `rarity_slices.csv` | T slice with a lesion ≥ 4 px (3,804) | `key`, `patient_id`, `n_informative`, `e1_best_other` (best shifted Dice against other patients' T slices), `e1_best_other_slice`, `e2_best_other` (best configuration score C against other patients' slices; empty without an informative lesion, 0 without a candidate), `rarity_e1`, `rarity_e2` (1 − the best score; decision V) |
| `patients.csv` | T patient (105) | `patient_id`, the decision V features `n_lesions3d` (main method), `volume_px` (E0b lesion pixels over all slices), `median_area_px`, `max_lesions_per_slice`, `position_spread` (√ of the summed variances of the brain-relative centroids), `rarity` (mean distance to the k nearest other patients on the z-scored features), `t_slice_share` (w_p, the prior of pi_p; decision R), `t_slices` |
| `heldout_nn.csv` | (detector, stratum, T patient) named as nearest neighbour by H-eval | `detector` (`e1`, `e2`), `stratum` (`single`, `multi`; E1 by lesions ≥ 4 px, E2 by informative lesions), `patient_id`, `count` (scorable H-eval masks whose nearest neighbour belongs to the patient; the n_p of pi_p) |

### `results/<dataset>/e6/reference/`

Written by `reference`. In git.

| File | Content |
| --- | --- |
| `review.txt` | T slices used and left out; 3D lesion counts per method (with an E0b lesion, informative, joined neighbours, span quartiles) and how many 26-connected lesions linking splits; slice-rarity quartiles per detector; the five rarest patients with their features; the coverage of the held-out nearest neighbours per detector and stratum |
| `reference.png` | Slice span of the informative 3D lesions per method; slice rarity (E1, E2); patient rarity against the number of 3D lesions |
| `run_info.json` | Run log |

### Analysis of a query set: `results/<dataset>/e6/rehearse/<case>/` and `results/<pool>/e6/exposure/`

Written by `rehearse` (cases `null`: H-cal; `positive`: H-cal + planted P1, P2, P4; `placement`,
`placement_compact`: H-cal with every lesion moved to a random T lesion's position, step 3b) with
`e6.exposure.analyze_query` and `draw_figures`, the same code `exposure` will run on S. In git.
"Cell" = class × stratum (`e1_copy`, `e2_full`, `e2_partial`, `e2_flagged` from `exposure.classes`;
`single`, `multi`); primary cells from `exposure.primary`.

| File | One row per | Columns / content |
| --- | --- | --- |
| `exposure.csv` | cell × smoothing m × T patient | `patient_id`, `observed` (flags with this source patient, E5 decision J), `heldout_nn` (n_p), `pi`, `expected` (pi × flags), `ratio`, `ratio_low`, `ratio_high` (both sides resampled by cluster), `groups` (chained source groups, E5 decision N), `null_low`, `null_high` (the 2.5–97.5% band of the ratio under the multinomial null), `cell`, `m` (decision R and its sensitivity) |
| `concentration.csv` | cell × m | `primary`, `flags`, `patients` (105), `patients_flagged`, `dispersion` with `dispersion_null_median`, `dispersion_null_p95`, `dispersion_null_pct` (the configured percentile), `dispersion_p`; `top5_share` with its null median, 95th percentile and p; `above_null_band` (patients above their band) and `above_null_band_expected` |
| `margins.csv` | cell | `primary`, `flags_q`, `flags_h` (query and H-eval chance flags); `margin_median_q`, `margin_median_h`, `margin_auc` (P(query margin > H-eval margin)), `margin_auc_low`, `margin_auc_high`; the same with `rarity_` for the source slice's rarity on the cell's detector scale; `source_slices` and `rarity_vs_t_*` (distinct source slices against all other T slices, both by patient: H3 at slice level) |
| `h3.csv` | cell | `primary`, `n` (patients), `rho` (Spearman of patient rarity and exposure ratio), `rho_low`, `rho_high` (patients and flags resampled), `p_permutation` |
| `h1.csv` | 3D method × size bin (+ `weighted`, `all (plain rho_s)`) | `bin`, `lesions3d` (informative 3D lesions), `credited_s`, `credited_h` (3D lesions with at least one credit), `rho_s`, `rho_h` (Spearman of span and credits), `difference`, `difference_low`, `difference_high` (H1 holds if the weighted row's interval is above 0), `method` |
| `flag_details.csv` | flag per cell, of the query set and of H-eval | `set` (`query`, `H-eval`), `cell`, `key`, `cluster`, `source`, `source_patient`, `margin` (to the second-best patient on the cell's scale), `source_rarity`; read by E7 (Panel C) |
| `top_patients.csv` | primary cell × the 10 most exposed patients (by `ratio_low`) | the `exposure.csv` columns with the patient's features from `patients.csv` |
| `review.txt` | — | the tables above and the reading of decision U per primary cell and overall; in the positive case also the recovery of the planted copies' base patients |
| `h1_rates.csv` | 3D method × size bin × span group × set (query, H-eval) | `bin`, `span`, `lesions3d`, `rate_per_1000` (mean credits per 3D lesion per 1,000 scorable masks) with `rate_low`, `rate_high`, `relative` (to the size bin's mean rate) with `relative_low`, `relative_high`, `set`, `method` (descriptive view of H1, added after the S run) |
| `h1_rates.png` | — | per size bin, the relative rate by span group, query set (red) and H-eval (grey), with intervals (slice-linking 3D lesions) |
| `exposure_sorted.png` | — | per primary cell, each T patient's observed against expected flags, with its 95% chance range (grey bar) and the diagonal; red = ratio interval above 1; ringed = planted base patients (positive case) |
| `margins.png` | — | margin and source rarity of the query flags and the H-eval chance flags as side-by-side boxes with every flag as a dot; AUC in the titles |
| `rarity_exposure.png` | — | observed / expected flags per fifth of the patients by rarity (H3), Spearman ρ in the titles |
| `h1_span.png` | — | ρ_S and ρ_H per size bin and method (H1) |
| `e5_rates.csv`, `e5_auc.csv` | — | placement mocks only, exploratory: E5's rates (excess over H-eval with intervals, main set) and AUCs (columns as E5's `rates.csv`, `auc.csv`) |
| `run_info.json` | — | run log, with the case and the reading |

## 10. E7: match reports (`e7_reports.py`; `docs/e6-e7-plan.md`)

Every E7 choice is in the config section `reports:` (decisions X, Y, Z). The report generator is
built and checked on E4's planted copies (`planted`) before any report on S is drawn.

### `results/<dataset>/e7/planted/`

Written by `planted`. In git.

| File | One row per | Columns / content |
| --- | --- | --- |
| `reports.csv` | planted mask with an informative lesion (698) | `key`, `variant`, `base_key`, `base_patient`, `gated`, `stratum` (E2), `e2_class`, `flagged`, `source` (E5 decision J), `strongest_train`, `e1_nn`, `n_query`, `n_kept` (pairs ≥ tau_L on the source slice), `all_copied`, `taxonomy`, `patient_ok`, `slice_ok`, `either_ok`, `lesions_ok` (decision Z), `verdict` (decision X) |
| `correctness.csv` | variant | `gated`, `flagged` (E2 full or partial), shares over the flagged masks: `patient`, `slice_within_window`, `either_source`, `lesions` (P6 only); `patient_all_gated` (an unflagged mask counts as not named) |
| `criteria.csv` | criterion of `reports.correctness` | `criterion`, `value`, `min`, `pass` |
| `deletion.csv` | deletion-test mask (`set`: planted controls, H-eval chance flags) | `key`, `source`, `top_lesion`, `small_lesion` (lesion numbers), `C_before`, `C_top`, `C_small` (against the source slice), `e1_before`, `e1_top`, `e1_small` (whole-mask score against the source slice), `flag_before`, `flag_top`, `flag_small` (still flagged with the same source patient after a full re-match) |
| `deletion_summary.csv` | set | `masks`, mean drops of C and of the E1 score after removing the top and the smallest contributor with one-sided paired Wilcoxon p, flips with the exact McNemar p |
| `taxonomy_confusion.csv` | variant | gated, E2-flagged masks per taxonomy category (exact, shifted, deformed, partial) |
| `report_<variant>.png` | variant | one example match report (Panel A images, Panels B and C as text, verdict) |
| `review.txt`, `run_info.json` | — | the tables above with the criteria verdict; run log |

### `results/<pool>/e7/reports/`

Written by `reports`. In git.

| File | One row per | Columns / content |
| --- | --- | --- |
| `report_index.csv` | report drawn (decision X) | `report` (file name), `set` (`S`, `H-eval`), `key`, `reason` (near-exact, beyond held-out, exposed <cell>, random S flag, H-eval chance flag), `group` (chained, E5 decision N), `e1_score`, `e2_class`, `C`, `source`, `source_patient`, `taxonomy`, `verdict` (patients named by their dataset IDs) |
| `reports/<nn>_<reason>.png` | report | the match report (Panel A: lesions outlined, source lesions in their query partner's colour, the named training slice whole with the zoom as a box, the overlay; Panels B and C as text; the verdict) |
| `deletion.csv`, `deletion_summary.csv` | S E2-flagged multi-lesion mask with ≥ `min_paired` contributing pairs / set (S, H-eval chance flags) | as in `results/<dataset>/e7/planted/` |
| `taxonomy.csv` | set × stratum × category | `set`, `stratum`, `taxonomy`, `count`, `share` (every E2-flagged mask of S, main set, and of H-eval) |
| `patient_profiles.csv` | 2D lesion of a profiled patient (the 3 most exposed per primary cell of E6; descriptive, added after the S run) | `lesion_id`, `z`, `area`, `lesion3d_id` (slice linking), `credits` (S masks with a lesion pair ≥ tau_L to this lesion), `e1_flags`, `e2_flags` (S flags naming the lesion's slice), `patient_id` |
| `patients/<patient_id>.png` | profiled patient | the patient-level report: 3D lesions over their slices (labelled with span and largest area) with the S matches per lesion and slice, S flags per slice, the most-matched slice whole (zoom as a box) and zoomed next to up to 4 of its S masks, lesions outlined |
| `review.txt`, `run_info.json` | — | reasons, the report list with verdicts, the deletion test, the taxonomy shares S vs H-eval; run log with E5's hashes |

### `results/<pool>/e6/check_report.txt` and `results/<pool>/e7/check_report.txt`

Written by the `check` stages. In git. One PASS/FAIL line per check (E6: decisions R, U, V, W; run
logs of `reference`, `rehearse`, `exposure` in order with S read by `exposure` only; E5's lock; every
reference and exposure table and the reading reproduced; the rehearsal readings; the unit tests. E7:
decisions X, Y, Z; run logs of `planted`, `reports`; E5's lock; the planted criteria and every planted
table; on S the selection, every verdict and taxonomy, the deletion test, taxonomy and profile tables
reproduced and every figure present; the unit tests). Exit code 1 on any failure. Run log:
`check.run_info.json`.
