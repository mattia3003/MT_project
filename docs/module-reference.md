# Module reference

Every code module: what it does, its main functions, what it imports from `memtk` and which
modules use it (so the effect of a change can be traced), and its external libraries. The short
overview of how everything fits together is `docs/architecture.md`; every file the pipeline
reads or writes is in `docs/data-dictionary.md`. Function details are in each function's
docstring.

Layers (a module only imports from its own layer or the layers below):
1. **Scripts** (`scripts/`): the command line, one per experiment.
2. **Stage packages** (`src/memtk/e0b/`, later one per experiment): one experiment's stages,
   one module per topic; they read inputs, call library modules, and write outputs.
3. **Library modules** (`src/memtk/*.py`): reusable building blocks with no knowledge of stages
   or experiments.
4. **Foundation:** `config` (the only module that knows the YAML config and the repository root)
   and `io` (files, tables, stacks).

`scripts/e0a_prepare_data.py` predates the stage packages: its stages are small and call library
modules directly.

Imports and "used by" were taken from the code on 2026-11-03 (all `from memtk...` imports;
updated after `memtk.stages` was added, E1–E3 step 1), re-taken on 2026-11-04 after
`memtk.sets` was added (E4 step 1) and on 2026-11-15 after `memtk.audit` was added and E4's
shared helpers moved to the library (E5 step 1).

---

## Library modules and foundation (`src/memtk/*.py`)

| Module | Purpose | Main functions | Imports from memtk | Used by | External libraries |
| --- | --- | --- | --- | --- | --- |
| `config` | Load the YAML config, look up datasets and pools, resolve paths from the repo root, write run logs | `load_config`, `value` (dotted key), `real_dataset`, `synthetic_pool`, `output_dir`, `results_dir`, `split_file`, `frozen_file` (an experiment's frozen file, `<experiment>_<dataset>.yaml`: E3 thresholds by default, E5's pre-S values), `write_run_info` | — | `io`, `synthetic`, `checks`, `stages`, `sets`, `calibration`, every `e0b`, `e3`, `e4` and `e5` module, every script | PyYAML |
| `stages` | Helpers every stage package and its script share: the command-line options of a stage, decisions read from the config (stops with the review command to run while a value is `null`), the overwrite and input guards, random picks for example figures | `add_stage`, `decided` (dotted key, e.g. `decided(cfg, "brain.method", "e0b_audit.py brain-review")`), `refuse_overwrite`, `require`, `pick` | `config` | `sets`, `calibration`, every `e0b` module except `e0b.common`, every `e3`, `e4` and `e5` module, `e0b_audit.py`, `e3_calibrate.py`, `e4_validate.py`, `e5_audit.py` | NumPy, pandas |
| `io` | Read and write NIfTI; a patient's model-space volumes (moved from `e0b.common` on 2026-11-04); BRAVO slices by key for figures (moved from `e4.common` on 2026-11-15); locate patients and their files; mask stacks (many 2D boolean masks in one bit-packed `.npz`) and label stacks (many 2D label maps in one sparse `.npz`), looked up by key | `load_image`, `save_nifti`, `load_real` (bravo and seg volumes of one patient), `slice_images` (model-space BRAVO slice per real slice key, one volume read per patient), `read_table` (pipeline CSVs with ID columns kept as text; `ID_COLUMNS` covers the detector tables' `query`, `nn`, `train` and lesion-ID columns since 2026-11-15, for numeric-looking S sample IDs), `list_patients`, `patient_files`, `MaskStackWriter`, `MaskStack`, `LabelStackWriter`, `LabelStack` | `config`, `ids` | `split`, `synthetic`, `checks`, `sets`, every `e0b` module, `e3.common`, `e3.freeze`, `e3.check`, `e4.common`, `e4.plant`, `e4.validate`, `e4.check`, every `e5` module, `lesionmatch`, `e0a_prepare_data.py` | nibabel, pandas |
| `ids` | The ID convention shared by every experiment: slice keys (`Mets_005/z127`, sample ID) and lesion IDs (`<slice key>/L<n>`); unique per source, `(source, lesion_id)` globally | `slice_key`, `sample_key`, `parse_slice_key`, `lesion_id`, `parse_lesion_id` | — | `io`, `checks`, `lesionmatch`, `calibration`, `audit`, `planted`, `e0b.common`, `e0b.brain`, `e0b.lesions`, `e0b.stats`, `e3.bound`, `e4.plant`, `e4.validate`, `e4.check`, `e5.prepare`, `e5.analyze` | — |
| `transforms` | The original training transform (model space) and slice counts | `model_space_transform`, `convert_volume_pair`, `binarize_mask`, `lesion_px_per_slice` | — | `split`, `checks`, `e0a_prepare_data.py` | MONAI, PyTorch |
| `split` | Seeded matched-pair split into T / H-cal / H-eval | `make_split`, `paired_split`, `summarize_split` | `io`, `transforms` | `e0a_prepare_data.py` | pandas |
| `synthetic` | Read synthetic samples in place; sample IDs; index a pool; parse the legacy QC list | `index_pool`, `load_sample`, `sample_id`, `name_scheme`, `read_legacy_exclusions` | `config`, `io`, `lesions` | `checks`, `e0b.common`, `e0b.brain`, `e0b.check`, `e0b.lesions`, `e5.score` (`name_scheme`), `e5.analyze` (`load_sample`), `e0a_prepare_data.py` | pandas, tqdm |
| `lesions` | 8-connected labelling; attach-or-drop (specks below the minimum area become strays of a lesion within the gap, or are dropped; lesions numbered by area); lesion features (centroid in three frames, bbox, shape, fraction outside the brain); brain masks (six candidate methods), tissue area, brain box; distance and growing helpers | `label_components`, `count_components`, `attach_or_drop` (→ `LesionLabels`), `speck_gaps`, `lesion_features`, `brain_mask`, `tissue_px`, `brain_box`, `fraction_outside`, `chessboard_distance`, `grow` | — | `plotting`, `synthetic`, `checks`, `planted`, `sets`, `e0b.brain`, `e0b.check`, `e0b.filters`, `e0b.lesions`, `e0b.qc`, `e0b.stats` | SciPy, scikit-image |
| `qc` | Quality classes of synthetic samples: mask class from the mask alone (`no_lesion`, `noisy`, `minor`, `clean`), the combined class with a failed MRI first, the edge count per slice | `mask_class`, `qc_class`, `edge_lesions`, `MASK_CLASSES`, `QC_CLASSES` | — | `checks`, `e0b.qc` | pandas |
| `plotting` | Shared figure helpers and colours (no pyplot, so importing memtk never changes the plotting backend). `save` wraps the figure title to the figure width and uses the constrained layout, so the title never covers tile titles or runs off the image. `draw_lesions` draws a labelled slice (lesion colours with L-numbers, lighter strays, specks as ×, brain outline and box) | `tile_figure`, `show_slice`, `draw_box`, `hide_unused`, `save`, `draw_lesions`, `lesion_sheet`; zoomed label maps (moved from `e4.plant` / `e4.validate` on 2026-11-15, E5 uses them): `zoom_window`, `fill`, `lesion_colors`, `draw_labels`; colours `BRAIN_COLOR`, `LESION_COLOR`, `BOX_COLOR`, `HIGHLIGHT_COLOR`, `SPECK_COLOR`, `LESION_COLORS`, `SET_COLORS`; `ZOOM_MARGIN`, `ZOOM_MIN_CROP` | `lesions` | `checks`, `e0b.brain`, `e0b.lesions`, `e0b.qc`, `e0b.stats`, `e3.bound`, `e3.calibrate`, `e4.plant`, `e4.validate`, `e5.analyze` | matplotlib |
| `wholemask` | E1, the whole-mask nearest-neighbour detector: plain Dice to every training mask (one sparse matrix product, queries in blocks), candidates (best by plain Dice plus nearest by the centroid of lesion L1), best Dice over shifts of ±`max_shift` px (zero padding, no wrap-around; ties: smallest shift), nearest neighbour, runner-up patient and margin. Refuses masks without a lesion | `match` (one row per query), `dice_matrix`, `shifted_dice`, `shift_order`, `candidate_indices`, `largest_lesion_centroids` | — | `calibration` (`run_frozen`), `e3.calibrate`, `e3.check`, `e5.prepare` | NumPy, SciPy (sparse), pandas |
| `lesionmatch` | E2, the lesion-level detector: lesion records from label maps (informative lesions only, so E4's planted copies take the same path), lesion score s = shape (IoU of bit-packed crops centred on the rounded centroids) × position (pixel distance, decision F) × size, exact shortcuts (KD-tree prefilter; shape only where position × size reaches the candidate cut, then for every pair of a candidate slice), Hungarian pairing per candidate slice pair (vectorized when one side has one lesion). The configuration score C counts every Hungarian pair from the candidate cut (independent of tau_L); tau_L and tau_C are applied afterwards (one value each, or one per query slice for stratified thresholds); classes: full = C(q, NN) >= tau_C, partial = not full but a lesion pair >= tau_L | `match` (→ `LesionMatch` with `queries`, `lesions`, `slice_pairs`, `lesion_pairs`; methods `slice_scores(tau_L)` (adds `C`, `n_kept`, `all_copied`), `summary(tau_L, tau_C)`), `lesion_records` (→ `Lesions`), `packed_crops`, `position_factor`, `size_factor`, `shape_factor`, `assign_pairs`, `save_tables` / `load_tables` (the four result tables as `e2_<tag>_*.csv` / `.parquet`; added 2026-11-15, used by E3, E4, E5) | `ids`, `io` | `calibration`, `audit`, `e3.common`, `e3.bound`, `e3.calibrate`, `e3.check`, `e4.plant`, `e4.validate`, `e4.check`, `e5.prepare`, `e5.score`, `e5.analyze`, `e5.rehearse`, `e5.check` | NumPy, SciPy (KDTree, Hungarian), pandas |
| `calibration` | E3 building blocks: area bins, the chance-match rate of held-out lesions by area bin and tau_L, percentiles (no match = 0), the bound rule (smallest bin edge from which every larger bin stays below the cut), the single/multi stratum of a slice, the bootstrap-by-patient interval of a percentile. For E4 and E5: reading a frozen file with its SHA-256 (`load_frozen`, moved from `e4.common` on 2026-11-15) and each slice's threshold by its stratum, its detector settings as keywords for both detectors (E4 and E5 take settings only from there), both detectors run with them (`run_frozen`, moved from `e4.validate.run_detectors` on 2026-11-15), and the classes the frozen thresholds give the detectors' results (E1 copy; E2 summary with stratum, thresholds and the patients of the nearest neighbour and strongest-pair slice). Thresholds from H-cal (moved from `e3.calibrate` and `e3.freeze` on 2026-11-15, so E5 `prepare` re-derives the sensitivity thresholds with E3's own code): per-slice E2 scores, percentile thresholds pooled and per stratum, the single/multi values under decision H, the conditions that keep the E2 shortcuts exact, and the H-cal chance rates at given thresholds | `area_bins`, `bin_label`, `chance_rates`, `percentile_threshold`, `propose_bound`, `stratum`, `bootstrap_interval`, `read_frozen`, `load_frozen`, `FROZEN_BY`, `frozen_threshold`, `frozen_settings`, `run_frozen`, `e1_classes`, `e2_classes`, `bootstrap_share` (share of flagged slices with a bootstrap-by-patient interval); `STRATA`, `THRESHOLDS`, `lesion_slices`, `threshold_rows`, `thresholds`, `frozen_thresholds`, `shortcut_problems`, `chance_rates_at` | `config`, `ids`, `lesionmatch`, `sets`, `stages`, `wholemask` | `audit`, `e3.bound`, `e3.calibrate`, `e3.freeze`, `e3.check`, `e4.plant`, `e4.validate`, `e4.check`, every `e5` module | NumPy, pandas, PyYAML |
| `audit` | E5 building blocks (`docs/e5-plan.md`, step 1, 2026-11-15): per-slice classes of both detectors under the frozen thresholds and the slices each class is counted on (moved from `e4.validate`); a cluster bootstrap of column sums (clusters drawn with replacement = a multinomial over cluster types, exact, so 20,000 independent masks cost as little as 26 patients), percentile intervals, one-sided bootstrap p, the excess of a query set over the held-out rate (decision M), Holm's adjustment, standardised rates; AUC (= Mann–Whitney U / n₁n₂, ties ½) with a two-level bootstrap; chained adjacent-slice groups (decision N); the copied-area share (E5 step 7); held-out maxima per stratum and size bin and the scores beyond them (decision Q) | `class_table` (was `e4.validate.heval_classes`), `class_flags` (was `fp_flags`), `share_row`, `CLASSES`, `resample_sums`, `interval`, `one_sided_p`, `excess_test`, `holm`, `standardized_rate`, `cell_rates` (rates standardised to given cell weights, point and bootstrap draws; added in E5 step 3 for the size-confound check and H2), `auc`, `auc_bootstrap`, `chain_groups`, `copied_area_share`, `cell_maxima`, `beyond_max` | `calibration`, `ids`, `lesionmatch`, `sets` | `e4.validate`, `e4.check`, `e5.common`, `e5.prepare`, `e5.score`, `e5.analyze` | NumPy, pandas |
| `planted` | E4 planted copies with known origin (decision I): base selection (a group of patients with many multi-lesion slices and one of other patients; slices spaced in z; a patient who cannot meet the quota is skipped), the variants P1 exact, P2/P3 shift (one axis direction, exact distance, never off the image), P4 rotation about the lesion-pixel centroid, P5 erode/dilate with the 3×3 cross, P6-remove, P6-add (held-out lesion at its own position, inside the brain, ≥ the gap from the base lesions); each planted mask labelled with attach-or-drop and each planted lesion traced to its source lesion (largest overlap) | `choose_bases`, `variants_of` (→ list of `Planted`, skipped variants with reasons), `planted_key`, `shift`, `shift_direction`, `rotate`, `rotate_labels`, `morph`, `add_lesion`, `provenance`, `lesion_centroid`; `VARIANTS`, `AXIS_DIRECTIONS`, `CROSS` | `ids`, `lesions` | `e4.plant`, `e4.validate` (`VARIANTS`; later E7) | NumPy, SciPy (ndimage), pandas |
| `sets` | The slice sets the detectors compare as label-map stacks from the E0b lesion tables: the real sets (T, H-cal, H-eval; held-out sets after decision E, T never filtered) and a synthetic pool by `mask_class` (E5, decision L; uint8 label maps, each sample its own cluster); slices without a lesion left out and counted; label maps made again with other attach-or-drop settings (E5's drop-only variant). Moved from `e3.common` on 2026-11-04 (E4 step 1); `load_pool`, `relabel` added 2026-11-15 (E5 step 1); `load_planted` moved from `e4.common` 2026-11-15 (E5 `rehearse`, later E7) | `SliceSet`, `load_set`, `load_pool`, `load_planted` (the E4 planted table and label maps), `relabel`, `decision_e` (`max_mask_px` from `slices.csv`; other filters stop with a message) | `config`, `io`, `lesions`, `stages` | `calibration`, `audit`, `e3.bound`, `e3.calibrate`, `e3.check`, `e4.plant`, `e4.validate`, `e4.check`, every `e5` module | NumPy, pandas |
| `checks` | E0 consistency checks. E0a: split, volumes, independent reference transform, synthetic index, sample grid. E0b: lesion tables vs label maps, label maps vs masks as saved, label volumes, QC table, stack keys, run logs vs the current config | `CheckReport`, `check_split`, `check_real_volumes`, `check_recompute`, `check_synthetic_index`, `sample_grid`, `check_lesion_tables`, `check_labels_vs_masks`, `check_label_volumes`, `check_qc_table`, `check_stack_keys`, `check_run_logs` | `config`, `ids`, `io`, `lesions`, `plotting`, `qc`, `synthetic`, `transforms` | `e0b.check`, `e3.check`, `e4.check`, `e5.check` (`CheckReport`, `check_run_logs`), `e0a_prepare_data.py` | PyTorch, matplotlib, tqdm |

## Stage package `e0b` (`src/memtk/e0b/`)

Every stage function is `run_<stage>(cfg, **options)`, with the options named like the
command-line options; each module has its stages first, then the computations, tables and
figures they use, under section headers.

| Module | Stages (command) | Also contains | Imports from memtk |
| --- | --- | --- | --- |
| `e0b.common` | — | `boxes_of`, `Slices`, `BOX_COLUMNS` (the helpers every stage package uses moved to `memtk.stages`, 2026-11-03; `load_real` moved to `memtk.io`, 2026-11-04) | `config`, `ids`, `io`, `synthetic` |
| `e0b.brain` | `run_brain_review` (`brain-review`, decision A), `run_brain` (`brain`) | `brain_row`, `evaluate_brain_methods`, `threshold_sensitivity`, `summarize_brain_methods`, `propose_brain_method`, `summarize_threshold_sensitivity`, `tissue_counts`, `summarize_brains`, `method_grid`, `sorted_sheet`, `tissue_histogram`, `brain_sheet` | `e0b.common`, `config`, `ids`, `io`, `lesions`, `plotting`, `stages`, `synthetic` |
| `e0b.lesions` | `run_lesions_review` (`lesions-review`, decisions B, C), `run_lesions` (`lesions`), `run_view` (`view`) | `Labeller`, `component_table`, `gap_to_nearest_lesion`, `propose_min_area`, `size_counts`, `mask_specks`, `summarize_attach_or_drop`, `gap_counts`, `summarize_lesions`, `size_figure`, `crop_sheet`, `lesion_view` | `e0b.common`, `config`, `ids`, `io`, `lesions`, `plotting`, `stages`, `synthetic` |
| `e0b.qc` | `run_qc_review` (`qc-review`, decision D), `run_qc` (`qc`) | `SPECK_STRATA`, `speck_stratum`, `summarize_speck_strata`, `summarize_qc` | `e0b.common`, `config`, `io`, `lesions`, `plotting`, `qc`, `stages` |
| `e0b.stats` | `run_stats` (`stats`) | `adjacency_pairs`, `native_vs_model`, `real_brain_heights`, `describe_sets`, `bound_impact`, `mask_dice`, `brain_by_height`, `cumulative_lines`, the figure functions | `config`, `ids`, `io`, `lesions`, `plotting`, `stages` |
| `e0b.filters` | `run_filters_review` (`filters-review`, decision E) | `FILTERS`, `filter_checks` (also used by E3/E4 to apply decision E), `summarize_filters`, `filter_cost` | `config`, `io`, `lesions`, `stages` |
| `e0b.check` | `run_check` (`check`) | `STAGE_KEYS`, `DECISION_KEYS` (the checks themselves are in `memtk.checks`) | `checks`, `config`, `io`, `lesions`, `stages`, `synthetic` |

## Stage package `e3` (`src/memtk/e3/`)

Same conventions as `e0b`. E3 reads T and H-cal only; S is never loaded (the cost of a bound for
S comes from the E0 bound-impact table).

| Module | Stages (command) | Also contains | Imports from memtk |
| --- | --- | --- | --- |
| `e3.common` | — | `load_hcal_tables` (the stored H-cal E1 table and E2 `LesionMatch`, via `lesionmatch.load_tables`), `lesion_settings` (E2 settings in pixels from the live config; checks decision F). `SliceSet`, `load_set`, `decision_e` moved to `memtk.sets` (2026-11-04) | `config`, `io`, `lesionmatch`, `stages` |
| `e3.bound` | `run_bound_review` (`bound-review`, decision G) | `EDGES`, `TAU_GRID`, `PROVISIONAL_MIN_AREA`, `CUTS`, `review_text`, `left_out_text`, `rates_figure`, `lesion_crop`, `example_sheet` | `e3.common`, `calibration`, `config`, `ids`, `lesionmatch`, `plotting`, `sets`, `stages` |
| `e3.calibrate` | `run_calibrate` (`calibrate`, evidence for decision H) | `STRICT_BOUND`, `SIGMA_ALT`, `PERCENTILE_ALT`, `FLAG_RANGE`, `MIN_MULTI_PATIENTS`, `stratum_threshold`, `class_rates`, `propose_strata`, `exclusion_counts`, `review_text`, `score_figure` (`lesion_slices`, `threshold_rows`, `thresholds`, `STRATA` moved to `calibration`, 2026-11-15) | `e3.bound` (`EDGES`), `e3.common`, `calibration`, `config`, `lesionmatch`, `plotting`, `sets`, `stages`, `wholemask` |
| `e3.freeze` | `run_freeze` (`freeze`, needs decision H) | `repo_relative`, `compare_tables` (`DETECTORS` → `calibration.THRESHOLDS`; `frozen_thresholds`, `shortcut_problems`, `chance_rates_at` moved to `calibration`, 2026-11-15) | `e3.common`, `calibration`, `config`, `io`, `stages` |
| `e3.check` | `run_check` (`check`) | `DECISION_KEYS`, `DETECTOR_KEYS`, `CALIBRATE_KEYS`, `FREEZE_KEYS`, `brute_force_differences`, `self_match` | `e3.common`, `e3.freeze`, `calibration`, `checks`, `config`, `io`, `lesionmatch`, `sets`, `stages`, `wholemask` |

## Stage package `e4` (`src/memtk/e4/`)

Same conventions as `e0b` and `e3`. E4 reads T, H-eval, the planted set and the frozen E3 file;
S is never loaded, and detector settings and thresholds come only from the frozen file.

| Module | Stages (command) | Also contains | Imports from memtk |
| --- | --- | --- | --- |
| `e4.common` | — | `STREAMS`, `streams` (one generator per random choice of the planted set), `gate_status` (PASS, ACCEPTED or FAIL from `gate.csv`; for E5). Moved on 2026-11-15: `load_frozen` to `calibration`, `slice_images` to `io`, `load_planted` to `sets` | `config`, `io`, `stages` |
| `e4.plant` | `run_plant` (`plant`, needs decision I; evidence for decision K) | `PlantedSet`, `plant_set` (bases and variants, deterministic for a seed), `informative_lesions` (P6-add donors), `planted_table`, `planted_lesion_table`, `lesion_fate`, `variant_summary`, `area_change`, `stratum_or_none`, `describe`, `by_base`, `draw_change` (base lesions dashed, centre moves as arrows), `source_colors`, `variants_grid`, `variant_sheet`, `area_figure`, `review_text` (shows decision K from the config); `SIZE_BINS` is also used by `e4.validate`. `zoom_window`, `fill`, `draw_labels` moved to `plotting` (2026-11-15) | `e4.common`, `calibration`, `config`, `ids`, `io`, `lesionmatch`, `planted`, `plotting`, `sets`, `stages` |
| `e4.validate` | `run_validate` (`validate`, needs decision K; the gate) | `planted_outcomes` (caught by decision J), `validation_table`, `group_recall`, `fp_table`, `rq2_table`, `criterion_variants`, `criterion_caught`, `gate_table`, `gate_verdict` (PASS / ACCEPTED / FAIL under `validation.accepted_deviation`), `score_figure`, `recall_figure`, `factors_figure`, `misses_figure`, `flag_examples`, `flags_figure`, `review_text`, `verdict_text`; used by `e4.check` to reproduce the gate. Moved on 2026-11-15 (E5 needs them): `run_detectors` → `calibration.run_frozen`; `heval_classes`, `fp_flags`, `share_row` → `audit.class_table`, `audit.class_flags`, `audit.share_row`; `lesion_colors` → `plotting` | `audit`, `e4.common`, `e4.plant`, `calibration`, `config`, `ids`, `io`, `lesionmatch`, `planted`, `plotting`, `sets`, `stages` |
| `e4.check` | `run_check` (`check`) | `DECISION_KEYS`, `PLANT_KEYS`, `VALIDATE_KEYS`, `CAUGHT`, `stored_match` (E2 results rebuilt from the stored tables), `same_gate` | `audit`, `e4.common`, `e4.plant`, `e4.validate`, `calibration`, `checks`, `config`, `ids`, `io`, `lesionmatch`, `sets` |

## Stage package `e5` (`src/memtk/e5/`)

Same conventions. E5 (`docs/e5-plan.md`) reads S only in `score` (planned); every setting and
threshold comes from the frozen E3 file and the E5 file written by `prepare`; the `audit:` config
section (decisions L–Q) is hashed so that later stages refuse to run on a changed analysis.

| Module | Stages (command) | Also contains | Imports from memtk |
| --- | --- | --- | --- |
| `e5.common` | — | `REVIEW`, `RERUN` (what each variant kind runs again), `audit_config` (the `audit:` section, every value set and implemented), `audit_hash`, `parse_variant`, `variant_frozen` (the frozen E3 dict with a variant's thresholds and settings), `check_prepared` (the E5 file made with the current `audit:` and E3 file), `drop_only_t` (T with its drop-only labels), `score_marker`, `refuse_after_score` (pre-S stages stop once S has been scored) | `audit`, `calibration`, `config`, `io`, `sets`, `stages` |
| `e5.prepare` | `run_prepare` (`prepare`, needs decisions O, P, Q) | `E3_COLUMN`, `TOL`, `real_masks` (model-space masks as saved, by key), `drop_only_sets` (gap-0 label maps; H-cal and H-eval must be unchanged), `prepare_variant` (thresholds re-derived on H-cal and compared with E3; H-eval under the variant), `e1_table`, `tau_d_star` (decision P), `heldout_maxima` (decision Q), `heval_rates`, `review_text` | `e5.common`, `audit`, `calibration`, `config`, `ids`, `io`, `lesionmatch`, `sets`, `stages`, `wholemask` |
| `e5.score` | `run_score` (`score`, step 4: S once; needs the E4 gate, `prepare`, `rehearse`; the only E5 stage that reads S; writes the completion marker) | `query_table` (one row per query: cluster, mask class, scheme, lesion count, largest lesion), `score_query` (the one scoring function, run by `rehearse` on mock sets and by `score` on S: main and every variant of decision O; raw tables only), `run_row` | `e4.common` (`gate_status`), `e5.common`, `audit`, `calibration`, `config`, `io`, `lesionmatch`, `sets`, `stages`, `synthetic` |
| `e5.analyze` | `run_analyze` (`analyze`, step 5: needs `score`; refuses unless `audit:` and both frozen files equal those `score` recorded) | `Scored`, `load_scored`, `heldout` (H-eval as a `Scored` set), `informative_areas` (T lesion areas for the copied-area share), `subsets` (decision L), `analyze_scored` (the one analysis function, run by `rehearse` and later on S), `rates_table` (with the Holm primary family, decision M), `auc_table`, `size_table` (size-confound check), `scheme_table`, `informative`, `rq2_table`, `h2_table` (decision P), `sources` (decision J), `chained` (decision N), `groups_table`, `strong_flags`, `near_exact_table` (decision Q, E8 exit count), `flagged_table`, `outcome` (mechanical reading), `score_figure`, `forest_figure`, `copied_area_figure`, `examples_figure`, `draw_figures` (every figure; moved from `rehearse.figures` 2026-11-15 so S gets the rehearsed figures), `review_text`, `write_tables`; `STRATUM_COL`, `SCORES`, `MAX_EXAMPLES` | `e5.common`, `e5.score` (`query_table`), `audit`, `calibration`, `config`, `ids`, `io`, `lesionmatch`, `plotting`, `sets`, `stages`, `synthetic` (`load_sample`, figure backgrounds) |
| `e5.check` | `run_check` (`check`) | `LOG_KEYS`, `INPUTS` (the inputs each stage may read), `TOL`, `differing_tables` (recomputed tables against the stored CSVs, after the same write and read) | `e5.analyze`, `e5.common`, `e5.prepare`, `calibration`, `checks`, `config`, `io`, `lesionmatch`, `sets` |
| `e5.rehearse` | `run_rehearse` (`rehearse`, needs `prepare`) | `PLANTED_VARIANTS`, `EXPECTED`, `mock_set`, `planted_recovery` (positive case: flagged, right source patient, near-exact, base patients recovered) (`figures` → `e5.analyze.draw_figures`, `check_prepared`, `drop_only_t` moved to `e5.common`, `heldout`, `informative_areas` to `e5.analyze`, 2026-11-15, for `score` and `analyze`) | `e5.analyze`, `e5.common`, `e5.score`, `calibration`, `config`, `io`, `lesionmatch`, `sets`, `stages` |

## Scripts (`scripts/`)

| Script | Stages | Imports from memtk |
| --- | --- | --- |
| `e0a_prepare_data.py` | `split`, `real`, `synthetic`, `check` (small, call library modules directly) | `checks`, `config`, `io`, `split`, `synthetic`, `transforms` |
| `e0b_audit.py` | the E0b stages; only the command line (`stages.add_stage` adds the shared options) | `config`, `stages`, the `e0b` modules |
| `e3_calibrate.py` | the E3 stages `bound-review`, `calibrate`, `freeze`, `check`; only the command line | `config`, `stages`, the `e3` modules |
| `e4_validate.py` | the E4 stages `plant`, `validate`, `check`; only the command line | `config`, `stages`, `e4.plant`, `e4.validate`, `e4.check` |
| `e5_audit.py` | the E5 stages `prepare`, `rehearse`, `score`, `analyze`, `check`; only the command line | `config`, `stages`, `e5.prepare`, `e5.rehearse`, `e5.score`, `e5.analyze`, `e5.check` |

## Tests (`tests/`)

Pytest, on small hand-built inputs: `tests/test_lesions.py` covers labelling,
attach-or-drop (gaps, drop-only, nearest lesion and ties, numbering, nothing moved), lesion
features (all three frames, no brain, partly outside), brain masks, tissue area and the brain
box; `tests/test_io.py` covers the mask- and label-stack round trips and `read_table`;
`tests/test_ids.py` covers building, parsing and refusing IDs; `tests/test_stages.py` covers
the shared stage helpers (dotted-key decisions and their stop message, the overwrite and input
guards, seeded picks, the shared command-line options); `tests/test_qc.py` covers the
class precedence, the speck cut-off and the edge count; `tests/test_filters.py` covers the
sampling-filter checks; `tests/test_wholemask.py` covers E1 (known Dice values, the L1 centroid,
a shifted exact copy found with its shift, no wrap-around at the border, shifts beyond the range
not found, centroid candidates catching a copy with no plain overlap, the margin against another
patient, the smallest shift among ties, masks without a lesion refused); `tests/test_lesionmatch.py`
covers E2 (an identical lesion scores 1, the factors at known distances and sizes, a translated
lesion keeps its shape, the centred-crop IoU, symmetry, Hungarian beating greedy pairing, the
experiment plan's worked example C = 0.37, C counting pairs from the candidate cut and not
depending on tau_L, the classes (including a configuration copy without any lesion above tau_L)
and the nearest neighbour, thresholds per query slice, tau_L below the candidate cut and tau_C of
0 refused, lesions below the bound, the prefilter, and the shortcuts against brute
force on random label maps); `tests/test_calibration.py` covers area bins and labels, chance-match
rates (lesions, patients, missing scores never match), the percentile with missing scores as 0,
the bound rule (including a later bin above the cut and no bound at all), the stratum, and the
bootstrap interval (repeatable with a seed, collapsing for constant scores, resampling patients
rather than slices), reading frozen thresholds back per stratum, the frozen detector settings
as keywords (an unknown position frame refused), and the E1 and E2 classes under the frozen
thresholds of each query's stratum (with the patients of the nearest neighbour and strongest
pair, and queries without an informative lesion left out), and the bootstrap share (resampling
patients, repeatable, empty input); `tests/test_e4_validate.py` covers the gate verdict (PASS,
ACCEPTED only when the accepted deviation lists exactly the failed criteria, FAIL otherwise) and
the criterion's caught flag per variant (boolean, so `~` negates it: a regression test for the
first run's object-dtype bug); `tests/test_audit.py` covers the E5 building blocks (chained
groups joining runs within the gap and keeping patients apart, Holm against known values, AUC
equal to Mann–Whitney U / n₁n₂ with ties, the cluster bootstrap drawing whole clusters and equal
to an ordinary bootstrap for one row per cluster, the excess test finding no excess for equal
samples and one for a clear shift, interval and one-sided p, the AUC bootstrap centred on the
point estimate, standardised rates, the copied-area share on the source slice only, held-out
maxima and scores beyond them at bin edges); `tests/test_sets.py` covers `load_pool` (mask
classes kept, sample IDs as text and as clusters, uint8 label maps, counts left out) and
`relabel` (drop-only drops a stray that gap 2 attaches); `tests/test_e5_common.py` covers the E5
helpers (variant names, the audit config refusing null or unimplemented values, the audit hash
independent of key order, a variant's thresholds and settings applied to a copy of the frozen
dict); `tests/test_e5_analyze.py` covers the E5 analysis rules (analysis sets by mask class,
main + noisy only with both parts, the source slice of each class by decision J, chained group
labels, the three branches of the outcome reading); `tests/test_audit.py` also covers `cell_rates`
(the crude rate with a set's own weights, reweighting with others); `tests/test_io.py` also checks
that the detector tables' ID columns stay text;
`tests/test_planted.py` covers the
E4 planted copies (exact shifts that refuse to leave the image, shift directions only where there
is room, rotation by 0° as the identity and ±5° keeping area and centroid, cross erosion and
dilation, the added lesion's brain and gap rules, provenance by largest overlap with ties, every
variant of a two-lesion base with its parameters and sources, repeatable with the same seeds, the
P5 survivor rule, skipped P6 variants, and the base quotas, z spacing, skipped patients and
repeatability of the base draw); `tests/test_e3_freeze.py`
covers the freeze helpers (a changed threshold is found, decision H picks stratum or pooled
values, thresholds that would break the E2 shortcuts are refused); `tests/test_checks.py` runs the E0b consistency checks on consistent
outputs (pass) and on damaged copies (fail: wrong areas, centroids, IDs, counts, numbering,
labels outside the mask, wrong QC class, stale or out-of-order run logs).
Run with `.venv\Scripts\python.exe -m pytest`.

## Planned modules

E1–E4 are complete; E5's library module `audit` exists (2026-11-15; it took the planned
`stats` module's bootstrap and AUC part), and `e5.common`, `e5.prepare`, `e5.score`,
`e5.analyze`, `e5.rehearse`, `e5.check` with `scripts/e5_audit.py`; E5 is complete (2026-11-15). Later (names may change when each experiment is planned):

| Module | For | Purpose |
| --- | --- | --- |
| `exposure` | E6 | Exposure ratio, rarity, 3D lesion span, permutation tests (may extend `audit`) |
| `reports` | E7 | Match-report figures and verdict lines |
