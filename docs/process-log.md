# Process log: preliminary project

A dated log of how the work was done: stages built, re-runs and verifications, refactors, bugs
fixed, numbers that were replaced and why, and things tried and removed. It is the history
behind `docs/report-notes.md`, which keeps only what the report may use (decisions with their
evidence, findings, limitations, open questions). Per-experiment plans and stage status live in
`docs/e0b-plan.md` and its successors.

Entry format: date heading, newest at the bottom; each entry says **what was done or changed**,
why, and its *source* (commit, file in git, or pipeline stage). Replaced numbers are kept here
in full, with what replaced them.

---

## 2026-09-22 to 2026-10-29: earlier segmentation work

- **Segmentation experiments before this project** (ResNet50 metrics, synthetic-data statistics
  with and without corrupt files). Removed from the working tree on 2026-11-01 and kept under the
  git tag `segmentation-final`. *Source: commits 175b919 to 2270376, a80cbd6.*

## 2026-11-01: project setup and E0a

- **Library and config set up** (`src/memtk/`, `configs/default.yaml`, one seed 2026, run logs per
  stage). *Source: commits 694137c, 3f735e7.*
- **E0a built and checked:** `split`, `real`, `synthetic`, `check`; 32/32 checks pass. A code
  review fixed naming issues for new datasets. *Source: commits 3f735e7 to 3001571;
  `results/stanford_brainmets/e0a/check_report_amibekk_v1.txt`.*
- **Earlier per-slice native-resolution files were not reused.** The model space must be
  converted from the 3D volumes (per-volume min–max), so the old per-slice files could not
  serve. See "all comparisons happen in model space" in `report-notes.md` section 2.
- **Docs added:** `report-notes.md`, `architecture.md`, `e0b-plan.md`; null model dropped from the
  plans. *Source: commits 20ef4fc, d478e76, ac30d45, 04341cf.*
- **Preliminary observations, later measured on the full data** (moved from `report-notes.md`
  on 2026-11-03; the final numbers are in its sections 2 and 3):
  - In a sample of 1,600 synthetic masks, about 11% had no component under 40 px, so the
    sampling script's "< 40 px" filter did not hold for the whole pool. *Source: inspection
    sample.* Final: 10.7% of the masks as saved (decision E, `filters-review`, 2026-11-03).
  - Synthetic brains often looked smaller or off-centre than real lesion slices; some synthetic
    masks were dominated by scattered specks (e.g. sample `2750`). *Source: check grid.*
    Final: brain size by slice height (`stats`), speck counts (`lesions-review`, `qc-review`).
  - Synthetic lesions looked spatially concentrated: centroid spread (lesions ≥ 4 px) row SD
    14.9 vs 24.1 px, column SD 18.6 vs 25.3 px, around the same centre; brain boxes median 63×66
    vs 103×95 px; Otsu brain area median ~3,000 vs ~7,200 px; 3.8% of synthetic lesions more
    than half outside the Otsu brain (T 1.4%), 1.6% with the proposed `low_threshold_all` mask
    (T 0.02%). *Source: read-only sample, all T lesion slices vs 3,000 synthetic samples (old-QC
    and near-blank excluded); brain-review `review.txt`, 2,000 random samples.* Final: full-pool
    centroid spread 15.0/18.8 px (`stats`), 2.0% of lesions outside (`lesions`, 2026-11-02).

## 2026-11-02: E0b stages `brain` to `qc`, decisions A to D

- **Stages built:** `brain-review` → decision A, `brain`, `lesions-review` → decisions B, C,
  `lesions` + `view`, `qc-review` → decision D, `qc`. Pyright set up in standard mode.
  *Source: commits 5a49d50 to 0c30421; `docs/e0b-plan.md` status table.*
- **A stricter failed-MRI rule was tried and removed.** Implementation notes from the test (the
  rule and its rejection are in `report-notes.md` section 2, "a failed synthetic MRI"):
  - measured from the raw mask, the specks of noisy masks, grown by the 5 px margin, covered
    normal brains; ignoring components < 4 px fixed that;
  - the code (a temporary `brain-review` extension) was removed after the decision.
- **Open question resolved at the `qc` stage: does a failed MRI exclude its mask?** Answer: no
  (separate `mask_class` and `mri_failed` columns; see "QC classes" in `report-notes.md`
  section 2). The question as first recorded: the mask comes from stage 1 whatever stage 2
  does, so a blank or failed MRI does not make its mask invalid for E1–E5, which use masks
  only. A failed MRI matters where the brain is used: the edge check, brain-relative positions,
  and E8. Option: `mri_failed` as a flag that excludes a sample from those analyses only,
  rather than a QC class that removes the mask from S. The experiment plan did not say. The
  failed images come in at least two kinds: a lesion with a thin rim of tissue painted around
  it, and a small faint patch away from the lesion, each on an otherwise black image. *Source:
  `results/amibekk_v1/e0b/brain/outlines.png` (row 3) and a read-only check of its 8 samples.*

## 2026-11-03: E0b `stats`, `filters-review`, `check`, restructuring, full re-run

- **Stages built:** `stats` (items 1–6), `filters-review` → decision E, `check`. *Source:
  commits 444fcc6 to f785804.*
- **Brain-threshold noise levels re-measured.** "More than 3 px from the brain" was first
  measured in the four straight directions only; it now grows diagonally too (Chebyshev), like
  every other distance in the project. Old near-worst background levels 0.006 (T) / 0.021 (S),
  new 0.0004 / 0.013. The area changes and the conclusion did not change. *Source:
  `results/stanford_brainmets/e0b/brain_review/review.txt`, `threshold_sensitivity_amibekk_v1.csv`.*
- **Code restructured (E0b step 8b):** the stages moved out of `audit.py` (split up and removed)
  into the stage package `src/memtk/e0b/` (one module per topic); `scripts/e0b_audit.py` is only
  the command line; the code conventions were written into `CLAUDE.md`. *Source: commit 6850543.*
- **Full re-run of E0a and E0b from scratch (E0b step 9), verifying the restructuring.**
  - All 13 stages run in order with `--overwrite` (about 20 min); every stage exits 0. E0a check
    32/32, E0b check 43/43, pytest 60 passed.
  - `processed_data/model_space/`: all 15 tables and caches and all 624 volume files (156
    patients × `bravo`, `seg_soft`, `seg`, `labels`) are byte-identical to the run before the
    restructuring (compared against a copy and file hashes taken just before the re-run).
  - `results/` (`git diff`): no table changed. Run logs changed (time, full config; `script`
    now names the stage module, e.g. `src\memtk\e0b\stats.py`, the command is in `argv`). The
    E0b check report changed only in timestamps. Four figures changed with the same examples
    in the same order: the E0a check grid (layout from `plotting.save`, last drawn 2026-11-01),
    the QC class sheet and both lesion overview sheets (tile subtitles in the singular,
    outlines on the lesion numbers). All other figures were rewritten byte-identical, as they
    had already been drawn after the title fix of 2026-11-02. All figures checked by eye: no
    issues.
  - Not re-run: the `view` helper pictures in `results/*/e0b/views/` (on demand, not a stage).
- **Docs split:** process entries moved from `report-notes.md` into this log.
- **Two report-notes tables recomputed with the final QC classes** (they were made on
  2026-11-02 with a provisional cut at 3 isolated specks, before decision D set 9). A read-only
  script first reproduced the old numbers with the provisional cut (confirming the
  definitions), then recomputed with `qc.csv`. Old numbers, by provisional group:
  - *Lesions more than half outside the brain* (masks / lesions / mostly outside / share /
    masks affected): clean 32,137 / 51,816 / 838 / 1.6% / 829; minor (1–3 specks) 3,423 / 5,410
    / 71 / 1.3% / 70; noisy (> 3) 1,912 / 4,325 / 220 / 5.1% / 172; failed MRI 152 / 84 / 84 /
    100% / 82. With the final classes, clean and failed MRI are unchanged; minor 1.4%, noisy 6.6%.
  - *Stray growth* (masks / lesions with a stray / median growth / grown > 25% / stray px over
    lesion px): clean 32,219 / 5.9% / +7.7% / 0.4% / 0.4%; minor 3,038 / 36% / +9.1% / 4.8% /
    3.2%; noisy 1,904 / 78% / +16% / 23% / 9.3%. The old mask counts left out masks without a
    lesion (463); the final `mask_class` does the same. With the final classes, clean is
    unchanged; minor 43% / +10% / 6.9% / 4.1%, noisy 80% / +18% / 27% / 10.5%.

  Conclusions unchanged. *Source: `lesions.csv`, `qc.csv` of `amibekk_v1`; script kept outside
  the repo (read-only).*
- **Duplicate entry merged:** the 2026-11-02 report-notes entry "Synthetic lesions are smaller
  than real ones" (from the `lesions` summaries, all of S) was merged into the `stats` entry
  "Synthetic masks: smaller lesions, more two-lesion masks, no large lesions" (S main). It added
  the 5th percentile (4 vs 6 px) and the share with a lesion under 40 px, now given for S main
  (86.5%) in the table and for all of S (85.4%) in the text.
- **E0 closed; minimal lock file.** `requirements-lock.txt` pins the 39 packages the project
  needs (the direct dependencies from `pyproject.toml` plus their requirements, from the
  installed package metadata, no optional extras) at the versions that produced E0.
  - Packages from the earlier segmentation work (mlflow, clearml, transformers, itk, Flask, ...;
    183 in the project venv) were left out, and the old `docs/requirements.txt` was removed.
    `tqdm`, used in seven modules, was added to `pyproject.toml`.
  - `pip check` reports that monai 1.6.0 declares torch>=2.8 while torch is 2.6.0+cu124, so the
    lock is installed with `--no-deps`.
  - MONAI's `LoadImage` tries nibabel before ITK for `.nii` files (readers are tried from the
    last registered), so dropping itk cannot change loading.
  - Verified: a fresh Python 3.11 venv built from the lock with the documented commands re-ran
    all 13 E0 stages into a scratch folder (checks 32/32 and 43/43, pytest 60); all 692 outputs
    (volumes, tables, caches, figures, split) were byte-identical to the repo's; the E0b check
    report differed only in timestamps.
  - E0 marked done in `docs/experiment-plan.md` with its deviations; the CLAUDE.md E0 summary
    updated to the decided values.

  *Source: `requirements-lock.txt`, `pip check`, fresh-venv run.*
- **E0b plan closed.** `docs/e0b-plan.md` is marked reference only (no longer updated): the
  record of how E0b was built (decision gates A–E, definitions, stage details). Its open items
  moved to where they will be used:
  - position frame for copy detection, lesion crop and table formats (CSV / Parquet) →
    `docs/experiment-plan.md`, E2, "Carried over from E0"; the position frame also added to
    the experiment plan's open decisions;
  - 3D lesion linking from the real masks as saved, not the cleaned labels →
    `docs/experiment-plan.md`, E6, duplication test step 1;
  - per-pool decisions for a new synthetic pool (re-run `brain-review` and `qc-review`, per-pool
    overrides) → `scripts/README.md`, Notes.

  `CLAUDE.md` gets a working-plans rule: one plan document per experiment group, in the E0b
  form, closed when done with its open items moved on. Found while moving the per-pool item:
  the code reads only the global E0b decision values, so per-pool overrides need a small code
  change first (noted in `scripts/README.md`). Noted for when E6 is planned: its step 5 counts
  only training slices that pass the "< 40 px" sampling filter, which E0 found does not hold
  for 10.7% of the pool (decision E). *Source: `docs/e0b-plan.md` ("Carried forward"),
  `src/memtk/e0b/` (`decided(...)` calls).*
- **E1–E3 plan written** (`docs/e1-e3-plan.md`, the active working plan): decisions F (set by
  the user: pixel coordinates), G (bound) and H (threshold strata); library modules `stages`
  (helpers moved from E0b), `wholemask` (E1), `lesionmatch` (E2), `calibration` (E3); E3
  stages `bound-review`, `calibrate`, `freeze`, `check`; tests; schedule to the 9 Nov freeze.
  Decision F recorded in `docs/experiment-plan.md` (E2 position factor and prefilter in pixels,
  E3 step 3, E0 status, preprocessing text, open decisions), `CLAUDE.md`, and the
  `docs/thesis-plan.md` risk table, which still named brain-relative coordinates as the fix for
  unregistered slices. Choices made in the plan and accepted by the user: provisional τ_L from
  H-cal lesions ≥ 10 px; chance-match rate per lesion (for the bound) but τ_L as a per-slice
  statistic (for the thresholds); strata counted by the lesions each detector sees (≥ 4 px for
  E1, informative for E2); area bins, 2,000 bootstrap draws, frozen file in `configs/frozen/`.
  The architecture rework is scheduled before the code of step 1. *Source: `docs/e1-e3-plan.md`.*
- **Architecture docs reworked for readability.** `docs/architecture.md` (550 lines; its
  module diagram drew every import as an arrow and was hard to read) became a short overview
  for people (146 lines): the experiment flow E0–E8 with the freeze and the gate, the data flow
  between experiments, the code in four layers, folders and conventions. The detail moved,
  complete, to `docs/data-dictionary.md` (every file and column, in pipeline order, with an
  index of which stage reads and writes what) and `docs/module-reference.md` (every module with
  its functions, imports from memtk and "used by" as tables, taken from the code; they match the
  old diagram exactly). A script confirmed that every name in the old document appears in the
  new ones, except the two old placeholder module names the E1–E3 plan replaced (`wholemask`,
  `lesionmatch`) and two doc links whose rule is in CLAUDE.md. The CLAUDE.md update rule now
  says which document to change for which kind of change; the E0b and E1–E3 plans point to the
  new files. *Source: `git show HEAD:docs/architecture.md` against the three new documents.*
- **E1–E3 step 1: shared stage helpers and config sections.** `decided`, `refuse_overwrite`,
  `require`, `pick` (from `e0b/common.py`) and `add_stage` (from `e0b_audit.py`) moved to the
  library module `memtk/stages.py`, because E3 needs them too. `decided` now takes a dotted key
  and the full review command (`decided(cfg, "brain.method", "e0b_audit.py brain-review")`), so
  E3 can read nested keys such as `calibration.strata.lesion` and the stop message names the
  right script; the 19 E0b calls were updated (the diff contains only those lines and the
  imports). `config.value` and `decided` are annotated `-> Any`: the dotted lookup had made
  Pylance infer `dict | Unknown` and report 34 errors in the E0b modules. New config sections
  `detectors:` (fixed plan values; decision F = pixel, set with its reason) and `calibration:`
  (percentile 95, 2,000 bootstrap draws; decisions G and H `null`). Verified: pytest 66 passed
  (6 new in `tests/test_stages.py`), E0b check 43/43 (report changed only in the pytest line),
  no Pylance problems. *Source: `git diff`, pytest, `e0b_audit.py check`.*
- **E1–E3 step 2: E1 library `memtk/wholemask.py`.** Plain Dice to every training mask by one
  sparse matrix product (queries in blocks of 512); shortlist of the 20 best by plain Dice plus
  the 20 nearest by the centroid of lesion L1; best Dice over ±3 px shifts of the query (zero
  padding, no wrap-around; the query's full area stays in the denominator); ties to the smallest
  shift, then the higher plain Dice, then the first training slice; runner-up patient and margin;
  masks without a lesion refused (the stages leave them out and report them). 10 tests (76 in
  total); one test first failed because its hand-built map did not number the largest lesion
  L1, fixed in the test. Self-match sanity on 300 random T slices against T: all scored 1 at no
  shift; 296 found themselves, 4 an identical neighbouring slice of the same patient. 4.4 ms per
  query (S main about 3 min). One Pylance issue (NumPy index type) fixed with an explicit
  `int(...)`. *Source: pytest; sanity script (scratch, not in the repo).*
- **E1–E3 step 3: E2 library `memtk/lesionmatch.py`.** Lesion records from label maps
  (informative lesions only, so E4's planted copies take the same path); s = shape × position ×
  size (IoU of bit-packed crops centred on the rounded centroids, pixel distance with σ = 6.4 px,
  area ratio); KD-tree prefilter at 25.6 px; shape computed only where position × size ≥ 0.1,
  then for every pair of a candidate slice pair; Hungarian pairing per candidate slice pair
  (vectorized when one side has one lesion); τ_L and τ_C applied afterwards from the stored
  tables (τ_L below 0.1 refused, where the shortcut would stop being exact). 14 tests (90 in
  total), including the shortcuts against brute force on random maps. On real T (no
  calibration): self-match C = 1 for 300/300 random slices (296 themselves, 4 the identical
  neighbouring slices also found by E1); shortcuts = brute force on 30 real slices against all of
  T (5,608 candidate slice pairs, 0 differences in C); 10 ms per query slice. `cKDTree` replaced
  by `KDTree` (same implementation since SciPy 1.6; the old name is missing from the type
  stubs). `pyarrow` deferred to step 5, where tables are first written. *Source: pytest; sanity
  script (scratch, not in the repo).*
- **E1–E3 step 4: `bound-review` and decision G.** Built `memtk/calibration.py` (area bins,
  chance-match rates, percentile with no match as 0, the bound rule; 7 tests), the stage package
  `memtk/e3/` (`common`: T and H-cal after decision E, E2 settings in pixels; `bound`: the
  stage) and `scripts/e3_calibrate.py`. Decision E is applied from `slices.csv` (all mask pixels
  = lesion px + speck px; removes no H-cal slice, largest 514 px); 3 H-cal and 19 T slices have
  no lesion ≥ 4 px and are left out. First run of a detector on H-cal: provisional τ_L 0.77,
  bound 12 px at the 10% cut or 25 px at the 5% cut. The user set decision G: 0.77, 10%, 12 px;
  the 25 px bound is added to `calibrate`'s sensitivity table. Pylance: 3 rewrites (array type
  in a signature, `str(...)` on an ID, `list(...)` in a test) and 1 stub ignore (pandas group
  key); output identical after the fixes. The user confirmed (2026-11-03) that parameter
  choices are theirs; no supervisor sign-off is needed for them. *Source:
  `results/stanford_brainmets/e3/bound_review/`, pytest.*
- **E1–E3 step 5: `calibrate`, decision H, and the configuration score.**
  - Built `memtk/e3/calibrate.py` (E1 and E2 on H-cal against T, thresholds pooled and per
    stratum with 2,000-draw bootstrap-by-patient intervals, classes on H-cal, sensitivity table,
    exclusions, histograms) and added `stratum` and `bootstrap_interval` to
    `memtk/calibration.py` (3 tests; one first failed because `np.r_` reads the string "B" as a
    directive, fixed in the test). First Parquet tables: `pyarrow` 25.0.1 added to
    `pyproject.toml` and `requirements-lock.txt` (no further requirements).
  - *First run (planned C, pairs ≥ τ_L only):* τ_C was degenerate: 0.020 pooled [0.000, 0.202],
    0.064 single [0.000, 0.373], 0.062 multi [0.000, 0.208]; C was 0 for ~95% of H-cal slices,
    and τ_C flagged exactly the τ_L slices. τ_D and τ_L as now.
  - *Detour:* the user first chose a structural rule (full = every lesion of both slices paired
    with s ≥ τ_L; `summary` without τ_C; NN preferring full copies); one run gave flagged
    5.1% / 5.2%, full 2.5% / 0% (single / multi).
  - *Final, after discussion with the user:* C counts every Hungarian pair with s ≥ 0.1; τ_C back
    as its 95th percentile; classes by the plan's rule A (full = C ≥ τ_C, partial = a pair ≥ τ_L);
    `n_kept` and `all_copied` descriptive. A read-only check on the stored H-cal tables first
    showed the new C to be non-degenerate (τ_C 0.719 / 0.406). Decision H set by the user:
    stratified for both detectors. `lesionmatch.summary(tau_L, tau_C)` takes one value or a
    Series per query for each threshold; τ_C ≤ 0 is refused; tests updated (worked example now
    C = 0.37) and added; brute-force test now checks the new C. 102 tests pass.
  - Pylance: `wholemask.match` accepts an array of patients; an unused argument removed; a
    pandas fill warning removed with `.eq(True)`.

  *Source: `results/stanford_brainmets/e3/calibrate/`, pytest, scratch check scripts.*
- **Lesion groupings across slices checked (for E6).** Prompted by the user's observation in the
  T overview sheet (Mets_266: neighbouring metastases merge and split between slices). A
  read-only check labelled each T patient's mask as saved in 3D (26-connectivity) and counted
  3D lesions holding two or more separate 2D lesions in a slice: 38 of 1,428 (2.7%), with 630 of
  6,101 2D lesions (10.3%), 19 of 105 patients. Recorded in `docs/experiment-plan.md` (E6
  duplication test step 1, with options), `report-notes.md` (section 3 finding, section 5 open
  question) and the CLAUDE.md H1 line. No code change; E1–E5 are slice-wise and unaffected.
  *Source: scratch script (not in the repo).*
- **Lesion weighting in C discussed; equal weights kept** (user decision, 2026-11-03). Added to
  the experiment plan: the reason in E2 step 3, a new E5 step 7 (report `n_kept`, strongest
  pair and the copied-area share per flagged slice; optional area-weighted C as a sensitivity)
  and the copied-area share in E7 Panel B; reasons in report-notes section 2. No code change.
- **E1–E3 step 6: `freeze` and `check`; E1–E3 done** (milestone of 9 Nov met on 3 Nov). `freeze`
  recomputes the thresholds from the stored H-cal tables, requires them to equal
  `thresholds.csv` (else `calibrate` must run again), applies decision H, confirms the shortcut
  conditions (τ_L ≥ 0.1 and above the largest score at the prefilter distance, exp(−8); τ_C > 0)
  and writes `configs/frozen/e3_stanford_brainmets.yaml` with intervals, class rule, bound,
  detector settings and H-cal chance rates. `check` passes 16/16: decisions, run logs (in order,
  only T and H-cal read), the frozen file reproduced, shortcuts = brute force on 20 random H-cal
  slices (2,428 candidate slice pairs), self-match on 50 T slices, unit tests. New:
  `config.frozen_file` and the `frozen_dir` config key, `calibration.read_frozen` /
  `frozen_threshold`, `e3.common.load_hcal_tables`, `tests/test_e3_freeze.py` (3 tests, one
  first failed on float rounding in the test) and a read-back test; 106 tests pass. *Source:
  `results/stanford_brainmets/e3/check_report.txt`, pytest.*
- **E1–E3 plan closed.** `docs/e1-e3-plan.md` is marked reference only, with a "Carried forward"
  table. Moved to `docs/experiment-plan.md`: E4 notes (tools in place; what "full" means under
  rule A, with the finding that most P6 variants come out full, since adding one lesion to a
  one-lesion base gives C = 0.5 ≥ τ_C multi 0.406, to decide before the E4 criteria; P3 at the
  edge, s ≈ 0.737 vs τ_L 0.739; false-positive criterion per class) and E5 notes (run time on S,
  drop-only re-run needs drop-only label maps and recalibrated thresholds, S under the E3
  sensitivity settings, strata for the noisy-mask E1 run). CLAUDE.md: the plan listed as
  closed; no active plan until E4 is planned. *Source: `docs/e1-e3-plan.md` ("Carried forward").*

## 2026-11-04: E4 planned

- **E4 plan written** (`docs/e4-plan.md`, step 0). Its layout follows `docs/e1-e3-plan.md`: steps
  0–6, decision gates I–K, the stages `plant`, `validate` and `check` of a new
  `scripts/e4_validate.py`, library modules `memtk/sets.py` (moved from `memtk/e3/common.py`),
  additions to `memtk/calibration.py`, and a new `memtk/planted.py`, plus tests and figures. The
  figure list (a variants-vs-original grid, one sheet per variant, area change, and debugging
  figures for `validate`) was added at the user's request. Base availability was checked first
  with a read-only query of `lesions.csv` / `slices.csv`: 3,172 T slices with an informative
  lesion (805 multi-lesion), 92 eligible patients (≥ 5 such slices), 45 of them with ≥ 3
  multi-lesion slices; informative lesion area median 28 px (quartiles 17 / 64).
- **Decisions I and J set by the user** in the planning session (each chosen from options with a
  recommendation): bases 10 + 10 patients, P4 about the lesion-pixel centroid, P5 erode and
  dilate gated on surviving lesions, caught = flagged and the right source patient. One detail
  changed while writing the plan: P6-add places the added lesion inside the base's brain *mask*
  (`brain_masks.npz` exists for every real lesion slice) rather than its brain box. Decision K
  (criteria) left open with a proposal.
- **Docs updated:** CLAUDE.md (the E4 plan as the active plan; decisions I and J in the E4
  bullet); `docs/experiment-plan.md` E4 (status line; typo "P6 or P7" in the Outputs line fixed
  to "P6"); `docs/report-notes.md` (section 2: decisions I and J; the C-weighting entry's claim
  that P6 "expects" the partial class corrected; section 5: P6 question resolved, criteria item
  updated). No code or config changed. *Source: `docs/e4-plan.md`.*
- **E4 step 1: shared helpers and config.** `SliceSet`, `load_set`, `decision_e` moved unchanged
  from `memtk/e3/common.py` to the new library module `memtk/sets.py` (E3, E4 and E5 use them);
  E3 imports updated; E3 check 16/16 after the move. New in `memtk/calibration.py`:
  `frozen_settings` (detector keywords from the frozen file), `e1_classes`, `e2_classes` (classes
  under each query's frozen stratum thresholds, with the patients of the nearest neighbour and
  strongest-pair slice); 3 tests, pytest 109. Verified on the stored H-cal tables: the class
  helpers reproduce the frozen `hcal_chance_rates` exactly in both strata. Config: `validation:`
  section with decisions I and J, `criteria: null` (K). Module reference: "used by" lists
  re-taken from the code, which also added a few entries missing since E3 (`e3.check`,
  `e3.common`, `e3.freeze`). *Source: pytest, `results/stanford_brainmets/e3/check_report.txt`,
  scratch check (not in the repo).*
- **E4 step 2: `memtk/planted.py`.** Base selection, the variants, attach-or-drop labelling and
  provenance as pure functions; 16 tests (pytest 125). While writing it, the base draw was
  simplified twice before the tests ran (a patient missing the quota first crashed instead of
  being skipped; then a for/else replaced by a count check), and a patient skipped in the multi
  group may still be drawn as an "other" patient. One Pylance false alarm (scipy stubs type
  `affine_transform`'s `offset` as float) silenced with a reasoned ignore. Dry run on real T in
  memory (no detector, nothing written): 100 bases from 20 patients, 37 multi-lesion (30 from the
  multi group, 7 from the other); 4 multi-group patients skipped for z spacing; all P1–P5 and
  P6-add made, P6-remove for the 37 multi-lesion bases; P6-add needed at most 8 tries (median 1);
  every informative base lesion still ≥ 12 px for P4 94, P5-erode 50, P5-dilate 98 of 100
  (relevant to decision K). *Source: pytest, scratch script (not in the repo).*
- **E4 step 3: `plant` stage.** New `scripts/e4_validate.py` (stage `plant`) and
  `src/memtk/e4/` (`common`, `plant`). `load_real` moved from `e0b.common` to `memtk.io` (E4
  needs the BRAVO slices for its figures); E0b check 43/43 after the move. Run: 737 planted masks
  (100 bases × 7 variants + 37 P6-remove), 1,264 planted lesions, all traced to a source; every P1
  map equals its base (read-back check of the written stack and tables). The first run's figures
  used yellow for added H-eval lesions, the same colour as the zoom box, and called lesions merged
  by dilation "gone"; both fixed (orange; "gone or merged") and the stage re-run with `--overwrite`
  (same planted set). Pylance: values of a `.loc` row were typed as possibly a Series, so the base
  rows are now read as dicts. As built, beyond the plan: three tiles per base in the variant
  sheets instead of an inset, and the extra tables `lesion_fate.csv`, `area_change.csv`,
  `skipped_variants.csv`. *Source: `results/stanford_brainmets/e4/plant/review.txt`, scratch
  check (not in the repo).*
- **E4 `plant` figures improved after the user's review.** The user found the change between
  base and variant hard to follow side by side (mainly P3's 5 px shift, also erosion and
  dilation) and P6 unclear in the grid. Now the third tile of each variant sheet draws the base
  lesions dashed in their colour under the filled planted lesions, with a dot at the old centre
  and an arrow to the new one where a centre moved by ≥ 1 px (black edge for contrast on the
  fill); in the grid, base lesions with no planted lesion traced to them (removed or eroded away)
  are filled red and added H-eval lesions orange. The user's note that P4 barely moves some
  lesions was kept as designed: a single-lesion base rotates in place (shape only), multi-lesion
  bases also move their lesions relative to each other. Stage re-run with `--overwrite` (same
  planted set; figures only changed); pytest 125. Accepted by the user as documentation of the
  variants, with known limits left as they are: the dashed base outline inside a dilated lesion
  is hard to see, and arrows on small lesions add little; figures for the report are picked by
  hand from these sheets.
- **E4 step 4: decision K set, decision I amended** (user, after a summary of the open choices
  with recommendations). Before any detector ran on the planted set, the expected behaviour of
  each variant was worked out from the score definitions and the `plant` area changes (no
  scores computed): P5 would fail any recall criterion by construction (s ≈ r²), so it became
  reported only, like P3; the gate rule (`keeps_base_lesions`) now covers every variant but
  P6-remove (P4 94 of 100 counted). Config: `validation.gate` and `validation.criteria` with
  dated comments. `plant`'s review text no longer hardcodes the proposal; it prints decision K
  from the config (or where to set it); docstrings and review text say "gate rule" instead of
  "P5 gate". `plant` re-run with `--overwrite` (same planted set); pytest 125. Docs: E4 plan
  (status, decisions I and K, definitions), experiment plan E4, CLAUDE.md, data dictionary,
  report-notes (section 2 decisions, section 4 detection limit, section 5 item closed).
  *Source: `configs/default.yaml`, `docs/e4-plan.md`.*
- **E4 step 5: `validate` stage; gate run, FAIL, accepted as a recorded deviation.** New
  `src/memtk/e4/validate.py` and stage `validate`; `calibration.bootstrap_share` (1 test);
  `e4.common` gains `load_planted`, `slice_images` (replaces `plant`'s `base_images`) and later
  `gate_status`. `plant` re-run first because its code changed (same planted set).
  - *Gate run 1* (frozen file of 2026-11-03, unchanged): every recall criterion passed (P1 100%,
    P2 100%, P4 98.9%, P6 97.1%); E1 copy 7.2% / 7.4% and E2 full single 7.1% failed the 7%
    limit; verdict FAIL. The review also reported 431 missed masks: a bug. `criterion_caught`
    assigned into a `pd.Series(False)` through a boolean-Series mask, which upcast it to object,
    so `~` turned `True` into −2 (truthy) and every gated mask counted as missed. It affected only
    the misses count and `misses.png`, not the gate values. Fixed with a numpy array, regression
    test added (`tests/test_e4_validate.py`); re-run: 5 misses, same gate values.
  - *Read-only checks after the run* (scratch scripts, not in the repo): H-eval flags cluster in
    patients (E1 copy single: 50 flags from 15 of 26 patients, 44% from the top 3; E2 full multi:
    8 from 6 of 14, 62% from the top 3); two H-eval slices match a T slice with Dice 1.00 (single
    lesions of 8 and 4 px, shift 1–2 px): chance, not leakage. The user's question about the one
    P4 miss (its nearest neighbour is its own base slice): checked, correct by the definitions
    (13 px lesion rotated to 12 px, shape 0.667, s 0.613 below the single-lesion thresholds;
    E1 0.615 below τ_D multi); `misses.png` captions now say "not flagged (below thresholds)" or
    "flagged, wrong patient".
  - *Decision (user, after the run):* the three failed false-positive criteria accepted as a
    recorded deviation (`validation.accepted_deviation`, criteria and reason). `validate` now
    names criteria uniquely by class and stratum in `gate.csv`, adds an `accepted` column, and
    gives ACCEPTED only when the failed criteria are exactly the listed ones;
    `e4.common.gate_status` reads the verdict for E5. Two Pylance errors fixed by rewriting
    (`gate_table` reuses the `pass` column of `fp_table`; `verdict_text` guards the deviation).
    Gate runs 2–4 (the bug fix, the deviation mechanism, the Pylance rewrite; one accidental
    duplicate run) gave the same values; final verdict ACCEPTED. pytest 128.
  - The user agreed not to add a combined variant (remove a lesion and move the others); reason
    in report-notes section 4.
  *Source: `results/stanford_brainmets/e4/validate/review.txt`, `gate.csv`, `fp_heval.csv`.*
- **E4 step 6: `check`; E4 done** (milestone of 13 Nov met on 4 Nov). New
  `src/memtk/e4/check.py` and stage `check` (reuses `checks.CheckReport` and
  `check_run_logs`, and the `plant` / `validate` functions to reproduce their outputs). First
  run 16/16. Because every check passed at once, a read-only test confirmed that they can fail:
  a changed gate value, a flipped `accepted` flag, a flipped caught flag and a deviation list
  missing one failed criterion are all detected (scratch script, not in the repo). Two checks
  beyond the plan: bases are T slices and every P6-add lesion comes from H-eval (no leakage
  through the planted set). Plan closed: `docs/e4-plan.md` marked reference only with a "Carried
  forward" table; CLAUDE.md says no active plan until E5 is planned. *Source:
  `results/stanford_brainmets/e4/check_report.txt`.*
- **E5–E8 plan grouping suggested** (at the user's request, not yet decided): E5 alone (scores S
  once, all choices fixed first), E6 + E7 in one plan (shared tables, decisions and deadline;
  report generator built on the E4 planted copies before S), E8 alone (optional, own go/no-go).
  Recorded in `docs/experiment-plan.md` ("Plan documents for E5–E8"), with the E4 milestone and
  the E4 criteria item marked done there; pointer in CLAUDE.md.

## 2026-11-15: E5 planned; step 1 (library and config)

- **E5 planned in `docs/e5-plan.md`** (E5 alone, as suggested on 2026-11-04). The user set
  decisions L–Q in the planning session, in two rounds of questions, before any E5 code existed
  (values and rejected options in report-notes section 2); the secondary analyses, the cut of
  the area-weighted C and the E8 exit rule were approved with the plan. Pointers updated in
  CLAUDE.md (active plan) and `docs/experiment-plan.md` (E5 status line).
  - *Read-only queries while planning* (inline scripts, not in the repo), whose results changed
    the plan: stray counts per set (`slices.csv`, real and pool: H-cal and H-eval 0, T 4 slices,
    S main 4,481), which removed the drop-only re-calibration the E1–E3 carry-over asked about;
    held-out E1 maxima by stratum and size bin (`e3/e1_hcal.csv`, `e4/e1_heval.csv`), which
    replaced the plan's "Dice > 0.9 is unambiguous" with decision Q; S main strata (E1 19,117 /
    16,803; E2 19,911 / 6,006 / 10,003 without an informative lesion), informative lesion sizes
    (median S main 24, H-eval 22, T 28 px), main masks per naming scheme, and the E2 maxima on
    H-eval (`heval_classes.csv`: C 0.952 / 0.488).
- **E5 step 1: shared helpers to the library, `memtk/audit.py`, config `audit:`.**
  - *Moved unchanged* (E5 needs them): `e4.common.load_frozen` → `calibration.load_frozen`
    (`config.frozen_file` now takes the experiment, default "e3", for the coming
    `e5_<dataset>.yaml`); `e4.validate.run_detectors` → `calibration.run_frozen`;
    `e4.common.slice_images` → `io.slice_images`; `e4.validate.heval_classes`, `fp_flags`,
    `share_row` → `audit.class_table`, `audit.class_flags`, `audit.share_row` (renamed: they now
    serve S too). E4 modules import them from there.
  - *Verified:* pytest 128 before, 141 after (13 new tests: `tests/test_audit.py` 11,
    `tests/test_sets.py` 2); E3 check 16/16 and E4 check 16/16 re-run (their check reports and run
    logs rewritten); the E4 check rebuilds the gate from stored tables and never calls
    `run_frozen`, so it was spot-checked separately: on 60 random H-eval slices it reproduces the
    stored `heval_classes.csv` (scores equal to 1e-16, every other column identical; inline
    script, not in the repo).
  - *New:* `memtk/audit.py` (cluster bootstrap of column sums as an exact multinomial over
    cluster types, intervals, one-sided p, excess test, Holm, standardised rate, AUC and its
    two-level bootstrap, chained groups, copied-area share, held-out maxima and `beyond_max`);
    `sets.load_pool` (S label maps by `mask_class`, uint8 to keep ~36,000 maps at ~0.6 GB; the
    detectors use only `labels > 0` and the lesion numbers) and `sets.relabel` (drop-only);
    config section `audit:` with the values of L–Q.
  - Pylance flagged five issues in the new code (a parameter reassigned with another type,
    pandas-stub complaints on `Series.map` / `Index.map` with a Series, `itertuples` scalars);
    each fixed by a clearer rewrite (`reindex`, plain lists), no ignores.
  *Source: `docs/e5-plan.md` (status, step 1), `results/stanford_brainmets/e3/check_report.txt`,
  `results/stanford_brainmets/e4/check_report.txt`.*
- **E5 step 2: `prepare` (no S).** New `scripts/e5_audit.py`, `src/memtk/e5/common.py`,
  `src/memtk/e5/prepare.py`, `tests/test_e5_common.py` (4 tests).
  - *Refactor first:* `prepare` re-derives the sensitivity thresholds, so E3's threshold code moved
    unchanged to `memtk.calibration` (`lesion_slices`, `threshold_rows`, `thresholds`, `STRATA`
    from `e3.calibrate`; `frozen_thresholds`, `shortcut_problems`, `chance_rates_at` from
    `e3.freeze`, its `DETECTORS` as `THRESHOLDS`); `lesionmatch.save_tables` / `load_tables`
    replace four copies of the E2 table naming in E3 and E4. pytest 142 and E3 / E4 check 16/16
    after this refactor, before any E5 code ran.
  - *Bug prevented:* `io.read_table` kept only `key`, `sample_id` and a few other columns as text,
    not the detector tables' `query`, `nn`, `train` or lesion-ID columns. Real keys
    (`Mets_005/z127`) survive either way, but an S sample ID such as `1033` would have been read
    back from `e2_*_queries.csv` as the number 1033 and no longer matched its key. `ID_COLUMNS`
    extended, with a test in `tests/test_io.py`.
  - *Run:* first run complete; every check inside the stage passed (the configured gap
    reproduces every E0b label map; gap 0 changes 4 T slices and no H-cal or H-eval slice; every
    sensitivity threshold equals E3 `sensitivity.csv`; drop-only thresholds equal the frozen
    ones; the E2 shortcuts stay exact under every variant). Re-run once with `--overwrite` (no S
    yet) after two review-text changes: size bins labelled as ranges, and a note that the E2
    beyond-held-out tier is empty for single-lesion masks. Values unchanged. A third run without
    `--overwrite` refused, as intended.
  - *Read-only check after the run* (inline script): the single-lesion E2 C maximum of 1.0 is H-cal
    `Mets_316/z88`, a 12 px lesion pixel-identical and at the same position as the L1 of T
    `Mets_249/z101` (another patient); the next highest are 0.85 and 0.82 (18 and 15 px lesions
    of the same patient).
  - pytest 146; Pylance clean on every changed file.
  *Source: `results/stanford_brainmets/e5/prepare/review.txt`, `configs/frozen/e5_stanford_brainmets.yaml`.*
- **E5 step 3: `rehearse` (no S).** New `src/memtk/e5/score.py` (`score_query`: the one scoring
  function), `src/memtk/e5/analyze.py` (`analyze_scored`: the one analysis function, every table,
  test and figure of decisions L–Q), `src/memtk/e5/rehearse.py`, `tests/test_e5_analyze.py` (4
  tests); `audit.cell_rates` (+1 test). Built as functions of any query set so that the `score`
  and `analyze` stages run on S exactly the code rehearsed here.
  - *Refactor:* E4's figure helpers `zoom_window`, `fill`, `draw_labels` (`e4.plant`) and
    `lesion_colors` (`e4.validate`) moved unchanged to `memtk.plotting`; `load_planted` from
    `e4.common` to `memtk.sets` (E7 will need it too). E4 check 16/16 afterwards.
  - *Fixed before the first run:* `audit.auc_bootstrap` used one `einsum` over draws × S clusters ×
    H clusters, which for S (about 26,000 masks) would have built an intermediate of about 400 MB
    per block; now `(ws @ wins) * wh`. `standardized_rate` returns NaN quietly where a resample
    holds no slice of the weighted cells (a test warning showed it).
  - *Runs:* first run complete on both cases. One change after it: with no noisy masks in the mock
    sets, every table repeated "main" as "main+noisy"; `subsets` now forms main + noisy only when
    both parts exist. Re-run with `--overwrite`; values unchanged.
  - Results in `docs/e5-plan.md` (status, step 3) and report-notes section 3.
  - pytest 151; Pylance clean.
  *Source: `results/stanford_brainmets/e5/rehearse/null/review.txt`, `positive/review.txt`.*
- **E5 step 4: S scored once** (2026-11-15, completed 13:48, 16.7 min; the user committed and
  pushed steps 0–3 first). Stage `score` added to `src/memtk/e5/score.py` (`run_score`); before it
  ran, `check_prepared` and `drop_only_t` moved from `rehearse` to `e5.common`, `heldout` and
  `informative_areas` to `e5.analyze` (shared with `analyze`), and `rehearse` was re-run: tables
  identical to the committed rehearsal, only timings differ.
  - *Run:* one run, no re-run, exit code 0, run in the background (32 GB RAM, 16.5 GB free at the
    start). 37,161 masks (clean 32,219, minor 3,701, noisy 1,241; the 463 `no_lesion` masks left
    out); main and the five variants of decision O. Per run (`score_log.csv`): main 258 s, 5.0 M
    lesion pairs; percentile 99 15 s (re-threshold); σ 0.08 270 s, 11.3 M pairs; bound 10 194 s,
    6.3 M; bound 25 26 s, 1.6 M; drop-only 237 s, 5.0 M. E2-scorable masks: 27,078 (main), 29,917
    (bound 10), 15,458 (bound 25), 26,857 (drop-only: 221 fewer than main, masks whose only
    informative lesion reaches 12 px through attached strays). No rate was computed or shown.
  - *Structural check afterwards* (inline script; counts and IDs only): every class table has
    37,161 rows in the key order of `queries.csv`; 8,187 numeric-looking sample IDs (scheme `<n>`)
    are read back as text, which the `ID_COLUMNS` extension of step 2 made possible.
  - *Meanwhile* (with the user's agreement that it could not affect the running process): the
    `analyze` stage was written (`run_analyze`, not run), the figure code moved from `rehearse` to
    `analyze.draw_figures`, and `rehearse` re-run once more: tables identical. `analyze` was called
    once on purpose before the marker existed and refused, as intended.
  *Source: `processed_data/model_space/amibekk_v1/e5/score_log.csv`, `score.complete.json`,
  `score.run_info.json` (not in git).*
- **E5 step 5: `analyze` run once on S** (2026-11-15, at the user's request after step 4; exit code
  0; hashes equal to `score.complete.json`). Results in report-notes section 3, the near-exact
  caveat in section 4, the status in `docs/e5-plan.md`.
  - *Exploratory checks after the run* (inline scripts, nothing written, not part of the fixed
    analysis; prompted by `near_exact.png`, where every example is a small single blob): (1) the
    chance expectation of the near-exact count under exchangeability, P(a new score exceeds the
    maximum of n held-out scores) = 1/(n + 1), summed over the cells of decision Q, from the cell
    counts in `configs/frozen/e5_stanford_brainmets.yaml` and the S main class table; (2) the
    excess intervals of the primary cells under every sensitivity variant
    (`results/amibekk_v1/e5/analyze/sensitivity.csv`); (3) the spread of E1 multi-lesion copies
    over source patients (`flagged.csv`).
  *Source: `results/amibekk_v1/e5/analyze/review.txt`, `rates.csv`, `sensitivity.csv`, `flagged.csv`.*
- **E5 step 6: `check`; E5 done** (milestone of 17 Nov met on 15 Nov). New `src/memtk/e5/check.py`
  and stage `check` (reuses `checks.CheckReport` and `check_run_logs`, `prepare`'s `tau_d_star` and
  `heldout_maxima`, and `analyze_scored`). First run 26/26. Because every check passed at once, a
  scratch test confirmed that the table comparison can fail: a copy of the analysis folder with
  one excess changed by 0.001, one group label changed and one row removed was reported as
  `rates.excess`, `flagged.e2_group` and `near_exact` (scratchpad only, not in the repo). Plan
  closed: `docs/e5-plan.md` marked reference only with a "Carried forward" table; "Carried over
  from E5" notes added to E6, E7 and E8 in `docs/experiment-plan.md`; CLAUDE.md says no active
  plan until E6 + E7 are planned. *Source: `results/amibekk_v1/e5/check_report.txt`.*

## 2026-11-15: E6 + E7 planned

- **E6 + E7 planned together in `docs/e6-e7-plan.md`** (as suggested on 2026-11-04: shared tables,
  decisions and deadline; the E7 work on planted copies runs in parallel to E6, only the E7 reports
  on S wait for E6). The user set decisions R and U–Z in the planning session, in one round of
  questions (copying vs placement evidence, source of π_p, 3D lesion identity, report selection),
  before any E6 or E7 code existed; the other values were approved with the plan. Pointers updated in
  CLAUDE.md (active plan, open decisions), `docs/experiment-plan.md` (E6 and E7 status) and
  report-notes section 5 (three questions resolved, one with its method fixed).
  - *Read-only queries while planning* (inline scripts, not in the repo; T and held-out tables only,
    no statistic of S computed, so no E6 result was previewed before its analysis was fixed):
    multi-lesion E1 nearest-neighbour assignments, H-eval 393 masks from 26 patients, H-cal 303, 94
    of 105 T patients covered by H-cal ∪ H-eval, max share 5.2% (`e3/e1_hcal.csv`,
    `e4/heval_classes.csv`), which led to the smoothed π_p of decision R; 62 T patients with
    multi-lesion slices (1,220 of 3,823; real `slices.csv`); H-eval multi-lesion chance flags E1 copy
    29, E2 full 8, E2 flagged 23 (`heval_classes.csv`); largest T slice lesion area 873 px, so the
    H1 sampling-filter step is moot; the E1 margin is already stored in every E1 table; E5's mock
    scored sets (`processed_data/model_space/stanford_brainmets/e5/rehearse/`) are in S's format and
    reusable for an E6 rehearsal; planted set 737 masks (`planted_outcomes.csv`).
  - *Changed against the experiment plan by reasoning, not data:* the exposure ratio alone cannot
    separate copying from placement (S places lesions centrally, E0b), so decision U adds margin and
    source rarity against H-eval chance flags; the deletion test as written (ΔC against the source
    slice) nearly follows from C being a sum of pair scores, so decision Y adds a full re-match and
    the whole-mask Dice change.
- **E6 + E7 step 1: shared helpers to the library, `memtk/lesions3d.py`, config `exposure:` /
  `reports:`.**
  - *Moved unchanged* (E6 and E7 read the same scored sets): `Scored`, `load_scored`, `heldout`,
    `sources`, `chained` from `e5.analyze` and `query_table` from `e5.score` (needed by `heldout`;
    not in the plan's list) → `audit`. `e5.analyze`, `e5.score`, `e5.rehearse`, `e5.check` and
    `tests/test_e5_analyze.py` import them from there. pytest 151 and E5 check 26/26 after the move
    (the check rewrote `results/amibekk_v1/e5/check_report.txt`; run again after the config change:
    26/26, since `exposure:` and `reports:` are outside the hashed `audit:` section).
  - *New:* `memtk/lesions3d.py` (slice-to-slice linking, 26-connected components, per-lesion table,
    2D → 3D map) and `ids.lesion3d_id` (`<patient_id>/D<n>`); `tests/test_lesions3d.py` (5 tests),
    one test in `tests/test_ids.py`; pytest 157. Config sections `exposure:` and `reports:` with the
    values of decisions R, U–Z.
  - *Read-only check on T* (inline script, not in the repo; T volumes only): 26-connectivity on the
    masks as saved gives 1,446 3D lesions, of which 1,428 contain a lesion ≥ 4 px (the recorded count;
    the other 18 consist only of 1–3 px components) and 38 hold two pieces in one slice (the recorded
    merged count); slice-to-slice linking gives 1,565 chains, none with two pieces in a slice, median
    span 3 slices under both, max 43; 17 s for 105 volumes.
  - Pylance flagged the renumbering in `components_26` (scipy's label array typed as int or float);
    fixed by an explicit int array and `np.unique(..., return_index=True)`, clearer than the first
    `pd.unique` version. No ignores.
  *Source: `docs/e6-e7-plan.md` (status, step 1), `results/amibekk_v1/e5/check_report.txt`.*
- **E6 + E7 step 2: `memtk/exposure.py`, E6 `reference` (no S).** New `src/memtk/exposure.py`,
  `src/memtk/e6/` (`common`, `reference`), `scripts/e6_exposure.py`, `tests/test_exposure.py` (11
  tests; pytest 168).
  - *Found while testing:* the slice-rarity test first used a 4×4 lesion shifted by 1 px and got an
    E2 shape factor of 0.6, not 1: `lesionmatch` centres crops on `np.rint` of the centroid, which
    rounds halves to even (11.5 → 12, 12.5 → 12), so an even-width lesion moved by an odd number of
    pixels is compared one pixel out of alignment. Read-only check on T (inline script, not in the
    repo; each informative T lesion against itself shifted 1 px in columns): 3.9% of 4,469 lesions
    affected (8% at 12–19 px, 3% at 20–39, 2% at 40–79, 0% at ≥ 80 px), shape factor median 0.57
    for those. The frozen detector is not changed (E3–E5 depend on it); the test uses a 2 px shift.
    E4's P2 (2 px) could not reveal it.
  - *First run* of `reference` (2 min 53 s), then two changes and a re-run with `--overwrite`: the
    review now states the 19 T slices without a lesion ≥ 4 px (left out by `sets.load_set`, as in
    E1–E5; first noticed as "3,804 of 3,823 slices" in the rarity table), and Pylance fixes
    (`slice_rarity` builds its rows by reindexing instead of a per-key loop; `pd.Index` for
    `get_indexer`; test assertions on dicts). Every table byte-identical to the first run
    (compared in the scratchpad).
  - *As built, against the plan:* `reference` does not store H-eval margins and chance-flag sources;
    the shared analysis function of step 3 computes them, so mock sets and S go through one code path.
  *Source: `results/stanford_brainmets/e6/reference/review.txt`, `reference.png`.*
- **E6 + E7 step 3: E6 `rehearse` (no S).** New `src/memtk/e6/exposure.py` (the one analysis:
  `analyze_query`, `draw_figures`), `src/memtk/e6/rehearse.py`, loaders in `e6.common` (`load_query`,
  `load_heval`, `load_reference`), `tests/test_e6_exposure.py` (3 tests; pytest 171).
  - *Changed before the first run:* `exposure.exposure_table` drew the query set's flags as one
    multinomial (independent flags, right for S), but E5's mock sets are clustered by patient (H-cal),
    so it now resamples the flags by their clusters (`resample_sums`), as the held-out side already was.
  - *Runs:* first run complete (13 s). Changes after it, each followed by a re-run with `--overwrite`:
    the positive-case check counted all 20 base patients in every cell, now only those whose planted
    copies are flagged in the cell (with the planted flags naming their base); H1 bins in size order
    (`order` argument of `exposure.h1_table`; values unchanged); exposure and rarity figures with
    symlog axes from 0; Pylance fixes (one helper `cell_exposure` per cell and smoothing instead of a
    loop that left variables possibly unbound; array arguments; bin labels from an array). Readings
    and tables unchanged by these rewrites (same random draws in the same order).
  *Source: `results/stanford_brainmets/e6/rehearse/null/review.txt`, `positive/review.txt`.*
- **E6 + E7 step 3b: placement mock, readable E6 figures** (at the user's request: decision U's
  "typical placement" branch had only a unit test, and the null and positive rehearsal figures
  looked alike). The user also asked for a report-synthesis document and a project walkthrough
  after E8; added to the schedule in `docs/experiment-plan.md`.
  - *New:* `planted.relocate_lesions` (+1 test in `tests/test_planted.py`; pytest 172), config
    `exposure.placement_mock`, `rehearse` cases `placement` and `placement_compact` (built from
    H-cal, scored with `calibration.run_frozen`, written in E5's layout to
    `<output_root>/stanford_brainmets/e6/rehearse/`), E5's rates and AUCs on them (exploratory,
    `e5_rates.csv`, `e5_auc.csv`; `e6.rehearse` imports `e5.analyze.rates_table` and `auc_table`).
  - *First test failure:* the relocation test expected the moved lesion's *rounded* centroid on the
    drawn position; with `np.rint` (halves to even) the exact centroid is within half a pixel but
    the rounded one can tip over (the same rounding as in E2's crops, step 2). Docstring and test
    now state the half-pixel guarantee.
  - *Figures redesigned* (the old ones plotted ratios of tiny counts, so most patients sat at 0 in
    both cases, and margins only against H-eval's median line, so half of the null case's flags lay
    above it by definition): observed vs expected flags per patient with each patient's chance range
    and the planted base patients of the cell ringed; margins and source rarity of query and H-eval
    flags as side-by-side boxes (`flag_details.csv` now has H-eval's flags, column `set`);
    observed / expected flags per fifth of the patients by rarity. Runs: first run, then two re-runs
    with `--overwrite` for figure-only changes (lighter chance bars; rings only for patients flagged
    in the cell). Tables unchanged.
  *Source: `results/stanford_brainmets/e6/rehearse/{placement,placement_compact}/review.txt`.*
- **Decision U amended by the user** (2026-11-15, after step 3b, before E6 read S): margins wholly
  below chance (margin AUC interval below 0.5) also read "typical placement". `e6.exposure.cell_reading`
  (+1 assertion in `tests/test_e6_exposure.py`), the config comment and `docs/e6-e7-plan.md`
  (decision U) changed; `rehearse` re-run with `--overwrite`: readings null undetermined, positive
  patient copying, placement undetermined, placement_compact mixed (E1 copy multi typical placement).
  Tables unchanged except the reading. *Source: `results/stanford_brainmets/e6/rehearse/*/review.txt`.*
- **E6 + E7 step 4: E7 `planted` (no S).** New `src/memtk/reports.py`, `src/memtk/e7/` (`common`,
  `planted`), `scripts/e7_reports.py`, `tests/test_reports.py` (8 tests; pytest 180). Pseudonyms
  P001–P105 written to `<output_root>/stanford_brainmets/e7/pseudonyms.csv` on first use.
  - *Runs:* first run complete (29 s), criteria PASS. Changes after it, each followed by a re-run
    with `--overwrite` (no S involved; these are the rules fixed on the planted copies):
    the taxonomy called 91 of 96 flagged P3 masks (5 px shifts) "partial", because "partial" was "a
    pair below tau_L"; now "a lesion without a partner", which gives P3 "shifted"; the "lesions"
    share is shown only for the P6 variants it is defined for; the verdict said "1 not matched ->
    full copy" for a mask E2 calls full while one pair stays below tau_L, now "configuration copy
    ... not every lesion", and it separates "without a partner (novel)" from "paired but not closer
    than chance"; report layout (wider figure, two-line titles, wrapped verdict, lesion-ID columns
    sized to the IDs). Pylance: dicts instead of `.at` lookups, a covariant `Sequence` in the test
    helper, one `# pyright: ignore` for scipy's Wilcoxon result (`pvalue` missing from the stub).
  *Source: `results/stanford_brainmets/e7/planted/review.txt`.*
- **E6 + E7 step 5: E6 `exposure` on S, one run** (2026-11-15, after the user committed steps 0–4).
  `run_exposure` added to `src/memtk/e6/exposure.py` (refuses if E5's lock changed or `rehearse` ran
  with another `exposure:` section) and to `scripts/e6_exposure.py`. Before the run: one syntax error
  from a script edit (a `\n` written as a line break), fixed; pytest 180, Pylance clean. The analysis
  code was unchanged since the last `rehearse`. Run: 16 s, exit 0, no re-run. Results in report-notes
  section 3 and `docs/e6-e7-plan.md` (status, step 5).
  - *Read-only checks after the run* (not part of the fixed analysis): the H1 weighted rows of the
    four rehearsal cases (all intervals cover 0); the source slices of the top E1 multi-lesion
    patient `Mets_230` (`flag_details.csv`).
  *Source: `results/amibekk_v1/e6/exposure/review.txt`, `run_info.json`.*
- **E6 + E7 step 6: E7 `reports` on S.** New `src/memtk/e7/reports.py` and the stage in
  `scripts/e7_reports.py`; `reports.pair_colors` (source lesions in their query partner's colour,
  the to-do of step 4); E5's lock check moved from E6 `exposure` to `e5.common.lock_unchanged`
  (used by both). pytest 180, then 181 (+1 verdict test); Pylance: report rows looked up as plain
  dicts of `iterrows` rows (the stubs type `.loc[key]` as Series or DataFrame).
  - *First run* (6 min 57 s, exit 0; nothing S-derived changed afterwards). After reading the
    reports, report text only was changed: the verdict now starts with the whole-mask match when E1
    flags the mask (E1-only flags, e.g. the exposed patients' and some near-exact masks, read only
    "no lesion-level match"), and missing margins print "-"; MRI slices cached. `planted` re-run with
    `--overwrite` (its tables byte-identical to before, compared in the scratchpad), then `reports`
    re-run with `--overwrite` (6 min 53 s).
  - *Read-only checks after the run:* report 02's mask against its E1 nearest neighbour (E2 shape
    0.39 after a 1 px diagonal shift: the rounding found in step 2); the count of S main masks with
    an informative lesion and E1 score 1.0 (4; 2 E2-novel).
  *Source: `results/amibekk_v1/e7/reports/review.txt`, `run_info.json`.*
- **Three changes at the user's request after step 6** (2026-11-15):
  - *Patient names:* reports use the dataset's IDs (`Mets_xxx`, already pseudonymous) instead of
    P001–P105; `e7.common.pseudonyms` replaced by `patient_name` (identity), the column
    `source_pseudonym` renamed `source_patient`, `<output_root>/stanford_brainmets/e7/pseudonyms.csv`
    deleted; decision X amended; CLAUDE.md and the experiment plan say the dataset IDs count as
    pseudonymous.
  - *H1 as rates* (`exposure.span_rates`, `e6.exposure.h1_rates_figure`, `h1_rates.csv` / `.png`):
    descriptive, from the same credits and bootstrap draws as the tested H1 (no new random draws).
    E6 `rehearse` and `exposure` re-run with `--overwrite`: every existing table byte-identical
    (compared in the scratchpad), only the new files added.
  - *Patient profiles* (`reports.profile_tables`, `draw_patient_profile`, `e7.reports.patient_profiles`;
    `patients/<patient>.png`, `patient_profiles.csv`) for the 3 most exposed patients per primary
    cell (6 patients). E7 `planted` and `reports` re-run with `--overwrite`: every table identical
    except the two with patient names (`reports.csv`, `report_index.csv`), whose other columns are
    identical and whose verdicts differ only in the names (checked in the scratchpad); then one more
    `reports` run for profile layout (whole-number slice ticks, no empty space for one lesion).
  - Tests +3 (`span_rates`, `profile_tables`, the E1 verdict part earlier): pytest 183; Pylance clean
    (profile rows drawn from `to_dict("records")`, the top slice by `argmax`).
  *Source: `results/amibekk_v1/e6/exposure/h1_rates.png`, `results/amibekk_v1/e7/reports/patients/`.*
- **Report and profile figures refined** (2026-11-15, at the user's request): lesions outlined, not
  filled, in report and profile tiles (`plotting.draw_labels(filled=False)`; E4/E5 figures keep the
  default fill), a whole-slice tile of the named training slice with the zoom window as a box
  (`reports.overview_tile`), and the largest 2D area next to each 3D lesion in the profiles
  (`profile_tables` keeps `area`). E7 `planted` and `reports` re-run with `--overwrite`: every table
  byte-identical except `patient_profiles.csv` (new `area` column). One test updated (toy lesion
  table needs `area`); pytest 183. Pylance then flagged `np.stack` of plain `np.ndarray` lists in
  three places, including a line in `tests/test_exposure.py` that had been clean since step 2 (the
  same construct elsewhere is not flagged): a numpy-stub overload false alarm, one-line
  `# pyright: ignore[reportCallIssue]` each with the reason.
  *Source: `results/amibekk_v1/e7/reports/patients/Mets_058.png`, `reports/50_exposed_e1_copy_multi.png`.*
- **E6 + E7 step 7: checks; plan closed** (2026-11-15). New `src/memtk/e6/check.py`,
  `src/memtk/e7/check.py` and their `check` stages. Refactors first: `differing_tables` moved from
  `e5.check` to `checks` (E5 check re-run: 26/26); `e7.reports` split into `deletion_tables`,
  `profile_data`, `profile_table` so the check recomputes exactly what `reports` wrote.
  - *E6 check:* first run 17/18. The `reference` run log predates `exposure.placement_mock` (added in
    step 3b), and the log check compared the whole `exposure:` section; `reference` uses only
    `exposure.lesions3d` and `exposure.rarity`, so the check now compares those keys
    (`REFERENCE_KEYS`; the reference tables themselves are reproduced by the check). Second run 18/18
    (~4 min).
  - *E7 check:* 14/14 at the first run (~8 min).
  - *Scratch test* (not in the repo): a damaged copy (one margin AUC +0.001, one taxonomy label
    changed, one profile row removed) was reported as `margins.margin_auc`, `taxonomy.taxonomy` and
    `patient_profiles` by `checks.differing_tables`.
  - Plan closed: `docs/e6-e7-plan.md` marked reference only, with a "Carried forward" table; E8
    "Carried over from E6 + E7" in `docs/experiment-plan.md`; report-notes sections 4 (four
    limitations) and 5 (E8 context, the master-thesis question); CLAUDE.md: no active plan until the
    E8 go/no-go.
  *Source: `results/amibekk_v1/e6/check_report.txt`, `results/amibekk_v1/e7/check_report.txt`.*
- **E8 decided: go, lean and exploratory** (the user, 2026-11-15, after the E6 + E7 close; not started
  in this session). A "Ready for planning" brief added to `docs/experiment-plan.md`, E8 (proposed
  design, the choices to settle, every input file and the code to reuse; inputs checked to exist:
  pool `brain_masks.npz`, `masks_raw.npz`, `qc.csv` with `mri_failed` (82 main masks), scikit-image
  0.26 for SSIM). CLAUDE.md points the next session at it; report-notes section 5 marks the go/no-go
  resolved.

## 2026-11-16: E8 planned

- **E8 planned in `docs/e8-plan.md`** (decisions AA–AE). The user chose group A, the held-out reading,
  the primary measure and the rehearsal in one round of questions; the other values (k = 5 matched
  controls, brain-size tertile as matching cell, Gaussian SSIM σ 1.5, z-score in each image's own
  brain) were approved with the plan. Pointers updated in CLAUDE.md (active plan; the "still open: E8
  go/no-go" line corrected, since report-notes had marked it resolved on 2026-11-15) and
  `docs/experiment-plan.md` (E8 status).
  - *Read-only queries while planning* (inline scripts, not in the repo; no MRI statistic of S
    computed): group A = 56 masks from 41 source patients, none with a failed MRI
    (`report_index.csv`, `qc.csv`); synthetic/source brain-area ratio median 0.51 (10th–90th pct
    0.12–1.42; `brain.csv` of both sets); H-eval 148 flagged / 775 novel by both detectors
    (`e4/heval_classes.csv`); about 22,000 novel S masks with an informative lesion and a valid MRI
    (main and noisy counted together; `e5/classes_main.csv` holds both sets; the main-only count comes
    from `select`); 29 novel masks with C = 0.
  - *Found:* E7's `report_index.csv` `source` is the E2 source (`reports.report_data`), so E8 takes the
    decision-J source per tier from `flagged.csv`; `planted.shift` refuses to push a non-zero pixel off
    the image and cannot align MRIs (synthetic brain boxes reach row 127), so `paired.align` is a
    separate zero-filled shift with E1's sign convention.
- **E8 step 1: library and config** (2026-11-16). New `src/memtk/paired.py` and `tests/test_paired.py`
  (9 tests; pytest 192). The plan's `comparison_shift` and `runner_up_slice` became one function,
  `best_match` (the same search). Config `paired:` written with the approved values; the rehearsal key
  `null` renamed `null_set` (YAML reads `null:` as an empty key). Read-only measurement on T and H-eval
  (inline script, no S): brain masks recomputed with `lesions.brain_mask` equal E0b's stored ones on
  every slice (3,823 T, 1,094 H-eval); z-scored whole-image range median 9.80 (T) / 9.26 (H-eval);
  lesion neighbourhood (+8 px, in the brain) median 513 px on T (5th pct 282), so no minimum region
  size. `paired.measure.data_range` set to 10 by the user. Module reference: `exposure` and `reports`
  added to `wholemask`'s "used by" (missing before).
- **E8 step 2: `select`** (2026-11-16). New `src/memtk/e8/` (`common`, `select`) and
  `scripts/e8_paired.py`; reads E7's `report_index.csv`, E5's `flagged.csv` and S tables (E5's lock
  checked), no MRI; seconds, re-run byte-identical. Group A 56 masks from 41 source patients; group B
  280 (5 per A mask, none short) from a pool of 20,875 novel S main masks (12 with a failed MRI left
  out); brain-area tertile edges 2,053 / 3,666 px; whole-mask Dice with the comparison slice A 0.93 vs
  B 0.62 (median); recomputed shifts equal E1's stored ones (109 of 109); runner-up slices median
  Dice 0.76. `io.ID_COLUMNS` extended with `case`, `comparison`, `comparison_patient` (sample IDs such
  as `419` stay text). Type hints of `e8.common` widened to accept pandas Series (Pylance).
  *Source: `results/amibekk_v1/e8/select/review.txt`.*
- **E8 step 3: `rehearse`** (2026-11-16). New `e8.paired` (the measurement and analysis that `paired`
  runs unchanged) and `e8.rehearse`; `e8.common.real_slices`, `synthetic_samples`;
  `tests/test_e8_paired.py` (4 tests; pytest 196). ~20 s, no S read; a re-run reproduces every table.
  Readings as required: null (H-cal) not supported (lesion AUC 0.479), positive (planted P1/P2 with the
  base patient's BRAVO 2 slices away) supported (0.985). All 100 P2 shifts recovered exactly. Held-out
  cases without a matched control (H-eval 25 of 131, H-cal 6 of 91, positive 64 of 200) kept in the AUC
  as the rule says; read-only check with matched cases only: H-eval 0.545 vs 0.532. `test_table`
  renamed `auc_tests` (pytest collected it as a test). After the user's question on the positive
  `examples.png` (why the P1 image differs from its comparison slice, why P1 and P2 both show 0.46):
  the caption now names the image's slice (base z + 2) and the positive `measures.csv` has an `image`
  column; re-run: every other table and the readings byte-identical; P1 and P2 of a base give identical
  SSIM for all 100 bases (same image after alignment), as expected. *Source:
  `results/stanford_brainmets/e8/rehearse/review.txt`.*
- **E8 step 4: `paired`** (2026-11-16), one run, 13 s: `run_paired` in `e8.paired` (guards: E5's lock,
  `select` and `rehearse` made with the current `paired:`, rehearsal as required; checked read-only
  before the run). 336 synthetic samples, 952 pairs, no empty region. Reading "not supported" (lesion
  AUC 0.481). Example figure: the median case per reason, the strongest case and E7 report 50 (the
  median of its tier, so 5 cases), captions with E7 report numbers. *Source:
  `results/amibekk_v1/e8/paired/review.txt`.*
- **E8 step 5: `check`; plan closed** (2026-11-16). New `e8.check` (17/17, ~1 min; reproducing
  `paired`'s measures re-reads S's MRIs, verification only). Refactor for the check: `select_pairs`,
  `rehearse_cases` (`Case`), `measure_s` split from the stage functions; the check reproduces the
  outputs written before the refactor. Scratch test (not in the repo): a damaged copy (one AUC +0.001,
  one row removed, one comparison key changed) was reported as `tests.auc`, `within_a`,
  `measures.comparison`. Plan closed; E8 status in CLAUDE.md, experiment plan, architecture. *Source:
  `results/amibekk_v1/e8/check_report.txt`.*
