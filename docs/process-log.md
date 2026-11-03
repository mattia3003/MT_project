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
