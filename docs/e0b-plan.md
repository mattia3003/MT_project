# E0b plan: data audit and mask quality control (`scripts/e0b_audit.py`)

> **Closed 2026-11-03. Reference only, no longer updated.** E0 is done (see the status table and
> "E0 closed" below). This file stays as the record of how E0b was built: the decision gates
> A–E, the definitions (gap, speck, stray, lesion IDs and numbering) and the stage details.
> Open items were carried forward on 2026-11-03 (see "Carried forward" at the end); the next
> work is planned in a new plan document for E1–E3.

The experiment itself is specified in `docs/experiment-plan.md` (E0); this file records how it
was implemented, stage by stage, and what was decided.

## Status (2026-11-03)

| # | Stage | Kind | Status |
| --- | --- | --- | --- |
| 0 | Drop the null model in the docs | docs | ✅ done, committed |
| 1 | `brain-review` | review → decision A | ✅ done; **decision A set**, confirmed 2026-11-02 |
| 2 | `brain` | production (needs A) | ✅ done (2026-11-02) |
| 3 | `lesions-review` | review → decisions B, C | ✅ done; **decisions B, C set** (2026-11-02): 4 px, gap 2 |
| 4 | `lesions` + `view` | production (needs B, C) | ✅ done (2026-11-02) |
| 5 | `qc-review` | review → decision D | ✅ done; **decision D set** (2026-11-02): 9 isolated specks |
| 6 | `qc` | production (needs D) | ✅ done (2026-11-02); mask class and failed-MRI flag kept apart |
| 7a | `stats` | E0 description (no decision) | ✅ done (2026-11-03: items 1–6) |
| 7b | `filters-review` | review → decision E (filters for H-cal/H-eval) | ✅ done; **decision E set** (2026-11-03): `[max_mask_px]` |
| 8 | `check` | consistency checks | ✅ implemented (2026-11-03) |
| 8b | Code restructuring | readability | ✅ 2026-11-03: stages moved to `src/memtk/e0b/`, `audit.py` dissolved, conventions in CLAUDE.md; verified by step 9 |
| 9 | Full re-run of E0a + E0b from scratch | end-to-end check | ✅ 2026-11-03: all 13 stages exit 0; outputs identical (below) |
| 10 | Close E0 | requirements lock, docs | ✅ 2026-11-03: minimal `requirements-lock.txt` (39 packages), verified by a full E0 re-run in a fresh venv; E0 marked done in the experiment plan; CLAUDE.md E0 summary updated |

**Step 9 result (2026-11-03).** Every E0a and E0b stage was re-run in order with `--overwrite`
(about 20 min in total). The E0a check passes 32/32 and the E0b check 43/43, with pytest at 60 passed.
- `processed_data/model_space/`: every table and cache is byte-identical to the run before the
  restructuring (15 CSV/NPZ files), and so are all 624 volume files (156 patients ×
  `bravo`, `seg_soft`, `seg`, `labels`).
- `results/` (`git diff`): no table changed. Run logs changed (time, full config, and `script`
  now names the stage module, e.g. `src\memtk\e0b\stats.py`; the command is in `argv`). The E0b
  check report changed only in timestamps. Four figures changed, with the same examples in
  the same order: the E0a check grid (layout from `plotting.save`), the QC class sheet and
  both lesion overview sheets (tile subtitles in the singular, outlines on the lesion numbers).
- This also verifies the code restructuring (8b). Full record in `docs/process-log.md`
  (2026-11-03). All figures checked by eye by the user: no issues.
- Docs split the same day: process entries moved from `docs/report-notes.md` into the new
  `docs/process-log.md`.

**E0 closed (2026-11-03).** `requirements-lock.txt` at the repository root pins only what this
project needs: the direct dependencies (`pyproject.toml`, now including `tqdm`) and everything
they require, at the versions that produced E0 (39 packages; the project venv also holds
packages from the earlier segmentation work, left out). Install with `--no-deps`, because
monai 1.6.0 declares torch>=2.8 while torch is 2.6.0+cu124; see the file header. Verified: a
fresh venv built from the lock re-ran all of E0 (13 stages, checks 32/32 and 43/43, pytest 60)
with all 692 outputs byte-identical to the repo's. The old `docs/requirements.txt` (from the
segmentation work) was removed. E0 is marked done in `docs/experiment-plan.md`, with its
deviations from the plan.

**Working rules (as followed for E0b; now in CLAUDE.md):**
- Small reviewed steps, one stage at a time.
- The user does all git operations.
- Decision values are set only after the user agrees to that specific value.
- After each stage, update `docs/architecture.md` and `scripts/README.md`, and suggest
  entries for `docs/report-notes.md` (what the report may use) and `docs/process-log.md` (how
  the work was done).

## Context

E0a delivered three things, and 32/32 checks pass:
- the split;
- the model-space real volumes;
- the synthetic index.

E0b completes E0. It covers:
- lesion labelling (8-connectivity, attach-or-drop) with a learned minimum lesion area;
- brain masks and boxes;
- the cached lesion table that E1–E7 reuse;
- fresh mask QC of the synthetic pool (clean / minor / noisy);
- the E0 statistics.

Each stage uses only the outputs and decisions of the stages before it.

**Evidence from read-only exploration (2026-11-01) that shaped the design:**
- **Real T component sizes** (8-conn, model space): the 1st percentile is **4 px**; 28 of 6,162
  components are ≤ 2 px and 61 are ≤ 3 px. So the minimum lesion area will be about 4 px.
- **Synthetic sample of 3,000** (min area 4, gap ≤ 2 px):
  - 86% have no isolated speck; 191 have 1; 57 have 2–3; 116 have 10 or more (a clear noisy group);
  - 14% have stray pixels to attach;
  - 85% of isolated specks lie outside the brain;
  - **1% have no component ≥ 4 px**;
  - 12% have no lesion under 40 px after cleaning.
- **Synthetic lesions are spatially concentrated.** Centroid spread: row SD 14.9 vs 24.1 px,
  column SD 18.6 vs 25.3 px, around the same centre. Synthetic brains are smaller: median
  2,975 vs 7,765 px with the chosen brain mask, and brain boxes 63×66 vs 103×95 px. This is a
  finding about stage 1 for the report; `stats` quantifies it on the full pool.

## Decision gates: review stage → user decision → production stage

| Review stage (evidence) | Decision | Production stage |
| --- | --- | --- |
| `brain-review` | A: brain method, failed-MRI tissue cut-off | `brain` |
| `lesions-review` | B, C: minimum lesion area, attach gap | `lesions` |
| `qc-review` | D: speck cut-off | `qc` |
| `filters-review` | E: which sampling filters apply to H-cal/H-eval | E3, E4 (no E0b stage) |

- A review stage computes candidates and writes tables, figures and a proposed value
  (`review.txt`). It never writes production data.
- Each decision is a config value that starts as `null`. It is set in `configs/default.yaml`
  with a dated comment, only after the user agrees.
- A production stage refuses to run while its value is `null`, and names the review stage to
  run first.
- **No leakage.** Decisions A–C use T only; H-cal and H-eval are described, never used to choose
  anything. Decision D views synthetic masks, which is allowed because no detector has touched
  S yet.

Config (`configs/default.yaml`) as of 2026-11-02:
```yaml
brain:
  low_threshold: 0.02         # set 2026-11-02; re-check with brain-review on new data
  method: low_threshold_all   # decision A, set 2026-11-01
  min_tissue_px: 20           # decision A, set 2026-11-01
lesions:
  connectivity: 8
  min_lesion_area: 4          # decision B, set 2026-11-02 (largest m with <= 1% of T components smaller)
  attach_gap: 2               # decision C, set 2026-11-02 (gap 1-2 specks far above the noise floor)
qc:
  max_isolated_specks: 9      # decision D, set 2026-11-02 (end of the falling counts; then a flat noise tail)
  edge_outside_frac: 0.5      # plan: flag a real-sized lesion if > 50% of it is outside the brain
  small_lesion_px: 40         # plan: the sampling filter's "a lesion under 40 px"
filters:
  max_mask_px: 2000           # the sampling script's pixel-sum limit (a fact, not a decision)
  real_sets: [max_mask_px]    # decision E, set 2026-11-03 (only filter the pool shows; removes no real slice)
```

**Decided at the `qc` stage (2026-11-02):** edge flags only flag, never exclude; a failed MRI
does not exclude its mask (separate `mask_class` and `mri_failed` columns); the main audit
(E1–E5) uses `mask_class` clean or minor (35,920 masks). Reasons in `docs/report-notes.md`.

### Decision A (done, 2026-11-01)

**Brain mask.**
- Method: `low_threshold_all`: pixels > `brain.low_threshold` (0.02), keep every 8-connected
  region ≥ 30 px, fill holes.
- The threshold is a config value, not a code constant, because it depends on the data.
  `brain-review` reports how much the brain area changes at 0.25–3× the threshold, plus the
  background and darkest-brain intensity levels. On this data it changes about 1%, so 0.02 sits
  in an empty gap.
- Six candidates were compared on 6,101 T lesions and 2,000 synthetic samples:

  | Method | T lesions >50% outside | T lesions partly outside |
  | --- | --- | --- |
  | `otsu` | 74 | 9.2% |
  | `otsu_dilate1` | 69 | 1.7% |
  | `otsu_dilate2` | 66 | 1.2% |
  | `otsu_all` | 13 | 8.5% |
  | `low_threshold` | 33 | 2.0% |
  | `low_threshold_all` | **1** | 1.5% |

  Full table in `results/stanford_brainmets/e0b/brain_review/review.txt`.
- Proposal rule: among methods with ≤ 0.5% of T lesions more than half outside, take the one
  with the fewest lesions even partly outside.

**Failed MRI.**
- Judged by brain tissue area (`memtk.lesions.tissue_px` = area of the `low_threshold_all`
  brain), not by maximum intensity. A bright lesion on a blank image has a high maximum.
- `min_tissue_px: 20`. Below 20 px all viewed samples have no brain at all (152 samples), and
  the smallest real T lesion slice has 31 px.
- A small brain alone is never a failure: synthetic small brains look like genuine slices from
  the top or bottom of the head.
- Since every kept tissue region has ≥ 30 px, the cut of 20 means "no brain region at all". It
  needs no tuning for a new pool.
- Confirmed 2026-11-02 after the `brain` outline sheet showed small images that pass: a lesion
  with a thin rim of tissue, or a small faint patch, on black. A stricter rule (tissue farther
  than 5 px from the lesions < 20 px) was tested and rejected; it was removed from the code.
  Reasons in `docs/report-notes.md` section 2. Such images are kept; later analyses that use
  the brain decide how to treat small brains, using `brain_px` / `tissue_px` in `brain.csv`.

## Definitions (fixed in code and docs)

- **Gap:** background pixels between the nearest pixels of two components (Chebyshev). Touching
  components are already one component under 8-connectivity, so gap 1 means one empty pixel in
  between. Implemented as dilation of the real-sized lesions by `gap + 1`.
- **Speck:** a component smaller than `min_lesion_area`.
- **Stray pixel:** a speck within `attach_gap` of a real-sized lesion. It is attached to the
  nearest lesion; ties go to the larger lesion.
- **Isolated speck:** any other speck. It is dropped and counted.
- **Two real-sized lesions are never merged.**
- Real and synthetic masks are treated identically.
- **Nothing is moved or deleted.** Attach and drop only decide labels in a derived label map.
  - A stray pixel keeps its position and takes its lesion's label, so it counts in that lesion's
    area, shape and centroid.
  - A dropped speck is left out of the label map and the lesion table, and counted in QC.
- **Every mask has two versions:**
  - *as saved*, never modified: cached for E8 and the "noisy masks included" checks in E5;
  - *cleaned* (`labels > 0`): used by the detectors.

## Lesion IDs and label maps (for human checking)

- **Label map per slice:** 0 = background, 1..k = lesions. Attached strays carry their lesion's
  number; dropped specks are 0 and recorded separately.
- **Numbering within a slice:** by area, largest first (L1 = largest). Ties go to the top-left
  lesion first. Deterministic.
- **Readable unique ID:** `Mets_005/z127/L1` (real; z is 0-based) and `file_2_1033/L2`
  (synthetic). The same ID is used in every table and later in the match reports.
- **Labels are per slice (2D).** The 3D lesion linking for H1 (26-connectivity) happens in E6:
  it adds a `lesion3d_id` column without changing the 2D IDs.

## Library modules

All implemented; `docs/module-reference.md` lists every function (until 2026-11-03 this was in
`docs/architecture.md`). In short:
- **`lesions.py`:** labelling; `attach_or_drop` (→ `LesionLabels`), `speck_gaps`;
  `lesion_features` (centroid in three frames: pixels, image fraction, brain box); brain masks,
  `tissue_px`, `brain_box`, `fraction_outside`; helpers `chessboard_distance`, `grow`.
- **`ids.py`:** slice keys and lesion IDs (`<slice key>/L<n>`), building and parsing.
- **`io.py`:** NIfTI, `read_table` (ID columns kept as text), mask stacks (bit-packed) and label
  stacks (sparse).
- **`qc.py`:** `mask_class` (from the mask alone), `qc_class` (failed MRI first, for
  reporting), `edge_lesions`. The edge check and the < 40 px rule are flag columns, not classes.
- **`plotting.py`:** shared figure helpers and colours; `save` wraps titles and reserves room for
  them; `draw_lesions` / `lesion_sheet` draw labelled slices (used by the overview sheets, QC
  sheets and `view`).
- **`checks.py`:** the E0a and E0b consistency checks.
- **`e0b/` (stage package, restructured 2026-11-03):** one module per topic with its stages,
  then the computations, tables and figures they use: `common`, `brain`, `lesions` (incl.
  `view`), `qc`, `stats`, `filters`, `check`. `scripts/e0b_audit.py` is only the command line.
  The former `audit.py` was split into these modules and removed.
- Not built: `lesion_crop` (a crop centred on a lesion), deferred to E2, where it is used.

## Position frame: open, decided when E2 is planned

E0b stores the lesion centroid in three frames (pixels, image fraction, brain-relative). Which
frame E2 uses for the position factor is open; the question, the reasoning and the
recommendation (pixel coordinates for copy detection) moved on 2026-11-03 to
`docs/experiment-plan.md`, E2, "Carried over from E0". The evidence is in `docs/report-notes.md`
(section 3, "Synthetic lesion positions"; section 5, open question).

## Stages

1. **`brain-review`** ✅. Evidence for decision A.
   - Methods evaluated on every T lesion slice and on 2,000 random synthetic samples.
   - Tissue area measured on all T lesion slices and all of S.
   - Figures: method grids (real, synthetic), the near-blank sheet, the tissue histogram.
   - Output: `results/<dataset>/e0b/brain_review/`.
2. **`brain`** ✅ (production; needs A; about 6–8 min). Mask stacks are read with
   `memtk.io.MaskStack` (bit-packed `.npz`, `stack[key]`).
   - Brain masks and boxes with the chosen method, for all real lesion slices (T, H-cal,
     H-eval) and all synthetic samples, plus `tissue_px` and the `mri_failed` flag.
   - Reads the pool once and caches the synthetic masks as saved.
   - Outputs: `<output_root>/<dataset>/brain.csv` + `brain_masks.npz`, and
     `<output_root>/<pool>/brain.csv` + `brain_masks.npz` + `masks_raw.npz`.
   - Includes a visual check that the brain outline follows synthetic brains.
3. **`lesions-review`** (evidence for B, C; proposal from T).
   - The T component-size distribution (figure + table) with the proposed minimum area
     (1st percentile).
   - Sensitivity: lesion counts per set for minimum area (m−1, m, m+1) × gap (1, 2), plus
     drop-only, with stray pixels and specks for each setting.
4. **`lesions`** (production; needs B, C). Final attach-or-drop for T, H-cal, H-eval and all of S.
   - `lesions.csv`: one row per lesion, with its features.
   - `slices.csv`: one row per slice or sample, with lesion count, isolated specks (inside and
     outside the brain), stray components and pixels, and a `has_lesion_lt40` flag.
   - `labels.npz`: label-map stack with its index.
   - For real patients, also `volumes/<pid>/labels.nii.gz`, which opens in ITK-SNAP next to
     `bravo.nii.gz`.
   - Overview sheets `results/<dataset|pool>/e0b/lesions_overview_<set>.png`:
     - lesions coloured, with their L-number at the centroid;
     - attached strays in a lighter shade, dropped specks marked ×;
     - brain outline and box;
     - the slice key as title.
   - Unit tests, including the numbering rule.
   - **`view`** helper: `view --sample file_2_1033` or `view --patient Mets_005 --z 127`.
     - It writes the full-size picture with that slice's `lesions.csv` rows under it, to
       `results/.../e0b/views/`.
     - It can also export a synthetic sample's labels as NIfTI.
5. **`qc-review`** (evidence for D).
   - Montage sheets of about 100 synthetic masks, stratified by isolated-speck count
     (0, 1, 2, 3, 4–5, 6–9, 10+), plus the brain-failure candidates.
   - Each tile shows the MRI, the brain outline, lesions, stray pixels and isolated specks in
     different colours.
6. **`qc`** (production; needs D).
   - Classes for every synthetic sample (`<pool>/qc.csv`).
   - Edge check on S and on the real sets. If real slices fail, report that as the reason to
     loosen the rule.
   - Summary counts: clean, minor, noisy, no_lesion, mri_failed, edge-flagged, < 40 rule, strays
     attached.
   - Cross-table against the old QC list (2,243 masks).
   - Adds `memtk/qc.py` and `tests/test_qc.py`.
7. **`stats` and `filters-review`**: see "Stage 7 details" below (planned and approved
   2026-11-02).
8. **`check`**. Consistency checks, as in e0a:
   - lesion areas and centroids match the label maps;
   - every non-zero label has exactly one row in `lesions.csv`, and the numbering follows the
     size rule;
   - label maps match the saved masks: lesion pixels = real component + attached strays, and
     dropped specks are absent;
   - every synthetic sample has exactly one QC class, and the counts add up;
   - the pytest suite passes.

## Stage 7 details (approved 2026-11-02)

The experiment plan asks E0 for a data description table, three distribution figures, the
median adjacent-slice Dice and the native-vs-model lesion area, and says to apply to the real
sets only the sampling filters the pool shows. The description has no decision, so it is one
stage (`stats`); the filter choice is decision E, so it gets its own review stage, like A–D.

### 7a `stats` (descriptive; no decision)

Outputs: `results/<dataset>/e0b/stats/`, with the pool name in the file names (the stage
compares the real sets with a pool, like `brain-review`); tables as CSV, figures as PNG. Items 1, 2 and 6 come from existing tables (`lesions.csv`, `slices.csv`,
`qc.csv`, `brain.csv`); items 3–5 read volumes again. Implement as two reviewed steps:
step 1 = items 1, 2, 6; step 2 = items 3, 4, 5.

1. **Data description table.** Columns: T, H-cal, H-eval, S main audit (`mask_class` clean or
   minor), S all. Rows: patients/samples and slices; lesions and lesions per slice; share of
   single- and multi-lesion slices; lesion area (median, IQR); share of slices with a lesion
   under 40 px; spread of lesion positions (centroid SD in pixels); native in-plane sizes (real).
2. **Three distribution figures**, T, H and S side by side: lesions per slice; lesion area (log
   scale); 2D density of lesion centroids **in two frames, pixels and brain-relative** (the
   position frame for E2 is still open; seeing both helps decide it).
3. **Adjacent-slice Dice** (plan E0 step 6): per T patient, Dice between consecutive lesion
   slices on the **cleaned** masks (`labels > 0`, as E1 uses); per-patient medians and the
   overall median. Measures how near-duplicate neighbouring slices are (E5 groups copies by
   ±2 slices because of this).
4. **Native vs model-space lesion area** (plan E0 step 5): on slices with exactly one lesion in
   both spaces, model-space area against native area scaled to 128×128, by size bin. Reads
   the native masks once. E0a measured the enlargement per patient (median 1.5×).
5. **Brain size by slice height:** real brain area over *all* real slices (about 23,000; brain
   mask with the decided method) against relative height in the head, with the synthetic
   brain areas alongside. Tests the report-notes finding that small synthetic brains look like
   slices near the top or bottom of the head. Not strictly required by the plan: the first
   item to drop if time is short.
6. **Bound-impact table (input for E3, not decided here):** for candidate informative-lesion
   bounds 5, 8, 10, 12, 15, 20, 25, 30, 40 px, per set: share of lesions kept and share of
   slices left with no informative lesion. The shape and size finding (lesions under ~40 px
   are hard to tell apart) suggests the E3 bound may land high; this shows the price.

### 7b `filters-review` (evidence for decision E)

The sampling script kept a pair only if (1) the mask has under 2,000 px, (2) at least one
lesion is under 40 px, (3) every lesion pixel lies inside the generated brain.
- **Show each filter both ways:** on the masks as saved, and after cleaning. Filter 2 was
  applied to raw masks, where a speck counts as "a lesion under 40 px"; that is why 15% of the
  pool has no small lesion after cleaning (their only small "lesion" was a speck).
- **Per filter:** share of S that passes (does the filter hold?), and share of each real set
  that would pass (what would applying it cost?).
- **Filters apply to H-cal and H-eval only** (chance rates, calibration, planted copies), never
  to T: T is the reference copies are matched against, so it must stay complete.
- **Decision value:** a config key such as `filters.real_sets: null`, set by the user to a list
  of filters; E3 and E4 apply it.
- Early expectation: filter 1 holds in S but removes no real slice (harmless); filter 3 does
  not hold (2.6% of masks have edge lesions) and should not be applied; filter 2 is the real
  question.

### 8 `check` (as before), then 9 the full re-run

## Verification (every stage)

- Each stage prints its summary, writes `run_info.json`, and refuses to overwrite without
  `--overwrite`. Production stages refuse to run while their decision is `null`.
- pytest passes (`.venv\Scripts\python.exe -m pytest`), with toy masks covering:
  - a speck at gap 1 or 2 is attached, at gap 3 it is dropped;
  - two 4 px lesions at gap 1 stay separate;
  - a 4×4 square has area 16 and its centroid at the centre;
  - QC precedence gives the expected class.
- The user views the figures and sheets for each decision.
- The `check` stage passes, and a negative test on damaged copies shows the checks can fail.
- Counts are cross-checked against the exploration above: about 86% clean, about 1% no_lesion,
  and a noisy group near 4%.

## Carried forward (2026-11-03)

The open items of this plan moved to the documents where they will be used:

| Item (first noted) | Moved to |
| --- | --- |
| Position frame for copy detection (2026-11-01) | `docs/experiment-plan.md`, E2, "Carried over from E0" |
| Lesion crop (`lesion_crop`), not built in E0 | the same E2 notes |
| Table formats: CSV for readable tables, Parquet for the large E2/E5 pair tables (2026-11-02) | the same E2 notes |
| 3D lesion linking for H1 from the real masks as saved, not the cleaned labels (2026-11-02) | `docs/experiment-plan.md`, E6, duplication test step 1 |
| Per-pool decisions when a new synthetic pool arrives (re-run `brain-review`, per-pool overrides) | `scripts/README.md`, Notes |

Done before closing: the pinned `requirements-lock.txt` and the full re-run of E0a + E0b
(steps 9 and 10 above).
