# E0b plan: data audit and mask quality control (`scripts/e0b_audit.py`)

Working plan for E0b, kept in the repo so work can resume from here. The experiment itself is
specified in `docs/experiment-plan.md` (E0); this file records how it is implemented, stage by
stage, and what has been decided. Update the status table when a stage is done.

## Status (2026-11-01)

| # | Stage | Kind | Status |
| --- | --- | --- | --- |
| 0 | Drop the null model in the docs | docs | ✅ done, committed |
| 1 | `brain-review` | review → decision A | ✅ done; **decision A set** |
| 2 | `brain` | production (needs A) | ⏳ next |
| 3 | `lesions-review` | review → decisions B, C | ⏳ |
| 4 | `lesions` + `view` | production (needs B, C) | ⏳ |
| 5 | `qc-review` | review → decision D | ⏳ |
| 6 | `qc` | production (needs D) | ⏳ |
| 7 | `stats` | E0 outputs | ⏳ |
| 8 | `check` | consistency checks | ⏳ |

**Working rules:**
- Small reviewed steps, one stage at a time.
- The user does all git operations.
- Decision values are set only after the user agrees to that specific value.
- After each stage, update `docs/architecture.md` and `scripts/README.md`, and suggest
  `docs/report-notes.md` entries.

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

- A review stage computes candidates and writes tables, figures and a proposed value
  (`review.txt`). It never writes production data.
- Each decision is a config value that starts as `null`. It is set in `configs/default.yaml`
  with a dated comment, only after the user agrees.
- A production stage refuses to run while its value is `null`, and names the review stage to
  run first.
- **No leakage.** Decisions A–C use T only; H-cal and H-eval are described, never used to choose
  anything. Decision D views synthetic masks, which is allowed because no detector has touched
  S yet.

Config (`configs/default.yaml`) as of 2026-11-01:
```yaml
brain:
  method: low_threshold_all   # decision A, set 2026-11-01
  min_tissue_px: 20           # decision A, set 2026-11-01
lesions:
  connectivity: 8
  min_lesion_area: null       # decision B: smaller components are specks (from T component sizes)
  attach_gap: null            # decision C: max background px between speck and lesion (1 or 2)
qc:
  max_isolated_specks: null   # decision D: minor if 1..N isolated specks, noisy if more (plan suggests 3)
  edge_outside_frac: 0.5      # plan: flag a real-sized lesion if > 50% of it is outside the brain
  small_lesion_px: 40         # plan: the sampling filter's "a lesion under 40 px"
```

### Decision A (done, 2026-11-01)

**Brain mask.**
- Method: `low_threshold_all`: pixels > 0.02, keep every 8-connected region ≥ 30 px, fill holes.
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
- Caveat: a bright synthetic lesion ≥ 30 px on an otherwise blank image would count as tissue.
  The `qc-review` sheets give a second look.

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

**`src/memtk/lesions.py`.**
- *Exists:*
  - `label_components`, `count_components`;
  - `brain_mask(image, method)` with the six `BRAIN_METHODS`;
  - `tissue_px`, `brain_box`, `fraction_outside`.
- *To add in stage 4:*
  - `attach_or_drop(mask, min_area, gap)`: returns the label map plus counts (isolated specks,
    stray components and pixels).
  - `lesion_features(labels, brain, box)`:
    - area, equivalent diameter, eccentricity and solidity (`skimage.measure.regionprops`);
    - bbox and fraction outside the brain;
    - the centroid in **three frames**: pixels `(row, col)`; the same scaled to [0, 1] by
      image size; and brain-relative `(x, y)` in the brain box, with the box stored too.
  - `lesion_crop(labels, lesion_id, size=16)`: a crop centred on the centroid, enlarged if the
    lesion doesn't fit. Computed on demand, not stored.

**`src/memtk/audit.py`** (exists). Review-stage evidence: method evaluation and summary, the
proposal rule, method grids, sorted sheets, tissue counts and histogram.

**`src/memtk/plotting.py`** (exists). Shared figure helpers, using `matplotlib.figure.Figure`
without pyplot.

**`src/memtk/qc.py`** (stage 6). Classifies synthetic samples. Precedence, first match wins:
1. `mri_failed` (`tissue_px < brain.min_tissue_px`);
2. `no_lesion` (no real-sized component after cleaning);
3. `noisy` (> N isolated specks);
4. `minor` (1..N);
5. `clean` (0).

The edge check and the < 40 px rule are separate flag columns, not classes.

## Position frame: open, decided when E2 is planned

The plan measures lesion position relative to the brain box of the paired MRI. But stage 1
generates masks without seeing a brain, so a memorized mask reproduces its training mask at the
same *pixel* positions. The synthetic brain box comes from stage 2, and synthetic brains are
smaller, so even a pixel-perfect copy can get shifted brain-relative coordinates. E4 cannot
reveal this, because planted copies are real masks with their own real MRI.

Recommendation:
- **Copy detection (E2/E4/E5):** absolute model-space coordinates (e.g. σ = 0.05 × 128 ≈ 6.4 px),
  consistent with E1's pixel-wise Dice.
- **Anatomical questions** (density figure, E6 rarity): brain-relative coordinates.

E0b stores all three frames. The open question is recorded in `docs/report-notes.md` section 5.

## Stages

1. **`brain-review`** ✅. Evidence for decision A.
   - Methods evaluated on every T lesion slice and on 2,000 random synthetic samples.
   - Tissue area measured on all T lesion slices and all of S.
   - Figures: method grids (real, synthetic), the near-blank sheet, the tissue histogram.
   - Output: `results/<dataset>/e0b/brain_review/`.
2. **`brain`** (production; needs A; about 6–8 min).
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
7. **`stats`**. E0 outputs into `results/<dataset>/e0b/` and `results/<pool>/e0b/`:
   - data description table (T / H-cal / H-eval / S clean+minor);
   - three distribution figures: lesions per slice, lesion area, 2D centroid density;
   - brain size: synthetic vs real over *all* real slices by height (z), to test whether
     synthetic slices resemble slices from the top or bottom of the head;
   - lesion position spread on the full pool;
   - median adjacent-slice Dice per training patient;
   - per-lesion native vs model area on single-lesion slices;
   - filter table: the share of S passing each sampling filter, i.e. which filters to apply to
     the real sets (decision E, recommended in the output);
   - bound-impact table: for informative-lesion bounds of 5–40 px, the share of S and T slices
     left without an informative lesion. This is input for E3, not decided here.
8. **`check`**. Consistency checks, as in e0a:
   - lesion areas and centroids match the label maps;
   - every non-zero label has exactly one row in `lesions.csv`, and the numbering follows the
     size rule;
   - label maps match the saved masks: lesion pixels = real component + attached strays, and
     dropped specks are absent;
   - every synthetic sample has exactly one QC class, and the counts add up;
   - the pytest suite passes.

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

## Later (outside E0b)

- Decide the position frame when E2 is planned, then update `docs/experiment-plan.md`.
- Write a pinned `requirements-lock.txt` at the end of E0.
