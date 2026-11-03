# E1–E3 plan: both detectors and chance calibration (`scripts/e3_calibrate.py`)

Working plan for E1, E2 and E3, kept in the repo so work can resume from here. The experiments
are specified in `docs/experiment-plan.md` (E1, E2, E3); this file records how they are
implemented, stage by stage, and what has been decided. Update the status table when a step is
done. E1 and E2 have no scripts of their own: they are library modules that E3 runs on H-cal,
E4 on planted copies and E5 on S.

## Status (2026-11-03)

| # | Step | Kind | Status |
| --- | --- | --- | --- |
| 0 | This plan; decision F recorded in the docs | docs | ✅ 2026-11-03, reviewed by the user (choices accepted) |
| 1 | Shared stage helpers → `memtk/stages.py`; config sections `detectors:`, `calibration:` | refactor + config | ✅ 2026-11-03: `decided` takes a dotted key and the full review command; F set in the config, G and H `null`; pytest 66 passed, E0b check 43/43 |
| 2 | E1 library `memtk/wholemask.py` + tests | library | ✅ 2026-11-03: 10 tests; self-match sanity on 300 random T slices: all score 1 at no shift (4 matched an identical neighbouring slice); 4.4 ms per query (S main ≈ 3 min) |
| 3 | E2 library `memtk/lesionmatch.py` + tests | library | ✅ 2026-11-03: 14 tests (shortcuts = brute force on random maps); on real T: self-match C = 1 for 300/300, shortcuts = brute force on 30 real slices (5,608 candidate slice pairs, 0 differences); 10 ms per query slice |
| 4 | `bound-review` | review → decision G | ✅ 2026-11-03 (`memtk/calibration.py`, `memtk/e3/`, `scripts/e3_calibrate.py`, 7 tests); **decision G set** by the user: provisional tau_L 0.77, cut 10%, bound 12 px (the 5% cut gives 25 px) |
| 5 | `calibrate` | production (needs G) → evidence for decision H | ✅ 2026-11-03. First run: the planned τ_C was degenerate (≈ 0.02) because C counted only pairs ≥ τ_L. Fixed (user, 2026-11-03): **C counts every paired lesion from the candidate cut 0.1; classes as planned** (full = C ≥ τ_C, partial = a pair ≥ τ_L). **Decision H set**: stratified for both detectors. τ_D 0.905 / 0.729, τ_L 0.739 / 0.816, τ_C 0.719 / 0.406 (single / multi) |
| 6 | `freeze` + `check` | production (needs H) + consistency checks | ⏳ |

**Next: start here.** Step 6, `freeze` (writes `configs/frozen/e3_<dataset>.yaml` from the stored
H-cal tables with decision H) and `check`. (The architecture docs were reworked first,
2026-11-03: new modules and files go into `docs/module-reference.md` and `docs/data-dictionary.md`.)

**Milestone:** 9 Nov 2026, both detectors run on H-cal; informative-lesion bound and thresholds
frozen. E4 (gate) by 13 Nov. Indicative schedule: step 1–2 on 4 Nov, step 3 on 5 Nov, step 4 and
decision G on 6 Nov, step 5 and decision H on 7 Nov, step 6 on 8 Nov, 9 Nov as buffer.

**Working rules** (as in CLAUDE.md):
- Small reviewed steps, one at a time; the user does all git operations.
- Decision values are set only after the user agrees to that specific value.
- After each step, update `scripts/README.md`, `docs/data-dictionary.md` and
  `docs/module-reference.md` (and `docs/architecture.md` if the structure changes), and suggest entries for
  `docs/report-notes.md` (what the report may use) and `docs/process-log.md` (how the work was
  done).
- **S is never scored in E1–E3.** Everything here runs on T and H-cal only. The only facts about
  S used here are E0 descriptions (how many slices a bound would exclude), which use no detector.

## Context

E0 delivered everything the detectors need, so E1–E3 read caches and never touch raw data:

| Input | From | Used for |
| --- | --- | --- |
| `lesions.csv` (one row per lesion: ID, area, centroid in pixels and two other frames, bbox) | E0b `lesions` | E2 lesion records, strata, the bound |
| `labels.npz` (label map per slice; cleaned mask = `labels > 0`) | E0b `lesions` | E1 masks, E2 lesion pixels (shape factor) |
| `slices.csv`, `brain.csv` | E0b `lesions`, `brain` | slice lists, lesion counts, decision E filter |
| Split: T 105 patients / 3,823 slices / 6,101 lesions; H-cal 25 / 992 / 1,548 | E0a, E0b `stats` | query and reference sets |
| Decision E: only the pixel-sum filter (< 2,000 px) applies to H-cal and H-eval; it removes no real slice | E0b `filters-review` | applied to H-cal before calibration |

Facts from E0 that shape the design (`docs/report-notes.md`):
- **Small lesions carry little shape information.** For random pairs of T lesions of similar
  size, median shape overlap × size ratio is 0.40–0.57 below 40 px and 0.27 at ≥ 40 px (section
  3). Lesion-level matching must lean on position and multi-lesion configurations, and the
  informative-lesion bound may land at the upper end of the expected 10–20 px or above.
- **What a bound costs** (E0b `stats`, `bound_impact_amibekk_v1.csv`): share of slices with no
  informative lesion at 10 / 15 / 20 / 30 / 40 px: H-cal 10 / 24 / 35 / 51 / 63%; S main 20 /
  37 / 50 / 67 / 77%. H-cal patients keeping an informative slice: 24 / 24 / 23 / 23 / 18 of 25.
  At 20 px half the main audit leaves the lesion-level analysis (still scored by E1).
- **Neighbouring slices are near-duplicates** (median adjacent-slice Dice 0.80 in T), so a
  single H-cal patient contributes many similar queries: confidence intervals are bootstrapped by
  patient, not by slice.
- **Multi-lesion strata are small:** H-cal has 303 multi-lesion slices from 17 patients (≥ 2
  lesions ≥ 4 px); fewer with informative lesions only. Hence decision H (stratified or pooled).
- **19 T slices and 3 H-cal slices have no lesion of ≥ 4 px**: their cleaned masks are empty;
  neither detector scores them (reported as excluded; T limitation in report-notes section 4).

## Decision gates: review stage → user decision → production stage

Lettering continues from E0b (A–E).

| | Decision | Evidence | Needed by |
| --- | --- | --- | --- |
| F | Position frame for the E2 position factor | E0b `stats` centroid density | all E2 code |
| G | Provisional τ_L, chance-rate cut (5% or 10%), informative-lesion bound | `bound-review` | `calibrate` |
| H | Thresholds per stratum (single / multi-lesion) or pooled, per detector: **stratified for both, set 2026-11-03** | `calibrate` | `freeze` |
| — | Configuration score and classes: **C counts every Hungarian pair with s ≥ 0.1** (not only ≥ τ_L); **full = C(q, NN) ≥ τ_C, partial = not full but a pair ≥ τ_L** (rule A, the plan's own), set 2026-11-03 after a one-run detour via a structural rule | `calibrate` (τ_C degenerate with the old C) | `lesionmatch`, E4, E5 |

**Decision F (set 2026-11-03 by the user): pixel coordinates.** Stage 1 generates masks without
seeing a brain, so a memorized mask reproduces its training mask at the same pixel positions;
synthetic brains are smaller (median 3,009 vs 7,765 px) and the brain-relative position of a
synthetic lesion mostly reflects how stage 2 drew the brain around it. Pixel coordinates are also
consistent with E1's pixel-wise Dice. With the plan's σ = 0.05 of the field: σ = 0.05 × 128 =
6.4 px; the prefilter distance 0.2 × 128 = 25.6 px, where the position factor is exp(−8) ≈
0.0003. Brain-relative coordinates stay for anatomical questions (E6 rarity, density figures).
Config: `detectors.lesion.position_frame: pixel`, set with a dated comment in step 1.

Each decision is a config value that starts as `null` (F is set directly, as decided); the
production stage refuses to run while its value is `null` and names the review stage to run.

## Definitions (fixed in code and docs)

- **Query set** Q: the masks being scored (H-cal here; planted copies in E4; S in E5).
  **Reference set**: T. Masks are compared as cleaned masks (`labels > 0`); the masks as saved
  are only used where the plan says so (E5's noisy-mask check, E8).
- **E1 score:** the highest Dice between the query mask and a candidate training mask over shifts
  of up to ±3 px in rows and columns (49 positions). The query is moved; pixels moved off the
  image are lost (no wrap-around), and Dice keeps the query's full area in its denominator.
  Ties: the smallest shift, then the higher plain Dice, then the first training slice.
  **Margin:** best score minus the best score of any slice from another patient (shift-searched
  for candidates, plain Dice for the rest).
- **Lesion score** s(a, b) = shape × position × size, each in [0, 1]:
  - *shape:* IoU of the two lesions' pixel sets with their centroids aligned (rounded to whole
    pixels); the same as IoU of centred crops large enough for both;
  - *position:* exp(−d² / 2σ²), d = pixel distance between centroids, σ = 6.4 px;
  - *size:* smaller area / larger area.
- **Informative lesion:** area ≥ the bound (decision G). Only informative lesions enter E2.
- **Slice-pair (configuration) score** C(q, t): Hungarian one-to-one pairing of the informative
  lesions on cost 1 − s; pairs with s ≥ 0.1 (the candidate cut) are kept; C = sum of kept s /
  max(informative lesions in q, in t). C does not depend on τ_L (changed 2026-11-03).
- **NN:** the training slice with the highest score (E1: Dice; E2: C).
- **E2 classes (the plan's rule A, confirmed 2026-11-03):** full copy = C(q, NN) ≥ τ_C; partial
  = not full, but the strongest lesion pair over all candidate slices ≥ τ_L; novel otherwise. NN =
  the slice with the highest C (ties: higher strongest pair, then the first training slice). Also
  recorded: `n_kept` (NN lesions passing τ_L), `all_copied` (all of them do), the strongest single
  lesion pair and its slice. (With C counting only pairs ≥ τ_L, τ_C came out at about 0; a
  structural "all lesions ≥ τ_L" rule was used for one run before C was redefined.)
- **Strata:** single- or multi-lesion slice, by the number of lesions ≥ 4 px for E1 and by the
  number of informative lesions for E2 (the lesions each detector actually sees).
- **Chance-match rate** (for the bound): the share of H-cal lesions in an area bin whose best
  lesion score against any T lesion is ≥ τ_L. H-cal patients were never trained on, so every such
  match is chance.
- **Thresholds** at the 95th percentile of H-cal (each one an H-cal slice statistic): τ_D of the
  E1 score, τ_L of the strongest single lesion-pair score, τ_C of C(q, NN) (no match counts as
  0); stratified single / multi (decision H, set 2026-11-03).

## Library modules (building blocks, no knowledge of stages)

- **`memtk/stages.py` (step 1, moved, not new):** `decided`, `refuse_overwrite`, `require`,
  `pick` from `memtk/e0b/common.py`, and `add_stage` from `scripts/e0b_audit.py`; `decided`
  names the right script in its message. E0b modules import them from here; E0b `check` and
  pytest are re-run to confirm nothing changed.
- **`memtk/wholemask.py` (E1, step 2):**
  - Dice of every query mask against every training mask in one matrix product (flattened
    masks, Q·Tᵀ; float32, in blocks of queries);
  - candidates per query: top 20 by Dice plus top 20 by the centroid distance of the largest
    lesion (catches shifted copies with low plain Dice);
  - shift search ±3 px on the candidates (49 positions, shifted masks padded with zeros, not
    wrapped);
  - result per query: NN slice and patient, score, shift, margin; plus the best score per
    training patient (needed for the margin and later for E6).
- **`memtk/lesionmatch.py` (E2, step 3):**
  - lesion records from label maps plus `lesions.csv` (pixel sets, area, centroid); E4's planted
    copies are not in `lesions.csv`, so the same records can be built from any mask through
    `lesions.attach_or_drop` + `lesion_features`, exactly as E0b did;
  - lesion score s with its three factors, vectorized over many pairs;
  - exact shortcuts: the position prefilter (pairs more than 25.6 px apart are skipped) and
    candidate slices (C is computed only for training slices sharing a lesion pair with
    s ≥ 0.1; all others have C = 0). Exact only while τ_L ≥ 0.1 and s < τ_L beyond the prefilter
    distance; both are checked at `freeze`;
  - C(q, t) by `scipy.optimize.linear_sum_assignment`, the classes, and the three tables of the
    experiment plan: slice pairs, lesion pairs (both lesion IDs, the three factors, s, passed
    τ_L) and synthetic summary. Every candidate pair is stored; thresholds are applied at
    analysis time.
  - Tables: Parquet (pairs, many rows) with the readable lesion IDs; `pyarrow` is added to
    `pyproject.toml` and `requirements-lock.txt` (with its requirements) in step 5, where the
    tables are first written (the library returns DataFrames and writes nothing).
  - As built (step 3): crops are centred on the centroid rounded with `np.rint` (half to even,
    so s(a, b) = s(b, a)); a query lesion's best single match counts only pairs with
    s >= 0.1 (NaN otherwise); pairs beyond the prefilter enter the Hungarian pairing as s = 0.
- **`memtk/calibration.py` (E3, steps 4–5):** percentile thresholds per stratum and pooled;
  bootstrap by patient (resample H-cal patients with replacement, seed 2026, 2,000 draws) for a
  confidence interval of every threshold; chance-match rate by area bin; the bound rule.

## Stages: `scripts/e3_calibrate.py` → `src/memtk/e3/`

The script is only the command line (as `e0b_audit.py`); stages live in the stage package, one
module per topic (`bound`, `calibrate`, `check`, plus `common` for loading T and H-cal).

1. **`bound-review`** (evidence for decision G; step 4).
   - Every H-cal lesion (≥ 4 px, i.e. all lesions) scored against every T lesion within the
     prefilter distance; per H-cal lesion its best s, the three factors and the T lesion.
   - Chance-match rate by area bin (4–5, 6–9, 10–14, 15–19, 20–29, 30–39, ≥ 40 px) for a grid of
     τ_L (0.3 to 0.8), as a table and figure.
   - Proposal, the plan's one-round rule: provisional τ_L = 95th percentile of the best lesion
     scores of H-cal lesions ≥ 10 px (the lower end of the expected bound); bound = smallest bin
     edge from which the chance-match rate stays below the cut (5% and 10% both shown); final
     τ_L is computed in `calibrate` from the informative lesions only.
   - The cost of each candidate bound for T, H-cal and S main (E0b `bound_impact`, no detector on
     S), and H-cal patients kept.
   - Example sheet: H-cal lesions with their best T match, by area bin and score (drawn with
     `memtk.plotting`), so the user can see what a chance match looks like.
   - Output: `results/<dataset>/e3/bound_review/` (`review.txt` with the proposal).
   - If the bound lands at 20 px or above, half of the main audit leaves the lesion-level
     analysis: the review says so, and the user weighs it when setting G (parameter choices are
     the user's; no supervisor sign-off, confirmed 2026-11-03).
2. **`calibrate`** (needs G; F is set; step 5).
   - H-cal after decision E (pixel-sum filter; no slice expected to drop), slices without a
     lesion reported and left out.
   - E1 on H-cal against T: one row per H-cal slice (NN, patient, score, shift, margin).
   - E2 on H-cal against T with the bound: the three tables.
   - τ_D, τ_L and τ_C at the 95th percentile, per stratum and pooled, each with its
     bootstrap-by-patient interval, the number of slices and patients behind it, and the share
     of each stratum the pooled threshold flags; the E2 classes on H-cal (flagged, full,
     partial, all lesions copied) under the pooled and the stratified thresholds.
   - Sensitivity, one alternative per setting: σ = 0.08 × 128 ≈ 10.2 px; the 99th percentile;
     the bound one bin lower (10 px); and, added with decision G, the 5% cut's bound (25 px).
   - Exclusion counts: lesions and slices the bound removes from T and H-cal (and S main from
     E0b `bound_impact`).
   - Score histograms of H-cal (both detectors) with the thresholds marked.
   - Outputs: E1 table (CSV) and E2 tables (Parquet) in `processed_data/model_space/<dataset>/e3/`;
     thresholds, intervals, sensitivity table and figures in `results/<dataset>/e3/calibrate/`
     with a proposal for decision H (pool a stratum when its interval is wide or rests on few
     patients).
3. **`freeze`** (needs H; step 6). Writes `configs/frozen/e3_<dataset>.yaml`: τ_D, τ_L and τ_C
   (per stratum or pooled, as decided), the class rule, the bound, the position frame, σ, the prefilter
   distance, top-k, the shift range, the percentile, and the run log reference. In git; refuses
   to overwrite. Before writing it confirms the shortcuts' conditions (τ_L ≥ 0.1; s < τ_L at the
   prefilter distance for every lesion size). E4 and E5 read only this file.
4. **`check`** (step 6). Exit code 1 on any failure:
   - the frozen file is reproduced from the stored H-cal tables;
   - the shortcuts give the same results as brute force on a random subsample of H-cal slices;
   - self-match sanity: random T slices queried against T find themselves (E1 score 1, C = 1);
     a sanity check of the code only, not part of E4 and not used to tune anything;
   - the run logs of every E3 stage list only T and H-cal as inputs (S never read before the
     freeze);
   - pytest passes.

## Tests (pytest, small hand-built inputs)

- `tests/test_wholemask.py`: Dice values on toy masks; a shifted exact copy is recovered with
  score 1 and the right shift; no wrap-around at the border; the margin uses another patient.
- `tests/test_lesionmatch.py`: an identical lesion scores 1; position and size factors at known
  distances and areas; the experiment plan's worked example (A–X 0.9, B–Z 0.2, three lesions in
  t → C = (0.9 + 0.2) / 3 = 0.37, a partial copy below τ_C); the shortcuts equal brute force on random toy sets; classes.
- `tests/test_calibration.py`: percentiles per stratum; the bootstrap is repeatable with a seed
  and resamples patients, not slices; the bound rule on a constructed rate table.

## Verification (every step)

- Each stage prints its summary, writes a run log and refuses to overwrite without
  `--overwrite`; production stages refuse to run while their decision is `null`.
- pytest passes; E0b `check` still passes after step 1.
- The user views the evidence of each review stage before setting a decision.
- Before E4: the frozen file exists, `check` passes, and the share of S excluded by the bound is
  accepted by the user, with the reason recorded (decision G: 28% of S main, 2026-11-03).

## Related work

- **Architecture docs rework: done 2026-11-03** (before the code of step 1). `docs/architecture.md`
  is a short overview for people (experiment flow, data flow, code layers, folders,
  conventions); the detailed material moved, complete, to `docs/data-dictionary.md` (pipeline
  order) and `docs/module-reference.md` (functions, imports and "used by" as tables).
- **Later, not in E1–E3:** E4 planted copies (P1–P6) use the same detector code on masks built
  from T; the go/no-go criteria must be fixed before E4 runs (open decision in the experiment
  plan).
