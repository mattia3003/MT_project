# E8 plan: paired synthetic MRI check (`scripts/e8_paired.py`)

> **Closed 2026-11-16. Reference only, no longer updated.** E8 is done: every analysis choice fixed in
> `paired:` (decisions AA–AE) and rehearsed on the H-eval reference, an H-cal null and a planted positive
> mock before S's MRIs were read, in one `paired` run; E8 check 17/17. Result: reading **not supported**:
> synthetic MRIs of the 56 strongest mask resemblances are no closer to their source slice's BRAVO than
> those of matched novel masks (lesion-neighbourhood AUC 0.48 [0.40, 0.57], H-eval 0.53), and no closer
> to the source than to another patient's slice with a similar mask. Open items were carried forward on
> 2026-11-16 (see "Carried forward" at the end).

Working plan for E8, kept in the repo so work can resume from here. The experiment is specified in
`docs/experiment-plan.md` (E8, including its "Carried over" notes and the "Ready for planning" brief);
this file records how it is implemented, stage by stage, and what has been decided. Update the status
table when a step is done. E8 reads E5's and E7's stored tables of S and never scores S again.

E8 asks the RQ4 question (H4, observational): is the paired synthetic MRI of a strongly matched mask
closer to the matched training slice's BRAVO than the paired MRI of a novel mask is to its nearest
training slice? It is a lean, exploratory pilot for the master thesis's "mask as key" question (RQ-M1),
decided by the user on 2026-11-15. E6 reads the E1 multi-lesion excess as typical placement, and E7
found no exact lesion-level copy, so group A holds **resemblances**, not proven copies. E8 asks whether
stage 2 draws tissue closer to the matched training slice for them than for novel masks. **Synthetic
MRIs are measured for the first time here:** the E7 report tiles show them, but no MRI statistic of S
has been computed.

## Status (2026-11-16)

| # | Step | Kind | Status |
| --- | --- | --- | --- |
| 0 | This plan; decisions AA–AE recorded in the docs | docs | ✅ 2026-11-16 (AA, AD, AE and the primary measure of AC chosen by the user in the planning session; the remaining values of AB and AC approved with the plan) |
| 1 | `memtk/paired.py` + tests; config section `paired:` with the values of AA–AE, including the SSIM `data_range` set from T's z-scored intensities (no S) | library + config | ✅ 2026-11-16 (`data_range` 10 set by the user): `memtk/paired.py` (`best_match`, `align`, `zscore_in_brain`, `lesion_region`, `regions`, `ssim_map`, `region_means`, `match_controls`, `size_cell`; `comparison_shift` and `runner_up_slice` of the plan became one function, `best_match`, since both are the same search); `tests/test_paired.py` 9 tests (pytest 192). Config `paired:` written with every approved value; `data_range` `null` until the user sets it. Read-only measurement on real slices (no S; inline script): brain masks recomputed with `lesions.brain_mask` equal E0b's stored masks for all 3,823 T and 1,094 H-eval slices; whole-image range of the z-scored BRAVO median 9.80 on T (5th–95th pct 6.82–12.74), 9.26 on H-eval; background about −2.9; lesion neighbourhood (+8 px, in the brain) median 513 px on T (5th pct 282), so no minimum region size is proposed. `data_range` = 10 z-units (about T's median range; fixed for every pair; per-pair max − min rejected, it changes with each lesion's brightness) |
| 2 | E8 `select` (reads no MRI): group A, matched group B, comparison slices, alignment shifts, runner-up slices; the user reviews | production | ✅ 2026-11-16 (user review of `results/amibekk_v1/e8/select/` pending): `e8.common`, `e8.select`, `scripts/e8_paired.py`; seconds; re-run byte-identical. **Group A** 56 masks (30 near-exact, 19 beyond held-out, 4 + 3 exposed) from 41 source patients. **B pool** 20,875 S main masks (20,887 novel by both detectors with an informative lesion, 12 with a failed MRI left out; 25 use the E1 nearest neighbour because C = 0); brain-area tertile edges 2,053 / 3,666 px; **group B** 280 masks, 5 per A mask, no A mask short (16 cells; no largest lesion ≥ 80 px in A). Whole-mask Dice with the comparison slice: A median 0.93 [IQR 0.53, 0.95], B 0.62 [0.46, 0.77] (the mask-resemblance gap decision AD judges against H-eval). Shifts on an E1 nearest neighbour equal E1's stored shift in 109 of 109. Within A: 56 runner-up slices (median Dice 0.76, below A's 0.93: the runner-up control is not matched on mask similarity, so a gap there is partly mask resemblance) and 560 Δz slices. Added for it: `case`, `comparison`, `comparison_patient` to `io.ID_COLUMNS` (sample IDs such as `419` stay text) |
| 3 | E8 `rehearse` (reads no S MRI): the H-eval reference, the H-cal null mock, the positive mock; the user reviews (changes still allowed) | rehearsal | ✅ 2026-11-16 (user review of `results/stanford_brainmets/e8/rehearse/` pending): `e8.paired` (the measurement and analysis `paired` will run unchanged: `measure_pairs`, `analyze`, figures, review), `e8.rehearse`, `common.real_slices` / `synthetic_samples`; ~20 s; re-run reproduces every table. **Readings as required:** null (H-cal) *not supported* (lesion AUC 0.479 [0.390, 0.563], difference to H-eval −0.053 [−0.169, +0.057]); positive *supported* (AUC 0.985 [0.969, 0.996], difference +0.453 [+0.377, +0.525]). **H-eval reference:** lesion AUC 0.532 [0.465, 0.605], whole brain 0.468, ring 0.515: in real data, mask resemblance alone makes the lesion region only slightly more alike (medians 0.079 vs 0.066). SSIM between different patients is low (medians 0.05–0.08), same tissue 2 slices away about 0.46. **Alignment:** all 100 P2 masks matched their base slice with the planted shift recovered exactly (sign included); P1 and P2 read alike (AUC 0.986 / 0.984). **Short matching in held-out sets** (S had none): cases with no control H-eval 25 of 131, H-cal 6 of 91, positive 64 of 200 (novel pools of 500–800 slices); every case counts in the AUC, as the rule says; read-only check: matched cases only gives H-eval 0.545 [0.473, 0.619] (vs 0.532), so the rule does not drive the reference. The positive mock tests correctness (alignment, sign, analysis), not sensitivity to faint resemblance: same-patient tissue is far above chance |
| 4 | E8 `paired` (reads S's MRIs, one run): measures, tests, the reading, figures | analysis | ✅ 2026-11-16, one run (13 s; E5's lock unchanged; `select` and `rehearse` made with the current `paired:`, rehearsal as required): `run_paired` in `e8.paired` (with `report_numbers`, `example_keys`, `review_text`; `draw_examples` takes caption notes). 336 synthetic samples, 952 pairs, no empty region. **Reading: not supported.** Lesion neighbourhood: A median SSIM 0.031 vs B 0.049, AUC 0.481 [0.401, 0.565] (Mann–Whitney p 0.68), H-eval 0.532, difference −0.052 [−0.156, +0.054]; whole brain 0.482 [0.406, 0.568]; ring 0.466 [0.386, 0.544]. Secondary (lesion): near-exact 0.533 [0.412, 0.652], beyond held-out 0.419 [0.286, 0.558], exposed 0.55 (4) and 0.20 (3); brain halves 0.456 / 0.515. **Within A:** SSIM to t minus to the runner-up patient's slice +0.002 [−0.018, +0.023] (Wilcoxon p 0.70); the Δz profile (descriptive) peaks next to t in the lesion region (0.057 at −1, 0.054 at 0, 0.025 at −5) and is flatter for whole brain and ring: with the runner-up result, a sign of lesions in the same place in neighbouring slices, not of patient tissue. Synthetic brains median 3,030 px vs comparison 6,558 px; S's SSIM levels (0.03–0.05) lower than held-out real pairs (0.06–0.08). Report 50 is the median case of its reason, so the example figure shows 5 cases |
| 5 | E8 `check`; close the plan; carry items forward (report synthesis) | consistency checks + docs | ✅ 2026-11-16: `e8.check` (17/17, ~1 min: decisions; run logs of `select`, `rehearse`, `paired` in order with S's MRIs read by `paired` only; E5's lock as recorded; rehearsal readings reproduced and as required; `select`'s pairs, every rehearsal table and `paired`'s measures, tables and reading reproduced with the seed; brain masks equal E0b's for 3,047 images; 9 figures; pytest 196). Reproducing `paired`'s measures re-reads S's MRIs (verification only). Refactor for the check: `select_pairs`, `rehearse_cases` (with `Case`) and `measure_s` split from the stage functions; the check reproduces the stored outputs, so the refactor changed nothing. Scratch test (not in the repo): a damaged copy (one AUC +0.001, one row removed, one comparison key changed) was reported as `tests.auc`, `within_a`, `measures.comparison`. Plan closed |

**Next: start here.** Plan closed. Next project step: report synthesis and the project walkthrough
(`docs/experiment-plan.md`, schedule).

**Milestone:** 24 Nov 2026 (optional RQ4). Schedule: step 1 on 17 Nov; step 2 on 17–18 Nov; step 3 on 18
Nov; step 4 on 19 Nov; step 5 on 20 Nov; 21–22 Nov buffer. Report synthesis and walkthrough follow on
25–26 Nov (`docs/experiment-plan.md`, schedule). **Cut in this order if behind:** the positive mock; the
ring sensitivity; the Δz profile. Never cut: A vs B with the H-eval reference and the pre-stated reading
(decision AD). E8 as a whole remains the first item to cut from the project (it then moves to the master
thesis as the phase-1 pilot).

**Working rules** (as in CLAUDE.md):
- Small reviewed steps, one at a time; the user does all git operations.
- Decision values are set only after the user agrees to that specific value.
- After each step, update `scripts/README.md`, `docs/data-dictionary.md` and
  `docs/module-reference.md` (and `docs/architecture.md`, since E8 adds a script and a stage package),
  and suggest entries for `docs/report-notes.md` and `docs/process-log.md`.
- **The analysis is fixed before S's MRIs are read.** Every choice is a config value in the section
  `paired:` (decisions AA–AE), set before `e8_paired.py paired` runs. That stage refuses while a value is
  `null` (`stages.decided`), unless E5's lock is unchanged (`e5.common.lock_unchanged`), and unless
  `rehearse` ran with the current `paired:`. No new hash lock: S is not re-scored. An analysis added
  after `paired` ran is labelled *exploratory* in report-notes and in its file names.
- **Rehearse first.** The analysis runs on the H-cal null and on the positive mock before it reads S, so
  no rule is tuned on S's MRIs.
- **S-derived masks and sources come only from E5's and E7's stored tables.** Masks as saved (specks
  included) are used for anything that relates a mask to its MRI, since stage 2 was conditioned on that
  version.

## Context

E5–E7 delivered everything E8 needs:

| Input | From | Used for |
| --- | --- | --- |
| `results/amibekk_v1/e7/reports/report_index.csv` (S rows: reason, group, key) | E7 `reports` | group A (AA) |
| `results/amibekk_v1/e5/analyze/flagged.csv` (per flagged S mask: `e1_nn`, `e2_source`, strata, `l1_area`, `near_exact`, `beyond_heldout_e2`) | E5 `analyze` | comparison slices of A (AA) |
| `processed_data/model_space/amibekk_v1/e5/`: `classes_main.csv` (novel masks, E1/E2 NN, `n_informative`), `e1_main.csv` (`shift_row`, `shift_col`, `runner_up_patient`) | E5 `score` | group B (AB), alignment (AC), runner-up slices (AD) |
| `processed_data/model_space/amibekk_v1/`: `qc.csv` (`mask_class`, `mri_failed`), `brain.csv` (`brain_px`), `labels.npz`, `masks_raw.npz`, `synthetic_index.csv` + pool files (`synthetic.load_sample`) | E0a, E0b | B pool, brain-size cells, lesion regions, synthetic MRIs |
| `processed_data/model_space/stanford_brainmets/`: `volumes/<pid>/bravo.nii.gz` (`io.load_real`), `labels.npz`, `brain_masks.npz`, `slices.csv` | E0a, E0b | T and held-out BRAVO, brain masks, lesion regions |
| `processed_data/model_space/stanford_brainmets/e4/`: `heval_classes.csv`, `e1_heval.csv`; `e3/e1_hcal.csv`, `e3/e2_hcal_*` | E3 `calibrate`, E4 `validate` | H-eval reference and H-cal null (AD, AE) |
| `processed_data/model_space/stanford_brainmets/e4/`: `planted.csv` (`base_key`, `shift_row`, `shift_col`), `planted_labels.npz`, `e1_planted.csv`, `e2_planted_*` | E4 `plant`, `validate` | positive mock (AE) |
| `configs/frozen/e3_stanford_brainmets.yaml`, `configs/frozen/e5_stanford_brainmets.yaml` | E3 `freeze`, E5 `prepare` | size bins, informative bound, held-out maxima |

Facts that shape the design (read-only queries on stored tables, 2026-11-16; no MRI statistic of S was
computed, so no E8 result was seen before its analysis was fixed):
- **Group A as E7 selected it:** 56 S masks (30 near-exact, 19 E2 beyond the held-out maximum, 4 + 3 top
  flags of E6's most exposed patients), one per chained group, from 41 source patients (at most 3 from
  one patient, `Mets_024`). E2 strata: single 34, multi 22 (1 or 2 informative lesions). Largest lesion
  16 / 29.5 / 62.5 px (10th / 50th / 90th pct). None has `mri_failed`. *Source: `report_index.csv`,
  `flagged.csv`, `qc.csv`.*
- **Brain-size mismatch.** An A synthetic brain covers a median 0.51 times its source slice's brain area
  (10th–90th pct 0.12–1.42; medians 2,927 vs 6,150 px). Synthetic brains are the size of real end-of-head
  slices (report-notes section 3). So whole-brain SSIM mostly compares brain outlines, and the lesion
  neighbourhood is the primary measure (AC). Report-notes section 4 asks every brain-based analysis to
  report its sensitivity to brain size (AD secondary). *Source: `amibekk_v1/brain.csv`,
  `stanford_brainmets/brain.csv`.*
- **The mask-resemblance confound.** When two masks overlap, both images have a bright lesion in the
  same place, so the lesion region looks alike without any memorization of tissue. A vs B alone cannot
  separate this from a "key" effect. H-eval measures it where no copying is possible: real held-out
  BRAVO against the BRAVO of its T comparison slice, 1,089 scorable slices, 148 flagged (E1 copy or E2
  full or partial), 775 novel by both detectors, 26 patients. Hence the reading against H-eval (AD).
  *Source: `e4/heval_classes.csv`.*
- **The B pool is large:** S novel masks (neither detector flags them) with an informative lesion and
  an MRI that did not fail. About 22,000 when main and noisy are counted together; the main-only count
  comes from `select`. 29 of them have C = 0 and use the E1 nearest neighbour as comparison slice.
  *Source: `e5/classes_main.csv`, `qc.csv`.*
- **E7's `source` column is the E2 source** (`report_data`: E2 NN for full, strongest-pair slice for
  partial). E8 instead takes the tier's decision-J source: E1 NN for near-exact and E1-copy masks, E2
  source for the E2 tiers (AA).
- **Stage 2 follows the mask channel**, even outside the brain (bright dots at specks; report-notes
  section 3). All synthetic MRIs started from one shared initial noise tensor, so MRI differences come
  from the mask and the per-step sampling noise, not from the starting point.
- **The planted copies store their shift** (`planted.csv` `shift_row`, `shift_col` for P2), so the
  positive mock can shift the BRAVO exactly as the mask was shifted (AE).
- **`planted.shift` cannot align images:** it refuses to move a non-zero pixel off the image, and MRIs
  can have tissue at the edge (synthetic brain boxes reach row 127). `paired.align` is a zero-filled
  shift with E1's convention (query pixel (r, c) goes to (r + dr, c + dc); pixels moved off are lost).

## Decision gates: decision → stage

Lettering continues after Z (E6 + E7: R, U–Z). The choice of group A, the held-out reading, the primary
measure and the rehearsal were made by the user on 2026-11-16, in the planning session. The remaining
values (B's matching rule and ratio, normalisation, alignment, SSIM settings, the 8 px region) were
proposed in the plan and approved with it. All were set before any code of E8 ran.

| | Decision | Evidence | Needed by |
| --- | --- | --- | --- |
| AA | Group A and its comparison slices | E7 selection (decision X); E5 decision J | E8 `select` |
| AB | Group B: pool, matching, ratio, comparison slice | the B-pool and brain-size facts above | E8 `select`, `rehearse` |
| AC | Normalisation, alignment, SSIM, regions; the primary measure | the brain-size mismatch above | E8 `rehearse`, `paired` |
| AD | Test, held-out reference, the reading; within-A controls | the mask-resemblance confound above | E8 `rehearse`, `paired` |
| AE | Rehearsal: null and positive mock | experiment plan E8 ("rehearse first") | E8 `rehearse` |

**Decision AA (set 2026-11-16): group A and comparison slices.**
- **A** = the S rows of E7's `report_index.csv` with reason near-exact, beyond held-out, exposed e1_copy
  multi or exposed e2_full multi: 56 masks, one per chained group (E5 decision N), already selected by a
  rule fixed before S was read (decision X). The 10 random S flags of E7 are not in A (they are typical
  flags, mostly chance).
- **Comparison slice t** = the tier's decision-J source: near-exact and exposed e1_copy → the E1 nearest
  neighbour (`flagged.csv` `e1_nn`); beyond held-out and exposed e2_full → `e2_source`.
- Rejected: the strong tiers only (49 masks; drops the most exposed patients, e.g. `Mets_230` and report
  50); adding the top flags of more exposed patients (a larger A of mostly placement-type matches, which
  E6 already reads as typical placement).

**Decision AB (set 2026-11-16): group B.**
- **Pool:** S main masks (E5 decision L) that neither detector flags (not E1 copy, E2 novel), with at
  least one informative lesion and `mri_failed` False.
- **Matching:** to each A mask, k = **5** B masks from the same cell, drawn with the seed, without
  replacement. Cell = E2 stratum and informative-lesion count (1, 2, ≥ 3) × size bin of the largest
  lesion (E5 bins: 12, 20, 40, 80 px) × tertile of synthetic brain area (`brain.csv` `brain_px`,
  tertiles of the B pool). An A mask without enough candidates gets what its cell has; such masks are
  reported in `select`'s `review.txt`.
- **Comparison slice** of a B mask: the E2 nearest neighbour (highest C), or the E1 nearest neighbour when
  C = 0 (as the experiment plan, step 2).
- **Held-out sets** (H-eval reference, H-cal null): flagged slices (E1 copy or E2 full or partial) take
  the place of A, with the decision-J source as t. Novel slices take the place of B, matched by the same
  rule (brain tertiles within the set, up to k per flag where the cell allows).
- Rejected: unmatched B (the size mix and brain sizes of A and B would differ); matching on the brain
  ratio between query and comparison slice (it depends on the comparison slice, chosen after matching).

**Decision AC (set 2026-11-16): measures.**
- **Brain:** `lesions.brain_mask` with the `brain:` config (E0b decision A), computed for every image
  (also for T slices without lesions, used in the Δz profile). `check` confirms it equals E0b's stored
  `brain_masks.npz` where those exist.
- **Normalisation:** each image is z-scored with the mean and SD of its own brain. The transform is
  applied to the whole image, so the background stays one constant value and brain edges are kept. Real
  BRAVO is already in model space (per-volume min–max to [0, 1], area resize: the training transform).
  Rejected: z-scoring over the union of both brains (it would mix background into the smaller brain's
  statistics).
- **Alignment:** the query image (synthetic MRI, held-out BRAVO, planted mock image) is moved by the E1
  shift (`shift_row`, `shift_col`) when t is the E1 nearest neighbour. Otherwise the shift is recomputed
  on the cleaned masks with `wholemask.shifted_dice` against t alone (±3 px, smallest move among ties).
  Moved with `paired.align` (zero-filled), as are the query's brain and lesion region.
- **SSIM:** `skimage.metrics.structural_similarity` with `gaussian_weights=True`, `sigma=1.5`,
  `use_sample_covariance=False`, `full=True` (the settings of Wang et al.) and a fixed `data_range`
  (z-units; value set in step 1 from T, no S). The SSIM map is averaged over a region:
  1. **Lesion neighbourhood (primary, chosen by the user):** the query's informative lesions (E0b label
     map, ≥ 12 px) grown by **8 px** (`lesions.grow`), inside both brains after alignment. The cleaned
     labels define the region; specks do not.
  2. **Whole brain (secondary):** inside both brains.
  3. **Ring (sensitivity):** region 1 without the lesion pixels of either mask (query as saved, t's
     mask): only the surrounding tissue counts, the strictest guard against the lesion-overlap effect.
  A pair whose region is empty after alignment is reported and left out (a minimum region size, if
  needed, is proposed in step 1 from T and set by the user).
- Rejected: a crop of 2× the lesion diameter (the plan's wording: for the median lesion, about 12 px
  across, barely larger than the SSIM window); whole-brain SSIM as primary (dominated by the brain-size
  mismatch).

**Decision AD (set 2026-11-16): test, held-out reference and the reading.**
- **AUC_S** = P(an A score > a B score) on the primary measure (`audit.auc`). Its 95% interval comes
  from the two-level bootstrap (`audit.auc_bootstrap`: A resampled by source patient, B by the patient of
  its comparison slice). **AUC_H** is the same for H-eval flagged vs novel, both resampled by patient.
  **Δ = AUC_S − AUC_H** comes from independent draws, `calibration.bootstrap_draws` each. The one-sided
  Mann–Whitney p (`scipy.stats.mannwhitneyu`, alternative "greater") is reported as the experiment plan
  asks, with the caveat that it ignores clustering.
- **Reading** (written mechanically to `review.txt`):

  | Reading | Rule |
  | --- | --- |
  | *supported (observational)* | the AUC_S interval lies above 0.5 **and** the Δ interval lies above 0 |
  | *explained by mask resemblance* | the AUC_S interval lies above 0.5 and the Δ interval covers 0 or lies below it |
  | *not supported* | the AUC_S interval covers 0.5 or lies below it |

  Even *supported* means no more than "observational": both stages may have learned the same patient
  independently, and common configurations may give similar tissue. Only RQ-M1 (master thesis) can
  separate these.
- **Within A** (experiment plan step 5): (i) SSIM to t vs SSIM to the **runner-up patient's**
  best-matching slice. Runner-up patient = E1 `runner_up_patient`; slice = its best E1 match among that
  patient's T slices (`wholemask.match`), aligned the same way. Its mask is nearly as close, so this
  controls the lesion-overlap effect within the same synthetic image. Paired Wilcoxon on the primary
  measure. (ii) **The Δz profile:** SSIM to the slices of t's patient at z ± 1 … ± 5, aligned with t's
  shift, mean with a bootstrap interval per Δz (descriptive). A peak at Δz = 0 suggests slice-specific
  tissue; a flat profile suggests patient-level or no resemblance.
- **Secondary:** the whole-brain and ring measures with the same AUCs; AUC_S per tier (near-exact,
  beyond held-out, exposed); AUC_S per half of synthetic brain area (the brain-size sensitivity
  report-notes asks for).
- Rejected: A vs B without a held-out reference (the experiment plan's test; it cannot separate a "key"
  effect from mask resemblance); a bootstrap that resamples S masks iid (A has several masks from some
  source patients).

**Decision AE (set 2026-11-16): rehearsal.** `rehearse` runs the analysis function that `paired` will
run on S, unchanged, on two mock sets:
- **null:** H-cal flagged vs matched H-cal novel slices, real BRAVO against the BRAVO of its T comparison
  slice, judged against the H-eval reference. No slice can carry memorized tissue. **Must not read
  *supported*.**
- **positive:** A = planted P1 and P2 masks (E4, gated, flagged by E2 with the base patient as
  decision-J source). Their "MRI" is the base patient's BRAVO at z + 2 (z − 2 if z + 2 lies outside the
  volume): same tissue as the base slice, not identical. For P2 it is shifted by the planted shift, so
  the alignment and its sign are tested on data. B = matched H-cal novel slices. **Must read
  *supported*.**
- The user reviews `results/stanford_brainmets/e8/rehearse/` before `paired` runs. Changes are still
  allowed then, since S's MRIs are unread.
- Rejected: a null mock only (no known positive case, so an analysis that can never read *supported*
  would pass); no rehearsal (against the project's rule to rehearse where a known answer exists).

## Definitions (fixed in code and docs)

- **Sets:** S main (E5 decision L); T; H-eval (the reference); H-cal (the null mock); planted P1/P2 (the
  positive mock).
- **Group A, comparison slice t:** decision AA. **Group B, cell:** decision AB.
- **Query image:** the image that is aligned onto t: synthetic MRI (S), real BRAVO (held-out), z ± 2
  BRAVO (positive mock).
- **Lesion neighbourhood, whole brain, ring:** decision AC. **AUC_S, AUC_H, Δ, the reading, runner-up
  slice, Δz profile:** decision AD.

## Library modules (building blocks, no knowledge of stages)

- **`memtk/paired.py` (new, step 1):** `zscore_in_brain`, `align` (zero-filled shift, E1's convention),
  `comparison_shift` (one candidate: `wholemask.shifted_dice`, `shift_order`), `lesion_region`
  (`lesions.grow`, clipped to both brains), `region_ssim` (SSIM map averaged over a region),
  `match_controls` (seeded, per cell, without replacement), `runner_up_slice`.
- **Reused unchanged:** `audit` (`auc`, `auc_bootstrap`, `interval`, `sources`); `wholemask` (`match`,
  `shifted_dice`, `shift_order`); `lesions` (`brain_mask`, `grow`); `io` (`read_table`, `load_real`,
  `slice_images`, `LabelStack`); `synthetic.load_sample`; `sets` (`load_set`, `load_pool`,
  `load_planted`); `calibration` (`load_frozen`); `plotting` (`draw_labels(filled=False)`, `zoom_window`,
  `draw_box`, `save`); `checks` (`CheckReport`, `check_run_logs`, `differing_tables`); `stages`
  (`add_stage`, `decided`, `refuse_overwrite`, `require`, `pick`); `e5.common.lock_unchanged`.

## Stages

### `scripts/e8_paired.py` → `src/memtk/e8/`

One module per stage (`select`, `rehearse`, `paired`, `check`) plus `common` (the `paired:` config,
loading the inputs, the analysis function shared by `rehearse` and `paired`).

1. **`select`** (step 2; needs AA, AB; reads no MRI).
   - Group A and t (AA); group B and its comparison slices (AB); the alignment shift of every pair (AC);
     the runner-up slice of every A mask (AD).
   - `results/amibekk_v1/e8/select/pairs.csv` (set, group, key, cell, comparison slice, shift, runner-up
     slice and its shift); `review.txt` (counts per cell, A masks with fewer than k matches, the B pool
     size).
2. **`rehearse`** (step 3; needs AB–AE; reads no S MRI).
   - The H-eval reference (stored for `paired`), the H-cal null and the positive mock, each through the
     shared analysis function.
   - `results/stanford_brainmets/e8/rehearse/{heval,null,positive}/` (`pairs.csv`, `measures.csv`,
     `tests.csv`, the figures below, `review.txt` with the reading). The user reviews.
3. **`paired`** (step 4; reads S's MRIs, one run; needs every `paired:` value).
   - Measures for A, B and the within-A controls; tests and the reading against the stored H-eval
     reference.
   - `results/amibekk_v1/e8/paired/`: `measures.csv` (one row per image pair: set, group, query,
     comparison slice, kind (t, runner-up, Δz), shift, region sizes, SSIM per region, brain areas),
     `tests.csv` (AUCs with intervals, Δ, Mann–Whitney p, per measure and secondary split),
     `within_a.csv`, the figures, `review.txt` ending with the reading.
4. **`check`** (step 5). Exit code 1 on any failure: every `paired:` value set; run logs in order
   (`select`, `rehearse`, `paired`), with S's MRIs read by `paired` only; E5's lock unchanged; the brain
   masks equal E0b's where stored; the rehearsal readings as required (null not *supported*, positive
   *supported*); every table reproduced with the seed (`checks.differing_tables`); every figure present;
   pytest.

## Figures (PNG, drawn with `memtk.plotting`)

- **`ssim_groups.png`:** the primary measure, A vs B next to H-eval flagged vs novel (points and box),
  with AUC_S, AUC_H and Δ and their intervals; the same panel for the whole-brain and ring measures.
- **`within_a.png`:** SSIM to t vs to the runner-up slice (paired lines); the Δz profile.
- **`examples.png`:** 3–4 A cases (one per tier where possible, including report 50): synthetic MRI,
  aligned t BRAVO, SSIM map, lesion neighbourhood outlined (`draw_labels(filled=False)`), with the E7
  report number. One may go in the written report.
- `rehearse` writes the same figures for each mock set.

## Tests (pytest, small hand-built inputs)

- `tests/test_paired.py`: identical images give SSIM 1 in any region; SSIM is unchanged under a·x + b
  after z-scoring; an image and its mask shifted by (dr, dc) are realigned to SSIM 1 by
  `comparison_shift` + `align` (shift sign); `align` loses pixels moved off the image instead of
  wrapping; the lesion region grows by exactly 8 px and is clipped to both brains; `match_controls`
  respects cells and k, is reproducible with the seed, and never reuses a B mask.
- `tests/test_e8_paired.py` (added in step 3): the three branches of the reading; AUC against a
  reference set and their difference; a query that is the comparison image moved by a shift measures SSIM 1
  after alignment, and the ring leaves out the lesion pixels; the within-A runner-up difference and z-offset
  profile.

## Verification (every step)

- Each stage prints its summary, writes a run log and refuses to overwrite without `--overwrite`;
  `paired` refuses while a decision value is `null`, if E5's lock changed, or if `rehearse` ran with
  another `paired:`.
- pytest passes; E5 `check` stays 26/26 (nothing in E5 changes).
- The user reviews `select` and `rehearse` before `paired` reads S's MRIs.
- After `paired`: `audit:`, the frozen files and `exposure:` / `reports:` unchanged; `e8_paired.py check`
  exits 0.

## Carried forward (2026-11-16)

| Item (first noted) | Moved to |
| --- | --- |
| E8 result: reading *not supported*; within A, source slice no closer than another patient's slice with a similar mask; the Δz profile (descriptive) peaks next to the source in the lesion region | report-notes section 3; `docs/experiment-plan.md`, E8 result |
| Limitations: brain-size mismatch (synthetic brains about half the comparison slices' area) and the mask-resemblance confound, handled by H-eval; observational only | report-notes section 4 (written 2026-11-16) |
| Limitations to add (suggested at the close): the positive mock shows correctness, not sensitivity to faint resemblance; the Δz profile has no held-out reference; some held-out cases had no matched control (H-eval 25 of 131; the rule does not drive the reference) | report-notes section 4 (written 2026-11-16) |
| The mask as key, by intervention: fixed noise seeds, training vs held-out vs synthetic masks, perturbation series (RQ-M1); a graded positive mock (e.g. BRAVO at z ± 1, 2, 5, 10) would show how sensitive the measure is | master thesis (RQ-M1); report-notes section 5 (written 2026-11-16) |
| Report synthesis and project walkthrough | `docs/experiment-plan.md`, schedule (25–26 Nov) |

Done before closing: decisions AA–AE set before the stages that use them (`data_range` set from T, no
S), the rehearsal as required before `paired` read S's MRIs in one run, the check passes, and the docs
(data dictionary, module reference, scripts README, architecture) describe every stage and file.
