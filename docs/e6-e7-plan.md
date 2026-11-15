# E6 + E7 plan: patient exposure (`scripts/e6_exposure.py`) and match reports (`scripts/e7_reports.py`)

Working plan for E6 and E7, kept in the repo so work can resume from here. The experiments are
specified in `docs/experiment-plan.md` (E6 and E7, including their "Carried over" notes); this file
records how they are implemented, stage by stage, and what has been decided. Update the status table
when a step is done. E6 and E7 read E5's stored tables of S and never score S again.

E6 asks which training patients are exposed and why, and answers the question E5 left open: is the
multi-lesion excess **copying of patients** or a generator that places lesions **where training lesions
typically are**? E7 explains flags in match reports and evaluates the explanations (faithfulness by
deletion, correctness on planted copies with known source). One plan for both: they share the frozen
E5 pair tables, the per-patient counting, the 3D lesion identity and the deadline, and E7's Panel C
shows E6's exposure ratio. The E7 work on planted copies needs neither S nor E6 and runs in parallel
to E6; only the E7 reports on S wait for E6.

## Status (2026-11-15)

| # | Step | Kind | Status |
| --- | --- | --- | --- |
| 0 | This plan; decisions R, U–Z recorded in the docs | docs | ✅ 2026-11-15 (decisions R, U–Z set by the user in the planning session) |
| 1 | Shared E5 helpers to the library; `memtk/lesions3d.py` + tests; config sections `exposure:` and `reports:` with the values of R, U–Z | refactor + library + config | ✅ 2026-11-15: `Scored`, `load_scored`, `heldout`, `sources`, `chained` (`e5.analyze`) and `query_table` (`e5.score`, needed by `heldout`) → `audit`, moved unchanged; E5 check 26/26, pytest 151 after the move. `memtk/lesions3d.py` (`link_slices`, `components_26`, `lesion3d_table`, `map_2d_to_3d`) and `ids.lesion3d_id`; 6 tests (pytest 157). Read-only check on T: 26-connectivity reproduces the recorded 1,428 3D lesions with a real-sized lesion (plus 18 made only of 1–3 px components) and the 38 merged ones; linking gives 1,565 chains, none holding two pieces in one slice (17 s for all 105 volumes). Config `exposure:`, `reports:` |
| 2 | `memtk/exposure.py` + tests; E6 `reference` (no S): 3D lesions, rarity, π_p, held-out margins | library + production | ✅ 2026-11-15: `memtk/exposure.py` (11 tests; pytest 168), `e6.common`, `e6.reference`, `scripts/e6_exposure.py`. `reference` (~3 min; re-run after a Pylance rewrite: byte-identical tables): T 3,804 slices (19 with only 1–3 px components left out, as in E1–E5); 3D lesions: linking 1,565 (1,018 informative), 26-connectivity 1,446 (940; 38 joined), 52 26-connected lesions split by linking (705 2D lesions); informative span median 4 slices (quartiles 3 / 6, max 43) under both; slice rarity quartiles E1 0.16 / 0.26 / 0.39, E2 0.40 / 0.53 / 0.68 (10 T slices have an identical whole mask in another patient); H-eval NN coverage: E1 multi 82 of 105 patients (393 masks), E2 multi 69 (272). As built: the H-eval margins and chance-flag sources are computed in the shared analysis function (step 3), not stored by `reference`, so mock sets and S go through the same code. Found while testing: E2 centres lesion crops on `np.rint` of the centroid (halves round to even), so an even-width lesion shifted by 1 px can lose shape overlap (4×4 block: 0.6); on T, 3.9% of informative lesions are affected by a 1 px shift (8% at 12–19 px), shape factor median 0.57 for those: a detection limit of the frozen detector, not changed |
| 3 | E6 `rehearse` (no S): the exposure analysis on E5's null and positive mock sets; the user reviews | rehearsal | ✅ 2026-11-15 (user review of `results/stanford_brainmets/e6/rehearse/{null,positive}/` pending). The analysis is one function (`e6.exposure.analyze_query` + `draw_figures`) that `exposure` will run on S unchanged; `e6.common` loads a scored set (`load_query`, `load_heval`) and the reference (`load_reference`); `exposure_table` now resamples flags by their clusters (mock sets are clustered by patient); 3 tests (pytest 171). **Null** (H-cal, 989 slices): reading *undetermined* (not *patient copying*): margin AUC 0.45 [0.22, 0.66] (E1 copy multi) and 0.53 [0.16, 0.80] (E2 full multi), source-rarity AUC intervals cover 0.5, H3 ρ 0.15 / 0.02 with intervals covering 0, H1 weighted difference −0.03 [−0.09, 0.05]; the E1 copy multi dispersion is beyond its null (893 vs 95th pct 255) because H-cal's flags come in adjacent-slice runs of one patient: the margin rule is what keeps it from reading as copying. **Positive** (H-cal + 294 planted): reading *patient copying* in both primary cells: margin AUC 0.88 [0.81, 0.94] and 0.95 [0.88, 0.99], dispersion far beyond null; every planted flag names its base patient; 7–9 base patients per cell have a ratio interval above 1 and no other patient does (5–9 copies per patient are visible only for patients with a modest expected share). Seen on the way: the plain Spearman(span, credits) of the plan is positive in both mocks (0.10, 0.23) without any duplication effect, which is why H1 is judged against H-eval (decision W). Not exercised by a mock: the *typical placement* branch (unit test only) |
| 3b | Placement mock (added 2026-11-15 at the user's request): H-cal masks with every lesion moved to the position of a random T lesion (variant *placement*) or of a random T lesion inside S's central area (variant *placement_compact*, box from E0b); scored with the frozen detectors and analysed by E6 (and, exploratory, by the E5 rates) inside `rehearse`. Tests the untested *typical placement* branch of decision U and whether placement alone produces an E5-style multi-lesion excess. Also: the E6 figures redesigned so the null and positive cases differ visibly (observed vs expected flags per patient; flags by rarity group; margins of query and H-eval flags side by side) | rehearsal | ✅ 2026-11-15 (user review pending; one decision open, below). `planted.relocate_lesions` (+1 test, pytest 172); config `exposure.placement_mock`; `rehearse` builds, scores and analyses both mocks (989 masks each, none left out; positions from 6,101 T lesions, compact: 2,401). **Neither placement mock reads "patient copying"** (the flaw that would have mattered most is ruled out): *placement* E1 copy multi margin AUC 0.43 [0.24, 0.58], E2 full multi 0.42 [0.12, 0.66]; *placement_compact* 0.33 [0.19, 0.49] and 0.45 [0.12, 0.78]. **But both read "undetermined", not "typical placement"**: the source-rarity criterion stays inside its interval (compact E1 multi 0.41 [0.29, 0.54]), although the compact mock's E1 multi margins are clearly *smaller* than chance. Exploratory: the compact mock gives an E5-style excess for *single*-lesion masks (E1 copy +5.4 points [+1.9, +9.4], E2 full +5.8 [+1.6, +10.0]) and not a clear one for multi-lesion masks (E1 +1.2 [−3.4, +6.9]); S shows the opposite pattern (multi only). Figures redesigned (`flag_details.csv` now holds H-eval's flags too). **Decision U amended by the user** (margins wholly below chance also read "typical placement"; see decision U): rehearse re-run, readings null undetermined, positive patient copying, placement undetermined, placement_compact mixed (E1 multi typical placement) |
| 4 | `memtk/reports.py` + tests; E7 `planted` (no S): reports, correctness, deletion test on planted copies | library + production | ✅ 2026-11-15 (user review of `results/stanford_brainmets/e7/planted/` pending): `memtk/reports.py` (8 tests; pytest 180), `e7.common`, `e7.planted`, `scripts/e7_reports.py`; ~30 s. 698 planted masks with an informative lesion, 679 gated, 597 of them flagged by E2. **Correctness criteria PASS** (decision Z): source patient P1, P2, P4 100%, P6-add 99.0%, P6-remove 91.9% (min 90%); lesions P6-add 99.0%, P6-remove 81.1% (min 80%); reported only: P3 95.8%, P5-dilate 88.1%, P5-erode 96.7% (slice within ±2: 62% / 60%, the eroded lesion fits a neighbouring slice better, as in E4). **Deletion test** (decision Y), 106 planted multi-lesion controls: C drops 0.43 (top) vs 0.41 (smallest), the whole-mask score 0.33 vs 0.15 (Wilcoxon p < 0.001), flips 23 vs 6 (McNemar p 0.001): faithful by both added measures; in exact copies every pair has s ≈ 1, so the "top" contributor is a tie broken by lesion number (L1 = largest), and ΔC hardly separates them. H-eval chance flags: only 7 multi-lesion masks qualify (ΔC 0.21 vs 0.08). **Taxonomy** (rules fixed here): P1 exact 100/100, P2 shifted 100/100, P3 shifted 91/96, P4 exact 51 / deformed 36 (a 5° rotation leaves many small lesions pixel-identical), P5 deformed, P6-add partial 99/100, P6-remove partial 30/37. Changed on the planted results (before S): "partial" = a lesion without a partner (first: a pair below tau_L, which called 5 px shifts partial); verdict wording ("configuration copy ... not every lesion" when E2 says full but not every pair passes tau_L; "without a partner (novel)" vs "paired but not closer than chance"); report layout. To do for S reports: colour lesions by pair (now each slice's own numbering) |
| 5 | E6 `exposure` (S tables): exposure, concentration, margins, rarity, H3, H1, the reading | analysis | ⬜ |
| 6 | E7 `reports` (S tables): selected reports, deletion test on S, taxonomy | analysis | ⬜ |
| 7 | E6 and E7 `check`; close the plan; carry items forward (E8 plan) | consistency checks + docs | ⬜ |

**Next: start here.** The user reviews E7 `planted` (`results/stanford_brainmets/e7/planted/`). Then
step 5: the `exposure` stage (`analyze_query` on E5's S tables in
`processed_data/model_space/amibekk_v1/e5/`, set main; the first time E6 reads S). After E8: report
synthesis and project walkthrough (`docs/experiment-plan.md`, schedule).

**Milestone:** 24 Nov 2026, results for RQ3a, H1, H3 and RQ3b. Schedule: step 1 on 16 Nov; step 2 on
17 Nov; step 3 and the start of step 4 on 18 Nov; step 5 and the rest of step 4 on 19 Nov; step 6 on
20–21 Nov; step 7 on 22 Nov; 23–24 Nov buffer. **Cut in this order if behind:** the E7 taxonomy; the
lesion-level part of the correctness check; the 26-connectivity sensitivity of H1; the H1 bin
analysis (keep the plain Spearman). Never cut: the exposure reading (decision U), the deletion test,
source-patient correctness.

**Working rules** (as in CLAUDE.md):
- Small reviewed steps, one at a time; the user does all git operations.
- Decision values are set only after the user agrees to that specific value.
- After each step, update `scripts/README.md`, `docs/data-dictionary.md` and
  `docs/module-reference.md` (and `docs/architecture.md`, since E6 and E7 add two scripts and two stage
  packages), and suggest entries for `docs/report-notes.md` and `docs/process-log.md`.
- **The analysis is fixed before S is read.** Every choice of E6 and E7 is a config value in the
  sections `exposure:` (decisions R–W) and `reports:` (X–Z), set before `e6_exposure.py exposure` and
  `e7_reports.py reports` run; those stages refuse while a value is `null` (`stages.decided`). No hash
  lock: S is not re-scored, and E5's tables are already locked by `score.complete.json`. An analysis
  added after an S-reading stage ran is labelled *exploratory* in report-notes and in its file names.
- **Rehearse first.** E6 runs on E5's mock sets and E7 on the planted copies before either touches S,
  so no rule is tuned on S.
- **S-derived numbers come only from E5's stored tables.** E6 and E7 never re-score S and never change
  `audit:` or the frozen files (`check` confirms the E5 hashes). New detector runs (T against T for
  rarity, edited masks for the deletion test) use `calibration.frozen_settings`.

## Context

E0–E5 delivered everything E6 and E7 need:

| Input | From | Used for |
| --- | --- | --- |
| `results/amibekk_v1/e5/analyze/flagged.csv` (per flagged S mask: classes, scores, source slice and patient per detector, chained group, near-exact and beyond-held-out flags, copied-area share) | E5 `analyze` | exposure (R), margins and rarity (U, V), report selection (X) |
| `processed_data/model_space/amibekk_v1/e5/`: `classes_main.csv`, `e1_main.csv` (with `margin`, `runner_up_patient`), `e2_main_*` (every candidate slice and lesion pair) | E5 `score` | E2 margins, H1 lesion credits (W), report panels (X) |
| `processed_data/model_space/stanford_brainmets/e4/`: `heval_classes.csv`, `e1_heval.csv`, `e2_heval_*` | E4 `validate` | π_p (R), chance-flag baselines (U, V, W) |
| `results/stanford_brainmets/e4/validate/fp_heval.csv` | E4 `validate` | chance rate per class and stratum in each report (X) |
| `processed_data/model_space/stanford_brainmets/e5/rehearse/{null,positive}/` and `results/stanford_brainmets/e5/rehearse/{null,positive}/flagged.csv` | E5 `rehearse` | E6 `rehearse` (null and positive mock S) |
| `planted.csv`, `planted_lesions.csv`, `planted_labels.npz`, `e2_planted_*`, `e1_planted.csv`; `results/stanford_brainmets/e4/validate/planted_outcomes.csv` | E4 `plant`, `validate` | E7 correctness and deletion control (Y, Z) |
| `configs/frozen/e3_stanford_brainmets.yaml`, `configs/frozen/e5_stanford_brainmets.yaml` | E3 `freeze`, E5 `prepare` | thresholds, detector settings, held-out maxima |
| Real `labels.npz`, `lesions.csv` (brain-relative centroids), `slices.csv`, model-space `volumes/<pid>/seg.nii.gz` | E0b `lesions`, E0a `real` | 3D lesions (W), rarity (V), patient features |
| Pool `labels.npz`, `masks_raw.npz`, `synthetic_index.csv` | E0b, E0a | report panels; deletion test on S label maps |

Facts that shape the design (read-only queries on T and held-out tables, 2026-11-15; no statistic of
S was computed, so no E6 result was seen before its analysis was fixed):
- **The placement confound.** Synthetic lesions sit in a compact central pixel area (E0b, report-notes
  section 3, "Synthetic lesion positions"). Patients whose lesions are central would attract S matches
  without any copying, and π_p from H-eval does not reflect S's placement. The exposure ratio alone
  therefore cannot separate copying from placement; decision U adds two comparisons against held-out
  *chance flags* (margin and source rarity), which react differently to the two.
- **π_p is sparse.** Multi-lesion E1 nearest-neighbour assignments: H-eval 393 masks from 26 patients
  (H-cal 303); 94 of the 105 T patients appear as NN in H-cal ∪ H-eval (max share 5.2%). Only 62 T
  patients have multi-lesion slices (1,220 of 3,823), but the E1 NN of a multi-lesion query may be a
  single-lesion slice. Hence smoothing (decision R). *Source: `e3/e1_hcal.csv`, `e4/heval_classes.csv`,
  real `slices.csv`.*
- **H-eval chance flags, multi-lesion:** E1 copy 29, E2 full 8, E2 flagged 23 (`heval_classes.csv`).
  The E2 full multi baseline for margin and rarity rests on 8 flags: its intervals will be wide;
  stated, not fixed by pooling (H-cal stays the independent null of `rehearse`).
- **Margins:** the E1 margin (score minus the best other patient) is already stored in every E1 table
  (`margin`, `runner_up_patient`: S `e1_main.csv`, `e3/e1_hcal.csv`, `e4/e1_heval.csv`); the E2 margin
  is computed from the stored slice pairs (`LesionMatch.slice_scores`).
- **The deletion test as written in the experiment plan is nearly tautological.** C sums the paired
  lesion scores, so removing the lesion with the largest s lowers C (against the same slice) most by
  construction. Decision Y adds two measures that do not follow from the definition.
- **H1 step 5 (the sampling filter) is moot.** No T slice reaches 2,000 lesion px (max 873; real
  `slices.csv` `lesion_px`), and the "< 40 px" rule does not hold in the pool (E0b decision E). No T
  slice is filtered; the report states why.
- **E5's mock sets are stored in S's format** (class tables, E1 and E2 tables, `flagged.csv`), so the
  E6 analysis can be rehearsed on a null (H-cal) and a positive case (H-cal + planted P1, P2, P4 of 20
  base patients) before it reads S.
- **Planted set:** 737 masks (P1–P5 100 each, P6-add 100, P6-remove 37; 296 from multi-lesion bases);
  E2 catches P1 100, P2 100, P4 95, P6-add 99, P6-remove 34 (`planted_outcomes.csv`).

## Decision gates: decision → stage

Lettering continues from E5 (L–Q); S and T are skipped because they name data sets. All six were set by
the user on 2026-11-15, in the planning session, before any code of E6 or E7 ran.

| | Decision | Evidence | Needed by |
| --- | --- | --- | --- |
| R | Exposure counting and π_p | the π_p sparsity above; E5 decision J and N | E6 `reference`, `rehearse`, `exposure` |
| U | Copying vs placement: margin, source rarity, the reading | the placement confound above; E5 exploratory findings | E6 `rehearse`, `exposure` |
| V | Rarity per slice and per patient; H3 | experiment plan E6 steps 3–5 | E6 `reference`, `exposure` |
| W | H1: 3D lesion identity and lesion credits | report-notes section 3 (3D merging) | E6 `reference`, `exposure`; E7 Panel C |
| X | Which S flags get reports; NN vs strongest pair | E4 (4 P6 masks), E5 (C vs best pair diverge) | E7 `planted`, `reports` |
| Y | Deletion test | the tautology above | E7 `planted`, `reports` |
| Z | Correctness on planted copies; taxonomy | E4 planted set; thesis-plan criteria | E7 `planted`, `reports` |

**Decision R (set 2026-11-15): exposure counting.**
- **Flags:** set main; classes E1 copy, E2 full, E2 flagged; strata single and multi. **Primary cells:
  E1 copy multi and E2 full multi** (the two cells beyond chance in E5); every other cell secondary.
- **Source:** one source patient per flag, decision J (`sources`: E1 copy → E1 NN; E2 full → E2 NN;
  E2 partial → strongest-pair slice). Observed o_p = flags of the cell with source patient p.
- **Expected share π_p:** the share of **H-eval** scorable masks of the same stratum whose nearest
  neighbour (the detector's own: E1 NN for E1 copy, E2 NN for the E2 classes) belongs to p, smoothed
  toward p's share w_p of T lesion slices: π_p = (n_p + m·w_p) / (N + m), **m = 52.5** (half a
  pseudo-mask per patient). Sensitivity: m = 105.
- **Exposure ratio** r_p = o_p / (π_p · O), O = Σ o_p, with a 95% interval: S flags resampled iid, H-eval
  by patient, independently (`audit.resample_sums`, `audit.interval`).
- **Concentration:** the top-5 share and the dispersion D = Σ (o_p − e_p)² / e_p, against a null of O
  flags drawn multinomially from π̃_p, with π̃ redrawn from an H-eval patient bootstrap in every draw
  (2,000 draws, seeded); also the number of patients whose ratio interval lies above 1, against its
  null expectation.
- **Unit:** S masks (each an independent sample); chained groups per patient (decision N) reported
  alongside.
- **Deviation from E6 step 1** ("use every stored slice pair above threshold"): the ratio uses one NN
  source per flag, because π_p is defined by nearest neighbours; crediting every pair in the numerator
  only would inflate patients with many near-duplicate slices. Every-pair crediting is used where
  adjacency matters: H1 (decision W).
- Rejected: π_p from H-cal ∪ H-eval pooled (twice the data, but the rehearsal null case, H-cal, would
  then share π_p: circular); unsmoothed π_p (patients without a held-out NN get an infinite ratio).

**Decision U (set 2026-11-15): copying vs placement, with a pre-stated reading.**
- **Margin** per flag: E1 the stored `margin`; E2 = C(q, NN) − the highest C(q, t) over slices t of
  other patients. S flags against H-eval chance flags of the same class and stratum: AUC with the
  two-level bootstrap (`audit.auc_bootstrap`; S iid, H-eval by patient). Copying predicts a source that
  is uniquely close: AUC > 0.5. Placement predicts many patients nearly as close: AUC ≤ 0.5.
- **Source rarity** (decision V) per flag: S flags against H-eval chance flags, AUC as above. Copying
  of rare configurations predicts > 0.5; placement predicts sources more typical than chance picks,
  < 0.5.
- **Patient level:** the H3 Spearman (decision V): placement predicts ρ < 0 (typical patients exposed),
  long-tail memorization ρ > 0.
- **Reading** (primary cells, written mechanically to `review.txt`):

  | Reading | Rule |
  | --- | --- |
  | *patient copying* | dispersion beyond the 95th percentile of its null **and** the margin AUC interval above 0.5 |
  | *typical placement* | the margin AUC interval lies wholly below 0.5 (added 2026-11-15, see below), **or** it covers or lies below 0.5 **and** the source-rarity AUC interval lies below 0.5 |
  | *undetermined* | anything else; stated as such, with the pattern seen |

  The positive mock (planted copies) must read *patient copying* in `rehearse`; the null mock must not.
- Rejected: the exposure ratio and H3 only (the plan's procedure): under the placement confound a ratio
  above 1 does not distinguish the two readings.
- **Amended 2026-11-15 by the user, after the placement mock (step 3b) and before E6 read S:** margins
  wholly *smaller* than chance also read as *typical placement*. The compact placement mock had E1
  multi-lesion margins clearly below chance (AUC 0.33 [0.19, 0.49]) while its source rarity stayed
  near 0.5, so the first rule read it "undetermined". Flags that fit their best patient less uniquely
  than chance flags are a direct sign of generic placement. With the amendment the mocks read: null
  undetermined, positive patient copying, placement undetermined, placement_compact mixed (E1 multi
  typical placement, E2 full multi undetermined). Rejected: keeping the first rule (it recognised
  placement only through rarity, which the mocks showed to be weak).

**Decision V (set 2026-11-15): rarity and H3.**
- **Slice rarity**, on each detector's own scale: E1 = 1 − the best shifted Dice against T slices of
  *other* patients; E2 = 1 − the highest C against other patients' slices (NaN for a slice without an
  informative lesion). Computed per patient (T_p against T without p) with `wholemask.match` and
  `lesionmatch.match`, settings from `calibration.frozen_settings`.
- **Patient rarity:** features = number of 3D lesions (decision W), total lesion volume (model-space
  px), median 2D lesion area, most lesions on one slice, spread of brain-relative lesion centroids
  (√(var_row + var_col); brain-relative because this is an anatomical question, decision F). Counts and
  areas log1p-transformed, all z-scored; rarity = mean Euclidean distance to the 5 nearest other
  patients.
- **H3, slice level:** AUC of source-slice rarity, sources of S flags against all other T slices (the
  experiment plan) and against the sources of H-eval chance flags (decision U).
- **H3, patient level:** Spearman(rarity_p, r_p), p from 10,000 permutations, 95% interval from a
  bootstrap over patients (with S flags resampled). Effect sizes first: 105 patients give limited
  power.

**Decision W (set 2026-11-15): H1 with slice-to-slice 3D lesion identity.**
- **3D lesions:** 8-connected components of every slice of the model-space mask **as saved**
  (`seg.nii.gz`, all sizes, so the thin end slices of a metastasis count), linked between consecutive
  slices one-to-one by Hungarian assignment on pixel IoU (IoU > 0 required). A chain is a 3D lesion; a
  merge or split keeps separate identities. Each E0b 2D lesion maps to the chain holding most of its
  pixels; the 2D lesion IDs stay unchanged (column `lesion3d_id`).
- **Sensitivity:** 26-connected 3D labelling; report where the two differ (38 merged 3D lesions
  holding 10.3% of the 2D lesions expected, report-notes section 3).
- **Span:** the number of slices in the chain.
- **Lesion credits:** c_S(ℓ) = S main masks with at least one lesion pair s ≥ τ_L (its stratum) to any
  2D instance of ℓ, over **every stored slice pair** (E6 step 1); each mask counts once per 3D lesion.
  c_H(ℓ) the same for H-eval (chance). Only 3D lesions with an informative 2D instance enter.
- **Test:** within size bins of ℓ's largest 2D area (12–19, 20–39, 40–79, ≥ 80 px), ρ_S =
  Spearman(span, c_S) and ρ_H likewise. **H1 holds if the bin-weighted mean of ρ_S − ρ_H has a 95%
  interval above 0** (S iid, H-eval by patient). The plain ρ_S (the experiment plan) is reported too;
  without the held-out reference it would also rise by chance, since a long lesion offers more slices
  to match.
- **Step 5 (sampling filter) not applied:** moot (facts above).
- Rejected: 26-connectivity as the main definition with the merged lesions excluded as a sensitivity
  (the exclusion removes exactly the long lesions H1 is about).

**Decision X (set 2026-11-15): which S flags get reports; NN vs strongest pair.**
- **Selection**, one report per chained group, rule fixed before E6 `exposure` runs:
  1. every near-exact mask and every E2 mask beyond the held-out maximum (E5 decision Q);
  2. the top flag (highest score) per detector of the 5 most over-exposed patients (by the lower bound
     of the ratio interval, primary cells);
  3. a seeded random 10 S multi-lesion flags and 10 H-eval chance flags, as a contrast (what a typical
     flag and a chance flag look like).

  About 50–60 reports. Every report states the H-eval chance rate of its class and stratum; the report
  text says that most individual flags are chance matches.
- **Panels** (experiment plan E7 step 1):
  - *A:* the synthetic mask with its E1 NN and, when different, its E2 NN; overlay coloured by overlap,
    synthetic-only and training-only.
  - *B:* the paired lesions with their shape, position and size factors and contribution to C; the
    strongest-pair slice when it differs from the NN; the copied-area share.
  - *C:* source pseudonym and z, the slices of its chained group and of the same 3D lesions (W), the
    margin to the second-best patient (U), the exposure ratio (R), the chance rate
    (`fp_heval.csv`), whether the score is beyond the held-out maximum (Q).
- **Verdict line**, from rules: names the decision-J source; adds the strongest-pair patient when it
  differs ("the strongest single lesion points to P0xx"); says "configuration match", not "copy", when
  no lesion pair reaches τ_L (E5: C and the best pair diverge for multi-lesion masks).
- **Correctness** scores the decision-J source (consistent with E4 "caught"); "either source names the
  base" is reported as secondary.
- **Pseudonyms** P001–P105 by sorted patient ID; the mapping stays in `processed_data/` (not in git).
- Rejected: the top 50 by score (mostly small single blobs, E5 `near_exact.png`); the strong tiers
  only (no picture of a typical or a chance flag); a stratified random sample only (few strong cases).

**Decision Y (set 2026-11-15): the deletion test, made non-tautological.**
- **Masks:** S main E2-flagged multi-lesion masks whose source slice pair has ≥ 2 paired lesions;
  planted P1, P2, P4 multi-lesion masks as a positive control; H-eval chance flags as reference.
- **Contributors:** top = the paired query lesion with the largest s; smallest = the paired query lesion
  with the smallest s > 0 (both deletions remove a contributing lesion).
- **Measures:**
  1. ΔC against the source slice (Hungarian re-run): the experiment plan's paired Wilcoxon (RQ3b
     criterion p < 0.05), reported with the caveat that it nearly follows from the definition of C;
  2. after deletion, a full E2 re-match against T: is the mask still flagged with the same source
     patient? Flip rates after top vs smallest deletion, McNemar test;
  3. ΔE1 score against the source slice (whole-mask Dice with shift, independent of the lesion
     scores), paired Wilcoxon.

  The explanation is faithful if measures 2 and 3 also favour the top contributor. Cost: about 850 E2
  queries (seconds).
- Rejected: ΔC alone (the plan's wording; it would pass for any additive score).

**Decision Z (set 2026-11-15): correctness on planted copies; taxonomy.**
- **Correctness:** report data for all 737 planted masks; scored on the gated masks flagged by E2 (those
  that get a report), per variant:
  - source patient = base patient (decision-J source);
  - source slice within ±2 of the base slice (same patient);
  - P6 lesions: every kept base lesion paired with its source lesion (`planted_lesions.csv`
    `source_lesion`) at s ≥ τ_L in the reported slice pair; for P6-add, the added lesion not paired at
    s ≥ τ_L.

  Criteria (thesis plan): source patient ≥ 90% for P1, P2, P4, P6; correct lesions ≥ 80% for P6. P3 and
  P5 reported only (as E4 decision K). All gated masks, flagged or not, as secondary.
- **Taxonomy** (optional, first to cut): rule-based (no human rater). Rules fixed on the planted
  variants with a confusion table (P1 → exact, P2/P3 → shifted, P4/P5 → deformed, P6 → partial), plus
  *likely chance* (score not beyond the held-out maximum and margin within the H-eval chance range);
  then applied unchanged to the reported S flags and the H-eval contrast.
- Rejected: a manual taxonomy (a single unblinded rater; the blinded human check moves to the master
  thesis).

## Definitions (fixed in code and docs)

- **Sets:** S main (E5 decision L), T, H-eval (baseline), H-cal (the rehearsal null only).
- **Flag, class, stratum, source slice and patient, chained group:** as in E5 (decisions J, M, N).
- **Primary cells:** E1 copy multi, E2 full multi (decision R).
- **π_p, exposure ratio, dispersion:** decision R. **Margin, the reading:** decision U.
- **Slice rarity, patient rarity:** decision V. **3D lesion, span, lesion credit:** decision W.
- **Top and smallest contributor:** decision Y. **Correct source, correct lesions:** decision Z.

## Library modules (building blocks, no knowledge of stages)

- **Moved (step 1), because E5, E6 and E7 need them:** `Scored`, `load_scored`, `heldout`, `sources`,
  `chained` from `memtk.e5.analyze` and `query_table` from `memtk.e5.score` (as built: `heldout`
  needs it) → `memtk.audit`. E5 `check` (26/26) and pytest re-run: nothing changed.
- **`memtk/lesions3d.py` (new, step 1):** `link_slices` (Hungarian IoU chains through a volume),
  `components_26`, `lesion3d_table` (span, largest 2D area, slices), `map_2d_to_3d` (E0b 2D lesion →
  chain by majority of pixels).
- **`memtk/exposure.py` (new, step 2):** `expected_shares` (smoothed π_p), `exposure_table`,
  `concentration_null`, `e2_margins`, `other_patient_max` (rarity runs), `patient_features`,
  `knn_rarity`, `spearman_test` (permutation p, bootstrap interval), `lesion_credits` (H1),
  `rho_difference`.
- **`memtk/reports.py` (new, step 4):** `ReportData` (built from the stored tables), `verdict`,
  `draw_report`, `delete_lesion`, `deletion_scores`, `correctness_row`, `taxonomy`.
- **Reused unchanged:** `audit` (`resample_sums`, `interval`, `auc`, `auc_bootstrap`, `chain_groups`,
  `class_flags`); `calibration` (`load_frozen`, `frozen_settings`, `frozen_threshold`, `run_frozen`);
  `lesionmatch` (`match`, `load_tables`, `LesionMatch.slice_scores` / `summary`); `wholemask.match`;
  `sets` (`load_set`, `load_pool`, `load_planted`); `io` (`read_table`, `slice_images`, `LabelStack`);
  `synthetic.load_sample`; `plotting` (`zoom_window`, `fill`, `draw_labels`, `lesion_colors`, `save`);
  `stages` (`add_stage`, `decided`, `refuse_overwrite`, `require`, `pick`).

## Stages

### `scripts/e6_exposure.py` → `src/memtk/e6/`

One module per stage (`reference`, `rehearse`, `exposure`, `check`) plus `common` (the `exposure:`
config, loading E5's tables and the E6 reference tables).

1. **`reference`** (step 2; needs R, V, W; no S).
   - 3D lesions of T under both definitions (W); `lesions3d.csv` (one row per 3D lesion: span, slices,
     largest area, merged under 26-connectivity) and the 2D → 3D map.
   - Slice rarity (E1, E2) for every T slice; patient features and patient rarity (V).
   - π_p per stratum and detector from H-eval (R); H-eval margins and chance-flag sources (U).
   - `processed_data/model_space/stanford_brainmets/e6/`; `results/stanford_brainmets/e6/reference/`
     (`review.txt`: 3D counts and where the definitions differ, rarity distributions, π_p coverage).
2. **`rehearse`** (step 3; no S). The `exposure` code on E5's mock sets:
   - *null* (H-cal as S): ratios near 1, dispersion within its null, margin AUC about 0.5, reading not
     *patient copying*;
   - *positive* (H-cal + planted P1, P2, P4): the 20 base patients over-exposed, margin AUC above 0.5,
     reading *patient copying*.
   - `results/stanford_brainmets/e6/rehearse/{null,positive}/`: every table and figure `exposure` will
     write. The user reviews; changes are still allowed (S unread by E6).
3. **`exposure`** (step 5; reads E5's S tables; needs every `exposure:` value).
   - `results/amibekk_v1/e6/exposure/`: `exposure.csv` (per patient and cell: o_p, π_p, r_p with
     interval, groups), `concentration.csv`, `margins.csv`, `rarity_h3.csv` (slice level), `h3.csv`
     (patient level), `h1.csv`, `top_patients.csv` (the 10 most exposed, pseudonymous: lesion count,
     sizes, locations, slices); figures below; `review.txt` ending with the reading (U).
4. **`check`** (step 7). Exit code 1 on any failure: every `exposure:` value set; `rehearse` ran before
   `exposure`; every table reproduced with the seed; the E5 hashes unchanged; pytest passes.

### `scripts/e7_reports.py` → `src/memtk/e7/`

1. **`planted`** (step 4; needs X–Z; no S).
   - Report data for every planted mask; `correctness.csv` per variant against the criteria (Z).
   - Deletion test on planted multi-lesion masks (Y, positive control) and on H-eval chance flags.
   - Example report sheets per variant; the taxonomy rules and their confusion table (optional).
   - `results/stanford_brainmets/e7/planted/` (`review.txt` with the criteria verdicts).
2. **`reports`** (step 6; reads E5's S tables and E6 `exposure`; needs every `reports:` value).
   - Selection (X); one PNG per report and `report_index.csv` (selection reason, class, scores, source,
     verdict line); deletion test on S (Y); taxonomy counts (optional).
   - `results/amibekk_v1/e7/reports/` (`review.txt`).
3. **`check`** (step 7): as E6, plus the planted criteria reproduced.

## Figures (PNG, drawn with `memtk.plotting`)

- **`e6/exposure/exposure_sorted.png`:** sorted exposure ratios with intervals per primary cell, the
  null band of the concentration test (the experiment plan's sorted exposure plot).
- **`e6/exposure/rarity_exposure.png`:** patient rarity against exposure ratio, Spearman with interval.
- **`e6/exposure/margins.png`:** margin and source rarity, S flags against H-eval chance flags.
- **`e6/exposure/h1_span.png`:** lesion credits against slice span by size bin, S and H-eval.
- **`e7/planted/report_<variant>.png`, `e7/reports/report_<n>.png`:** match reports; 3–4 chosen for the
  written report.
- **`e7/*/deletion.png`:** ΔC, flip rates and ΔE1, top vs smallest contributor.

## Tests (pytest, small hand-built inputs)

- `tests/test_lesions3d.py`: two blobs that touch in one slice stay two chains under linking and merge
  under 26-connectivity; the span counts thin end slices; the 2D → 3D map follows the majority of
  pixels.
- `tests/test_exposure.py`: smoothed shares are positive and sum to 1; r_p = 1 when o_p ∝ π_p; the
  dispersion null is centred on its expectation; the Spearman permutation p on hand data; the E2 margin
  on a toy `LesionMatch`; `lesion_credits` counts a mask once per 3D lesion.
- `tests/test_reports.py`: verdict rules (NN and strongest pair differ; no pair ≥ τ_L → "configuration
  match"); top vs smallest deletion on toy label maps; correctness on a toy P6-add and P6-remove.

## Verification (every step)

- Each stage prints its summary, writes a run log and refuses to overwrite without `--overwrite`;
  `exposure` and `reports` refuse while a decision value is `null`.
- pytest passes; E5 `check` stays 26/26 after the moves in step 1.
- The user views the E6 `rehearse` and E7 `planted` outputs before any stage reads S.
- After E6 `exposure` and E7 `reports`: `audit:` and the frozen files unchanged; both `check` stages
  exit 0.

## Carried forward

(Filled when the plan closes: open items moved to `docs/experiment-plan.md` E8, report-notes, and the
E8 plan.)
