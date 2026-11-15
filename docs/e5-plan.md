# E5 plan: main audit, S scored once (`scripts/e5_audit.py`)

Working plan for E5, kept in the repo so work can resume from here. The experiment is specified in
`docs/experiment-plan.md` (E5, including its "Carried over from E1–E3" and "Carried over from E4"
notes); this file records how it is implemented, stage by stage, and what has been decided. Update
the status table when a step is done. E5 runs the E1 and E2 library detectors (`memtk.wholemask`,
`memtk.lesionmatch`) with the settings and thresholds of the frozen E3 file on the synthetic masks S,
**once**, and compares them with the held-out baseline H-eval already scored in E4.

E5 is the only step that scores S. Every analysis choice (sets, tests, the adjacent-slice window,
sensitivity runs, the drop-only re-run, noisy masks, H2, near-exact copies) is fixed below and in the
config section `audit:` **before** `score` runs, as E4's criteria were fixed before `validate`. Its
outputs are frozen evidence that E6–E8 only read.

## Status (2026-11-15)

| # | Step | Kind | Status |
| --- | --- | --- | --- |
| 0 | This plan; decisions L–Q recorded in the docs | docs | ✅ 2026-11-15 (decisions L–Q set by the user in the planning session) |
| 1 | Shared helpers to the library; `memtk/audit.py` + `tests/test_audit.py`; config section `audit:` with the values of L–Q | refactor + library + config | ✅ 2026-11-15: `load_frozen`, `run_frozen` → `calibration` (`config.frozen_file` takes the experiment, for `e5_<dataset>.yaml`); `slice_images` → `io`; `heval_classes`, `fp_flags`, `share_row` → `audit` as `class_table`, `class_flags`, `share_row` (moved unchanged; `run_frozen` on 60 random H-eval slices reproduces the stored `heval_classes.csv` to 1e-16); `sets.load_pool`, `sets.relabel`; `memtk/audit.py` (cluster bootstrap as an exact multinomial over cluster types, excess test, Holm, AUC and its bootstrap, chained groups, copied-area share, standardised rate, held-out maxima); 13 tests (pytest 141); E3 check 16/16, E4 check 16/16; `audit:` in `configs/default.yaml` |
| 2 | `prepare` (no S): drop-only real label maps, sensitivity thresholds, τ_D\*, held-out maxima, H-eval under every variant; `configs/frozen/e5_<dataset>.yaml` | production | ✅ 2026-11-15 (user review of `results/stanford_brainmets/e5/prepare/review.txt` pending): gap 0 changes only 4 T slices (H-cal, H-eval none), so drop-only uses the frozen thresholds (re-derived: equal); every sensitivity threshold re-derived on H-cal equals E3 `sensitivity.csv`; τ_D\* 0.897 / 0.700 (H-cal 7.43% / 8.91% flagged, target 7.43% / 8.66%: ties); held-out E1 maxima single 1.000 (4–11 px), 0.973, 0.967, 0.933, 0.962 (≥ 80 px), multi 0.855 overall; E2 C maxima single 1.000 (H-cal `Mets_316/z88`: a 12 px lesion pixel-identical to one of T `Mets_249/z101`, chance at the bound), multi 0.508, so the beyond-held-out tier exists for multi-lesion masks only (as the E8 rule assumes). Refactor in this step: E3's threshold helpers (`lesion_slices`, `threshold_rows`, `thresholds`, `frozen_thresholds`, `shortcut_problems`, `chance_rates_at`, `STRATA`) moved to `calibration`; `lesionmatch.save_tables` / `load_tables` (E3, E4, E5); `io.ID_COLUMNS` extended to the detector tables' ID columns (S sample IDs such as `1033` would otherwise be read back as numbers). pytest 146; E3 and E4 check 16/16 |
| 3 | `rehearse` (no S): `score` and `analyze` code on mock S (null and positive case); the user reviews the tables and figures | rehearsal | ✅ 2026-11-15 (user review of `results/stanford_brainmets/e5/rehearse/{null,positive}/` pending). The scoring (`e5.score.score_query`) and the analysis (`e5.analyze.analyze_scored`) are single functions that `score` and `analyze` will call on S unchanged. **Null** (H-cal, 989 slices): H-cal reproduces its own chance rates (E1 copy single 35/686 = 5.1%), no primary cell beyond chance (Holm p 0.52–1.0), 0 near-exact, reading "validated negative". **Positive** (H-cal + 294 gated planted P1/P2/P4): all four primary cells beyond chance, P1 and P2 flagged 100/100 by both detectors with the right patient, P4 93/94; 257 near-exact copies (none on H-eval, by construction); 80 E2 groups naming all 20 base patients; reading "memorization beyond chance". Seen on the way: the multi-lesion E1 Mann–Whitney p is 0.004 on the null case while the patient-bootstrap AUC interval [0.40, 0.74] covers 0.5 (the decision M caveat, live); H2 (ii) does not hold on the positive case, since whole-mask copies are caught by both detectors (H2 needs partial copies). Refactor: E4's `zoom_window`, `fill`, `draw_labels`, `lesion_colors` → `plotting`; `load_planted` → `sets`; new `audit.cell_rates`; `audit.auc_bootstrap` avoids a draws × S × H intermediate. pytest 151; E4 check 16/16 |
| 4 | `score`: S scored once, every variant of decision O; raw tables only, no rates printed | production (S) | ⬜ |
| 5 | `analyze`: every table and figure, `flagged.csv`, the outcome reading | analysis | ⬜ |
| 6 | `check`; close the plan; carry items forward to `docs/e6-e7-plan.md` (and the E8 exit counts) | consistency checks + docs | ⬜ |

**Next: start here.** The user reviews the rehearsal (`results/stanford_brainmets/e5/rehearse/null/`
and `positive/`: `review.txt`, the tables, the figures). Changes to the analysis are still allowed
now (S unseen); after them, re-run `rehearse --overwrite`. Then step 4: the `score` stage, a thin
wrapper that loads S (main + noisy, `sets.load_pool`) and its drop-only labels (from
`masks_raw.npz`), checks the gate and the hashes, calls `score_query` once and writes the
completion marker.

**Milestone:** 17 Nov 2026, main audit tables for RQ1 and RQ2. Realistic schedule: steps 0–1 on
15–16 Nov, steps 2–3 on 17 Nov, steps 4–5 on 18 Nov, step 6 on 19 Nov: about two days past the
milestone, leaving 19–24 Nov for E6 + E7. If time is short, cut first the per-scheme table and the
extra figures; the area-weighted C is already out (below). Never cut `rehearse` or the lock.

**Working rules** (as in CLAUDE.md):
- Small reviewed steps, one at a time; the user does all git operations.
- Decision values are set only after the user agrees to that specific value.
- After each step, update `scripts/README.md`, `docs/data-dictionary.md` and
  `docs/module-reference.md` (and `docs/architecture.md` if the structure changes), and suggest
  entries for `docs/report-notes.md` (what the report may use) and `docs/process-log.md` (how the
  work was done).
- **S is read only by `score`.** `prepare` and `rehearse` read T, H-cal, H-eval and the planted set;
  their run logs list their inputs and `check` confirms that S is not among them. Detector settings
  and thresholds come only from the frozen files, never from the live config.
- **The analysis is fixed before `score`** (the lock). `score` records the SHA-256 of the `audit:`
  section and of both frozen files (`e3_<dataset>.yaml`, `e5_<dataset>.yaml`); `analyze` refuses to
  run if any of them changed. An analysis added after `score` is allowed but is labelled
  *exploratory* in report-notes and in its file names.
- **`score` runs once.** It refuses to overwrite its tables; `--overwrite` is only for a run that
  crashed before writing its completion marker, and every such re-run is logged in
  `docs/process-log.md`. No rate is printed or written before `analyze`.

## Context

E0–E4 delivered everything E5 needs:

| Input | From | Used for |
| --- | --- | --- |
| `configs/frozen/e3_stanford_brainmets.yaml`: τ_D 0.905 / 0.729, τ_L 0.739 / 0.816, τ_C 0.719 / 0.406 (single / multi), bound 12 px, min area 4 px, detector settings, `hcal_chance_rates` | E3 `freeze` | every threshold and setting of the main run |
| `results/stanford_brainmets/e3/calibrate/sensitivity.csv` | E3 `calibrate` | the sensitivity thresholds (decision O), re-derived and asserted in `prepare` |
| `<output_root>/stanford_brainmets/e3/e1_hcal.csv` | E3 `calibrate` | τ_D\* (decision P), held-out maxima (decision Q) |
| H-eval E1 / E2 tables, `heval_classes.csv` in `<output_root>/stanford_brainmets/e4/` | E4 `validate` | the main baseline (not recomputed) |
| `results/stanford_brainmets/e4/validate/gate.csv` | E4 `validate` | `gate_status` must be PASS or ACCEPTED (now ACCEPTED) |
| `planted.csv`, `planted_labels.npz` | E4 `plant` | the positive case of `rehearse` |
| Pool `labels.npz`, `slices.csv`, `lesions.csv`, `qc.csv`, `masks_raw.npz` (`amibekk_v1`) | E0b `lesions`, `qc`, `brain` | S label maps, audit sets (decision L), drop-only S labels (from the masks as saved) |
| Real `labels.npz`, `slices.csv`, model-space seg volumes | E0b `lesions`, E0a `real` | T, H-cal, H-eval; drop-only real labels |

Facts that shape the design (read-only queries, 2026-11-15, unless stated):
- **The drop-only re-run needs no re-calibration.** H-cal and H-eval have 0 attached strays
  (`slices.csv`, `n_strays`, `stray_px`), so their label maps at gap 0 equal the E0b maps at gap 2,
  and every frozen threshold, a function of H-cal against T, is unchanged up to T's own change: T has
  strays in 4 slices (10 px of 280,410 lesion px). In S main 4,481 masks have strays (pool
  `slices.csv`). This answers the E1–E3 carry-over ("recalibrate or descriptive only"): neither is
  needed; `prepare` asserts the identity in code and re-scores H-eval against the drop-only T.
- **Whole-mask Dice > 0.9 is not unambiguous for single lesions.** Held-out single-lesion E1 scores
  reach a maximum of 1.0 (4–11 px), 0.973 / 0.960 (12–19 px), 0.933 / 0.967 (20–39 px), 0.933 /
  0.933 (40–79 px) and 0.909 / 0.962 (≥ 80 px) (H-cal / H-eval); 8% of H-eval single-lesion slices
  score above 0.9. Multi-lesion maxima are 0.855 (H-cal) and 0.828 (H-eval). *Source:
  `e3/e1_hcal.csv`, `e4/e1_heval.csv` with `slices.csv` `largest_lesion_px`.* Hence decision Q.
- **E2 chance maxima on H-eval:** C(q, NN) 0.952 single, 0.488 multi; strongest pair 0.952 / 0.976
  (`heval_classes.csv`).
- **S main (35,920 masks) by stratum:** E1 (lesions ≥ 4 px) 19,117 single / 16,803 multi; E2
  (informative lesions ≥ 12 px) 19,911 single / 6,006 multi / 10,003 without an informative lesion
  (28%, scored by E1 only). Median informative lesion area: S main 24 px, H-eval 22 px, T 28 px
  (quartiles 16 / 41, 16 / 37, 17 / 64). Six naming schemes with about 5,400–7,900 main masks each
  (pool `qc.csv`, `lesions.csv`).
- **Measured chance rates at the frozen thresholds** (H-eval, `fp_heval.csv`): E1 copy 7.2% / 7.4%,
  E2 full 7.1% / 2.9%, E2 partial 3.0% / 5.5%, E2 flagged 10.1% / 8.5% (single / multi), with
  flags clustered in few patients. At these rates **thousands of S masks are flagged by chance**
  (about 7–10% of 35,920); the question is the excess, not the count.
- **Run time** (E1–E3 carry-over): E1 about 4.4 ms and E2 about 10 ms per query, so about 15 min for
  the main run, about 1 h with every variant of decision O. The E2 tables of S hold several million
  lesion pairs: Parquet in `<output_root>/<pool>/e5/`.
- **Detection limit** (E4): copies whose small lesions carry about 1 px of boundary noise are not
  detected (P5: 0% at 12–19 px, 85% at ≥ 80 px). S rates are lower bounds for such copies.

## Decision gates: decision → stage

Lettering continues from E4 (I–K). All six were set by the user on 2026-11-15, in the planning
session, before any code of E5 ran.

| | Decision | Evidence | Needed by |
| --- | --- | --- | --- |
| L | Audit sets and the noisy masks | `qc.csv`; E0b decision D | `score`, `analyze` |
| M | RQ1 tests: primary family, multiplicity, secondary analyses | E4 `fp_heval.csv`; this plan | `analyze` |
| N | Adjacent-slice grouping | E0b adjacency Dice (report-notes section 3); E4 `nn_is_base` | `analyze` |
| O | Sensitivity runs on S | E3 `sensitivity.csv`; the stray counts above | `prepare`, `score` |
| P | H2: equal false-positive rate | frozen `hcal_chance_rates` | `prepare`, `analyze` |
| Q | Near-exact copies | held-out E1 maxima above | `prepare`, `analyze` |

Each decision is a config value under `audit:` (dated comments); `score` refuses to run while any of
them is `null`.

**Decision L (set 2026-11-15): audit sets.**
- **Main:** `mask_class` clean or minor (35,920 masks), as decided at the E0b `qc` stage. Failed MRIs
  (`mri_failed`) and edge-flagged masks stay in: the mask comes from stage 1 whatever stage 2 drew.
- **Noisy** (1,241 masks, > 9 isolated specks): scored by **both detectors**, strata counted on the
  cleaned mask, frozen thresholds. Reported as the rows *noisy* and *main + noisy* next to *main*.
  Chosen over E1 only (the experiment plan's wording): E2 sees the lesions without the specks, so it
  shows whether a copy hides in a noisy mask, where specks lower whole-mask Dice; the cost is about
  a minute.
- `no_lesion` masks have nothing to match and are counted only.
- **Strata** as for every query: E1 by lesions ≥ 4 px, E2 by informative lesions (≥ 12 px).

**Decision M (set 2026-11-15): RQ1 tests.**
- **Primary family (4 cells):** the excess = S rate − H-eval rate for **E1 copy** and **E2 full**,
  single and multi, on the main set. S masks are independent samples and are resampled directly;
  H-eval is resampled by patient (its slices are clustered); the two resamplings are independent,
  `calibration.bootstrap_draws` (2,000) draws, seeded. Per cell: the 95% percentile interval of the
  excess and a one-sided bootstrap p = (1 + draws with excess ≤ 0) / (draws + 1). **Holm**-adjusted
  over the 4 cells; a cell is *beyond chance* at adjusted p < 0.05.
- **Secondary (95% intervals, no test claims):**
  - E2 partial and E2 flagged against H-eval, per stratum.
  - AUC = P(S score > H-eval score), ties counting ½, for the E1 score, E2 C(q, NN) and E2 strongest
    pair, per stratum, with a two-level bootstrap interval (S iid, H-eval by patient). The one-sided
    Mann–Whitney p of the experiment plan is reported with the caveat that it treats slices as
    independent.
  - Size-confound check: rates by size bin of the largest informative lesion (12 / 20 / 40 / 80 px),
    and the H-eval rate reweighted to S's stratum × size-bin mix (standardised), next to the crude
    rate. Single small lesions match by chance often (report-notes section 3), so a different size
    mix alone could move the rate.
  - Rates per naming scheme (descriptive: the generation rounds may differ).
  - Estimated excess count = N_S × excess, with the interval.
  - Every analysis on *noisy* and *main + noisy*, and under each sensitivity variant (decision O).
- **Reading the outcome mechanically** (the experiment plan's "How to read the outcomes"):
  *memorization beyond chance* if at least one primary cell is beyond chance after Holm; *rare but
  real* if none is but at least one near-exact copy exists (decision Q); otherwise a *validated
  negative* (E4 shows the detectors work). `analyze` writes this reading in `review.txt`.
- Rejected: a 95% interval rule per cell without correction (four looks at the same question); a
  single pooled endpoint (E2 flagged, strata pooled; hides that the strata have different chance
  rates and that E1 is the baseline of RQ2).

**Decision N (set 2026-11-15): adjacent-slice grouping, chained ±2.**
- The source slice of a flag is the slice decision J names: E1 copy → the E1 nearest neighbour; E2
  full → the E2 nearest neighbour; E2 partial → the strongest-pair slice.
- Per detector class and source patient, sort the source z; a group is a maximal run in which
  consecutive z differ by at most 2. One lesion copied across many slices is one group.
- Reported per detector class: raw flags, distinct source slices, groups, distinct source patients;
  for S and for H-eval (where every group is chance). The gap between raw and grouped counts shows
  how much near-duplicate slices inflate the raw number.
- Rejected: fixed greedy 5-slice windows (a long lesion counts several times, depending on where the
  window starts). E6 still credits every stored pair above the threshold (experiment plan, E6 step 1).

**Decision O (set 2026-11-15): sensitivity runs on S, in the same single `score` run.**

| Variant | What changes | S | H-eval baseline | Thresholds |
| --- | --- | --- | --- | --- |
| percentile 99 | thresholds only | re-threshold the main tables | re-threshold E4's tables | E3 `sensitivity.csv` |
| σ 0.08 (10.2 px) | E2 position factor | E2 re-run | E2 re-run in `prepare` | E3 `sensitivity.csv` |
| bound 10 px | informative-lesion bound | E2 re-run | E2 re-run in `prepare` | E3 `sensitivity.csv` |
| bound 25 px | informative-lesion bound | E2 re-run | E2 re-run in `prepare` | E3 `sensitivity.csv` |
| drop-only (gap 0) | stray pixels dropped, not attached | E1 + E2 on S relabelled from `masks_raw.npz` | E1 + E2 against the drop-only T in `prepare` | the frozen thresholds (H-cal unchanged, asserted) |

Each variant reports the main rates, the excess and its interval for both strata; the primary family
and its Holm adjustment apply to the main run only. `prepare` re-derives every sensitivity threshold
on H-cal and asserts that it equals `sensitivity.csv`, then writes them to the E5 frozen file.
Rejected: the 99th percentile and drop-only only (σ and bound sensitivity would stay H-cal-only),
drop-only only (the plan's minimum).

**Decision P (set 2026-11-15): H2 at equal false-positive rate, matched on H-cal.**
- (i) E2 full against E1 copy at the frozen thresholds (both the 95th percentile of H-cal).
- (ii) E2 flagged (full or partial) against E1 at a matched τ_D\* per stratum: the percentile of the
  H-cal E1 scores that flags the H-cal rate of E2 flagged in the same-named stratum (7.43% single,
  8.66% multi; frozen `hcal_chance_rates`), i.e. τ_D\* = the (100 − rate)th percentile. Computed in
  `prepare` from `e3/e1_hcal.csv` and frozen before S.
- Both on the informative subset (slices with ≥ 1 informative lesion), each detector with its own
  stratum rule for its threshold, reported by E2 stratum and pooled (H-eval reweighted to S's mix).
- Statistic: the difference of excesses (S − H-eval for E2) − (S − H-eval for E1), with the same
  bootstrap as decision M (paired within each set, since both detectors score the same masks). **H2
  holds if its 95% interval is above 0**; (ii) is the main H2 comparison, (i) supporting. Also
  reported: discordant counts (E2 only, E1 only) on S and H-eval, and the RQ2 cross-tabulation
  (E1 flag × E2 class) on the informative subset.
- Rejected: the frozen thresholds only (comparison (i) alone; E2 flagged would stay descriptive).

**Decision Q (set 2026-11-15): near-exact copies, beyond the held-out maximum.**
- Near-exact = E1 score ≥ 0.9 **and** above the highest H-cal ∪ H-eval E1 score in the same stratum
  and size bin of the largest lesion (bins from 4, 12, 20, 40, 80 px). The maxima are computed in
  `prepare` and frozen before S. A single-lesion mask with its largest lesion under 12 px cannot be
  near-exact (held-out maximum 1.0).
- Reported next to the plan's plain count of E1 score > 0.9 and the H-eval rate above 0.9 per
  stratum, so the reader sees why the plain count is not "unambiguous".
- **Beyond-held-out tier for E2** (descriptive, for E7 examples and the E8 exit item): E2 C(q, NN)
  above the held-out maximum of its stratum (H-cal ∪ H-eval), computed and frozen the same way.
- Rejected: the plan's plain Dice > 0.9 (8% of single-lesion H-eval slices pass by chance);
  multi-lesion only (would discard single-lesion copies of large lesions that no held-out mask
  approaches).

**Also fixed with this plan (proposed with it and approved by the user, 2026-11-15):**
- **Copied-area share** (experiment plan, E5 step 7) for each flag's source slice: area of the paired
  lesions with s ≥ τ_L over the total informative lesion area of the two slices. Descriptive.
- **Not run: the area-weighted C** (optional in the experiment plan). It would need its own τ_C
  calibrated on H-cal; the copied-area share answers the same question descriptively.
- **E8 exit item.** At chance thousands of masks are flagged, so the experiment plan's "about 20 or
  more flagged masks" is always met and says nothing. Proposed rule for the E8 plan: E8 is eligible
  if the grouped count of near-exact copies plus E2 multi-lesion masks beyond the held-out maximum is
  ≥ 20. E5 reports these counts; the E8 plan applies the rule.

## Definitions (fixed in code and docs)

- **S main / noisy / main + noisy:** decision L. **Reference set:** T. **Baseline:** H-eval (decision
  E applied; 1,094 slices, 26 patients).
- **Scorable:** E1 needs ≥ 1 lesion ≥ 4 px; E2 needs ≥ 1 informative lesion. Unscorable masks are
  counted, not scored.
- **Flag classes:** E1 copy (score ≥ τ_D of its stratum); E2 full (C(q, NN) ≥ τ_C), partial (not full,
  strongest pair ≥ τ_L), flagged (full or partial); as in the frozen `class_rule`.
- **Rate:** flagged / scorable masks of a set, class and stratum. **Excess:** S rate − H-eval rate.
  **Interval:** S resampled iid, H-eval by patient (decision M).
- **Source slice / patient:** decision N (as decision J). **Group:** decision N.
- **Near-exact, beyond-held-out tier:** decision Q. **Copied-area share:** above.
- **Informative subset (RQ2):** masks with ≥ 1 informative lesion, scored by both detectors.

## Library modules (building blocks, no knowledge of stages)

- **Moved (step 1), because E5 needs them too** (a helper used by two experiments moves to a library
  module; E3 and E4 `check` and pytest are re-run to confirm nothing changed):
  - `memtk.e4.validate.run_detectors` → `memtk.calibration.run_frozen` (E1 + E2 with the frozen
    settings, and their classes);
  - `memtk.e4.common.load_frozen` → `memtk.calibration.load_frozen` (file and SHA-256);
  - `memtk.e4.common.slice_images` → `memtk.io.slice_images`;
  - `memtk.e4.validate.heval_classes`, `fp_flags`, `share_row` → `memtk.audit` as `class_table`,
    `class_flags` and `share_row` (as built: renamed because they now serve S as well as H-eval).
- **`memtk/sets.py` (additions):** `load_pool(cfg, pool, mask_classes)` (S label maps from the pool's
  `labels.npz`, filtered by `qc.csv` `mask_class`; a `SliceSet` whose "patients" are the sample IDs
  and with the naming scheme kept), and `relabel(masks, min_area, gap)` (label maps through
  `lesions.attach_or_drop`, for the drop-only variant).
- **`memtk/audit.py` (new):** `excess_bootstrap` (S iid, held-out by patient; interval and one-sided
  p), `holm`, `auc` and `auc_bootstrap`, `chain_groups(patients, z, max_gap)`, `copied_area_share`,
  `standardized_rate`, `beyond_max` (scores against the maxima per stratum and size bin), and the
  moved `fp_flags`, `share_row`.
- **Moved in step 2 (as built),** because `prepare` re-derives the sensitivity thresholds with
  E3's own code: `e3.calibrate.lesion_slices`, `threshold_rows`, `thresholds`, `STRATA` and
  `e3.freeze.frozen_thresholds`, `shortcut_problems`, `chance_rates_at` (`DETECTORS` as
  `THRESHOLDS`) → `memtk.calibration`. New `lesionmatch.save_tables` / `load_tables` (the four E2
  tables under one naming rule, used by E3, E4 and E5); `io.ID_COLUMNS` extended to the detector
  tables' ID columns.
- **Reused unchanged:** `calibration.e1_classes`, `e2_classes`, `frozen_settings`, `bootstrap_share`,
  `stratum`, `area_bins`, `bin_label`; `LesionMatch.summary` / `slice_scores`; `stages.decided`,
  `refuse_overwrite`, `require`, `pick`; `e4.common.gate_status`; `memtk.plotting`.

## Stages: `scripts/e5_audit.py` → `src/memtk/e5/`

The script is only the command line (as `e4_validate.py`); stages live in the stage package, one
module per topic (`prepare`, `rehearse`, `score`, `analyze`, `check`, plus `common` for loading the
frozen files, the audit config and its hash, and the stored tables).

1. **`prepare`** (step 2; needs O, P, Q; reads T, H-cal, H-eval; no S).
   - Drop-only label maps (gap 0) of T, H-cal and H-eval from the model-space seg volumes; assert
     that H-cal and H-eval equal the E0b label maps (expected: 0 strays) and list the T slices that
     change.
   - H-cal re-runs under σ 0.08 and bounds 10 and 25 px, and the 99th percentile; assert each value
     equals E3 `sensitivity.csv`.
   - τ_D\* per stratum (decision P) from `e3/e1_hcal.csv`; held-out maxima per stratum and size bin
     (decision Q) from the H-cal and H-eval tables.
   - H-eval E1 / E2 tables under every variant of decision O (E4's tables are the main baseline).
   - `configs/frozen/e5_<dataset>.yaml` (version-controlled; sensitivity thresholds, τ_D\*, maxima,
     the SHA-256 of the E3 frozen file); `<output_root>/<dataset>/e5/` (drop-only labels, H-eval
     tables); `results/<dataset>/e5/prepare/` (`review.txt`, the asserted values).
2. **`rehearse`** (step 3; no S). Runs the `score` and `analyze` code on a mock S:
   - *null case:* H-cal as S. Expected: AUC about 0.5 and no excess. H-cal sits at about 5% by
     construction (its own thresholds) against 7% on H-eval, so the excess may come out negative;
     this tests the code, not a hypothesis.
   - *positive case:* H-cal plus the gated planted P1, P2 and P4 masks. Expected: a positive excess
     carried by the planted masks, and groups that recover their base patients (bases of one patient
     are ≥ 2 apart in z, so chained groups may join them: check against a hand count).
   - `results/<dataset>/e5/rehearse/`: every table and figure `analyze` will write. The user reviews
     formats and content here; changes to the analysis are still allowed (S unseen).
3. **`score`** (step 4; S, once). Needs `gate_status` PASS or ACCEPTED, every `audit:` value set, and
   the E5 frozen file.
   - Records the SHA-256 of `audit:` and both frozen files in run_info.
   - E1 and E2 on main and noisy; every variant of decision O (drop-only S labels from
     `masks_raw.npz`); about 1 h.
   - `<output_root>/<pool>/e5/`: E1 tables (CSV) and E2 tables (queries and lesions CSV, slice and
     lesion pairs Parquet) per set and variant, and a completion marker. Prints counts, table sizes
     and run times only.
4. **`analyze`** (step 5). Refuses to run if the hashes differ from `score`'s.
   - `results/<pool>/e5/analyze/`:
     - `rates.csv` (set × class × stratum: S and H-eval rates with intervals, excess, interval, p,
       Holm-adjusted p for the primary cells, estimated excess count);
     - `auc.csv`, `rates_by_size.csv` (with the standardised H-eval rate), `rates_by_scheme.csv`,
       `sensitivity.csv`;
     - `rq2_crosstab.csv` (informative subset, S and H-eval), `h2.csv` (comparisons (i) and (ii),
       discordant counts);
     - `groups.csv` (decision N), `near_exact.csv` (decision Q, with the plain Dice > 0.9 count);
     - `flagged.csv`: one row per flagged S mask (sample ID, mask class, strata, both detectors'
       classes and scores, source slice and patient per detector, group ID, n_kept, all_copied,
       copied-area share, near-exact and beyond-held-out tiers). Read by E6, E7 and E8;
     - the figures below; `review.txt` ending with the outcome reading (decision M) and the E8 exit
       counts.
5. **`check`** (step 6). Exit code 1 on any failure:
   - the `audit:` section and both frozen files unchanged since `score` (hashes);
   - `score` ran once to completion (run_info, marker; any re-run is in the process log);
   - every `analyze` table reproduced from the raw tables;
   - the `prepare` values reproduced (sensitivity thresholds, τ_D\*, maxima);
   - the run logs of `prepare` and `rehearse` list no S input;
   - pytest passes.

## Figures (PNG, drawn with `memtk.plotting`)

- **`analyze/score_distributions.png`:** E1 score, E2 C(q, NN) and E2 strongest pair, S main against
  H-eval, by stratum, with the frozen thresholds (the experiment plan's score-distribution figure).
- **`analyze/rates_forest.png`:** S and H-eval rates with intervals per class and stratum, and the
  excess per sensitivity variant and set (main, noisy, main + noisy).
- **`analyze/near_exact.png`:** every near-exact S mask (or a seeded sample if many) next to its
  nearest training slice, with the scores; E7 candidates.
- **`analyze/groups_top.png`:** the largest groups (decision N): S masks and their source slices.
- **`analyze/copied_area.png`:** copied-area share of the flagged masks by class and stratum.

## Tests (pytest, small hand-built inputs)

- `tests/test_audit.py`: chained groups on toy z lists (z 1, 2, 4, 7, 8 → {1, 2, 4} and {7, 8};
  two patients never share a group); Holm against known values; AUC equals U / (n₁n₂) with ties;
  the excess interval covers 0 and p is large for identical samples, small for a clear shift;
  copied-area share and the standardised rate on hand examples; `beyond_max` at bin edges.
- `tests/test_sets.py` (or the existing sets tests): `load_pool` keeps only the requested mask
  classes; `relabel` at gap 0 drops a stray that gap 2 attaches.

## Verification (every step)

- Each stage prints its summary, writes a run log and refuses to overwrite without `--overwrite`;
  `score` refuses to run while an `audit:` value is `null` or the gate is FAIL.
- pytest passes; E3 and E4 `check` still pass after step 1.
- The user views the `prepare` review and the `rehearse` tables and figures before `score` runs.
- After `score`: nothing in `audit:` or the frozen files changes; `analyze` and `check` confirm it.
