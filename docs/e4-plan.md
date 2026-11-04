# E4 plan: planted-copy validation, the go/no-go gate (`scripts/e4_validate.py`)

Working plan for E4, kept in the repo so work can resume from here. The experiment is specified
in `docs/experiment-plan.md` (E4, including its "Carried over from E1–E3" notes); this file
records how it is implemented, stage by stage, and what has been decided. Update the status table
when a step is done. E4 runs the E1 and E2 library detectors (`memtk.wholemask`,
`memtk.lesionmatch`) with the settings and thresholds of the frozen E3 file on planted copies
(positives) and H-eval (negatives), and decides whether the main audit (E5) may run.

## Status (2026-11-04)

| # | Step | Kind | Status |
| --- | --- | --- | --- |
| 0 | This plan; decisions I and J recorded in the docs | docs | ✅ 2026-11-04 (decisions I and J set by the user in the planning session) |
| 1 | Shared helpers to the library (`memtk/sets.py`); frozen-file helpers in `memtk/calibration.py`; config section `validation:` | refactor + library + config | ✅ 2026-11-04: `SliceSet`, `load_set`, `decision_e` moved unchanged (E3 check 16/16); `frozen_settings`, `e1_classes`, `e2_classes` + 3 tests (pytest 109); the class helpers reproduce the frozen H-cal chance rates exactly from the stored tables; `validation:` with I and J, `criteria: null` |
| 2 | `memtk/planted.py` + `tests/test_planted.py` | library | ✅ 2026-11-04: 16 tests (pytest 125). Dry run on real T in memory (nothing written, no detector): 100 bases from 20 patients, 37 multi-lesion (30 from the multi group, 7 from the other); 4 multi-pool patients skipped (no spaced quota); all P1–P5 and P6-add made, P6-remove for the 37 multi bases; P6-add needed at most 8 tries (median 1). Every informative base lesion still ≥ 12 px: P4 94, P5-erode 50, P5-dilate 98 of 100 |
| 3 | `plant`: base masks, variants, provenance, figures (no detector) | production (needs I) → evidence for K | open |
| 4 | Decision K: go/no-go criteria in `validation.criteria` | decision | open |
| 5 | `validate`: E1 + E2 on the planted set and H-eval; validation table; gate verdict | production (needs K) | open |
| 6 | `check`; close the plan | consistency checks + docs | open |

**Next: start here.** Step 3: the `plant` stage (`scripts/e4_validate.py`, `src/memtk/e4/`):
bases and variants written with their provenance, `variant_summary.csv`, the figures, and
`review.txt` as the evidence for decision K.

**Milestone:** 13 Nov 2026, E4 gate passed. Indicative schedule: step 0 on 4 Nov, steps 1–2 on
5 Nov, step 3 on 6 Nov, decision K on 7 Nov, step 5 on 8 Nov, step 6 on 9 Nov; 10–13 Nov is
buffer for a failed gate (adjust on H-cal, re-freeze, re-run E4).

**Working rules** (as in CLAUDE.md):
- Small reviewed steps, one at a time; the user does all git operations.
- Decision values are set only after the user agrees to that specific value.
- After each step, update `scripts/README.md`, `docs/data-dictionary.md` and
  `docs/module-reference.md` (and `docs/architecture.md` if the structure changes), and suggest
  entries for `docs/report-notes.md` (what the report may use) and `docs/process-log.md` (how
  the work was done).
- **S is never read in E4.** Inputs are T, H-eval, the planted set and the frozen file. H-cal
  enters only through the frozen file. Detector settings come only from the frozen file, never
  from the live config.
- **The go/no-go criteria (decision K) are fixed before `validate` first runs**, from the
  planted set and its geometry, never from detector scores.

## Context

E0–E3 delivered everything E4 needs, so E4 reads caches and never touches raw data:

| Input | From | Used for |
| --- | --- | --- |
| `configs/frozen/e3_stanford_brainmets.yaml`: τ_D 0.905 / 0.729, τ_L 0.739 / 0.816, τ_C 0.719 / 0.406 (single / multi), bound 12 px, min area 4 px, detector settings, `hcal_chance_rates` | E3 `freeze` | every threshold and setting in E4 |
| `labels.npz`, `slices.csv`, `lesions.csv` | E0b `lesions` | base masks, H-eval, T reference |
| `brain_masks.npz` | E0b `brain` | P6-add placement (added lesion inside the base's brain) |
| Model-space BRAVO volumes | E0a `real` | backgrounds of the figures only |
| H-eval after decision E (pixel-sum filter; removes no slice): 1,094 slices, 26 patients | E0a split, E0b `filters-review` | negatives (false-positive rate); also E5's baseline |

Facts that shape the design:
- **Base availability** (read-only query of `lesions.csv` and `slices.csv`, 2026-11-04): 3,172
  T slices have an informative lesion (≥ 12 px), 805 of them multi-lesion (25%); 103 T patients
  have such a slice; 92 have ≥ 5, and of those 49 have ≥ 2 and 45 have ≥ 3 multi-lesion slices.
  A plain random draw would give ~25 multi-lesion bases, too few for P6-remove.
- **Informative T lesions are small:** median 28 px, quartiles 17 / 64 px (same query). A 1 px
  erosion can destroy a 12–17 px lesion or drop it below the bound; a 1 px dilation roughly
  doubles it (size factor ≈ 0.5, Dice ≈ 0.6–0.7). P5 can fail by construction for small lesions.
- **P3 sits at the edge** (E1–E3 carried-over note): a single lesion shifted 5 px keeps shape and
  size, its position factor is exp(−25 / (2 · 6.4²)) ≈ 0.737, just under τ_L single (0.739) and
  above τ_C single (0.719); E1's shift search reaches only 3 px. P3 is reported only, as planned.
- **"Full" under the frozen rule A** means C(q, NN) ≥ τ_C, the configuration closer than chance,
  not every lesion copied. Most P6 variants come out full: one H-eval lesion added to a
  single-lesion base gives C = 1.0 / 2 = 0.5 ≥ τ_C multi 0.406; removing one lesion from a
  three-lesion base gives C ≈ 2/3. Hence decision J.
- **Lesion-level chance rates on H-cal** (frozen file): full 5.1% / 5.2%, flagged (full or
  partial) 7.4% / 8.7% (single / multi). A false-positive criterion of 7% on "flagged" would be
  failed at chance; it must be stated per class.

## Decision gates: decision → production stage

Lettering continues from E1–E3 (F–H).

| | Decision | Evidence | Needed by |
| --- | --- | --- | --- |
| I | Planting design: base selection and variant parameters. **Set 2026-11-04** | this plan; base counts above | `plant` |
| J | What counts as caught. **Set 2026-11-04** | report-notes section 5; frozen rule A | `validate` |
| K | Go/no-go criteria (numbers) | `plant` review: the variants, their area changes, stratum changes (no scores) | `validate` |

Each decision is a config value under `validation:`; I and J are entered as decided in step 1
(dated comments), K starts as `null` and `validate` refuses to run while it is `null`.

**Decision I (set 2026-11-04 by the user): planting design.**
- **Base masks.** Eligible: T patients with ≥ 5 slices holding an informative lesion (92). With
  the seed: 10 patients drawn from those with ≥ 3 multi-lesion slices (45), giving 3 multi- and
  2 single-lesion slices each; 10 patients drawn from the other eligible patients, giving 5
  random informative slices each. Slices of one patient at least 2 apart in z (fewer
  near-duplicate bases). About 100 bases, ~35–40 of them multi-lesion. Strata here count
  informative lesions. Chosen over a plain random draw (~25 multi-lesion bases) and over half
  multi overall (over-represents multi-lesion slices against T's 25%).
- **Variants.** Each starts from the base's cleaned mask (`labels > 0`) and ends with
  `lesions.attach_or_drop` (min area 4, gap 2), exactly like real masks.

  | Variant | How it is made | Bases |
  | --- | --- | --- |
  | P1 exact | unchanged | all |
  | P2 shift 2 px | one random axis direction per base (up, down, left, right), so the distance is exact; no wrap-around (a direction that would push lesion pixels off the image is redrawn) | all |
  | P3 shift 5 px | as P2, same direction; reported only | all |
  | P4 rotate | ±5° (random sign) about the centroid of all lesion pixels; bilinear on the float mask, re-binarized at 0.5. Chosen over the image centre, which also moves peripheral lesions (≈ 4.4 px at 50 px from the centre) and so mixes rotation with the shift P2/P3 already test | all |
  | P5-erode, P5-dilate | 1 px with the 3×3 cross (the smallest 1 px change); two sub-variants. Gated only on planted masks that keep every base lesion at ≥ the bound; the others are reported as "lesion destroyed", and recall is shown by lesion size | all |
  | P6-remove | remove one random informative lesion, with its strays | ≥ 2 informative lesions |
  | P6-add | add one random informative H-eval lesion at its own pixel position, inside the base's brain mask and ≥ 3 background px from every base lesion (so it stays a separate lesion); redrawn up to a fixed number of tries, else skipped and reported | all |

- **Strata of a planted mask:** by its own lesion counts (E1: lesions ≥ 4 px; E2: informative
  lesions), as for any query; P6 can move a base between strata.
- **Keys:** `<variant>__<patient_id>_z<z>`, e.g. `P2__Mets_005_z127` (no "/", as `memtk.ids`
  requires); lesion IDs `<key>/L<n>` through `memtk.ids.lesion_id`.
- **Provenance:** every planted lesion gets the ID of the base lesion it comes from (largest
  overlap with the base label map put through the same transform, nearest-neighbour), or the
  H-eval lesion ID for an added lesion. E7 uses this for its correctness check.
- **Seeds:** one child generator per purpose (`np.random.SeedSequence(seed).spawn`: bases,
  directions, angles, P6 choices), so changing one draw leaves the others unchanged; logged in
  run_info.

**Decision J (set 2026-11-04 by the user): what counts as caught.** A planted mask is caught when
it is flagged *and* the source patient is the base patient:
- E1: score ≥ τ_D (its stratum) and the NN patient is the base patient.
- E2: class full with the NN patient = base patient, or class partial with the patient of the
  strongest-pair slice = base patient.

The full/partial split, `n_kept` and `all_copied` are reported per variant; that a P6 mask is not
a complete copy shows in `all_copied` = False, not in its class. Chosen over "flagged, any
patient" (a flag pointing at the wrong patient is not a detection) and over restating an
expected class per variant (P6's class depends on C, so the expectation would be arbitrary).

**Decision K (open, set at step 4 before `validate`): go/no-go criteria.** Proposal:
- P1: 100% caught by both detectors. A miss is a bug to fix in code, not a setting to tune.
- P2, P4, P5-erode, P5-dilate (gated masks): ≥ 90% caught by at least one detector, per variant.
- P6 (remove and add together; each also reported): ≥ 80% caught by E2.
- H-eval false-positive rate per stratum (point estimate; bootstrap-by-patient interval
  reported): E1 copy ≤ 7%, E2 full ≤ 7%, E2 flagged (full or partial) ≤ 12% (H-cal at chance
  7.4% / 8.7%).
- P3: reported only.
- To weigh at K: P5-dilate may miss small lesions in both detectors by construction (see
  Context). The `plant` review shows the area change of every planted lesion, so K can state the
  P5 criterion by lesion-size bin if needed, before any score exists.

## Definitions (fixed in code and docs)

- **Planted set:** the variants of the base masks, label maps after attach-or-drop. **Reference
  set:** T (which contains the bases themselves). **Negatives:** every H-eval slice.
- **Scorable:** E1 needs ≥ 1 lesion ≥ 4 px; E2 needs ≥ 1 informative lesion (≥ 12 px).
  Unscorable planted masks are counted, not scored.
- **Caught:** decision J. **Source patient:** E1: the NN patient; E2: the NN patient (full) or
  the strongest-pair slice's patient (partial). Also recorded: NN = the base slice itself, or
  another slice of the base patient.
- **Recall** of a variant: caught / scorable planted masks, per detector, by stratum and by base
  lesion-size bin, with a bootstrap-by-patient interval (resample the 20 base patients).
  **Per-patient recall:** base patients with all their planted masks of that variant caught.
- **Gated:** the planted masks the criteria count (P5: those keeping every base lesion at ≥ the
  bound; all others: every scorable mask).
- **False-positive rate:** the share of H-eval slices flagged, per class (E1 copy; E2 full,
  partial, flagged) and stratum, with a bootstrap-by-patient interval (26 patients).
- **RQ2 evidence:** planted masks caught by E2 and missed by E1, per variant.

## Library modules (building blocks, no knowledge of stages)

- **`memtk/sets.py` (step 1, moved, not new):** `SliceSet`, `load_set`, `decision_e` from
  `memtk/e3/common.py`, now needed by E3, E4 and E5. E3 imports them from here; E3 `check`
  (16/16) and pytest are re-run to confirm nothing changed.
- **`memtk/calibration.py` (step 1, additions):**
  - `frozen_settings(frozen)`: keywords for `wholemask.match` and `lesionmatch.match` (top-k,
    shift range, bound, σ and prefilter in px, candidate cut) read from the frozen file;
  - `e1_classes(e1, frozen, n_lesions)` and `e2_classes(lesion_match, frozen)`: one row per query
    with stratum, its thresholds and class under the frozen stratified thresholds (reused by
    E5). Built on `frozen_threshold`, `stratum`, `LesionMatch.summary`.
- **`memtk/planted.py` (step 2):** `shift`, `shift_direction`, `rotate` / `rotate_labels`,
  `morph` (erode / dilate), `add_lesion` (pure functions on masks); `variants_of` (every variant
  of one base: labelled with attach-or-drop, with parameters, sources and the P5 survivor flag;
  variants that cannot be made are returned with a reason); `choose_bases` (quotas, z spacing,
  seeded; a patient who cannot meet the quota is skipped and the next drawn patient used);
  `provenance` (planted lesion → source lesion); `planted_key`. P6-remove needs no function of
  its own (the mask without one label).

## Stages: `scripts/e4_validate.py` → `src/memtk/e4/`

The script is only the command line (as `e3_calibrate.py`); stages live in the stage package,
one module per topic (`plant`, `validate`, `check`, plus `common` for loading the frozen file,
T, H-eval and the planted set).

1. **`plant`** (needs I; step 3). No detector runs.
   - Bases, then every variant, then attach-or-drop and provenance.
   - `<output_root>/<dataset>/e4/`: `planted_labels.npz` (`io.LabelStackWriter`), `planted.csv`
     (one row per planted mask: key, base key, base patient, variant, parameters, lesion counts,
     E1 and E2 strata, all base lesions kept at ≥ the bound), `planted_lesions.csv` (one row per
     planted lesion: ID, source lesion ID, area before and after).
   - `results/<dataset>/e4/plant/`: `bases.csv`, `variant_summary.csv` (masks per variant,
     skipped, unscorable, stratum changes, lesions destroyed, area change by size bin), the
     figures below, `review.txt` (counts, and what to look at before setting K).
2. **`validate`** (needs K; step 5).
   - E1 and E2 on the planted set and on H-eval against T, settings and thresholds from the
     frozen file.
   - `<output_root>/<dataset>/e4/`: E1 tables (CSV) and E2 tables (queries and lesions CSV,
     slice and lesion pairs Parquet) for the planted set and for H-eval. The H-eval tables are
     E5's baseline, so E5 does not recompute them.
   - `results/<dataset>/e4/validate/`: `validation.csv` (variant × detector: planted, scorable,
     gated, caught, flagged, full / partial, source-patient accuracy, NN = base slice,
     `all_copied`, by stratum and lesion-size bin, per-patient recall, intervals);
     `fp_heval.csv`; `rq2_planted.csv`; `gate.csv` (each criterion: value, limit, pass); the
     figures below; `review.txt` ending in PASS or FAIL.
   - **If it fails:** nothing changes in E4. The user decides which detector setting to change
     (top-k first for a missed shifted copy, then shift range or σ); E3 `calibrate` and `freeze`
     are re-run on H-cal; `validate` runs again. The planted set stays fixed (it does not depend
     on detector settings). Every gate run is logged in `docs/process-log.md` with its frozen-file
     timestamp. S is never looked at.
3. **`check`** (step 6). Exit code 1 on any failure:
   - the frozen file is unchanged since the run (hash in run_info), and the detector values in
     the config equal the frozen ones;
   - the planted set is reproduced from the seed; every P1 label map equals its base map;
   - the gate table is reproduced from the stored tables;
   - the run logs of every E4 stage list only T, H-eval and the planted set as inputs (no S);
   - pytest passes.

## Figures (PNG, drawn with `memtk.plotting`)

Added 2026-11-04 at the user's request, for seeing what the variants do and for debugging results.

- **`plant/variants_grid.png`:** one row per base (about 12 bases, seeded, single and multi
  mixed), columns original, P1, P2, P3, P4, P5-erode, P5-dilate, P6-remove, P6-add. Each tile is
  a zoomed crop around the lesions on the base BRAVO slice, the original outline in one colour
  and the planted outline in another, titled with the variant's parameters (direction, angle,
  lesion removed or added).
- **`plant/variant_<P>.png`**, one per variant: about 20 bases, original next to planted (full
  128×128 slice with a zoomed inset), lesions coloured per lesion (`LESION_COLORS`) so the
  provenance can be checked by eye.
- **`plant/area_change.png`:** area after vs before for every planted lesion of P4, P5-erode and
  P5-dilate, by base lesion size, with the 12 px bound marked: which lesions the morphology
  destroys or doubles (evidence for K).
- **`validate/score_distributions.png`:** E1 score, E2 C(q, NN) and strongest pair per variant
  next to H-eval, with the frozen thresholds per stratum.
- **`validate/recall_by_size.png`:** recall per variant and detector by base lesion-size bin.
- **`validate/factors.png`:** shape, position and size factors of the matched lesion pairs per
  variant: where each variant loses its score (position for P3, size for P5-dilate).
- **`validate/misses.png`:** every gated planted mask that was not caught: original, planted, and
  the nearest neighbour found (patient, slice, E1 score, C, strongest pair with its three
  factors). The first place to look when a criterion fails.
- **`validate/heval_flags.png`:** a seeded sample of H-eval false positives with their nearest
  training neighbour, per class: what a chance match looks like (a report candidate).

## Tests (pytest, small hand-built inputs)

- `tests/test_planted.py`: a shift moves every pixel by exactly the given offset and refuses to
  push pixels off the image; rotation by 0° is the identity, by ±5° keeps the area within a
  tolerance and the centroid within 0.5 px; cross erosion and dilation of a known square;
  removing a lesion leaves the others identical; adding a lesion respects the gap and the brain
  mask and is skipped when it cannot; provenance maps each planted lesion to its source; base
  quotas and z spacing on a toy table; repeatable with a seed.
- `tests/test_calibration.py` (additions): `frozen_settings` on a small frozen dict;
  `e1_classes` and `e2_classes` give each query its stratum's thresholds.

## Verification (every step)

- Each stage prints its summary, writes a run log and refuses to overwrite without
  `--overwrite`; `validate` refuses to run while K is `null`.
- pytest passes; E3 `check` still passes after step 1.
- The user views the `plant` review and figures before setting K.
- Before E5: `gate.csv` records PASS for every criterion, and `check` passes.

## Carried forward (to fill in when the plan closes)

Known already:
- E5 reads `gate.csv` (refuses to run on FAIL) and the H-eval detector tables of `validate`.
- E7 reads `planted.csv` and `planted_lesions.csv` for its correctness check (source patient
  ≥ 90% of planted copies, correct lesions ≥ 80% of P6).
