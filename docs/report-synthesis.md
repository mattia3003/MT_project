# Report synthesis: preliminary project

*Written 2026-11-16, after E8 closed. Status: a plan for writing the report, not report text.*

This file maps the whole project onto the report's sections. For each subsection it says:
- what to cover and why the reader needs it;
- which maths and procedure must be stated;
- which decisions to name in the text and which to leave to the settings appendix;
- which figures and tables to use (main text, appendix or left out, and whether they need a compact
  redraw).

It does not contain finished paragraphs. The plain-language explanation of every step, for
understanding and oral defence, is in `docs/project-walkthrough.md`.

---

## 0. How to use this file

**Audience.** The author, and Claude (in this repo or in Claude chat elsewhere) when helping to write
the report. Read section 1 first (the story), then the section of the report being written. The
numbers here are for orientation: every number names its source file, and when writing, take the
number from that file.

**The project in ten lines.**
- A previous master thesis built a two-stage pipeline of pixel-space denoising diffusion models
  (DDPMs) at 128×128. Stage 1 generates brain-metastasis masks; stage 2 generates a BRAVO MRI slice
  from the mask. It produced 37,624 synthetic (mask, MRI) pairs.
- This project asks whether the mask generator memorized patient-specific lesion configurations, and
  whether that can be detected and explained from the outputs alone (no model weights).
- Two detectors compare each synthetic mask with the 3,823 training lesion slices (T): whole-mask Dice
  with a small shift search (E1), and lesion-by-lesion matching with a configuration score (E2).
- Their thresholds are set at chance level on 25 held-out test patients (H-cal) and frozen. They are
  proven on planted copies, with the false-positive rate checked on 26 other held-out patients
  (H-eval).
- The synthetic set S is then scored once, with every analysis choice fixed beforehand. The results
  are explained per patient (E6) and per match (E7), and the paired synthetic MRIs are checked (E8).

**Where things are.**

| What | Where |
| --- | --- |
| Research questions, hypotheses, criteria, master thesis | `docs/thesis-plan.md` |
| Experiment procedures E0–E8, schedule | `docs/experiment-plan.md` |
| Every decision with its evidence, every finding and limitation, with sources | `docs/report-notes.md` (sections 1–6) |
| How each experiment was built (decisions with letters A–AE) | `docs/e0b-plan.md`, `docs/e1-e3-plan.md`, `docs/e4-plan.md`, `docs/e5-plan.md`, `docs/e6-e7-plan.md`, `docs/e8-plan.md` (all closed) |
| How the work was done (re-runs, bugs, replaced numbers): not report material | `docs/process-log.md` |
| Every file and column | `docs/data-dictionary.md` |
| Code overview, modules | `docs/architecture.md`, `docs/module-reference.md` |
| Literature and the novelty argument | `docs/literature-timeline.md`, `docs/thesis-plan.md` ("Has it been done?") |
| All settings and decision values | `configs/default.yaml`, `configs/frozen/e3_stanford_brainmets.yaml`, `configs/frozen/e5_stanford_brainmets.yaml` |
| Tables, figures, review texts, check reports | `results/stanford_brainmets/<experiment>/` (real data, mock sets), `results/amibekk_v1/<experiment>/` (S) |

**Rules for writing from this file.**
- Every number comes from its source file. If this file and a source disagree, the source wins; flag
  the mismatch.
- E5's analysis is locked. Anything not in `audit:` or `docs/e5-plan.md` is described as exploratory.
- Patients are named by their dataset ID (`Mets_xxx`). Data use, including showing real masks, is
  authorized.

**Confirmed fact (previous author, 2026-11-16).** Both models were trained on the entire train set
(105 patients). No validation patients were held out from it, and the 51 test patients are the only
data neither model saw. Stage 1 saw lesion slices only. So T is everything the mask generator could
have memorized, and H-cal and H-eval are truly unseen. This replaces the earlier "stage-1 validation
set unknown" limitation in report-notes section 4.

---

## 1. The story (the spine every section serves)

1. **The risk.** Synthetic medical data is shared as privacy-safe. A mask generator could still
   reproduce a training patient's lesion configuration (number, position, size and shape of
   metastases), which identifies that patient in its own right. No published audit tests the mask
   stage of a mask→image cascade.
2. **The approach.** Use outputs only. Compare synthetic masks with training masks, and judge the
   similarity against *chance*: how close held-out patients' masks, which the model never saw, come
   to the same training set.
3. **The safeguards.**
   - Both detectors were calibrated on H-cal and frozen before S was touched.
   - The detectors were then shown to catch planted copies (E4 gate).
   - S was scored once, with every analysis choice fixed beforehand and rehearsed on mock sets.
4. **Main result (RQ1).** Synthetic masks are closer to training masks than held-out masks are, but only
   for **multi-lesion** masks: E1 copy 15.8% against 7.4% on H-eval (+8.5 points [+4.3, +11.1]), and
   E2 full 5.9% against 2.9% (+3.0 [+0.4, +4.6]). Single-lesion masks are at chance. *Exploratory
   (2026-11-16):* most of the E1 excess comes from S's lesion-count mix. Masks with one informative
   lesion plus small extras are twice as common in S and are judged against the lower multi
   threshold. Re-weighted, the E1 excess is +2.9 [−0.4, +6.7]; in masks with ≥ 2 informative
   lesions it is +5.2 [+2.8, +6.9], in line with E2.
5. **What the excess is (RQ3a).** E6 reads the multi-lesion excess as **typical lesion placement**, not
   copying of particular patients: flags fit their best patient *less* uniquely than chance flags do.
   E7 finds **no exact lesion-level copy**; multi-lesion flags resemble a training slice as a whole
   ("deformed" layouts).
6. **Duplication (H1).** Lesions that span many adjacent training slices are matched more often than
   held-out chance predicts. This is the duplication pattern known to drive memorization.
7. **Not supported:**
   - H2: lesion-level matching does not find more than whole-mask Dice on S, though it does on planted
     partial copies;
   - H3: exposure is unrelated to patient rarity.
8. **Explanations (RQ3b).** The match reports name the true source patient and lesions of planted
   copies (correctness criteria pass), and the lesion they rank first drives the match (faithfulness,
   deletion test).
9. **Images (RQ4, exploratory).** Synthetic MRIs of the strongest resemblances are not closer to the
   source slice than chance ("not supported").
10. **The bounds.** All of this holds within stated detection limits: 1 px of boundary noise on small
    lesions defeats both detectors, lesions under 12 px carry no lesion-level evidence, and the chance
    baseline rests on about 25 patients per held-out half. So the result is a validated *negative*
    for verbatim patient copying, not a certificate of privacy.

---

## 2. Section-by-section plan

Format per subsection: **Cover** (in order) · **Why** · **Maths / procedure** · **Name in text**
(decisions; everything else goes to Appendix A) · **Figures / tables** · **Sources**.

**Final files (2026-11-16).** Every figure and table below is drawn by `scripts/r_report_figures.py`
into `results/amibekk_v1/report/` (`method/`, `results/`, `appendix/`: PNG at 300 dpi and PDF, cropped
to content; `tables/`: LaTeX). The bullets below were written as the plan. Where they still name an
experiment's original figure, use the drawn file listed here. Each folder's `figures.csv` (or
`tables.csv`) gives the content and the source files of every item.

| Section | Figures | Tables |
| --- | --- | --- |
| 2.1 | (pipeline and flow: the author's own) | `tables/T1_data_sets` |
| 2.3 | `method/2-3_preprocessing_qc`; App. B1-B6 | — |
| 2.4 | `method/2-4_e1_shift_search` | — |
| 2.5 | `method/2-5_e2_lesion_pairs` | — |
| 2.6 | App. B8 | — |
| 2.7 | `method/2-7_planted_variants`; App. C1 | (criteria in `tables/T4b_gate`) |
| 2.8 | App. B7 | App. `tables/F_rehearsals` |
| 2.10 | App. D7-D9 (rehearsal margins) | (the reading rules: a small table to write by hand) |
| 2.11 | `method/2-11_match_report` | — |
| 2.13 | — | App. `tables/F_checks`, `tables/F_software` |
| 3.1 | `results/3-1_lesions_synthetic_vs_real`; App. B2, B6 | `tables/T2_description` |
| 3.2 | `results/3-2_chance_rates`; App. B8, B9, C4 | `tables/T3a_thresholds`, `tables/T3b_chance_rates` |
| 3.3 | `results/3-3_recall_by_size`; App. C2-C4 | `tables/T4_validation`, `tables/T4b_gate` |
| 3.4 | `results/3-4_score_distributions_multi`; App. D1-D3 | `tables/T5_main_audit`; App. `tables/D_h2`, `tables/D_rates_by_size` |
| 3.5 | `results/3-5a_exposure_margins`, `results/3-5b_h1_rates`; App. D4-D6 | `tables/T6_exposure`, `tables/T6b_h1_h3` |
| 3.6 | `results/3-6_report50_profile`; App. E1-E4 | `tables/T7a_correctness`, `tables/T7b_deletion` |
| 3.7 | `results/3-7_ssim_lesion`; App. E5-E6 | `tables/T8_paired` |
| App. A | — | `tables/A_decisions` |

Final outline (Method numbering as decided 2026-11-16):
- 1 Introduction 1.1–1.4.
- 2 Method:
  - 2.1 Data · 2.2 Pipeline under study · 2.3 Preprocessing and mask QC (E0)
  - 2.4 Whole-mask detector (E1) · 2.5 Lesion-level detector (E2) · 2.6 Calibration (E3) · 2.7
    Validation with planted copies (E4)
  - 2.8 Statistical framework and pre-registration · 2.9 Main audit (E5) · 2.10 Exposure, rarity
    and duplication (E6) · 2.11 Match reports and their evaluation (E7) · 2.12 Paired-MRI check (E8)
  - 2.13 Implementation and reproducibility.
- 3 Results 3.1–3.7.
- 4 Analysis 4.1–4.6.
- 5 Discussion 5.1–5.5.
- Appendix A–F.

Method shows only what the method is and how it is applied (method demos on planted or real
data are fine). Every result on H-cal, H-eval or S goes in Results.

### 1 Introduction (already written: checklist)

- **1.1 Motivation:**
  - synthetic data shared as privacy-safe;
  - diffusion models memorize (Carlini et al. 2023; Somepalli et al. 2022/2023), also in medical
    imaging (Akbar et al. 2023; Dar et al. 2023/2024);
  - a lesion map identifies a patient in its own right;
  - cascades are judged on realism and utility, not privacy.
- **1.2 Related work:**
  - copy detection by nearest neighbour and calibration against held-out data (Dar et al. 2024: copy =
    above the 95th percentile of train-to-held-out similarity, the rule reused here);
  - duplication drives copying (Somepalli et al. 2023), and long-tail memorization (Feldman and Zhang
    2020) as the reasons for H1 and H3;
  - local and region memorization (Bright Ending, FB-Mem) and xAI for memorization (Wen et al., NeMo),
    which read cross-attention or model internals and so do not apply here (no weights, and no
    cross-attention in either model);
  - the closest cascade study, Edirisooriya et al. 2026, which tests only the image stage;
  - re-identification from medical images (Packhäuser et al. 2022).
- **1.3 Research questions and scope:**
  - the overarching question, RQ1, RQ2, RQ3a, RQ3b and RQ4 (optional), H1–H4 and the RQ3b criteria,
    worded as in `docs/thesis-plan.md`;
  - out of scope: training, membership inference, model internals, mitigation, recombinations, and
    image analyses beyond RQ4.
- **1.4 Contributions:**
  1. two output-only detectors of mask copying, whole-mask and lesion-level, calibrated against
     held-out patients;
  2. a planted-copy validation with go/no-go criteria fixed beforehand and a measured detection limit;
  3. a pre-registered audit of 35,920 synthetic masks of a real pipeline;
  4. patient- and lesion-level explanations (match reports) evaluated for correctness and faithfulness;
  5. a reusable toolkit for the master thesis.

  Report structure: one paragraph.
- Sources: `docs/thesis-plan.md`, `docs/literature-timeline.md`. Re-check novelty claims against
  MICCAI/MIDL 2026 before submission (thesis-plan caution).

### 2.1 Data

- **Cover:**
  - BrainMetShare (Grøvik et al.), 156 patients, standard split 105 train / 51 test by patient,
    confirmed by the previous author;
  - 3D BRAVO and binary segmentation volumes;
  - the sets used here: T = the 3,823 lesion slices of the 105 training patients; the 51 test patients
    split into H-cal (25 patients, 992 lesion slices) and H-eval (26, 1,094) by matched pairs on
    lesion-slice count (seed 2026);
  - only lesion slices are compared everywhere (stage 1 saw only lesion slices);
  - S: 37,624 synthetic (mask, MRI) pairs in six naming schemes, most likely generation rounds that
    look alike; the sample ID is the full file stem.
- **Why:** the reader must know what "training" and "held-out" mean before any comparison. Two held-out
  halves give one half to set thresholds (H-cal) and an independent half to check them (H-eval).
- **Name:** the matched-pair split, and why: lesion burden ranges from 3 to 119 slices per patient.
- **Table 1 (main), data sets:** per set, patients, lesion slices, and role (T reference; H-cal
  calibration; H-eval false-positive check and baseline; S audited, with S main / noisy counts). Source:
  `splits/stanford_brainmets_seed2026.csv`, report-notes section 1.
- **Figure (optional, not drawn):** data overview grid (random T / H-cal / H-eval slices and synthetic
  samples), `results/stanford_brainmets/e0a/check_grid_amibekk_v1.png` at 90 dpi, if wanted.
- **Sources:** report-notes section 1 ("Real dataset", "Synthetic pool"), section 2 (H-cal/H-eval split).

### 2.2 Pipeline under study

- **Cover:**
  - two pixel-space DDPMs (MONAI `DiffusionModelUNet`, channels 128/256/256, self-attention at the
    lower two levels, `DiffusionInferer`); *not* latent diffusion;
  - stage 1 is unconditional (1 channel, masks); stage 2 takes the mask as a second input channel
    (concatenation), and only the image channel is noised; no cross-attention anywhere;
  - training: the whole train set (confirmed 2026-11-16); stage 1 on lesion slices, stage 2 on
    (nearly) all slices with their own masks;
  - training transform (per-volume min–max, area resize to 128×128, mask > 0.01); no augmentation;
    no seeds; checkpoints at epoch 149 of 200;
  - sampling: one shared initial noise tensor for every MRI; the repo's filters, of which only the
    pixel-sum filter holds on the data (decision E);
  - the repo is not exactly the code that ran (previous author), so where they differ the data wins.
- **Why:** the threat model. Stage 1 never sees a brain, so a memorized mask reappears at the training
  mask's *pixel* positions (this motivates decision F and E8's alignment). Stage 2 uses the mask as a
  condition, which motivates RQ4.
- **Figure:** the existing pipeline figure (section 1.3).
- **Sources:** `CLAUDE.md` "Models", report-notes section 4 (repo vs data, sampling filters), section 2
  (decision E).

### 2.3 Preprocessing and mask quality control (E0)

- **Cover:**
  1. *Model space.*
     - All comparisons happen in the 128×128 space the models saw.
     - Real volumes go through the exact training transform, built from the 3D volumes because the
       min–max is per volume.
     - Verified bit-identical (a fresh MONAI run and an independent re-implementation; 32 checks).
     - Consequence: the 0.01 threshold enlarges lesions (median 1.5× per patient; most for small
       lesions). This affects real and synthetic masks alike.
  2. *Brain mask* (decision A): threshold 0.02, keep every region ≥ 30 px, fill holes. Chosen because
     plain Otsu cuts dim edges and drops separate brain pieces. A failed MRI = no brain region
     (< 20 px of tissue).
  3. *Lesions* (decisions B, C):
     - a lesion is an 8-connected component ≥ 4 px;
     - smaller components are specks: attached as "strays" to a lesion within 2 px, otherwise
       dropped;
     - two lesions are never merged;
     - the same rule applies to real and synthetic masks;
     - masks as saved (with specks) are kept for anything relating mask and MRI.
  4. *QC classes* (decision D): clean (0 isolated specks), minor (1–9), noisy (> 9), no lesion. The
     MRI-failure flag is kept separate (a failed MRI does not exclude its mask).
     - **S main = clean + minor = 35,920 masks**; noisy masks (1,241) are reported separately.
     - The edge check (> 50% of a lesion outside the brain) only flags.
     - The old QC list (2,243 masks) is explained by two coarse rules and replaced.
- **Why:** the comparison is only fair if real and synthetic masks pass through the same definitions.
  Synthetic masks carry speck noise that real masks do not (62% of synthetic components are 1 px).
- **Name:** A–D in one sentence each, each with its data-driven reason:
  - B: the largest m with ≤ 1% of T components below it;
  - C: a speck halo at gaps 1–2, then flat noise;
  - D: the break in masks per speck count after 9.
  - E (the sampling filters) goes in 2.2.
- **Maths:** none beyond the definitions. Define *informative lesion* here as a forward reference
  (≥ 12 px, set in E3).
- **Figure 3 (main; drawn: `method/2-3_preprocessing_qc`):**
  - (a) `Mets_120/z40` at native resolution;
  - (b) the same slice in model space with E0's brain outline, brain box and numbered lesions;
  - (c) a zoom on one lesion's resized mask: the 0.01 threshold (15 px) against plain downscaling
    (5 px);
  - (d-g) one synthetic mask per QC class after attach-or-drop.
- **Appendix B (drawn):** B1 native vs model space, B3 component sizes (decision B), B4 specks by gap
  (decision C), B5 brain-mask methods (decision A), B6 QC classes.
- **Sources:** report-notes section 1 (conversion), section 2 (decisions A–D, QC classes), section 3
  (specks); `results/amibekk_v1/e0b/qc/summary.txt`.

### 2.4 Whole-mask detector (E1)

- **Cover:**
  - the score per synthetic mask q against every training slice t (cleaned masks, specks removed);
  - shortlist: the 20 best training masks by plain Dice plus the 20 nearest by the centroid of the
    largest lesion;
  - best Dice over shifts of ±3 px (49 positions, zero padding, no wrap);
  - nearest neighbour (NN) = the best slice; the runner-up patient and the margin;
  - strata single / multi by the number of lesions ≥ 4 px;
  - copy class if the score ≥ τ_D of its stratum.
- **Why:** the plain, recognisable measure ("is the whole mask a copy?"). The shift search covers
  copies moved by a few pixels.
- **Maths:** D(q, t) = max over |Δr|, |Δc| ≤ 3 of 2|q ∩ t_(Δr,Δc)| / (|q| + |t|). Ties go to the
  smallest shift, then the higher plain Dice. State that the shortlist is a speed-up whose effect E4
  measures through P2/P3 recall; unlike E2's shortcuts, it is not proven exact.
- **Sources:** `docs/e1-e3-plan.md`, `docs/experiment-plan.md` E1, `src/memtk/wholemask.py`.

### 2.5 Lesion-level detector (E2)

- **Cover:**
  1. Only *informative* lesions (≥ 12 px, from E3) take part.
  2. The lesion-pair score s(a, b) = shape × position × size:
     - shape = IoU of the two lesions' crops centred on their (rounded) centroids;
     - position = exp(−d² / 2σ²), with d the centroid distance in pixels and σ = 6.4 px (0.05 × 128);
     - size = smaller area / larger area.
  3. The slice-pair (configuration) score:
     - Hungarian one-to-one pairing of the lesions of q and t on cost 1 − s;
     - keep pairs with s ≥ 0.1;
     - C(q, t) = Σ kept s / max(n_q, n_t);
     - NN = the slice with the highest C.
  4. Classes:
     - *full* if C(q, NN) ≥ τ_C (the configuration is closer than chance);
     - *partial* if not full but the strongest lesion pair over all training slices ≥ τ_L;
     - *novel* otherwise.
     - Also stored: the strongest pair and its slice, the number of NN lesions passing τ_L, and
       `all_copied`.
  5. Two exact shortcuts:
     - a position prefilter at 25.6 px;
     - only training slices sharing a pair with s ≥ 0.1 are scored.

     Verified against brute force, 0 differences.
- **Why:** RQ2. A *partial* copy (some lesions copied, others new) can lower whole-mask Dice below its
  threshold while a lesion-level match still finds it. The factorised score also makes every match
  explainable lesion by lesion (used in E7).
- **Name:**
  - *F, pixel coordinates* (not brain-relative): stage 1 makes masks without a brain, and synthetic
    brains are about 0.4× the area of real ones, so a brain-relative position would measure stage
    2's drawing.
  - *Equal lesion weights in C* (not area-weighted): a big copied lesion among new small ones is caught
    by E1 and as "partial"; area weighting would turn C into a variant of Dice and blur RQ2.
- **Figures 4a and 4b (main; drawn, split after review):**
  - `method/2-4_e1_shift_search` (cited in 2.4): a planted copy shifted by 2 px, with Dice 0.78 as
    generated, the Dice of every shift, and 1.00 after the best shift;
  - `method/2-5_e2_lesion_pairs` (2.5): a planted copy rotated by 5°, its lesion pairs with shape ×
    position × size = s (0.80, 0.85, 0.58), and C = 0.74.

  Both use planted copies of the same base slice (`Mets_271/z122`).
- **Sources:** report-notes section 2 (decision F, "C weights every lesion equally", "E2's speed-ups are
  exact"); `docs/e1-e3-plan.md`; `src/memtk/lesionmatch.py`.

### 2.6 Calibration (E3)

- **Cover:**
  1. Chance = the scores of H-cal slices (never trained on) against T.
  2. *Informative bound* (G): for each H-cal lesion, its best lesion score against all T lesions.
     - The provisional τ_L is the 95th percentile of those scores for lesions ≥ 10 px (0.77).
     - The chance rate is tabulated per area bin; the bound is the smallest area from which every bin
       stays ≤ 10%, giving 12 px.
     - 25 px (the 5% cut) is kept as a sensitivity. Why not 25 px: the 15–24 px bins sit at 5.1–5.8%
       (noise decides), and S main would lose 59% of its masks from the lesion level.
  3. *Thresholds* (H): τ_D, τ_L and τ_C = 95th percentiles of H-cal's NN scores (strongest pair for
     τ_L), separately for single- and multi-lesion slices.
     - Each has a bootstrap-by-patient interval (2,000 draws).
     - Why stratified: a pooled threshold would flag the strata very unequally (pooled τ_D flags 7.3% of
       single and 0% of multi slices).
  4. *Freeze:* the values, bound and settings are written to `configs/frozen/e3_stanford_brainmets.yaml`
     before any detector touched S; E4 and E5 read only that file.
- **Name, as a stated deviation from the plan:** *C was redefined.*
  - The planned C kept only pairs with s ≥ τ_L, so it was 0 for about 95% of H-cal and τ_C fell to
    0.02, which is unusable.
  - The redefined C counts every pair with s ≥ 0.1, independent of τ_L.
  - Decided on H-cal, before S was touched.
- **Maths:** τ = Q_0.95 of {score(q, NN(q)) : q ∈ H-cal stratum}; the patient-bootstrap interval
  (see 2.8).
- **Results go to 3.2** (the chance-rate curve and the threshold table).
- **Sources:** report-notes section 2 (decisions G, H, C redefinition, freeze),
  `results/stanford_brainmets/e3/bound_review/review.txt`, `e3/calibrate/review.txt`.

### 2.7 Validation with planted copies (E4)

- **Cover:**
  1. *Bases* (I): 20 T patients × 5 lesion slices, at least 2 apart in z. Ten bases come from patients
     with ≥ 3 multi-lesion slices (3 multi + 2 single), so P6-remove has enough material, and ten from
     the rest.
  2. *Variants:*
     - P1 exact;
     - P2 shift 2 px; P3 shift 5 px;
     - P4 rotate ±5° about the lesion-pixel centroid;
     - P5 erode / dilate 1 px (3×3 cross);
     - P6 remove one informative lesion / add one H-eval lesion.
     - Altogether 737 masks.
  3. *Gate rule:* a mask counts only if every informative base lesion stays ≥ 12 px (all variants but
     P6-remove).
  4. *Caught* (J) = flagged (E1 copy; E2 full or partial) **and** the named source patient is the base
     patient.
  5. *Go/no-go criteria* (K), fixed after planting and before any detector ran:
     - P1 100% by both detectors;
     - P2 and P4 ≥ 90% by either;
     - P6 ≥ 80% lesion-level;
     - H-eval false positives per stratum: E1 copy ≤ 7%, E2 full ≤ 7%, E2 flagged ≤ 12%;
     - P3 and P5 reported only.
- **Why:** without proof that the detectors find known copies, a negative audit means nothing. This is
  the first question a reviewer asks.
- **Name:** why P5 is "reported only". For an eroded lesion with area ratio r, s ≈ r² and Dice ≈
  2r/(1+r), which is below the chance thresholds even for lesions ≥ 80 px. No permitted setting could
  pass it, so it is a *stated detection limit*.
- **Figure 5 (main; drawn: `method/2-7_planted_variants`):**
  - (a) every variant of `Mets_271/z122`;
  - (b) the same 1 px of erosion and dilation on a 14 px and a 90 px lesion (14 → 3 / 29 px,
    90 → 59 / 125 px).
- **Table:** the criteria are in `tables/T4b_gate` (Results 3.3); Method can list them in the text.
- **Appendix C (drawn):** C1 lesion area before and after P4 and P5 (the area-change chart).
- **Sources:** report-notes section 2 (decisions I, J, K), `docs/e4-plan.md`.

### 2.8 Statistical framework and pre-registration (shared by E3–E8)

- **Cover:**
  1. *Units.*
     - Real slices are clustered by patient: adjacent slices are near-duplicates, with median Dice
       0.80 between neighbours (report-notes section 3).
     - The patient is therefore the unit of resampling for every real set.
     - Synthetic masks are independent samples.
     - Copies are counted per source patient and per *chained group* (N): the source slices of one
       patient joined while consecutive z differ by ≤ 2.
  2. *Intervals:* a two-level bootstrap.
     - Each draw resamples H-eval (or H-cal) patients with replacement and S masks independently, and
       recomputes the statistic.
     - Percentile 95% interval; 2,000 draws.
     - Implemented exactly as a multinomial over cluster types (cheap for 36k masks).
  3. *Excess and the primary test* (E5):
     - excess = rate_S − rate_H-eval per class and stratum;
     - one-sided bootstrap p = (1 + #{draws with excess ≤ 0}) / (draws + 1);
     - Holm adjustment over the primary family of four cells.
  4. *AUC* = P(score of A > score of B), ties count ½, with the same two-level bootstrap.
     - The Mann–Whitney p is reported only as secondary, because it treats slices as independent.
     - On the H-cal null mock it gave p = 0.004 where the patient-bootstrap AUC interval [0.40, 0.74]
       covered 0.5. This is the strongest single argument for the patient bootstrap.
  5. *Pre-registration and the lock:*
     - every analysis choice of E5–E8 was written into the config (`audit:`, `exposure:`, `reports:`,
       `paired:`) and the plan before the data it concerns were read;
     - E5 scored S once and wrote a completion marker holding the hashes of `audit:` and the frozen
       files; later stages refuse to run if any of them changed;
     - outcome readings are applied mechanically from pre-stated rules;
     - anything added afterwards is labelled exploratory.
  6. *Rehearsal on mock sets:* the exact analysis code ran on sets with known answers before it read
     S. The mocks:
     - a null mock (H-cal, which cannot contain memorization);
     - a positive mock (H-cal plus planted copies);
     - two placement mocks (H-cal lesions moved to typical T positions; E6);
     - the E8 positive mock (planted copies with the source patient's BRAVO two slices away).
- **Why:**
  - Chance matches are common (about 7–10% of any mask set matches a training slice at the frozen
    thresholds), so only *excess over held-out* is evidence.
  - Slices of one patient are not independent.
  - Fixing the analysis first protects the result from being tuned towards an answer.
- **Name:** decision M (the primary family with Holm); N (chained groups); the lock.
- **Table (appendix F; drawn: `tables/F_rehearsals`):** each mock set with its required and obtained
  reading, for E5, E6 and E8. In the main text, one sentence: every mock gave the expected reading.
- **Sources:** report-notes section 2 (E5 decisions L–Q, "Intervals for S against H-eval"), section 3
  (rehearsal entries for E5, E6, E8); `src/memtk/audit.py`.

### 2.9 Main audit (E5)

- **Cover:**
  - *Sets* (L): S main (35,920; 25,917 with an informative lesion, so E2-scorable); noisy masks as
    separate rows.
  - *Primary family* (M): the excess of E1 copy and E2 full, single and multi (four cells),
    Holm-adjusted; "beyond chance" at adjusted p < 0.05.
  - *Mechanical reading:*
    - "memorization beyond chance" if any primary cell is beyond chance;
    - else "rare but real" if a near-exact copy exists;
    - else "validated negative".
  - *Secondary* (intervals only):
    - E2 partial and flagged;
    - AUCs of the scores;
    - rates by size of the largest lesion, with H-eval standardised to S's size × stratum mix (the
      size-confound check);
    - rates per naming scheme;
    - chained groups and source patients.
  - *Sensitivity on S* (O), each compared with H-eval at the same setting: 99th-percentile thresholds,
    σ 0.08, bounds 10 and 25 px, drop-only (attach gap 0).
  - *H2 at equal false-positive rate* (P):
    - (i) E2 full vs E1 copy at the frozen thresholds;
    - (ii), the main comparison: E2 flagged vs E1 at τ_D\*, the E1 threshold that flags as many H-cal
      slices as E2 flagged does (7.43% / 8.66%);
    - H2 holds if the interval of the difference of excesses (E2 − E1) lies above 0.
  - *Near-exact copies* (Q): E1 score ≥ 0.9 **and** above the highest held-out E1 score in the same
    stratum and size bin. The reason: Dice > 0.9 occurs by chance for single lesions; 8% of H-eval
    single-lesion slices exceed it.
- **Name, honestly:** the near-exact count has *no chance baseline*. S is about 17× larger than the
  held-out sets, so even without memorization a mask exceeds the maximum of n held-out scores with
  probability 1/(n+1). About 68 single-lesion masks would be expected by chance, against 35 observed.
  Found after the run; recorded, not repaired. Put it here in one sentence and in full in 5.4.
- **Sources:** report-notes section 2 (decisions L–Q), section 4 (Dice > 0.9, near-exact baseline);
  `docs/e5-plan.md`; `configs/frozen/e5_stanford_brainmets.yaml`.

### 2.10 Exposure, rarity and duplication (E6)

- **Cover:**
  1. *Exposure* (R). In each primary cell (E1 copy multi, E2 full multi):
     - o_p = the flags whose source patient is p (one source per flag, decision J);
     - π_p = the expected share: the share of H-eval masks of the stratum whose NN belongs to p,
       smoothed towards p's share w_p of T lesion slices;
     - exposure ratio r_p = o_p / (π_p · O), with an interval.
  2. *Concentration:* the dispersion D = Σ (o_p − e_p)² / e_p against a multinomial null (π redrawn
     from an H-eval patient bootstrap in each draw), and the number of patients whose ratio interval
     lies above 1.
  3. *Copying vs placement* (U). Two flag-level measures, each compared between S flags and H-eval
     chance flags by AUC:
     - the margin (how much better the best patient fits than the second-best);
     - the rarity of the source slice.

     The pre-stated reading:
     - *patient copying* = dispersion beyond its null **and** margin AUC above 0.5;
     - *typical placement* = margin AUC wholly below 0.5, **or** margin not above 0.5 and rarity AUC
       below 0.5;
     - *undetermined* otherwise.

     The "wholly below" branch was added after the placement mock and before S was read.
  4. *Rarity* (V):
     - slice rarity = 1 − the best score against other patients' slices, on each detector's scale;
     - patient rarity = the mean distance to the 5 nearest patients in a z-scored feature space
       (number of 3D lesions, total volume, median area, most lesions per slice, centroid spread);
     - **H3:** Spearman(rarity_p, r_p), with a permutation p and a bootstrap interval.
  5. *Duplication, H1* (W):
     - 3D lesions = 2D components linked between consecutive slices one-to-one (Hungarian on pixel
       IoU); 26-connectivity as the sensitivity;
     - span = the number of slices;
     - lesion credit c(ℓ) = the masks with a lesion pair ≥ τ_L to any 2D instance of ℓ, over every
       stored slice pair;
     - within size bins (12–19, 20–39, 40–79, ≥ 80 px): ρ_S = Spearman(span, c_S) and ρ_H likewise on
       H-eval;
     - **H1 holds if the bin-weighted mean of ρ_S − ρ_H has an interval above 0.**
- **Why:**
  - A raw exposure ratio above 1 does not distinguish copying from placement. S places lesions in a
    compact central area, so patients with central lesions attract matches without being copied.
  - The margin and rarity against chance flags make the distinction testable.
  - H1 needs the held-out reference: a long lesion offers more slices to match, so ρ_S > 0 also
    happens by chance (both mocks showed it).
- **Maths to state:** π_p = (n_p + m·w_p) / (N + m) with m = 52.5; r_p; D; the margin (E2: C(q, NN)
  minus the best C over other patients' slices); the H1 statistic.
- **Table (main, small):** the three readings and their rules (from decision U).
- **Sources:** `docs/e6-e7-plan.md` (decisions R, U, V, W), report-notes section 3 (E6 rehearsal).

### 2.11 Match reports and their evaluation (E7)

- **Cover:**
  1. *Report content:*
     - Panel A: the synthetic mask against its E1 NN and E2 NN, with an overlap overlay;
     - Panel B: the paired lesions with their shape, position and size factors and their contribution
       to C, plus the strongest-pair slice if it differs;
     - Panel C: the source patient and slice, its chained group and 3D lesions, the margin, the
       exposure ratio, the chance rate of its class, and whether the score is beyond the held-out
       maximum;
     - a rule-based verdict line, which says "configuration match", not "copy", when no single pair
       reaches τ_L, and names both patients when NN and strongest pair disagree.
  2. *Selection* (X), one report per chained group:
     - every near-exact and every E2 beyond-held-out mask;
     - the top flag per detector of the 5 most exposed patients;
     - 10 random S flags and 10 H-eval chance flags as contrast.
     - Total: 76 reports.
  3. *Correctness* (Z), on planted copies:
     - the named source patient = the base patient (≥ 90% for P1, P2, P4, P6);
     - the right lesions for P6 (≥ 80%);
     - P3 and P5 reported only.
  4. *Faithfulness: the deletion test* (Y). Remove the top contributor vs the smallest contributor
     and measure:
     - (1) ΔC against the source slice (paired Wilcoxon; nearly follows from the definition of C);
     - (2) whether the mask is still flagged with the same patient after a full re-match (McNemar);
     - (3) Δ whole-mask score (paired Wilcoxon), which is independent of the lesion scores.

     Faithful if (2) and (3) also favour the top contributor.
  5. *Taxonomy:*
     - classes: exact, shifted, deformed, partial, likely chance;
     - rules fixed on the planted variants, then applied unchanged to S.
- **Why:**
  - RQ3b is the xAI contribution: an explanation must name the true source (correctness) and the
    lesions it names must drive the score (faithfulness).
  - ΔC alone would pass for any additive score, hence the two extra measures.
- **Figure 6 (main; drawn: `method/2-11_match_report`):** the report's four images on the planted
  P6-add copy of `Mets_271/z122`, plus its verdict line ("configuration copy, not every lesion").
  The panel descriptions belong in the caption.
- **Sources:** `docs/e6-e7-plan.md` (decisions X, Y, Z), report-notes section 3 (E7 entries).

### 2.12 Paired-MRI check (E8, exploratory)

- **Cover:**
  1. *Groups:*
     - A = the 56 strong S cases of E7 (near-exact, E2 beyond held-out, most exposed; one per
       chained group), each with its decision-J source slice (AA);
     - B = 5 novel S masks per A case, matched on stratum and informative-lesion count, size bin of the
       largest lesion and synthetic brain-area tertile, each with its nearest training slice (AB);
     - the same design on H-eval (flagged vs novel) gives the reference.
  2. *Measure* (AC):
     - each image is z-scored with its own brain's mean and SD;
     - the synthetic image is moved by the E1 shift (no rescaling);
     - a Gaussian-weighted local SSIM map (σ 1.5 px, data range fixed at 10 z-units);
     - the map is averaged over a region inside both brains: *lesion neighbourhood* (informative
       lesions grown 8 px; primary), *whole brain* (secondary), *ring* = the neighbourhood without
       lesion pixels (sensitivity).
  3. *Test* (AD):
     - AUC of A vs B, judged against H-eval's AUC of flagged vs novel, with the two-level bootstrap;
     - mechanical reading: *supported* if the AUC interval is above 0.5 and the difference to H-eval is
       above 0; *explained by mask resemblance* if the AUC is above 0.5 but the difference is not;
       *not supported* otherwise;
     - within A: SSIM to the source vs to the best-matching slice of the runner-up patient (paired
       Wilcoxon), and a profile over z ± 5 (descriptive).
  4. *Rehearsal* (AE): an H-cal null (must not read supported) and a positive mock (must read
     supported).
- **Why:**
  - RQ4 asks whether a copied mask acts as a "key" that makes stage 2 reproduce the source tissue.
  - Overlapping masks put a bright lesion in the same place in both images, so lesion-region SSIM
    rises even without memorization. H-eval measures this mask-resemblance effect where no copying is
    possible.
  - No rescaling, because a stage-2 reproduction would sit at the mask's pixel positions (as in F).
  - Brain-size matching of B handles the size mismatch (A brains are a median 0.51× their source).
- **Maths:** SSIM(x, y) = [(2μ_xμ_y + C1)(2σ_xy + C2)] / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)], with
  local Gaussian-weighted moments, C1 = (0.01 L)² and C2 = (0.03 L)², L = 10; the z-score
  z = (x − μ_brain) / σ_brain.
- **Figure:** the measure is shown in Results figure `results/3-7_ssim_lesion` (rows b-d: a typical and
  the strongest S case, and the rehearsal positive for scale); the full example sheet is App. E5.
- **Sources:** `docs/e8-plan.md` (AA–AE), report-notes section 2 (E8 design), section 4 (E8 limits).

### 2.13 Implementation and reproducibility (about half a page)

- **Cover:**
  - the toolkit `memtk` in three layers: scripts (one command line per experiment), stage packages
    (one experiment's steps) and library modules (transforms, lesions, detectors, calibration, audit
    statistics, exposure, reports, paired SSIM), so it runs again on new models;
  - a small table: experiment → what it reads → what it writes (from `docs/architecture.md`
    section 2, shortened);
  - safeguards:
    - one config file and one seed (2026);
    - every stage writes a run log (seed, config, package versions) and refuses to overwrite;
    - review → decision → production stages (a setting chosen from data stays `null` until decided);
    - the S lock;
    - a `check` stage per experiment that reproduces every table from stored inputs (E0a 32/32, E0b
      43/43, E3 16/16, E4 16/16, E5 26/26, E6 18/18, E7 14/14, E8 17/17);
    - unit tests on hand-built inputs (pytest, 196 tests);
    - a pinned environment (Python 3.11.9, MONAI 1.6.0, PyTorch 2.6.0, …);
    - E0 re-run from scratch is byte-identical.
- **Why:** an audit is only credible if it can be re-run and if it is shown that S was not touched
  before the thresholds were frozen.
- **Not here:** module names, function lists, the folder layout. One sentence points to the repo.
- **Sources:** `docs/architecture.md`, check reports under `results/`, report-notes section 2
  ("Reproducibility setup", "Software").

### 3.1 Data and quality control

- **Cover:**
  - QC class counts: clean 32,137, minor 3,701, noisy 1,241, no lesion 393, failed MRI 152; S main
    35,920;
  - how synthetic masks differ from real ones:
    - smaller lesions (median 13 vs 19 px in T), none above about 330 px;
    - more two-lesion masks (34% vs 18%);
    - lesion centroids compact in pixels (SD 15/19 px vs 24/25 px in T) but real-like relative to the
      brain box;
    - smaller brains (median about 3,000 vs 7,765 px), the size of real end-of-head slices;
  - 2.0% of synthetic lesions lie mostly outside the brain (T: 1 of 6,101).
- **Why:** these differences shape everything later:
  - small lesions → many chance matches and the 12 px bound;
  - compact positions → the placement confound in E6;
  - small brains → pixel frame (F) and E8's design.
- **Table 2 (main), real vs synthetic description:** per set (T, H-cal, H-eval, S main), median lesion
  area (IQR), share with a lesion < 40 px, single- and two-lesion shares, most lesions per slice,
  centroid SD in pixels and relative, median brain area. Sources:
  `results/stanford_brainmets/e0b/stats/description_amibekk_v1.csv`, report-notes section 3.
- **Figure 7 (main, compact 2-panel redraw):** (a) lesion area distributions S vs T/H; (b) centroid
  density in pixels, S vs T. Sources: `e0b/stats/lesion_area_amibekk_v1.png`,
  `centroid_density_amibekk_v1.png`.
- **Appendix B:**
  - QC classes figure (`results/amibekk_v1/e0b/qc/qc_classes_amibekk_v1.png`);
  - brain size by head height (`e0b/stats/brain_by_height_amibekk_v1.png`);
  - adjacency Dice (`e0b/stats/adjacency_dice_stanford_brainmets.png`), which supports per-patient
    counting.
- **Sources:** report-notes section 2 (QC classes), section 3 (synthetic vs real entries).

### 3.2 Calibration

- **Cover:**
  - single small lesions match by chance very often: 73–93% of H-cal lesions of 5–39 px find a T
    lesion with s ≥ 0.5;
  - the chance lesion-match rate by area, and the 12 px bound;
  - the thresholds with their intervals (τ_D 0.905 / 0.729, τ_L 0.739 / 0.816, τ_C 0.719 / 0.406,
    single / multi);
  - chance rates on H-cal (E1 copy 5.1 / 5.3%, E2 full 5.1 / 5.2%, partial 2.3 / 3.5%);
  - lesions excluded at the bound.
- **Why:** the reader must see that "closer than chance" has a concrete, measured meaning, and that
  single-lesion masks need a far higher bar than multi-lesion ones.
- **Figure 8 (main):** `results/stanford_brainmets/e3/bound_review/chance_rates.png`. Check legibility;
  redraw if needed.
- **Table 3 (main), thresholds and chance rates:** threshold [interval] per detector and stratum, the
  H-cal rate and the H-eval rate (from E4) side by side. Sources: `e3/calibrate/thresholds.csv`,
  `e4/validate/fp_heval.csv`.
- **Appendix B:**
  - `e3/calibrate/score_histograms.png`;
  - `e3/bound_review/examples.png` (what chance lesion matches look like).
- **Sources:** report-notes section 2 (G, H), section 3 ("Single small lesions match by chance", "Chance
  whole-mask matches").

### 3.3 Detector validation (E4)

- **Cover:**
  - recall per variant and detector, with intervals. Either detector: P1 100%, P2 100%, P4 99%, P6
    97% → all recall criteria pass;
  - E2 alone catches what E1 misses: P3 79 of 100, P6-remove 19 of 37, P6-add 33 of 100, versus E1
    alone at most 1 per variant;
  - the detection limit: P5 recall by lesion size, from 0% at 12–19 px to 85% at ≥ 80 px;
  - H-eval false positives: three of six criteria exceed 7% by 0.1–0.4 points (E1 copy 7.2 / 7.4%,
    E2 full single 7.1%);
    - **accepted as a recorded deviation, decided after the run**, with the reasons (intervals contain
      both the limit and the H-cal rate; flags cluster in few patients; re-calibration could not help
      without leaking H-eval);
    - consequence: E5 compares S with the measured H-eval rates, not with 5%;
  - misses explained (the NN can name another patient while the strongest pair names the base).
- **Why:** the gate is what makes the later negative findings meaningful. The deviation must be stated
  openly, as decided after seeing the numbers.
- **Table 4 (main), validation:** variant (n) × E1 / E2 / either recall [interval], plus a criterion
  column with pass / fail, plus the H-eval false-positive rows. Sources: `e4/validate/validation.csv`,
  `gate.csv`, `fp_heval.csv`.
- **Figure 9 (main, compact):** recall by lesion size per variant and detector,
  `results/stanford_brainmets/e4/validate/recall_by_size.png`.
- **Appendix C:**
  - `factors.png` (where each variant loses its score);
  - `misses.png`;
  - `heval_flags.png` (what a chance match looks like, including two exact chance matches of 4 and 8
    px single lesions).
- **Sources:** report-notes section 2 (E4 gate accepted), section 3 (E4 entries), section 4 (detection
  limit).

### 3.4 Main audit (E5)

- **Cover, in this order:**
  1. *Primary family* (Holm):
     - E1 copy multi 15.8% vs 7.4%, +8.5 [+4.3, +11.1], p_holm 0.004;
     - E2 full multi 5.9% vs 2.9%, +3.0 [+0.4, +4.6], p_holm 0.036;
     - single-lesion cells at chance (E1 +1.3 [−1.4, +3.9], E2 −0.2 [−2.7, +2.2]);
     - reading: "memorization beyond chance", multi-lesion only.
  2. *Size-confound check:* H-eval standardised to S's size mix gives the same excess (+9.0, +3.0).
  3. *AUCs* (patient bootstrap):
     - multi-lesion E1 0.67 [0.56, 0.77], E2 C 0.68 [0.53, 0.78];
     - multi-lesion strongest single pair 0.38 [0.32, 0.49]: the configuration is closer, the best
       single lesion is *not*.
  4. *Partial copies* not above chance (single 3.3% vs 3.0%; multi 1.1% vs 5.5%).
  5. *H2 not supported:* E1 shows more excess than E2 in every comparison ((ii) all −9.4 points
     [−12.6, −5.8]).
  6. *Noisy masks* are flagged less than H-eval (e.g. E2 flagged single 1.7% vs 10.1%), so excluding
     them hides no copies.
  7. *Spread:* the 2,662 E1 multi-lesion flags name 96 source patients; the top 5 take 17%.
  8. *Near-exact:* 39 masks (31 groups), with the chance caveat in the same paragraph.
  9. *Exploratory, lesion-count composition (2026-11-16; after the lock, labelled so):*
     - E1 multi masks split by informative lesions (0 / 1 / 2+): S 22% / 42% / 36%, H-eval 10% /
       21% / 69%.
     - Copy rates per group: S 3.5% / 29.8% / 7.0%, H-eval 5.3% / 26.5% / 1.8%.
     - H-eval re-weighted to S's mix: 13.0%, so the excess is +2.9 [−0.4, +6.7]. The 2+ group alone:
       +5.2 [+2.8, +6.9].
     - Source: walkthrough 7.7 (read-only computation on `processed_data/model_space/amibekk_v1/e5/classes_main.csv`
       and `processed_data/model_space/stanford_brainmets/e4/heval_classes.csv`).
     - Put it in 3.4 as exploratory, and use it in 4.1.
- **Why:** this is RQ1's and RQ2's main evidence. The AUC of the strongest pair is the first hint that
  the excess is configuration-level, which sets up E6.
- **Table 5 (main), main audit:**
  - rows: E1 copy and E2 full × single / multi (primary), plus E2 partial and E2 flagged (secondary);
  - columns: n S, rate S, rate H-eval, excess [interval], p_holm (primary only).
  - Source: `results/amibekk_v1/e5/analyze/rates.csv` (main set, main variant), `review.txt`.
- **Figure 10 (main, compact redraw):** score distributions of S vs H-eval for multi-lesion masks only
  (E1 score and E2 C), with τ marked. From `results/amibekk_v1/e5/analyze/score_distributions.png`.
- **Appendix D:**
  - H2 table (`h2.csv`);
  - rates by size (`rates_by_size.csv`);
  - sensitivity forest (`rates_forest.png`);
  - near-exact sheet (`near_exact.png`), with the caveat;
  - `groups_top.png`;
  - per-scheme rates as one sentence (`rates_by_scheme.csv`).
- **Sources:** report-notes section 3 (E5 main audit, E5 exploratory), section 4 (near-exact baseline).

### 3.5 Patient exposure (E6)

- **Cover:**
  1. *E1 copy multi* (2,662 flags):
     - concentrated: 22 patients above their 95% chance band, against about 2.2 expected
       (`concentration.csv`, `above_null_band`); 20 have an exposure-ratio interval above 1 (red dots
       in `exposure_sorted.png`). report-notes says "20 … about 2 expected", which mixes the two
       counts; a correction is suggested;
     - but the margin AUC is 0.36 [0.24, 0.495] (flags fit their best patient *less* uniquely than
       chance flags);
     - source rarity AUC 0.42 [0.34, 0.498] (sources more typical);
     - reading **typical placement**.
  2. *E2 full multi* (357 flags): **undetermined** (margin 0.55 [0.27, 0.72], rarity 0.62 [0.49,
     0.81]; only 8 H-eval chance flags).
  3. *H3 not supported:* ρ 0.03 [−0.21, 0.25] (E1) and 0.00 [−0.20, 0.21] (E2).
  4. *H1 supported:*
     - +0.086 [+0.041, +0.163] (linking), +0.077 [+0.030, +0.163] (26-connectivity);
     - carried by 20–79 px lesions, reversed at ≥ 80 px;
     - no mock shows it.
  5. *Illustration:* `Mets_230` (77 E1 multi flags vs 1.5 expected; one metastasis over 18 slices).
- **Why:** this turns "closer than chance" into "how": copying would give unique, rare sources;
  placement gives typical sources with small margins.
- **Figure 11 (main, compact 2-panel):** (a) margins of S flags vs H-eval chance flags
  (`results/amibekk_v1/e6/exposure/margins.png`); (b) H1 as rates by slice span, S vs H-eval
  (`h1_rates.png`).
- **Table 6 (main), E6 readings:** per primary cell, the dispersion against its null, patients above
  the chance range (observed / expected), margin AUC, rarity AUC and the reading; plus H3 ρ and H1
  rows. Sources: `concentration.csv`, `margins.csv`, `h3.csv`, `h1.csv`.
- **Appendix D:**
  - `exposure_sorted.png` (observed vs expected flags per patient);
  - `rarity_exposure.png`;
  - `h1_span.png`;
  - the rehearsal margins (`results/stanford_brainmets/e6/rehearse/{null,positive,placement_compact}/margins.png`),
    which show the reading works.
- **Sources:** report-notes section 3 (E6 on S, E6 rehearsal), section 4 (E6 limits).

### 3.6 Match reports (E7)

- **Cover:**
  1. *On planted copies:*
     - correctness passes: source patient 100% for P1, P2, P4, 99.0% P6-add, 91.9% P6-remove; right
       lesions 99.0% / 81.1%;
     - deletion: Δ whole-mask 0.33 vs 0.15 (Wilcoxon p < 0.001); lost flag or source 23 vs 6
       (McNemar p 0.001); ΔC 0.43 vs 0.41 (hardly differs, as expected for exact copies).
  2. *On S:*
     - taxonomy: no exact lesion-level copy; multi-lesion flags deformed 50% vs 13% of H-eval chance
       flags; single-lesion flags look like chance;
     - deletion on 224 multi-lesion flags: ΔC 0.29 vs 0.16 and Δ whole-mask 0.16 vs −0.04 (both
       p < 0.001); the re-match measure is at its ceiling (224 vs 203 of 224 lose the flag; McNemar
       p < 0.001 on 21 discordant pairs, but almost every mask flips either way, since two-lesion
       masks become single-lesion masks), so it says little.
  3. *Example:* report 50, whose large lesion lies on `Mets_230`'s 18-slice metastasis (whole-mask
     0.89), while E2 finds no single lesion closer than chance → "configuration match".
  4. *Patient profile:* matches along the long metastases of the exposed patients (H1 seen in single
     patients).
- **Why:** RQ3b's criteria. The examples show what a "flag" in S actually looks like.
- **Figure 12 (main, compact):** (a) report 50 (panels A–B and the verdict), from
  `results/amibekk_v1/e7/reports/reports/50_exposed_e1_copy_multi.png`; (b) the `Mets_230` patient
  profile, from `results/amibekk_v1/e7/reports/patients/Mets_230.png`.
- **Table 7 (main), correctness and faithfulness:** correctness per variant against its criterion; the
  deletion measures (top vs smallest, test, p) for planted and S. Sources:
  `results/stanford_brainmets/e7/planted/correctness.csv`, `criteria.csv`, `deletion_summary.csv`;
  `results/amibekk_v1/e7/reports/deletion_summary.csv`, `taxonomy.csv`.
- **Appendix E:**
  - `02_near-exact.png` (the 1 px rounding limit of E2, caught by E1);
  - one random S flag and one H-eval chance flag report (57–76);
  - `patients/Mets_058.png`;
  - `report_P4.png` (E1 and E2 name different slices).
- **Sources:** report-notes section 3 (E7 entries), section 4 (E7 limits).

### 3.7 Paired MRI (E8)

- **Cover:**
  - the rehearsal: null 0.48 (not supported), positive 0.985 (supported), H-eval reference 0.53;
  - on S: lesion-neighbourhood AUC 0.48 [0.40, 0.57] against H-eval 0.53, difference −0.05
    [−0.16, +0.05] → **not supported**; whole brain and ring alike;
  - within A: the source equals the runner-up (+0.002 [−0.018, +0.023], p 0.70);
  - z-profile peak near the source (descriptive, consistent with mask resemblance);
  - scale anchors: the same patient two slices away ≈ 0.46, different patients ≈ 0.05.
- **Why:** answers RQ4 and sets up RQ-M1.
- **Figure 13 (main, compact):** the lesion-region panel of
  `results/amibekk_v1/e8/paired/ssim_groups.png` (four boxes: A, B, H-eval flagged, H-eval novel),
  optionally with two example rows from `examples.png`.
- **Table 8 (main if space; else appendix), E8 tests:** per region, AUC [interval], H-eval AUC,
  difference [interval], reading; the within-A row. Sources: `results/amibekk_v1/e8/paired/tests.csv`,
  `within_a.csv`, `results/stanford_brainmets/e8/rehearse/review.txt`.
- **Appendix E:** `within_a.png`, `examples.png`, the rehearsal `ssim_groups.png` (null, positive).

### 4 Analysis

Every subsection follows the same pattern:
1. the question or hypothesis restated;
2. the evidence for, with section links;
3. the evidence against, with section links;
4. the alternative explanation and how it was tested;
5. the verdict, worded as the pre-stated reading.

- **4.1 RQ1 and H1: memorization beyond chance; duplication.**
  - *For:*
    - the multi-lesion excess, which survives Holm and the size standardisation;
    - E1 is robust to every sensitivity variant (smallest +2.7 at the 99th percentile); E2 holds at
      bound 10 and drop-only but not at σ 0.08, bound 25 or the 99th percentile.
  - *Alternative 1:* typical placement. It is tested by E6 (margins below chance), E5's strongest-pair
    AUC of 0.38, and E7's taxonomy (deformed, not exact).
  - *Alternative 2 (exploratory):* lesion-count composition. E1's "multi" includes one-lesion-plus-extras
    masks that S has twice as often. Re-weighted, the E1 excess is +2.9 [−0.4, +6.7]; the robust part
    (+5.2 in masks with ≥ 2 informative lesions) agrees with E2's +3.0.
  - *Verdict:* "beyond chance" in the statistical sense, but the evidence points to the generator
    having learned *where* lesions usually are, not to verbatim copies of patients.
  - *Near-exact count:* not evidence on its own (the 1/(n+1) argument).
  - *H1:* supported. Lesions seen in many adjacent slices are matched more. This is the duplication
    mechanism (Somepalli et al.), but the held-out reference is sparse (58 credited lesions).
  - *Single-lesion masks* at chance: a single small blob has too little shape to be distinctive.
- **4.2 RQ2 and H2: lesion level vs whole mask at equal false-positive rate.**
  - The method *can* find what Dice misses: E4 P3, P6.
  - On S it does not: E1 shows more excess in every comparison.
  - Explanation:
    - H2 needs incomplete exact copies, and S has none (E7 taxonomy; the rehearsal showed H2 fails
      even on whole planted copies);
    - the S resemblance is a layout, which whole-mask Dice rewards and a lesion-by-lesion score with
      a chance-level τ_L does not;
    - E2's strict per-lesion thresholds and 1 px fragility also cost it recall.
  - Verdict: H2 not supported on this pool; the lesion level remains necessary for explanation (E7).
- **4.3 RQ3a and H3: exposure and rarity.**
  - Exposure is concentrated (22 patients above their chance band, about 2 expected), but the margins
    say placement.
  - H3 ρ ≈ 0, and placement predicts ρ ≤ 0, so the long-tail argument (Feldman and Zhang) is not
    seen.
  - `Mets_230` and `Mets_058` illustrate exposure through long, central metastases.
  - Limits: about 26 patients behind π_p; the E2 cell is undetermined; the placement mocks recognise
    placement only weakly.
- **4.4 RQ3b: faithfulness and correctness of the explanations.**
  - All thesis-plan criteria pass on planted copies.
  - Faithfulness holds on planted copies by all three measures, and on S by two (ΔC, Δ whole-mask);
    the re-match measure cannot separate them on two-lesion masks.
  - What the reports add: naming both NN and strongest-pair patients; "configuration match" wording;
    chance rate stated on every report.
  - Limits: no human rater; the taxonomy is rule-based.
- **4.5 RQ4 and H4: paired MRI.**
  - Not supported, and the runner-up comparison is the most direct evidence: the source patient's
    tissue is no closer than another patient's with a similar mask.
  - The profile peak fits mask resemblance.
  - Limits:
    - observational;
    - the positive mock shows correctness, not sensitivity to faint resemblance;
    - brain-size mismatch;
    - 56 cases.
  - Points to RQ-M1.
- **4.6 Sensitivity analysis.** One table (main) or the forest plot (appendix D), listing per
  variant whether each conclusion holds:
  - E5 variants (99th percentile, σ 0.08, bound 10 / 25, drop-only);
  - noisy masks included;
  - π_p smoothing m = 105;
  - H1 with 26-connectivity;
  - E8 whole brain, ring, brain-size halves.

  Conclusion: the E1 multi excess is robust to every pre-registered variant, but much smaller after
  the exploratory composition re-weighting; H1 is robust; the E2 multi excess is threshold-sensitive.
  Sources: `results/amibekk_v1/e5/analyze/sensitivity.csv`, `e6/exposure/review.txt`,
  `e8/paired/secondary.csv`.

### 5 Discussion

- **5.1 What the results mean.**
  - Stage 1 learned the *distribution of lesion placement* (compact, central, typical layouts) more
    than individual patients: there is no exact copy, and resemblances are diffuse over 96 patients
    and configuration-level.
  - A residual duplication signal (H1) shows the mechanism by which memorization would arise.
  - Within the validated detection range, this is a negative for verbatim patient copying.
- **5.2 What a copied mask would mean for privacy.**
  - A metastasis configuration (count, positions, sizes) is a quasi-identifier, linkable to a
    radiology record.
  - *Found:* no exact copy; single-lesion resemblances at chance; multi-lesion resemblances explained
    by placement; no tissue reproduction in stage 2 (observational).
  - *Not certifiable:*
    - copies with boundary noise or "drift" below chance are invisible to any output-only method;
    - small lesions carry no evidence;
    - the baseline rests on 51 patients.
  - An output-only, chance-calibrated audit is a practical *pre-release check*, not a guarantee.
  - Compare with membership-inference framing (out of scope).
- **5.3 The detectors and the explanations: strengths and blind spots.**
  - *Strengths:*
    - calibrated against held-out patients;
    - validated with pre-stated criteria;
    - analysis locked before S;
    - rehearsed on mocks;
    - explanations tested for correctness and faithfulness.
  - *Blind spots:*
    - 1 px boundary noise (P5);
    - E2's half-to-even centroid rounding at 1 px shifts (2 of 4 identical S masks called novel by E2,
      all caught by E1);
    - lesions < 12 px;
    - tiny-lesion chance matches in E1 (Dice 1.0 for 4–8 px lesions);
    - recombinations across patients;
    - 2D only;
    - E1's candidate shortlist;
    - combined perturbations not planted.
- **5.4 Limitations and threats to validity.**
  - *Provenance:*
    - stage-1 slice selection only bounded (≤ 2 px slices: 10 of 3,823 possibly unseen);
    - the repo is not exactly the code that ran;
    - no seeds, so the original models cannot be reproduced;
    - the sampling filters only partly hold;
    - the six generation rounds are assumed alike.
  - *Statistics:*
    - about 25 patients per held-out half (the H-eval false-positive rate is 7%, not 5%);
    - the E4 deviation was accepted after the run;
    - the multi-lesion strata rest on 12–17 patients;
    - the near-exact count and the E8 entry rule have no chance baseline;
    - the E1 strata are defined by lesions ≥ 4 px, so "multi" mixes compositions. The pre-registered
      size check did not cover this; the exploratory re-weighting does.
  - *Design:*
    - C was redefined after a look at H-cal (before S);
    - the placement mocks recognise placement only weakly;
    - E8 is observational with a brain-size mismatch;
    - one analyst made every decision (mitigated by pre-registration and mocks);
    - no human rater.
  - *Resolved:* the training/validation question. The whole train set was used (confirmed
    2026-11-16), so T is complete and the held-out sets are clean.
- **5.5 Future work (the master thesis):**
  - retrain both models with seeds and known splits (Branch B);
  - RQ-M1, the mask as a key by intervention (training vs held-out vs synthetic masks; perturbation
    series; a graded positive mock);
  - RQ-M2, does the full pipeline leak when each stage passes;
  - RQ-M3, training-data attribution (e.g. D-TRAK) for `Mets_230` and `Mets_058`, plus a
    position-only null generator to settle placement vs copying;
  - RQ-M4, attribution on the mask channel;
  - a blinded human check of the reports; recombinations; the toolkit reused.

### Appendix (proposed)

| Appendix | Content |
| --- | --- |
| A Settings and decisions | One table of the 14 critical decisions (B, C, D, F, G, H, J, K, M, N, Q, U, AC, AD) with their values, no config keys (curated 2026-11-17, `tables/A_decisions`); the rest is referenced to the repository in 2.13 |
| B Data and QC | slim set (2026-11-17) in `results/amibekk_v1/report/appendix/`: B1 native vs model space, B2 brain size by height, B3 component sizes, B4 specks by gap, B5 brain-mask methods (3 slices), B6 H-cal score histograms with the stratum thresholds |
| C Validation | C1 four missed planted copies (the variants are in Method figure 2-7, recall in 3-3) |
| D Audit and exposure | D1 sensitivity (excess per variant, E1 copy and E2 full), D2 six near-exact masks, D3 rarity fifths (H3), D4 rehearsal-mock and S margin and rarity AUCs; as tables: H2 (`D_h2`), rates by size (`D_rates_by_size`) |
| E Match reports and E8 | E1 report 02 compact (E2's 1 px limit, caught by E1), E2 the `Mets_058` profile, E3 E8 within group A |
| F Reproducibility | rehearsal table (mock → expected → obtained); check results per experiment; software versions |

---

## 3. Figure and table register

Status: *as is* = use the file; *compact* = redraw or crop the existing figure smaller and legible;
*new* = draw from stored data. *done* = drawn by `scripts/r_report_figures.py` into
`results/amibekk_v1/report/<stage>/` (PNG 300 dpi + PDF, 16 cm wide; decided 2026-11-16: a stage
package `src/memtk/report_figures/`, Results figures first). The file stem is given after "done".

### Main-text figures (13)

| # | Section | Content | Source | Status |
| --- | --- | --- | --- | --- |
| 1 | 1.3 | The pipeline producing the synthetic data | the author's figure (in the report) | exists |
| 2 | 2 | Experiment flow E0–E8 | the author's figure (in the report) | exists |
| 3 | 2.3 | Model-space conversion (native vs 128×128, the 0.01 threshold on the soft mask), E0's annotations (brain outline, box, numbered lesions), one synthetic sample per QC class | real volumes, E0b brain and QC outputs | done: `method/2-3_preprocessing_qc` |
| 4a | 2.4 | E1's shift search on a planted P2 copy: Dice as generated, per shift, after the best shift | E4 planted tables | done: `method/2-4_e1_shift_search` |
| 4b | 2.5 | E2's lesion pairs on a planted P4 copy: pairs, centred crops with shape × position × size = s, C | E4 planted tables, E7 planted report data | done: `method/2-5_e2_lesion_pairs` |
| 5 | 2.7 | Every planted variant of one base, and 1 px erosion and dilation of a small and a large lesion | E4 `planted_labels.npz`, T lesions | done: `method/2-7_planted_variants` (the area-ratio chart `e4/plant/area_change.png` goes to Appendix C) |
| 6 | 2.11 | The match report's images and verdict line on a planted P6-add copy | E4 planted tables, E7 planted report data | done: `method/2-11_match_report` |
| 7 | 3.1 | Lesion area distribution; centroid density in pixels, S vs real | `results/stanford_brainmets/e0b/stats/lesion_area_amibekk_v1.png`, `centroid_density_amibekk_v1.png` | done: `results/3-1_lesions_synthetic_vs_real` (pixels and brain-box frames) |
| 8 | 3.2 | Chance lesion-match rate by lesion area, and the bound | `results/stanford_brainmets/e3/bound_review/chance_rates.png` | done: `results/3-2_chance_rates` (with H-eval chance-match tiles, panel b) |
| 9 | 3.3 | Recall by lesion size per variant and detector | `results/stanford_brainmets/e4/validate/recall_by_size.png` | done: `results/3-3_recall_by_size` (P3–P6; P1, P2 in the caption) |
| 10 | 3.4 | Score distributions S vs H-eval, multi-lesion, with thresholds | `results/amibekk_v1/e5/analyze/score_distributions.png` | done: `results/3-4_score_distributions_multi` (S main only; the E5 figure included noisy masks) |
| 11a | 3.5 | Observed vs expected flags per patient with chance bands (E1 multi); margins and source rarity, S flags vs chance flags | `results/amibekk_v1/e6/exposure/exposure_sorted.png`, `margins.png` | done: `results/3-5a_exposure_margins` |
| 11b | 3.5 | H1 rates by slice span, S vs H-eval, per size bin | `results/amibekk_v1/e6/exposure/h1_rates.png` | done: `results/3-5b_h1_rates` |
| 12 | 3.6 | Report 50 and the `Mets_230` patient profile | `results/amibekk_v1/e7/reports/reports/50_exposed_e1_copy_multi.png`, `patients/Mets_230.png` | done: `results/3-6_report50_profile` |
| 13 | 3.7 | Lesion-region SSIM: A, B, H-eval flagged, H-eval novel | `results/amibekk_v1/e8/paired/ssim_groups.png` | done: `results/3-7_ssim_lesion` (with three example pairs: typical, strongest, rehearsal positive) |

If space is short, cut in this order: 7 → appendix; 9 → appendix (keep Table 4); 4a → appendix
(describe the shift search in words).

### Main-text tables (8)

| # | Section | Content | Source |
| --- | --- | --- | --- |
| 1 | 2.1 | Data sets: patients, lesion slices, role (T, H-cal, H-eval, S main, noisy) | `splits/stanford_brainmets_seed2026.csv`; report-notes sections 1–2 |
| 2 | 3.1 | Real vs synthetic description | `results/stanford_brainmets/e0b/stats/description_amibekk_v1.csv` |
| 3 | 3.2 | Frozen thresholds [interval]; chance rates H-cal and H-eval | `results/stanford_brainmets/e3/calibrate/thresholds.csv`, `e4/validate/fp_heval.csv` |
| 4 | 3.3 | Validation: recall per variant and detector, criteria, pass / fail | `results/stanford_brainmets/e4/validate/validation.csv`, `gate.csv` |
| 5 | 3.4 | Main audit: primary family and secondary classes | `results/amibekk_v1/e5/analyze/rates.csv`, `review.txt` |
| 6 | 3.5 | E6 readings, H3, H1 | `results/amibekk_v1/e6/exposure/concentration.csv`, `margins.csv`, `h3.csv`, `h1.csv` |
| 7 | 3.6 | Correctness and faithfulness (planted and S) | `results/stanford_brainmets/e7/planted/correctness.csv`, `criteria.csv`, `deletion_summary.csv`; `results/amibekk_v1/e7/reports/deletion_summary.csv` |
| 8 | 3.7 | E8 tests per region, within A | `results/amibekk_v1/e8/paired/tests.csv`, `within_a.csv` |

Plus a small rules table in 2.10 (the E6 readings). If space is short: Table 2 → appendix B; Table 8 →
appendix E (its numbers stay in the text).

### Left out of the report

- the `review.txt` files (sources, not figures);
- per-row CSVs (`flagged.csv`, `measures.csv`, `pairs.csv`, `deletion.csv`, `planted_outcomes.csv`,
  `exposure.csv`, `flag_details.csv`);
- rates per naming scheme (one sentence);
- the old-QC crosstab (one sentence);
- `views/*.png` except the one in Figure 3;
- the rehearsal versions of S figures (except in appendices D–E);
- the remaining 70 match reports.

---

## 4. Equations register

Define each formula once, in the section named. Symbols: q a query mask (synthetic or held-out), t a
training slice, T the training set.

| Quantity | Definition | Section |
| --- | --- | --- |
| Whole-mask score | D(q, t) = max over \|Δr\|, \|Δc\| ≤ 3 of 2\|q ∩ t_(Δr,Δc)\| / (\|q\| + \|t\|); D(q) = max over shortlisted t | 2.4 |
| Lesion-pair score | s(a, b) = IoU(crop_a, crop_b) · exp(−d²/2σ²) · min(A_a, A_b)/max(A_a, A_b), σ = 6.4 px | 2.5 |
| Configuration score | C(q, t) = Σ_{(a,b) ∈ M, s ≥ 0.1} s(a, b) / max(n_q, n_t), M = Hungarian matching on 1 − s | 2.5 |
| Classes | E1 copy: D(q) ≥ τ_D; E2 full: C(q, NN) ≥ τ_C; partial: not full and max_t max_(a,b) s ≥ τ_L | 2.4–2.5 |
| Thresholds | τ = 95th percentile of the H-cal NN score in the stratum (τ_L: of the strongest pair) | 2.6 |
| Bootstrap | per draw: resample patients (real sets) and masks (S) with replacement, recompute; 95% interval = 2.5th and 97.5th percentiles; 2,000 draws | 2.8 |
| Excess, p | Δ = r_S − r_H; p = (1 + #{Δ* ≤ 0}) / (B + 1); Holm over the 4 primary cells | 2.8 |
| AUC | P(X_A > X_B) + ½ P(X_A = X_B) (= Mann–Whitney U / n_A n_B) | 2.8 |
| Expected share | π_p = (n_p + m·w_p) / (N + m), m = 52.5; n_p = H-eval masks whose NN is in p; w_p = p's share of T lesion slices | 2.10 |
| Exposure ratio | r_p = o_p / (π_p · O) | 2.10 |
| Dispersion | D = Σ_p (o_p − e_p)² / e_p, e_p = π_p · O, against a multinomial null | 2.10 |
| Margin | E1: D(q, NN) − the best score of any other patient (shifted Dice for shortlisted slices, plain Dice otherwise); E2: C(q, NN) − the best C over other patients' slices | 2.10 |
| Rarity | slice: 1 − the best score against other patients' slices; patient: mean Euclidean distance to the 5 nearest patients in z-scored features | 2.10 |
| H1 statistic | Σ_b w_b (ρ_S,b − ρ_H,b) / Σ_b w_b, ρ = Spearman(span, credits) within size bin b, w_b = the number of 3D lesions in the bin | 2.10 |
| Deletion | ΔC, ΔD (top vs smallest contributor, paired Wilcoxon); flag/source flip after re-match (McNemar) | 2.11 |
| z-score | z = (x − μ_brain) / σ_brain, over the whole image | 2.12 |
| SSIM | [(2μ_xμ_y + C1)(2σ_xy + C2)] / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)], Gaussian window σ = 1.5, C1 = (0.01·10)², C2 = (0.03·10)² | 2.12 |

The E1 margin and the H1 weights were checked against `src/memtk/wholemask.py` and
`src/memtk/exposure.py` (`h1_table`), 2026-11-16.

---

## 5. Decision register (A–AE) and deviations from the plan

"Text" = name it in the Method section; "App A" = only in the settings appendix. The letters S and T
are skipped (they name data sets).

| Letter | Fixes | Value | Config key | Where | Mention |
| --- | --- | --- | --- | --- | --- |
| A | Brain mask; failed MRI | threshold 0.02, all regions ≥ 30 px, filled; failed < 20 px of tissue | `brain:` | 2.3 | Text (short) |
| B | Minimum lesion area | 4 px | `lesions.min_lesion_area` | 2.3 | Text |
| C | Attach gap for strays | 2 px | `lesions.attach_gap` | 2.3 | Text |
| D | Noisy-mask cut | > 9 isolated specks | `qc.max_isolated_specks` | 2.3 | Text |
| E | Sampling filters on held-out sets | pixel sum < 2,000 only | `filters.real_sets` | 2.2 | Text (short) |
| F | Position frame | pixel coordinates, σ 6.4 px, prefilter 25.6 px | `detectors.lesion` | 2.5 | Text (with reason) |
| G | Informative-lesion bound | 12 px (chance rate ≤ 10%) | `calibration.informative_bound_px` | 2.6 | Text |
| H | Stratified thresholds | single / multi for both detectors | `calibration.strata` | 2.6 | Text |
| I | Planting design | 20 × 5 bases; P1–P6; gate on base lesions ≥ bound | `validation:` | 2.7 | Text |
| J | What counts as caught | flagged + right source patient | `validation.caught` | 2.7 | Text |
| K | Go/no-go criteria | see 2.7 | `validation.criteria` | 2.7 | Text |
| L | Audit sets | S main; noisy as separate rows | `audit.sets` | 2.9 | Text |
| M | RQ1 tests | primary family of 4, Holm; secondary list | `audit.primary` | 2.8–2.9 | Text |
| N | Adjacent slices | chained groups, z gap ≤ 2 | `audit.window` | 2.8 | Text |
| O | Sensitivity on S | 99th pct, σ 0.08, bound 10/25, drop-only | `audit.sensitivity` | 2.9 | Text (list) |
| P | H2 at equal FPR | (i) frozen; (ii) τ_D\* | `audit` (H2) | 2.9 | Text |
| Q | Near-exact | ≥ 0.9 and above the held-out max per stratum × size bin | `audit.near_exact` | 2.9 | Text (with caveat) |
| R | Exposure | one source per flag; π_p from H-eval NN, m = 52.5 | `exposure.expected` | 2.10 | Text |
| U | Copying vs placement reading | margin + rarity AUC vs chance flags; amended rule | `exposure.reading` | 2.10 | Text |
| V | Rarity, H3 | slice and patient rarity; Spearman with permutation | `exposure.rarity` | 2.10 | Text |
| W | H1 | slice-to-slice linking; ρ_S − ρ_H in size bins | `exposure` (H1) | 2.10 | Text |
| X | Report selection and content | 76 reports; both NN and strongest pair | `reports.selection` | 2.11 | Text |
| Y | Deletion test | three measures | `reports.deletion` | 2.11 | Text |
| Z | Correctness; taxonomy | thesis-plan criteria; rule-based | `reports.correctness`, `reports.taxonomy` | 2.11 | Text |
| AA | E8 group A and comparison slice | E7's 56 strong cases; decision-J source | `paired.group_a` | 2.12 | Text |
| AB | E8 group B | k = 5 matched novel masks | `paired.controls` | 2.12 | Text |
| AC | E8 measure | z-score, E1 shift, SSIM σ 1.5, range 10, regions | `paired.measure` | 2.12 | Text |
| AD | E8 test and reading | AUC vs H-eval reference, mechanical reading | `paired.test` | 2.12 | Text |
| AE | E8 rehearsal | H-cal null, planted positive | `paired.rehearsal` | 2.12 | Text (one sentence) |

Numbers that are *not* method-defining go to Appendix A only: bootstrap draws, permutations, k for
patient rarity, top-k shortlist sizes, the provisional τ_L, region growth 8 px, m sensitivity 105,
seeds.

**Deviations from the plan, all stated openly in the report (they are strengths when stated):**

| Deviation | When | Section |
| --- | --- | --- |
| Position frame: pixel instead of brain-relative (F) | E2 planning, before any score | 2.5 |
| C redefined (τ_C was not calibratable) | E3, on H-cal, before S | 2.6 |
| P5 reported only (s ≈ r² below chance) | E4, before any detector ran | 2.7 |
| E4 false-positive criteria failed by 0.1–0.4 points and were accepted | E4, **after** the run | 3.3, 5.4 |
| Null model dropped; chance from H-cal only | E3 planning | 2.6, 5.4 |
| Rule U amended (margins below chance read as placement) | after the placement mock, before S | 2.10 |
| Near-exact definition lacks a chance baseline | found after the E5 run, not repaired | 2.9, 5.4 |
| Area-weighted C not run (optional sensitivity) | E5 planning | App A |
| Exposure ratio uses one source per flag, not every pair (E6 step 1 wording) | E6 planning | 2.10 |

---

## 6. Terminology and consistency

| Term | Meaning (use exactly) |
| --- | --- |
| model space | the 128×128 space the models saw, after the original training transform |
| T | the 3,823 lesion slices of the 105 training patients |
| H-cal / H-eval | the held-out test patients (25 / 26), lesion slices only; H-cal sets thresholds, H-eval checks them and is the baseline |
| S, S main | all synthetic samples (37,624); main = mask class clean or minor (35,920) |
| lesion, speck, stray | component ≥ 4 px; smaller component; a speck attached to a lesion within 2 px |
| informative lesion | lesion ≥ 12 px; the only ones in E2 |
| stratum | single- or multi-lesion (E1: lesions ≥ 4 px; E2: informative lesions) |
| flag | any mask in a detector class above chance (E1 copy; E2 full or partial) |
| E1 copy, E2 full / partial / novel | class names, not claims. Write "flagged as an E1 copy", never "is a copy" |
| source (patient, slice) | decision J: E1 copy → E1 NN; E2 full → E2 NN; partial → the strongest-pair slice |
| chained group | source slices of one patient joined while z differs by ≤ 2 |
| chance rate | the rate of a class on held-out masks at the frozen thresholds |
| excess | S rate − H-eval rate |
| near-exact | decision Q; always with the baseline caveat |
| exposure ratio, margin, rarity | decisions R, U, V |
| configuration match | an E2 flag where no single lesion pair reaches τ_L |
| mock set; reading | a rehearsal set with a known answer; a pre-stated rule that turns statistics into a verbal outcome |

Wording rules: "beyond chance" (not "significant memorization"); "typical placement" and "patient
copying" as the names of E6 readings; DDPM, never LDM; self-attention only; patients by `Mets_xxx`.

---

## 7. What not to put in the report

- How the work was done: bugs and their fixes, refactors, Pylance, re-runs, replaced intermediate
  numbers, scratch checks. These live in `docs/process-log.md`. Exception: a re-run that changes a
  reported result.
- Every individual number choice that does not define the method (Appendix A only, see section 5).
- Stage and module names, CLI commands, file paths (except in the reproducibility appendix).
- Preliminary observations later replaced by full measurements.

---

## 8. Open items for writing

- **Tables:** drawn as LaTeX by `r_report_figures.py tables` into `results/amibekk_v1/report/tables/` (18 files, placeholder captions `TODO: ...` to replace; needs `\usepackage{booktabs}`). The whole appendix is
  `results/amibekk_v1/report/appendix.tex` (sections A-F, every appendix figure and table, placeholder
  captions), for `\input{appendix}` (its own `appendices` environment gives an "Appendices" page and
  contents entry and keeps floats in their subsections; preamble lines are in its header; no `\appendix`).
- **Figures:** the Method and Results figures are drawn (`r_report_figures.py method`, `results`,
  2026-11-16; E1 and E2 as separate figures 4a and 4b). The appendix figures were cut to a slim set of 14
  readable at print size (2026-11-17; section 3's appendix table); `check` 21/21. All figures live in
  `results/amibekk_v1/report/{method,results,appendix}/`.
- **Appendix F** (reproducibility) may be cut later (e.g. the software table); left as it is for now.
- **Previous author:**
  - answered 2026-11-16: the whole train set was used, nothing held out from it, the test set never
    seen;
  - still open (report-notes section 5): the stage-1 slice filter actually used; whether all six
    sampling rounds used the same checkpoints and filters; the MONAI version used in training.
- **Report-notes:** the validation-set limitation and the previous-author questions were marked
  resolved, the E6 count corrected and the E5 composition finding added (2026-11-16).
- **Literature:** re-run the novelty search (MICCAI / MIDL 2026) before submission.
- **Numbers:** re-check each number against its source file when it is copied into the report.
