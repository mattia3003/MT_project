# Project walkthrough: memorization in a mask→MRI diffusion cascade

*Written 2026-11-16. Plain-language explanation of the whole preliminary project, for understanding
it and defending it orally.*

The companion file `docs/report-synthesis.md` says *what goes where in the report*. This file says
*what was done, why, and what it means*. Each section can be read on its own. Numbers name their
source file; the decisions carry the letters used everywhere else (A–AE).

---

## 0. Orientation

**How each experiment section is built** (sections 2–10):
1. **The question**: what the experiment had to answer.
2. **What was done**: the steps in order.
3. **The maths**: first in words, then the formula.
4. **The decisions**: each choice, the alternatives and why they lost.
5. **Reading the figures and tables**: what is on the axes and what to look at.
6. **The result**: one sentence, then the numbers with their source.
7. **How to say it**: wording for the report.
8. **Questions you may get**: short answers for the oral defence.

**Where to look things up.**

| Need | File |
| --- | --- |
| A decision's full evidence | `docs/report-notes.md` section 2 (search for the letter or the setting) |
| A finding or limitation with its source | `docs/report-notes.md` sections 3 and 4 |
| How an experiment was built, step by step | the closed plan: `docs/e0b-plan.md`, `e1-e3-plan.md`, `e4-plan.md`, `e5-plan.md`, `e6-e7-plan.md`, `e8-plan.md` |
| A file or column | `docs/data-dictionary.md` |
| The code behind a step | `docs/module-reference.md` |
| Where it goes in the report | `docs/report-synthesis.md` |

**Figures: experiment originals and report versions.** The "Reading the figures" sections below
describe the figures each experiment wrote (`results/<dataset|pool>/<experiment>/`, 90 dpi). The report
uses redrawn versions in `results/amibekk_v1/report/` (`method/`, `results/`, `appendix/`: 300 dpi PNG
and PDF, cropped; `tables/`: LaTeX). Their content is the same; panels and layout differ:

| Walkthrough | Original | Report version |
| --- | --- | --- |
| 2.3 | `e0b/stats/centroid_density_*.png`, `lesion_area_*.png` | `results/3-1_lesions_synthetic_vs_real` (both frames) |
| 2.3 | `e0b/stats/brain_by_height_*.png`, `adjacency_dice_*.png` | appendix B2, B7 |
| 2.2 (steps 2, 4-6) | `e0b` views and review sheets | `method/2-3_preprocessing_qc`; appendix B1, B3-B6 |
| 3, 4 | (no original) | `method/2-4_e1_shift_search`, `method/2-5_e2_lesion_pairs` |
| 4.5 | `e7/planted/report_P6-add.png` | `method/2-11_match_report` |
| 5.5 | `e3/bound_review/chance_rates.png`, `e3/calibrate/score_histograms.png` | `results/3-2_chance_rates` (with chance-match tiles); appendix B8, B9 |
| 6 | `e4/plant/variants_grid.png`, `area_change.png` | `method/2-7_planted_variants`; appendix C1 |
| 6.5 | `e4/validate/recall_by_size.png`, `heval_flags.png`, `factors.png` | `results/3-3_recall_by_size`; appendix C4, C2, C3 |
| 7.8 | `e5/analyze/score_distributions.png`, `rates_forest.png`, `near_exact.png` | `results/3-4_score_distributions_multi` (multi-lesion, S main only); appendix D1, D2 |
| 8.7 | `e6/exposure/exposure_sorted.png`, `margins.png`, `h1_rates.png` | `results/3-5a_exposure_margins`, `results/3-5b_h1_rates`; appendix D4-D9 |
| 9.4 | `e7/reports/reports/50_*.png`, `patients/Mets_230.png` | `results/3-6_report50_profile`; appendix E1-E4 |
| 10.6 | `e8/paired/ssim_groups.png`, `examples.png`, `within_a.png` | `results/3-7_ssim_lesion` (box plot and three example rows); appendix E5, E6 |

**Terms** are defined in `docs/report-synthesis.md` section 6 (glossary). The most important ones
appear in section 1 below.

---

## 1. The big picture

### 1.1 The problem

A previous master thesis built a pipeline that produces synthetic brain MRI with metastases, in two
stages:
- **Stage 1** draws a lesion *mask*: a 128×128 binary image saying where the metastases are. It is
  trained only on real masks and never sees an MRI.
- **Stage 2** draws a BRAVO MRI slice to fit that mask. The mask is fed in as an extra input channel.

It produced 37,624 synthetic (mask, MRI) pairs: the set **S**.

Synthetic data is often presented as privacy-safe, because "no real patient is in it". But
diffusion models can **memorize**: they sometimes reproduce training examples almost exactly.
For this pipeline the worry is concrete. A patient's metastasis configuration (how many, where, how
big, what shape) is a fingerprint. If stage 1 reproduced it, the synthetic data would carry
patient-identifying information.

**The overarching question.** Does the mask generator memorize patient-specific lesion
configurations, and can that be detected and explained from the generated outputs alone? There are
no model weights to inspect, only the outputs and the real data.

### 1.2 Why masks, and why outputs only

- **Masks first.** Nobody has audited the mask stage of such a cascade (the closest study,
  Edirisooriya et al. 2026, only tests the image stage). The mask is also the simpler object: binary,
  128×128, and directly comparable with training masks.
- **Stage 1 never sees a brain.** A copied mask would therefore reappear at the training mask's *pixel*
  positions, wherever stage 2 later draws the brain. This one fact shapes several decisions (F in
  E2, the alignment in E8).
- **Outputs only,** because the original weights are not available. That also makes the method
  reusable: anyone holding synthetic data and the training set can run it.

### 1.3 The four sets

Everything is compared in the 128×128 "model space" (section 2), and only slices with lesions are
compared, because stage 1 was trained only on lesion slices.

| Set | What | Role |
| --- | --- | --- |
| **T** | 3,823 lesion slices of the 105 training patients | what could have been memorized (the reference) |
| **H-cal** | 992 lesion slices of 25 test patients | sets the thresholds (calibration) |
| **H-eval** | 1,094 lesion slices of 26 other test patients | checks the thresholds and is the chance baseline |
| **S** | the synthetic masks (main audit: 35,920 after QC) | what is audited |

**Confirmed by the previous author (2026-11-16):** both models were trained on the whole train set,
with no validation patients held out, and the test patients were never used. So H-cal and H-eval are
genuinely unseen. That is what makes them a fair measure of chance.

### 1.4 The logic of the design (the order matters)

```
E0  prepare: model space, lesions, quality control
E1 + E2  build two detectors (whole mask; lesion by lesion)
E3  calibrate them on H-cal  →  FREEZE the thresholds (S untouched so far)
E4  prove them on planted copies (gate)          ← the detectors can find copies
E5  score S once, analysis fixed beforehand      ← is there more matching than chance?
E6  who is exposed, and is it copying or placement?
E7  explain each match; check the explanations
E8  do the paired MRIs follow the mask? (exploratory)
```

The point of this order: **every choice that could bend the result was made before S was looked at.**
Nothing was tuned towards a desired outcome.

### 1.5 Four ideas that run through everything

These come up in almost every examiner question. Learn them well.

1. **Chance is measured, not assumed.** Metastases are small and sit in common places, so any
   synthetic mask will look like *some* training mask. The question is never "is it similar?" but "is
   it *more* similar than a mask from a patient the model never saw?" Held-out patients (H-cal,
   H-eval) give that chance level. About 7–10% of *any* mask set gets flagged at the frozen
   thresholds, and only the **excess** over held-out is evidence.
2. **Patients, not slices, are the units.** Neighbouring slices of one patient are near-duplicates
   (median Dice 0.80 between neighbours). Ten flagged slices from one patient are not ten independent
   findings. So intervals come from resampling patients (a *bootstrap by patient*), and copies are
   counted per patient and per group of adjacent slices.
3. **Analysis fixed before data.** Thresholds were frozen before S was scored. E5's whole analysis was
   written into the config and *locked* (hashed) before S was scored once. Later stages refuse to run
   if any locked value changed. Readings ("beyond chance", "typical placement", …) come from rules
   stated in advance.
4. **Rehearsal on mock sets.** Before the analysis code touched S, it ran on sets with a known answer:
   - a *null* set (H-cal, which cannot contain memorization) must not read "memorization";
   - a *positive* set (with planted copies) must read it.

   If the code gets those right, a result on S can be trusted.

### 1.6 The result in one paragraph

Synthetic masks match training masks more than held-out masks do, but **only multi-lesion masks**, by
+8.5 points (whole mask) and +3.0 points (lesion level). Looking closer:
- The matches look like **typical lesion placement**, not copies of particular patients: each match
  fits its best patient *less* uniquely than chance matches do.
- **No synthetic mask is an exact lesion-level copy.**
- Lesions that span many training slices are matched more often than chance predicts (**H1**, the
  duplication effect).
- Lesion-level matching does not find more than whole-mask matching (**H2 not supported**).
- Exposure is not linked to rarity (**H3 not supported**).
- The match reports correctly name planted sources, and their explanations are faithful (**RQ3b
  passes**).
- The paired MRIs show no sign of the source patient's tissue (**H4 not supported**).

All of this holds within measured detection limits. It is a validated negative for verbatim copying,
not a privacy guarantee.

---

## 2. E0: Data, model space and mask quality control

*Scripts: `e0a_prepare_data.py` (split, real, synthetic, check) and `e0b_audit.py` (brain, lesions,
QC, statistics). Plan: `docs/e0b-plan.md` (closed). Checks: E0a 32/32, E0b 43/43.*

### 2.1 The question

Before comparing anything, two things had to be true:
- real and synthetic masks are compared in the *same* space, under the *same* definitions;
- the synthetic pool is understood: what is noise, what is a lesion, what is broken.

### 2.2 What was done

**Step 1: the split (E0a `split`).** The 51 test patients are divided into H-cal and H-eval.
- Patients are sorted by lesion-slice count, paired with their neighbour in that order, and one of
  each pair goes to each set at random (seed 2026). The odd patient goes to H-eval.
- Result: H-cal 25 patients / 992 slices, H-eval 26 / 1,094 (medians 35 and 36 slices per patient).
- *Why matched pairs:* lesion burden ranges from 3 to 119 slices per patient. A plain random split
  could put most heavy patients in one half, and then the two halves would have different chance
  levels.
- File: `splits/stanford_brainmets_seed2026.csv`.

**Step 2: model space (E0a `real`).**
- The models saw images after this transform, per 3D volume: scale intensities to [0, 1] using that
  volume's min and max; resize in-plane to 128×128 with area interpolation (the depth is kept, and the
  aspect ratio is not); binarize the mask at > 0.01.
- Every real volume is pushed through exactly the same transform. It must be done on the 3D volume,
  because the min and max are per volume.
- Verified bit-identical: a fresh MONAI run and an independent re-implementation both give the same
  result, and 32 consistency checks pass.
- **Side effect to know:** the 0.01 threshold makes lesions larger. Any pixel with more than 1% lesion
  coverage after resizing becomes lesion, which adds about a one-pixel rim. Per patient, lesion area
  grows by a median of 1.5× compared with plain downscaling, and small lesions grow most (2.6× for
  1–5 px lesions). Real and synthetic masks both went through this threshold, so the comparison is
  fair. It just means "12 px" in model space is a bit smaller in reality.

**Step 3: indexing S (E0a `synthetic`).**
- 37,624 files in six naming schemes (`<n>`, `file_<n>`, `file_2_<n>` … `file_5_<n>`), most likely one
  per generation round. They look alike on every statistic.
- The sample ID is the full file stem: `1033` and `file_1033` are different samples.

**Step 4: the brain mask (E0b, decision A).**
- Brain = pixels > 0.02, keep every region ≥ 30 px, fill holes.
- *Why not Otsu* (the usual automatic threshold): it fails in two ways.
  - It keeps only the largest region, which drops separate brain pieces in low slices (cerebellum,
    lower lobes). Lesions there then look "outside the brain".
  - Its threshold (about 0.10) lands at the intensity of dim brain tissue, so it trims the edge.
- Because the images are skull-stripped, background is near 0, and a fixed 0.02 sits in the gap.
  Result: only 1 of 6,101 training lesions is more than half outside the brain, against 74 with Otsu.
- A *failed* synthetic MRI = no brain region at all (< 20 px of tissue): 152 samples (0.4%).

**Step 5: what a lesion is (decisions B and C).** Synthetic masks are full of tiny specks. 62% of all
synthetic mask components are a single pixel, against 0.08% in T. They are probably an artefact of
how the sampling script scaled and thresholded each mask. So:
- **A lesion** = an 8-connected component of at least **4 px** (B). This is the largest size that
  calls at most 1% of real components "specks" (0.99%). Below it, real components are rare and
  synthetic ones are everywhere.
- **Attach-or-drop** (C): a speck within **2 px** of a lesion is attached to it as a *stray* (it is
  probably the lesion's own faint fringe, cut off by the threshold); any other speck is dropped.
  - The data showed a halo of specks at gaps 1–2 around lesions, then a flat noise level from gap 3
    on.
  - Two real-sized lesions are never merged.
- The same rule is applied to real and synthetic masks. Real masks are almost unaffected (only 4 T
  specks get attached).
- Each lesion gets an ID such as `Mets_005/z127/L1` (L1 = the largest lesion in the slice).

**Step 6: quality classes (decision D).** Each synthetic mask is classed from its mask alone:
- *clean*: 0 isolated specks;
- *minor*: 1–9 isolated specks, which are dropped (kept in the audit);
- *noisy*: more than 9 isolated specks (left out of the main audit, reported separately);
- *no_lesion*: nothing ≥ 4 px.

*Why 9:* masks per speck count fall steadily from 1 to 9 specks, then turn into a flat tail. That
looks like two populations, occasional specks and noise fields, and visually the masks change
character at 10.

Result: clean 32,137, minor 3,701, noisy 1,241, no lesion 393, plus 152 failed MRIs (a separate
flag). **The main audit, "S main", is clean + minor = 35,920 masks.**

**Two rules here that are easy to forget, and examiners like them:**
- *A failed MRI does not remove its mask from the mask audit.* The mask came from stage 1 whatever
  stage 2 did afterwards. 82 failed MRIs carry clean masks.
- *The edge check only flags, it never excludes.* Stage 2 draws smaller brains, so a genuine copy of a
  lesion near a real cortex can land outside the synthetic brain. Excluding such masks could remove
  exactly the copies being looked for.

**Step 7: sampling filters (decision E).** The repo's sampling script claims three filters. On the
data, only one holds:
- **mask pixel sum < 2,000:** holds for every synthetic mask (the largest is 731 px);
- **at least one lesion < 40 px:** fails for 10.7% of the masks, so it was not applied as written;
- **lesion pixels inside the brain:** not meaningfully applied.

Only the filter that demonstrably holds is applied to H-cal and H-eval, so that they describe the same
kind of masks as S. It removes no real slice. T is never filtered.

**Step 8: describe and compare (E0b `stats`).** What synthetic masks look like next to real ones:

| | T | S main |
| --- | --- | --- |
| Median lesion area | 19 px | 13 px |
| Single-lesion slices | 68% | 53% |
| Two-lesion slices | 18% | 34% |
| Lesion centroid spread in pixels (SD row/col) | 24 / 25 px | 15 / 19 px |
| Median brain area | 7,765 px | about 3,000 px |

*Source: `results/stanford_brainmets/e0b/stats/description_amibekk_v1.csv`, report-notes section 3.*

### 2.3 Reading the figures

*Report versions: `results/3-1_lesions_synthetic_vs_real`, `method/2-3_preprocessing_qc`, appendix B1-B7.*

**Centroid density** (`results/stanford_brainmets/e0b/stats/centroid_density_amibekk_v1.png`). Six
heat maps of where lesion centres lie, for T, H (held-out) and S main:
- **Top row, in pixels.** T and H spread over the whole round brain. **S fills a compact central
  rectangle** (roughly rows 40–85, columns 55–105) with sharp edges. Stage 1 puts lesions in a
  narrow central area.
- **Bottom row, as a fraction of each slice's own brain box.** All three look alike.
- *Reading:* stage 2 draws a brain *around* the lesions, so relative to the synthetic brain, lesions
  look normally spread.

This figure is the visual basis for decision F (pixel coordinates in E2) and for the "typical
placement" reading in E6.

**Lesion area** (`lesion_area_amibekk_v1.png`). The synthetic distribution lies below every real one
over the whole range, and stops near 330 px (T: about 700 px).

**Brain size by height** (`brain_by_height_amibekk_v1.png`). Real brain area rises from the top or
bottom of the head to about 9,900 px in the middle. The median synthetic brain (3,009 px) matches real
slices only in the outer 10–15% of the head. So synthetic brains look like real end-of-head slices,
but appear far more often than lesion slices occur there.

**Adjacency Dice** (`adjacency_dice_stanford_brainmets.png`). Dice between consecutive lesion slices
of one patient has a median of 0.80, and 86% of pairs have Dice ≥ 0.5. This is the reason behind "count
per patient, not per slice".

### 2.4 The result

Real data is in the exact space the models saw. Every mask, real or synthetic, is cut into lesions by
the same rule. S main is 35,920 masks. Synthetic masks have smaller lesions, more two-lesion slices,
compact central positions and smaller brains.

### 2.5 How to say it

"All comparisons were made in the 128×128 space in which the models were trained. The real volumes
were converted with the original training transform, which we verified to reproduce bit for bit.
Lesions were defined identically for real and synthetic masks as 8-connected components of at least
4 px. Smaller specks within 2 px of a lesion were attached to it, and the rest were dropped. Synthetic
masks with more than nine isolated specks were classed as noisy and reported separately."

### 2.6 Questions you may get

- *Why not compare at native resolution?* The models never saw native resolution. A memorized mask can
  only be a copy of what the model was trained on, which is the 128×128 version.
- *Doesn't the 0.01 threshold distort lesions?* It enlarges them by about one pixel at the edge. The
  same happens to real and synthetic masks, so comparisons stay fair. It is stated as a property of
  the space.
- *Why 4 px, why 2 px, why 9 specks?* Each was read off the data: 4 px from the share of real
  components below it, 2 px from the speck halo around lesions, 9 from the break in the speck-count
  distribution. Each was then reviewed and set before the detectors existed. Sensitivities: 3/5 px,
  attach gap 0 (drop-only) in E5, noisy masks reported separately.
- *Why not use the old QC list?* It dropped a mask for a single pixel outside the brain. It is fully
  explained by two coarse rules (more than 10 components; any pixel outside the brain) and was
  replaced by explicit classes.
- *Could excluding noisy masks hide copies?* No. E5 scored them as well, and they are flagged *less*
  often than held-out masks.

---

## 3. E1: The whole-mask detector

*Library: `src/memtk/wholemask.py`. Plan: `docs/e1-e3-plan.md` (closed). No script of its own: it
runs inside E3, E4 and E5.*

### 3.1 The question

For each mask, which training mask is it most like as a whole, and how alike?

### 3.2 What was done

For each query mask q (synthetic or held-out; cleaned, so dropped specks are gone):
1. Compute plain Dice against all 3,823 T masks.
2. Make a shortlist of up to 40 candidates: the 20 with the best plain Dice, plus the 20 whose largest
   lesion's centre is nearest.
3. For each candidate, slide q by up to ±3 px in rows and columns (49 positions) and take the best
   Dice.
4. The **nearest neighbour (NN)** is the candidate with the best score. Also stored: the shift, the
   NN's patient, the runner-up patient (the best other patient) and the **margin** (score minus
   runner-up score).
5. Stratum: *single* if q has one lesion ≥ 4 px, *multi* otherwise.
6. **E1 copy** = score ≥ τ_D of its stratum (τ_D comes from E3).

### 3.3 The maths

*In words:* Dice measures overlap: twice the shared pixels divided by the total pixels of both masks.
1 = identical, 0 = no overlap. The shift search allows the mask to have moved by up to 3 pixels.

*Formula:* D(q, t) = max over |Δr|, |Δc| ≤ 3 of 2·|q ∩ t shifted by (Δr, Δc)| / (|q| + |t|).
- The score of q is the maximum over the shortlist.
- Ties go to the smallest shift, then the higher plain Dice, then the first slice.

### 3.4 The decisions

- **Shift search ±3 px:** a copy may come out slightly displaced. 3 px is small relative to typical
  lesion spacing; larger shifts are left to E2, whose position factor degrades smoothly.
- **No flips or rotations:** training used no augmentation, so the model never saw flipped masks.
- **Cleaned masks:** specks would lower Dice for reasons that have nothing to do with copying.
- **The shortlist** (a speed-up; explained in the 2026-11-16 discussion): only up to 40 candidates get
  the shift search.
  - The "nearest centre" list exists because a shifted small lesion can have plain Dice near 0.
  - It is *not* proven exact (unlike E2's shortcuts). In rare cases E1 may miss the very best
    shifted match and report a slightly lower score.
  - It is applied identically to every set, so it cannot create an excess by itself. E4 measures its
    effect on real copies (P2, a 2 px shift: 100% caught).
  - Its only possible effect is to make E1 a little conservative.

### 3.5 The result

E1 alone is just a scorer. Its results appear in E3 (chance levels), E4 (recall) and E5 (S).

### 3.6 Questions you may get

- *Why Dice and not IoU?* They rank pairs identically (Dice = 2·IoU / (1 + IoU)). Dice is the
  standard in segmentation, so readers know it.
- *Why is a high Dice not enough to call something a copy?* Because single small lesions easily find a
  near-identical training mask by chance: 5% of held-out single-lesion slices reach Dice ≥ 0.905.
  See E3.
- *What does the margin tell you?* How uniquely the best patient fits. A copy of one patient should fit
  that patient much better than anyone else. Used in E6.

---

## 4. E2: The lesion-level detector

*Library: `src/memtk/lesionmatch.py`. Plan: `docs/e1-e3-plan.md`.*

### 4.1 The question

Can single lesions or lesion configurations be matched, so that a *partial* copy is found (some
lesions copied, others new) that whole-mask Dice would miss? Can each match be explained lesion by
lesion? This is RQ2, and it provides the basis for E7's explanations.

### 4.2 What was done

1. Only **informative lesions** take part: lesions ≥ 12 px (the bound comes from E3). Smaller blobs
   match each other by chance too often to be evidence. A slice with no informative lesion is left
   out of E2, but still scored by E1.
2. **Score every pair of lesions** (a from the query, b from a training slice) by three factors:
   - *shape:* the IoU of the two lesions after centring both on their centroids. Do they have the same
     outline?
   - *position:* how far apart the centroids are, in pixels, turned into a value from 1 (same place)
     down towards 0;
   - *size:* the smaller area divided by the larger.

   The lesion score s is the product of the three, between 0 and 1.
3. **Pair the lesions of two slices** one-to-one (Hungarian algorithm: the best overall pairing, so no
   lesion is used twice). Keep pairs with s ≥ 0.1.
4. **The configuration score C** = the sum of the kept pair scores divided by the larger lesion count
   of the two slices. If q has 2 lesions and t has 3, the best possible C is 2/3: one lesion is
   unexplained.
5. **NN** = the training slice with the highest C.
6. **Classes:**
   - *full* = C(q, NN) ≥ τ_C: the whole configuration is closer than chance;
   - *partial* = not full, but the strongest single lesion pair over all training slices ≥ τ_L: one
     lesion is closer than chance;
   - *novel* = neither.
7. **Stored:**
   - every candidate pair (slice-pair table and lesion-pair table), so thresholds are applied
     later and other experiments can reuse them;
   - `n_kept`: how many NN lesions pass τ_L;
   - `all_copied`: whether all of them do.

### 4.3 The maths

*In words:* two lesions are "the same" if they have the same shape, sit in the same place and have the
same size. Multiplying the factors means one bad factor ruins the score. Two slices are alike if
their lesions pair up well, and an extra lesion on either side counts against them.

*Formulas:*
- s(a, b) = IoU(centred crops) · exp(−d² / 2σ²) · min(A_a, A_b) / max(A_a, A_b), with d the centroid
  distance in pixels and σ = 6.4 px (5% of 128). At d = 6.4 px the position factor is 0.61; at 12.8 px
  it is 0.14.
- C(q, t) = Σ s over the Hungarian pairs with s ≥ 0.1, divided by max(n_q, n_t).

*Two exact speed-ups* (S has 35,920 masks × 3,823 slices):
- lesion pairs more than 25.6 px apart are skipped, since their position factor is below e⁻⁸;
- C is computed only for training slices that share at least one lesion pair with s ≥ 0.1, because all
  others have C = 0 by definition.

Both were verified against scoring every pair: 0 differences.

### 4.4 The decisions

- **F: pixel coordinates, not brain-relative** (2026-11-03). The experiment plan first said to measure
  position relative to the brain.
  - Stage 1 makes the mask without any brain, so a copy keeps the training mask's *pixel* positions.
    Synthetic brains are much smaller.
  - A perfect pixel copy would therefore get a *different* brain-relative position, which reflects how
    stage 2 drew the brain, not what stage 1 copied. The centroid-density figure shows exactly this.
  - Pixel coordinates are also consistent with E1's pixel-wise Dice.
  - Brain-relative coordinates are kept for anatomical questions (E6 rarity).
- **Equal weights for every lesion in C, not area-weighted.**
  - A large copied lesion next to new small ones is not missed: E1 sees it (high Dice), and E2 calls it
    *partial*.
  - Weighting by area would turn C into a version of Dice and blur the contrast RQ2 tests.
- **The candidate cut 0.1 in C** (not τ_L): see E3, the redefinition of C.

### 4.5 Reading a lesion match

*Report versions: `method/2-5_e2_lesion_pairs` (a P4 copy with partial overlaps) and `method/2-11_match_report` (the P6-add copy below).*

`results/stanford_brainmets/e7/planted/report_P6-add.png` shows a planted copy with one extra lesion
added:
- every base lesion pairs with s ≈ 1 (same shape, place and size);
- the added lesion has no partner;
- C = (sum of the pair scores) / (one more lesion than the training slice has), so it lands below 1 but
  above τ_C.

This is "the configuration was copied, but not every lesion".

### 4.6 Questions you may get

- *Why multiply the factors instead of averaging them?* Copying means all three agree. With an average,
  a lesion in the right place but the wrong shape would score well.
- *Why σ = 6.4 px?* 5% of the field. A shift of 2–3 px keeps the position factor high (0.95–0.90), and
  at 13 px it falls to 0.14. Sensitivity in E5: σ = 0.08 (10.2 px).
- *Why centre the crops for the shape factor?* So that shape and position are measured separately. The
  position factor handles displacement; shape only asks whether the outlines match.
- *What if q has three lesions and t has two?* Dividing by the larger count (3) caps C at 2/3. The
  unmatched lesion counts as evidence *against* a copy.
- *What is the "strongest pair"?* The best single lesion match over *all* training slices, not only
  the NN. It can point to another patient than the NN (seen in E4 and E5), and E7 shows both.

---

## 5. E3: Calibration on H-cal, and the freeze

*Script: `e3_calibrate.py` (bound-review, calibrate, check). Plan: `docs/e1-e3-plan.md`. Check: 16/16.
Frozen file: `configs/frozen/e3_stanford_brainmets.yaml`.*

### 5.1 The question

How close do masks from patients the model never saw come to T *by chance*? The thresholds that
separate "closer than chance" from "chance" are set from that.

### 5.2 What was done

1. **Score H-cal against T** with both detectors. H-cal was never trained on, so its scores are pure
   chance.
2. **Choose the informative-lesion bound (decision G).**
   - For each H-cal lesion, its best lesion score s against all T lesions.
   - A provisional τ_L = the 95th percentile of those scores for lesions ≥ 10 px, which is 0.77.
   - Per area bin, compute the share of H-cal lesions that reach it: the *chance-match rate*.
   - The bound is the smallest area from which every bin stays ≤ 10%: **12 px**.
3. **Set the thresholds (decision H).** For every H-cal slice, take its NN score. Each threshold is
   the **95th percentile**, computed separately for single- and multi-lesion slices:
   - τ_D (E1 score);
   - τ_L (the strongest lesion pair);
   - τ_C (C of the NN).
4. **Intervals:** each threshold gets a 95% interval from a bootstrap by patient (2,000 draws).
5. **Freeze:** values, bound and detector settings are written to the frozen file. E4 and E5 read only
   that file. Its hash is checked later, and the run logs show that no E3 stage read S.

### 5.3 The maths

*In words:* "the 95th percentile of held-out scores" means only 5% of chance matches score higher. So
a mask above the threshold is closer to a training mask than 95% of masks from unseen patients.

*Formula:* τ = Q₀.₉₅ { score(q, NN(q)) : q ∈ H-cal, same stratum }.

*The bootstrap by patient:* draw 25 patients from H-cal *with replacement* (some twice, some not at
all), take all their slices, recompute the percentile. Repeat 2,000 times. The middle 95% of these
values is the interval. Resampling patients rather than slices respects that neighbouring slices are
near-copies of each other.

### 5.4 The decisions

- **G: bound 12 px with a 10% chance-rate cut, not 25 px with 5%.**
  - At 5% the bound would be 25 px, but the 15–19 and 20–24 px bins sit at 5.1% and 5.8% (about
    ±1.7 points each). Whether they fall under 5% is decided by noise.
  - A 25 px bound would also remove 59% of S main from the lesion-level analysis (28% at 12 px).
  - The bound only decides which lesions may count. The false-positive rate is set afterwards by the
    thresholds.
  - 25 px is kept as a sensitivity (E5).
- **H: stratified thresholds.** Single- and multi-lesion slices reach very different chance scores.
  - A single small lesion easily finds a lookalike (τ_D single 0.905).
  - A layout of several lesions rarely overlaps by chance (τ_D multi 0.729).
  - One pooled threshold (0.898) would flag 7.3% of single and 0% of multi slices. The same holds for
    τ_L and τ_C.
- **C was redefined (a deviation from the plan, made on H-cal, before S).**
  - The plan's C kept only lesion pairs with s ≥ τ_L. But τ_L is itself the 95th percentile of the
    strongest pair, so about 95% of H-cal slices have no such pair: their C was exactly 0.
  - The 95th percentile of a score that is 0 for 95% of slices sits at the edge of the zeros: τ_C =
    0.02, meaning "any pair reaches τ_L". It was not a usable configuration threshold, and more data
    would not have helped, since the zeros follow from the definitions.
  - The new C counts every pair with s ≥ 0.1, independent of τ_L, and so measures how alike the
    whole configuration is.
  - The class rule stayed as planned.
- **The null model was dropped** (2026-11-01): fake masks from randomly placed H-cal lesions. It mainly
  served the recombination question, which is out of scope, and real held-out patients are the more
  credible chance baseline.

### 5.5 Reading the figures

*Report versions: `results/3-2_chance_rates` (with three H-eval chance matches), appendix B8, B9.*

**Chance-match rate by lesion area** (`results/stanford_brainmets/e3/bound_review/chance_rates.png`):
- *x-axis:* H-cal lesion area bins, with n in each bin.
- *y-axis:* the share of those lesions whose best match among all T lesions reaches a candidate τ_L.
- One line per candidate τ_L; the **bold black line** is the provisional 0.77; the dashed lines are the
  5% and 10% cuts.
- *What to see:* at τ_L 0.5, around 90% of small held-out lesions find a training lesion that similar,
  so small lesions match by chance all the time. On the bold line, every bin from 12–14 px onwards is
  below 10%, and the bins from 15 to 24 px hover just above 5%. That is why the bound is 12 px.

**Score histograms with thresholds** (`results/stanford_brainmets/e3/calibrate/score_histograms.png`):
- Three panels: E1 score, E2 strongest pair, E2 C of the NN.
- Blue = single-lesion H-cal slices, orange = multi-lesion. The dashed lines are the per-stratum
  thresholds; the solid black line is the pooled one.
- *What to see:* single-lesion slices reach far higher chance scores on E1 and C. A pooled line would
  cut almost nothing from the orange distribution, which is why the thresholds are stratified.

**The thresholds** (`results/stanford_brainmets/e3/calibrate/thresholds.csv`):

| | Single | Multi |
| --- | --- | --- |
| τ_D (E1) | 0.905 [0.896, 0.909] | 0.729 [0.696, 0.769] |
| τ_L (strongest pair) | 0.739 [0.730, 0.761] | 0.816 [0.781, 0.839] |
| τ_C (configuration) | 0.719 [0.709, 0.726] | 0.406 [0.391, 0.432] |

τ_L is higher for multi-lesion slices because a slice with several lesions gets several chances to
have one lucky pair.

**Chance rates on H-cal at these thresholds:** E1 copy 5.1% / 5.3%, E2 full 5.1% / 5.2%, partial
2.3% / 3.5%, so "flagged at all" (full or partial) is 7.4% / 8.7% (single / multi). This is why E4's
false-positive limits are stated per class.

### 5.6 The result

Single small lesions match by chance very often: 73–93% of H-cal lesions of 5–39 px find a training
lesion with s ≥ 0.5. Lesion-level evidence therefore needs informative lesions (≥ 12 px) and
configurations. The thresholds were frozen on 2026-11-03, before any detector touched S.

### 5.7 How to say it

"Following Dar et al. (2024), a match was considered closer than chance if its score exceeded the 95th
percentile of the nearest-neighbour scores of held-out masks against the training set. Thresholds
were set on 25 held-out patients (H-cal), separately for single- and multi-lesion slices, with
bootstrap-by-patient intervals, and frozen before the synthetic set was scored."

### 5.8 Questions you may get

- *Why the 95th percentile?* It is the convention from Dar et al. 2024, and it means a 5% false-positive
  rate on unseen data. Sensitivity: the 99th percentile in E5.
- *Why do you need two held-out halves?* If thresholds are set and checked on the same patients, the
  check is optimistic. H-cal sets them; H-eval checks them (E4) and is the baseline (E5).
- *Isn't 25 patients few?* Yes, it is a stated limitation. The intervals show the uncertainty, and on
  H-eval the frozen thresholds flag about 7% instead of 5%, which is exactly that sampling
  uncertainty (E4).
- *You changed C after seeing H-cal. Isn't that tuning?* It was changed because the planned definition
  could not produce a threshold at all, on H-cal, before S existed in any analysis. The class rule
  stayed the same. It is stated as a deviation.
- *Why is the bound separate from the 4 px lesion definition?* They answer different questions: "is
  this an object or noise?" (4 px) and "is this object distinctive enough to be evidence?" (12 px).
  One number cannot do both. 12 px as the lesion definition would delete real small metastases, and
  4 px as the bound would flood E2 with chance matches.

---

## 6. E4: Validation with planted copies (the gate)

*Script: `e4_validate.py` (plant, validate, check). Plan: `docs/e4-plan.md`. Check: 16/16. Gate:
ACCEPTED with a recorded deviation.*

### 6.1 The question

Do the frozen detectors actually find copies when copies are there? And do they stay near the chance
rate on held-out masks? Without a yes to both, a negative result on S would mean nothing. This is the
first question any reviewer asks.

### 6.2 What was done

1. **Bases (decision I).** 20 T patients × 5 lesion slices (at least 2 apart in z), so 100 base
   masks.
   - 10 patients come from those with ≥ 3 multi-lesion slices (3 multi + 2 single slices each) and 10
     from the others (5 slices each).
   - *Why:* a plain random draw would give only about 25 multi-lesion bases, too few for "remove one
     lesion" (P6-remove), which needs at least two lesions.
2. **Variants**, each made from every base:

   | Variant | What | Tests |
   | --- | --- | --- |
   | P1 | exact copy | is anything broken? |
   | P2 | shifted 2 px | small displacement |
   | P3 | shifted 5 px | beyond E1's ±3 px search (reported only) |
   | P4 | rotated ±5° about the lesions' centroid | small deformation |
   | P5-erode / P5-dilate | 1 px boundary removed / added | boundary noise (reported only) |
   | P6-remove | one informative lesion removed | incomplete copy |
   | P6-add | one H-eval lesion added at its own position, inside the brain, ≥ 3 px from base lesions | copy plus a new lesion |

   Altogether 737 planted masks, each with the source of every lesion recorded.
3. **Gate rule.** A planted mask counts only if every informative base lesion is still ≥ 12 px
   (except P6-remove, which removes one on purpose).
   - *Why:* erosion destroys small lesions (a 1 px erosion leaves none of the 12–19 px lesions at
     ≥ 12 px).
   - Rotation pushed 6 lesions just below 12 px. Those test the bound, not robustness.
   - All masks are still reported.
4. **What counts as caught (decision J):** flagged (E1 copy; E2 full or partial) **and** the named
   source patient is the base patient. A flag pointing at the wrong patient is not a detection.
5. **Go/no-go criteria (decision K),** fixed after planting and before any detector scored the planted
   set:

   | Criterion | Value |
   | --- | --- |
   | P1 exact | 100% by both detectors (a miss would be a bug) |
   | P2, P4 | ≥ 90% by at least one detector |
   | P6 (remove + add) | ≥ 80% by the lesion-level detector |
   | H-eval false positives, per stratum | E1 copy ≤ 7%, E2 full ≤ 7%, E2 flagged ≤ 12% |
   | P3, P5 | reported only |

6. **Run** both frozen detectors on the planted set and on H-eval.

### 6.3 The maths: why P5 can never pass

Erode a lesion so that its area becomes r times the original (median r = 0.43 for a 1 px erosion):
- its shape factor is about r and its size factor is r, so s ≈ r². With r = 0.43, s ≈ 0.18, far below
  τ_L;
- its Dice with the original is 2r / (1 + r), which for r = 0.43 is 0.60, far below τ_D.

Even for lesions ≥ 80 px, s ≈ 0.52–0.59 and Dice ≈ 0.84–0.87 stay below the single-lesion thresholds.
The thresholds are set by chance, so no allowed detector setting could make P5 pass. A P5 criterion
would fail with no permitted remedy. Hence P5 is "reported only": a **stated detection limit**,
decided from the definitions before any score existed.

### 6.4 The result

**Recall** (share caught, of the gated masks; `results/stanford_brainmets/e4/validate/validation.csv`):

| Variant (n) | E1 | E2 | Either |
| --- | --- | --- | --- |
| P1 exact (100) | 100% | 100% | 100% |
| P2 shift 2 px (100) | 100% | 100% | 100% |
| P3 shift 5 px (100) | 13% | 92% | 92% |
| P4 rotate 5° (94) | 78% | 98% | 99% |
| P6-remove (37) | 41% | 92% | 92% |
| P6-add (100) | 67% | 99% | 100% |

All recall criteria pass.

**Lesion level catches what whole mask misses** (first evidence for the RQ2 mechanism):
- E2 alone catches 79 of 100 masks shifted by 5 px, 19 of 37 with a lesion removed and 33 of 100 with a
  lesion added.
- E1 alone catches at most one mask per variant.
- *Why:* an added lesion swamps the Dice of a small base (E1 catches only 30% of P6-add at 12–19 px),
  while E2 still pairs the base lesions perfectly.

**The detection limit, measured:** recall of P5 rises with lesion size, from 0% (erode) and 13% (dilate)
at 12–19 px to 85% at ≥ 80 px.

**False positives on H-eval** (`fp_heval.csv`): E1 copy 7.2% / 7.4%, E2 full 7.1% / 2.9%, E2 flagged
10.1% / 8.5% (single / multi). **Three of the six false-positive criteria fail, by 0.1–0.4 points.**

**The accepted deviation** (decided by the user *after* the run, and stated as such):
- Every bootstrap-by-patient interval contains both the 7% limit and the H-cal rate.
- The flags cluster in few patients: the 3 patients with the most flags give 33–62% of a cell's
  flags. With 25–26 patients per half, the gap is patient sampling.
- It goes both ways: E2 full multi is 2.9% on H-eval against 5.2% on H-cal.
- The planned remedy (re-calibrate on H-cal) gives the same thresholds. Choosing thresholds so that
  H-eval passes would leak H-eval into calibration.
- **Consequence:** E5 compares S with the *measured* H-eval rates, never with the nominal 5%.

**Misses explained:**
- The 4 missed P6 masks were flagged, but the NN was *another patient's* slice, while the strongest
  single lesion pair (s = 1.00) still pointed to the base. A single remaining small lesion can fit
  another patient's slice as a whole better than its own incomplete slice. This is why E7 shows both
  the NN and the strongest pair.
- The one P4 miss: a 13 px lesion shrank to 12 px with shape overlap 0.67, which is not closer than
  chance.

### 6.5 Reading the figures

*Report versions: `results/3-3_recall_by_size` (P3-P6 only), `method/2-7_planted_variants`, appendix C1-C4.*

**Recall by lesion size** (`results/stanford_brainmets/e4/validate/recall_by_size.png`): one panel per
variant; x = the size of the base's largest lesion (bins, with n); y = the share caught. Blue = E1,
orange = E2, dashed = either.
- P1 and P2 are flat at 1.
- In P3, blue stays near 0 until ≥ 80 px: only large lesions keep overlap after a 5 px shift, while E2
  stays at about 0.9.
- In P5, both detectors climb with size: the detection limit.
- In P6-add, blue starts at 0.3 for small bases: one added lesion swamps a small base's Dice.

**What chance matches look like** (`results/stanford_brainmets/e4/validate/heval_flags.png`): two H-eval
slices (never trained on) match a training slice of *another* patient with Dice 1.00 after a 1–2 px
shift, with single lesions of 8 and 4 px. A few-pixel blob has almost no shape, so such matches are
chance. This is why a single small lesion is never evidence on its own.

**Where each variant loses its score** (`factors.png`): the shape, position and size factors of each
planted lesion against its own base lesion.
- P2 loses a little position.
- P5 loses shape and size together (about 0.6 each, so s ≈ 0.36), as the r² reasoning predicted.

### 6.6 How to say it

"Before scoring S, both detectors were validated on 737 planted copies of training masks with known
source, under go/no-go criteria fixed before any detector ran. All recall criteria were met (exact and
2 px-shifted copies 100%, 5° rotation 99%, one lesion removed or added 97%). The lesion-level detector
caught most of the copies the whole-mask detector missed. Copies with 1 px of boundary noise on small
lesions are below the chance thresholds by construction, which defines the method's detection limit.
Three false-positive criteria on H-eval were exceeded by 0.1–0.4 percentage points; this was accepted
as a recorded deviation, and S was compared with the measured H-eval rates."

### 6.7 Questions you may get

- *You accepted a failed gate. Isn't that moving the goalposts?*
  - It is stated openly as decided after the run.
  - Every recall criterion passed; only the false-positive limits were exceeded, by less than half a
    point, with intervals that contain the limit.
  - The fix that would have made it pass (tuning on H-eval) would have destroyed H-eval's independence.
  - And the consequence was made stricter, not looser: S is compared with the measured 7%, not with 5%.
- *Why is the H-eval false-positive rate 7% and not 5%?* The thresholds are the 95th percentile of 25
  patients. On 26 other patients the rate varies by a few points, because flags cluster in a few
  patients. It is sampling, not a broken detector.
- *Why plant copies of training masks? S copies might look different.* Planted copies are the only
  copies whose source is known. Each variant isolates one kind of imperfection (shift, rotation,
  boundary, one lesion), so the result says which imperfections the method tolerates. Combined
  perturbations follow from the parts: C averages lesion scores, each a product of three factors.
- *So you can't detect copies with boundary noise?* Not for small lesions, and no output-only method
  calibrated to chance can: a 1 px change moves a small lesion further from its original than chance
  matches lie. It is a measured limit (0% at 12–19 px, 85% at ≥ 80 px), stated in the limitations.
- *Why does the gate drop eroded masks that lost a lesion?* Then the planted mask no longer contains
  the lesion being tested. That tests the 12 px bound, not the detector's robustness. They are
  reported separately.

---

## 7. E5: The main audit (S scored once)

*Script: `e5_audit.py` (prepare, rehearse, score, analyze, check). Plan: `docs/e5-plan.md` (closed).
Check: 26/26. Locked file: `configs/frozen/e5_stanford_brainmets.yaml`; completion marker
`processed_data/model_space/amibekk_v1/e5/score.complete.json`.*

### 7.1 The question

- **RQ1:** do synthetic masks match training masks more closely than held-out masks do?
- **RQ2 / H2:** at the same false-positive rate, does lesion-level matching flag more than whole-mask
  Dice?

### 7.2 What was done

1. **Fix everything first (decisions L–Q).**
   - Every analysis choice was written into the config section `audit:` and `docs/e5-plan.md`
     *before any E5 code ran*.
   - `prepare` then wrote the E5 file: H-eval's rates per class and stratum (the baseline), the
     τ_D\* threshold for H2, and the held-out maxima for the near-exact rule.
2. **Rehearse** on mock sets with known answers (H-cal; H-cal + 294 planted copies), with the exact
   code that would analyse S.
3. **Score S once** (`score`). Both detectors run on S main, the noisy masks and every sensitivity
   variant in one complete run.
   - Then a completion marker is written, holding the hashes of `audit:` and both frozen files.
   - From then on `prepare` and `score` refuse to run, and `analyze` and `check` refuse if anything
     changed.
4. **Analyse** (`analyze`): rates, excesses, tests, groups, near-exact copies. The verbal reading comes
   from a pre-stated rule.
5. **Check** (`check`, 26/26): the lock is intact, S was read only by `score`, and every table is
   reproduced from the stored raw tables.

### 7.3 The maths

*Excess:* for each class (E1 copy, E2 full) and stratum (single, multi), excess = (share of S flagged)
− (share of H-eval flagged).

*Interval and p (two-level bootstrap):* in each of 2,000 draws:
- resample H-eval *patients* with replacement;
- resample S *masks* with replacement (synthetic masks are independent samples);
- recompute the excess.

The 95% interval is the middle 95% of the draws. The one-sided p = (1 + number of draws with excess
≤ 0) / (draws + 1).

*Holm:* four primary tests (two classes × two strata) look at one question, so the p-values are
adjusted:
1. sort them from smallest to largest;
2. multiply the smallest by 4, the next by 3, and so on, keeping the sequence increasing.

"Beyond chance" means an adjusted p < 0.05.

*AUC:* the probability that a random S mask scores higher than a random H-eval mask. 0.5 = no
difference; 1 = every S mask is higher.

*Size standardisation:* re-weight H-eval's rates to S's mix of strata and largest-lesion sizes. If S
were flagged more only because its lesions have different sizes, the excess would vanish.

### 7.4 The decisions

| | What | Why; rejected |
| --- | --- | --- |
| L | Main set = S main; noisy masks scored by both detectors, reported as separate rows | E2 sees lesions without the specks, so it tests whether excluding noisy masks hides copies; rejected: E1 only on noisy masks |
| M | Primary family = E1 copy and E2 full × single / multi, Holm; everything else secondary (intervals only) | one question, four looks → correct for it; rejected: uncorrected 95% intervals per cell; one pooled endpoint (hides different chance rates) |
| N | Count copies per source patient in *chained groups* (source z within 2 of each other) | one lesion copied across slices is one event; rejected: fixed 5-slice windows |
| O | Sensitivity on S in the same run: 99th percentile, σ 0.08, bound 10 / 25 px, drop-only | every variant compared with H-eval at the same setting |
| P | H2 at equal false-positive rate: (i) E2 full vs E1 copy at the frozen thresholds; (ii) E2 flagged vs E1 at τ_D\* | (ii) is the fair comparison: τ_D\* makes E1 flag as many H-cal slices as E2 flagged does (7.43% / 8.66%) |
| Q | Near-exact = E1 ≥ 0.9 *and* above every held-out E1 score in the same stratum and lesion-size bin | 8% of H-eval single-lesion slices already exceed 0.9; rejected: plain Dice > 0.9 |

The pre-stated reading:
- "memorization beyond chance" if any primary cell is beyond chance;
- else "rare but real" if a near-exact copy exists;
- else "validated negative".

### 7.5 The rehearsal (before S)

- **Null (H-cal):** reproduced its own chance rates; no primary cell beyond chance; no near-exact copy;
  reading "validated negative".
- **Positive (H-cal + 294 planted copies):**
  - all four cells beyond chance;
  - 100 / 100 / 93 of 94 planted copies (P1 / P2 / P4) flagged with the right patient;
  - 257 near-exact copies;
  - reading "memorization beyond chance".
- **A lesson worth quoting:** on the null set, the slice-level Mann–Whitney test gave p = 0.004 for
  multi-lesion E1 scores, as if there were an effect. The patient-bootstrap AUC interval [0.40, 0.74]
  covered 0.5. Treating slices as independent manufactures evidence, which is why the patient
  bootstrap is primary.
- **A second lesson:** H2 failed on the positive set too. Whole copies are caught by both detectors
  alike, so lesion-level matching can only win if the copies are *incomplete*.

*Source: `results/stanford_brainmets/e5/rehearse/{null,positive}/review.txt`.*

### 7.6 The result (pre-registered)

*Source: `results/amibekk_v1/e5/analyze/review.txt`, `rates.csv`.*

| Cell | S | H-eval | Excess [95% interval] | Holm p | |
| --- | --- | --- | --- | --- | --- |
| E1 copy, single | 8.5% | 7.2% | +1.3 [−1.4, +3.9] | 0.32 | chance |
| E1 copy, multi | 15.8% (2,662 / 16,803) | 7.4% | **+8.5 [+4.3, +11.1]** | 0.004 | beyond chance |
| E2 full, single | 6.9% | 7.1% | −0.2 [−2.7, +2.2] | 0.56 | chance |
| E2 full, multi | 5.9% (357 / 6,006) | 2.9% | **+3.0 [+0.4, +4.6]** | 0.036 | beyond chance |

**Reading: "memorization beyond chance", in the multi-lesion stratum only.** The secondary results:
- *Size-confound check:* H-eval standardised to S's size mix gives the same excess (+9.0, +3.0).
- *AUCs, multi-lesion:* E1 score 0.67 [0.56, 0.77], E2 C 0.68 [0.53, 0.78], **strongest single pair
  0.38 [0.32, 0.49]**. The configuration is closer than chance, but S's best single lesion match is
  *weaker* than held-out masks'. That is the first sign that the excess is about layout, not copied
  lesions.
- *Partial copies* are not above chance (single 3.3% vs 3.0%; multi 1.1% vs 5.5%).
- *H2 not supported:* E1 shows more excess than E2 in every comparison ((ii) all: −9.4 points
  [−12.6, −5.8]).
- *Noisy masks* are flagged *less* than H-eval (E2 flagged single 1.7% vs 10.1%), so excluding them
  hid nothing.
- *Groups:* E1 copy flags 4,278 on 1,180 source slices, in 455 chained groups from 99 patients.
- *Near-exact:* 39 masks in 31 groups, with the caveat below.

### 7.7 After the run (exploratory, labelled so)

- **Robustness:** the E1 multi excess holds under every sensitivity variant (smallest: +2.7 at the
  99th percentile). The E2 full multi excess holds at bound 10 and drop-only, but not at σ 0.08, bound
  25 or the 99th percentile.
- **Spread:** the 2,662 E1 multi flags name 96 of 105 patients; the top 5 take only 17%. Copying of a
  few patients would look concentrated, so this already hints at something generic (tested in E6).
- **The near-exact count has no chance baseline.**
  - "Above the highest held-out score" sounds strict, but S has about 17× more masks than a held-out
    set. If S came from the same distribution as the held-out masks, each S mask would still beat the
    maximum of n held-out scores with probability 1/(n + 1). With many S masks, many do.
  - Summed over the cells, about 68 single-lesion masks are expected above their maximum by chance,
    against 35 observed. About 12 E2 multi-lesion masks are expected above the held-out C maximum,
    against 29 observed.
  - So the 39 near-exact masks are *not* by themselves evidence of copying. The definition was
    proposed and approved in planning, and the gap was found after the run, so it is recorded, not
    repaired.
  - The primary family is unaffected: its baseline is a rate, not a maximum.
- **Lesion-count composition** (found while writing this walkthrough, 2026-11-16; read-only, not in
  the locked analysis):
  - E1 calls a mask "multi" when it has two or more lesions of ≥ 4 px. Many S multi masks are one
    informative lesion plus small extras (report 50 is one: a 58 px lesion and a 6 px second one).
    Such masks behave like single-lesion masks, since the small extra barely lowers Dice, but they are
    judged against the much lower multi threshold (0.729 instead of 0.905).
  - S has twice as many of them as H-eval: 42% of E1 multi masks against 21%.
  - Their copy rate is high in *both* sets (S 29.8%, H-eval 26.5%), against 7.0% vs 1.8% for masks
    with two or more informative lesions.
  - Re-weighting H-eval to S's composition (0 / 1 / 2+ informative lesions) gives an H-eval rate of
    13.0%. The E1 multi excess then shrinks from +8.5 to **+2.9 points [−0.4, +6.7]**.
  - The clear excess is in masks with **two or more informative lesions: +5.2 points [+2.8, +6.9]**,
    which is the same population as E2's multi stratum (+3.0).
  - Reading: about two thirds of the headline E1 excess comes from S's different lesion-count mix, not
    from closer matching. The genuine multi-lesion excess is smaller and agrees with E2.
  - The pre-registered result stands as reported; this is an exploratory refinement.
  - *Source: read-only computation on `processed_data/model_space/amibekk_v1/e5/classes_main.csv`
    (S main) and `processed_data/model_space/stanford_brainmets/e4/heval_classes.csv`; 2,000 draws,
    S by mask, H-eval by patient (14 patients in each H-eval group); 2026-11-16.*

### 7.8 Reading the figures

*Report versions: `results/3-4_score_distributions_multi` (multi-lesion panels, S main only; the original also plotted noisy masks), appendix D1-D3.*

**Score distributions** (`results/amibekk_v1/e5/analyze/score_distributions.png`): six histograms,
S (red) against H-eval (green); top row single-lesion, bottom row multi-lesion; dashed = frozen
threshold.
- *Top row:* red and green lie on top of each other. Single-lesion S masks score like held-out masks:
  chance.
- *Bottom left (E1, multi):* the red distribution is shifted right of the green, and more of it lies
  beyond the dashed line. That is the +8.5-point excess, part of it from the composition effect in
  7.7.
- *Bottom middle (E2 C, multi):* red is narrower, with more mass just above the threshold.
- *Bottom right (strongest pair, multi):* red lies slightly *left* of green, so the best single lesion
  matches of S are weaker (AUC 0.38). Whole layouts closer, individual lesions not.

**Sensitivity forest** (`rates_forest.png`): one row per variant and cell, the excess with its
interval. Read: does the interval stay right of 0 in every row? Yes for E1 multi; not for E2 multi in
three variants.

**Near-exact sheet** (`near_exact.png`): the top near-exact masks next to their training slice. They
look convincing, which is exactly why the 1/(n+1) caveat matters: striking examples are expected by
chance in a set this large.

### 7.9 How to say it

"Synthetic multi-lesion masks matched training masks more often than held-out masks did (whole-mask
15.8% vs 7.4%, excess +8.5 points [4.3, 11.1]; lesion-level full matches 5.9% vs 2.9%, +3.0 [0.4,
4.6]; Holm-adjusted p 0.004 and 0.036). Single-lesion masks were at chance. By the pre-stated rule
this reads as memorization beyond chance in the multi-lesion stratum. Two observations limit what
this means:
- the strongest single lesion match was weaker in S than in held-out masks (AUC 0.38);
- in an exploratory re-weighting by lesion-count composition, most of the whole-mask excess is
  explained by S's larger share of masks with one informative lesion and small extras.

What remains, in masks with two or more informative lesions, is examined in E6 and E7."

### 7.10 Questions you may get

- *Why compare with H-eval and not with 5%?* Because the frozen thresholds flag about 7% of H-eval, not
  5% (E4). Comparing S with 5% would invent an excess of 2 points out of sampling noise.
- *How can S and H-eval be resampled differently?* They are different kinds of data. S masks are
  independent draws from a generator; H-eval slices come in patients whose neighbouring slices are
  near-copies. Each set is resampled by its own independent unit, separately, and the statistic is
  recomputed in each draw.
- *Why only multi-lesion?* A single small blob has little shape, so chance matches are already
  frequent, and stage 1 producing typical single lesions is not distinguishable from chance. A layout
  of several lesions is more specific, so a resemblance there is rarer by chance. The composition
  check in 7.7 shows the "multi" label must be read carefully.
- *You found something after the lock. Does it change the result?* The pre-registered reading stays as
  reported. The new analysis is labelled exploratory. It changes the *interpretation*: the headline
  E1 number overstates how much closer S comes, and the robust part agrees with E2.
- *Why is H2 not supported, when E4 showed E2 catches more?* E2 wins on *incomplete* exact copies (a
  lesion added or removed). S has none (E7). The S resemblance is a layout that whole-mask Dice
  rewards and per-lesion thresholds do not.

---

## 8. E6: Patient exposure, rarity and duplication

*Script: `e6_exposure.py` (reference, rehearse, exposure, check). Plan: `docs/e6-e7-plan.md` (closed).
Check: 18/18. E6 never scores S; it reads E5's stored tables.*

### 8.1 The question

- **RQ3a:** which patients are most exposed, and is exposure linked to how unusual they are (H3)?
- **The question E5 left open:** is the multi-lesion excess **copying of particular patients**, or
  **typical lesion placement**? Stage 1 puts lesions in a compact central area, so patients whose
  lesions happen to be there would collect matches without being copied.
- **H1:** are lesions that span many adjacent training slices (that is, duplicated in training)
  matched more often?

### 8.2 What was done

1. `reference` (T only): the 3D lesions of every T patient, slice rarity, patient rarity, and the
   expected shares π_p from H-eval.
2. `rehearse`: the full analysis on E5's mock sets plus two placement mocks, before S. In the placement
   mocks, H-cal lesions are moved to the positions of random T lesions; in the compact variant, only to
   positions inside S's central area.
3. `exposure`: the analysis of S's stored flags, with the pre-stated readings.
4. `check` (18/18).

### 8.3 The maths

**Exposure (decision R).** In each primary cell (E1 copy multi, E2 full multi):
- *Observed:* o_p = the number of flags whose source is patient p (one source per flag, decision J).
- *Expected share:* π_p = how often patient p is the nearest neighbour of a *held-out* mask (chance).
  Held-out masks are few (393 multi-lesion masks name only 82 of 105 patients), so π_p is smoothed
  towards p's share of training slices:

  π_p = (n_p + m·w_p) / (N + m), with m = 52.5 (half a pseudo-mask per patient).
- *Exposure ratio:* r_p = o_p / (π_p · O), where O is all flags in the cell. 1 = as chance predicts;
  10 = ten times more.
- *Concentration:* the dispersion D = Σ (o_p − e_p)² / e_p. A large D means the flags pile up on some
  patients more than chance would. It is compared with D in 2,000 random multinomial draws (with π
  itself redrawn from an H-eval patient bootstrap).

**Copying vs placement (decision U).** Two measures per flag, each compared between S flags and H-eval
chance flags with an AUC:
- *Margin:* how much better the best patient fits than the second-best. A copy of one patient should
  fit that patient uniquely (a large margin). Placement in a common spot fits many patients almost
  equally (a small margin).
- *Source rarity:* how unusual the matched training slice is (1 − its best match to any other
  patient). Copying of memorable cases predicts rare sources; placement predicts typical ones.

The pre-stated reading:

| Reading | Rule |
| --- | --- |
| patient copying | dispersion beyond its null **and** margin AUC interval above 0.5 |
| typical placement | margin AUC interval wholly below 0.5, **or** margin not above 0.5 and rarity AUC interval below 0.5 |
| undetermined | anything else |

**Rarity and H3 (decision V).**
- Patient rarity is built from five features:
  - number of 3D lesions;
  - total lesion volume;
  - median lesion area;
  - most lesions on one slice;
  - spread of lesion positions (brain-relative).
- Each is z-scored; rarity = the mean distance to the 5 most similar patients.
- H3 = Spearman correlation of rarity with the exposure ratio (permutation p, bootstrap interval).

**H1, duplication (decision W).**
1. A 3D lesion = 2D lesions linked from slice to slice, one-to-one, by overlap (Hungarian on pixel
   IoU). Span = the number of slices it covers.
2. Credits = how many S masks match (s ≥ τ_L) any slice of that 3D lesion.
3. Within lesion-size bins: ρ_S = Spearman(span, credits) for S, and ρ_H the same for H-eval.
4. **H1 holds if the size-weighted mean of ρ_S − ρ_H has an interval above 0.**

*Why the comparison with H-eval:* a lesion spanning 10 slices offers 10 slices to match, so even
chance matching rises with span (both mocks showed ρ > 0). Only a rise *steeper than chance* is
duplication.

### 8.4 The decisions

- **R: one source per flag, π_p from H-eval nearest neighbours, smoothed (m = 52.5; m = 105 as
  sensitivity).**
  - Rejected: π_p from H-cal + H-eval pooled, which is circular because the rehearsal null *is* H-cal.
  - Rejected: unsmoothed π_p, which gives infinite ratios for patients no held-out mask names.
- **U: the margin and rarity readings,** because a ratio above 1 alone cannot tell copying from
  placement. *Amended before S was read:* the compact placement mock had margins clearly *below*
  chance but normal rarity, which the first rule called "undetermined". Margins below chance are a
  direct sign of generic placement, so they now read "typical placement".
- **V: rarity on the detectors' own scales** for slices; brain-relative features for patients, since
  rarity is an anatomical question.
- **W: slice-to-slice linking instead of 26-connected 3D labelling.** 26-connectivity merges
  neighbouring metastases that touch in some slice: 38 merged 3D lesions holding 10% of the 2D
  lesions. 26-connectivity is kept as the sensitivity.

### 8.5 The rehearsal (before S)

| Mock | Reading obtained | Expected |
| --- | --- | --- |
| Null (H-cal) | undetermined; margins like chance (AUC 0.45 / 0.53); no H1, no H3 | not "patient copying" ✓ |
| Positive (H-cal + planted copies of 20 patients) | patient copying; margin AUC 0.88 / 0.95; every planted flag names its base patient | "patient copying" ✓ |
| Placement (random T positions) | undetermined | not "patient copying" ✓ |
| Placement, compact | E1 multi: typical placement (margin AUC 0.33); E2 multi: undetermined | not "patient copying" ✓ |

- The reading never mistakes placement for copying.
- It recognises placement only weakly (one of four cells). That limitation is stated.
- A side result of the compact mock (exploratory): compact placement raises E5's *single*-lesion rates,
  not the multi ones, which is the opposite of S. So pure compact placement does not reproduce E5's
  pattern by itself (caveat: the mock keeps H-cal's lesion sizes and counts).

### 8.6 The result

*Source: `results/amibekk_v1/e6/exposure/review.txt`, `concentration.csv`, `margins.csv`, `h3.csv`,
`h1.csv`.*

- **E1 copy multi (2,662 flags):**
  - concentrated: dispersion 7,910 against a null 95th percentile of 2,027; 22 patients lie above
    their 95% chance band (about 2.2 expected); 20 have an exposure-ratio interval above 1;
  - margin AUC **0.36 [0.24, 0.495]**: flags fit their best patient *less* uniquely than chance flags;
  - rarity AUC 0.42 [0.34, 0.498]: their sources are *more typical*;
  - **reading: typical placement.**
- **E2 full multi (357 flags):** concentrated (D 647 vs 384), margin AUC 0.55 [0.27, 0.72], rarity
  0.62 [0.49, 0.81]. **Undetermined.** H-eval has only 8 chance flags here, so there is little power.
- **H3 not supported:** ρ = 0.03 [−0.21, 0.25] (E1 multi) and 0.00 [−0.20, 0.21] (E2 multi). Exposure
  is unrelated to rarity.
- **H1 supported:** weighted ρ_S − ρ_H = **+0.086 [+0.041, +0.163]**; with 26-connectivity +0.077
  [+0.030, +0.163].
  - Carried by lesions of 20–39 px and 40–79 px; reversed at ≥ 80 px.
  - No mock shows it.
  - Caveat: the H-eval side rests on 58 credited 3D lesions.
- **The most exposed patient:** `Mets_230`, 77 E1 multi flags against 1.5 expected. One metastasis
  spans 18 slices (z93–z110), and the flags name 17 of them.

*How to put them together:* the whole-mask excess is best explained by *where* the generator puts
lesions, not by reproducing particular patients. But at the level of single lesions, lesions seen in
many adjacent training slices are matched more than chance, which is the duplication pattern
memorization research predicts (Somepalli et al.).

### 8.7 Reading the figures

*Report versions: `results/3-5a_exposure_margins` (the E1 multi-lesion cell: red = above its chance band, 22), `results/3-5b_h1_rates`, appendix D4-D9.*

**Observed vs expected flags per patient** (`results/amibekk_v1/e6/exposure/exposure_sorted.png`):
- One dot per T patient; x = expected flags (chance, from H-eval); y = observed flags; log axes. The
  diagonal is "as chance predicts".
- Each grey bar is the 95% range of flag counts chance would give that patient. Red dots have an
  exposure-ratio interval above 1.
- *What to see:* many dots far above their bars at the left (patients chance rarely names, but S names
  often), for example `Mets_230`. This is the concentration.
- The figure alone cannot say whether that is copying: a patient with central lesions would also sit
  there.

**Margins** (`margins.png`): boxes of S flags (red) against H-eval chance flags (grey).
- *Top:* the margin to the second-best patient; *bottom:* source rarity. Left = E1 multi, right = E2
  multi.
- *What to see, top left:* the red box sits *lower* than the grey one. S flags fit their best patient
  less uniquely than chance flags do (AUC 0.36): the signature of placement, the opposite of copying.
- *Right:* only 8 grey points, so no conclusion.

**H1 rates** (`h1_rates.png`): one panel per lesion-size bin; x = slice span of the 3D lesion; y =
matches per lesion relative to the bin's average (1 = average); red = S, grey = H-eval.
- *What to see:* in 20–39 px and 40–79 px, the red line climbs steadily with span, and its intervals
  are tight.
- The grey line is noisy (few lesions), which is why the test compares correlations within bins and
  the effect size is uncertain.
- This is a descriptive view of the tested statistic in `h1.csv`.

### 8.8 How to say it

"Exposure was concentrated: 22 training patients received more flags than chance allows, against
about 2 expected. However, flagged masks fitted their best-matching patient less uniquely than
held-out chance flags did (margin AUC 0.36 [0.24, 0.50]), and their sources were more typical. By the
pre-stated rule this reads as typical lesion placement rather than copying of individual patients.
Exposure was unrelated to patient rarity (H3 not supported). Lesions spanning more training slices
were matched more often than held-out chance predicts (H1 supported, +0.086 [0.041, 0.163]),
consistent with duplication driving memorization."

### 8.9 Questions you may get

- *If 22 patients are over-exposed, isn't that memorization?* Not by itself. Patients whose lesions sit
  where S puts lesions (the central area) collect matches without being copied. The margin test
  separates the two, and it says placement.
- *Isn't "typical placement" also a kind of memorization?* It is learning the *distribution* of
  positions, which is what a generator should do. It is not reproducing an individual. Privacy risk
  comes from the individual, not the distribution.
- *Why should duplication matter?* In 2D training, a 3D metastasis spanning 10 slices appears as 10
  near-identical training examples. Duplicated training examples are known to be copied more
  (Somepalli et al. 2023). H1 tests that mechanism directly.
- *H1 is supported but there is no exact copy. Isn't that a contradiction?* No. H1 is about *which*
  lesions get matched more than chance (closer than τ_L), not about exact reproduction. It shows the
  mechanism is active at a modest level.
- *Why is H3 null?* Placement predicts typical patients are exposed (ρ < 0), long-tail memorization
  predicts rare ones (ρ > 0). ρ ≈ 0 fits neither strongly. With 105 patients the power is limited, so
  report the effect size and interval, not just "no".

---

## 9. E7: Match reports and their evaluation (the xAI part)

*Script: `e7_reports.py` (planted, reports, check). Plan: `docs/e6-e7-plan.md` (closed). Check:
14/14.*

### 9.1 The question

**RQ3b:** can each flagged match be explained at patient and lesion level so that:
- the explanation names the **true source** on planted copies with known origin (*correctness*);
- the lesions it names first actually **drive the score** (*faithfulness*)?

### 9.2 What was done

1. **Build and check the report generator on planted copies first** (`planted`), where the answer is
   known. The rules were fixed there and never tuned on S.
2. **Then apply it unchanged to S** (`reports`).

**What a match report shows** (decision X), using report 50 as the example
(`results/amibekk_v1/e7/reports/reports/50_exposed_e1_copy_multi.png`):
- *Images:*
  - the synthetic mask on its synthetic MRI;
  - the named training slice (zoomed, with partner lesions in the query's colours);
  - where that slice sits in the brain;
  - an overlap overlay (green both, red query only, blue source only).
- *Panel B:* the E2 class and stratum, C and the strongest pair against their thresholds; each paired
  lesion's shape / position / size factors, s, and whether it is closer than chance.
- *Panel C:*
  - the E1 score against its threshold;
  - whether it is near-exact or beyond the held-out maximum;
  - the margin;
  - the chained group;
  - the source patient's exposure ratio;
  - the 3D lesions the source lesions belong to.
- *Verdict line,* by rule:
  - it names the decision-J source;
  - it adds the strongest-pair patient if that differs;
  - it says "as a whole closer than chance" or "configuration match", never "copy", when no single
    lesion is closer than chance.

**Report 50 in words:**
- Synthetic mask `file_5944` matches `Mets_230`, slice 110 with whole-mask Dice 0.89 (shifted −2, 2).
  E1 calls the mask *multi* (two lesions ≥ 4 px), so the threshold is 0.73 and it is flagged.
- E2 sees only one informative lesion, so for E2 the mask is *single*. C = 0.71 is just under τ_C
  single (0.72), and the lesion pair (shape 0.88, position 0.92, size 0.88 → s 0.71) is under τ_L
  (0.74). E2 says *novel*.
- Verdict: "Matches `Mets_230`, slice 110 as a whole closer than chance; no lesion-level match closer
  than chance".
- `Mets_230` has 77 flags against 1.5 expected (ratio 51.9 [24.7, 87.3]), and its lesion spans 18
  slices.
- *Lesson:* this flag exists because a tiny second lesion moved the mask into E1's multi stratum (see
  E5, 7.7).

**Selection** (decision X), one report per chained group:
- 30 near-exact;
- 19 E2 beyond the held-out maximum;
- 7 top flags of the 5 most exposed patients;
- 10 random S flags and 10 H-eval chance flags as contrast.

76 reports in total. Every report states its class's chance rate.

**Correctness** (decision Z), on the gated planted masks that E2 flags:
- does the report name the base patient (≥ 90% for P1, P2, P4, P6)?
- for P6, does it pair every kept base lesion with its true source lesion at s ≥ τ_L, and leave the
  added lesion unpaired (≥ 80%)?

**Faithfulness, the deletion test** (decision Y): take multi-lesion flags and delete either the *top*
contributor (the paired lesion with the largest s) or the *smallest* contributor. Three measures:
1. **ΔC** against the named slice. It nearly follows from the definition (C is a sum), so alone it
   proves little.
2. After a **full re-match** against T: is the mask still flagged with the same patient? (McNemar test
   on the flips.)
3. **Δ whole-mask score** against the named slice, a measure that does not use the lesion scores at
   all.

Faithful = deleting the top contributor hurts more on measures 2 and 3 as well.

**Taxonomy** (decision Z), rule-based and fixed on the planted variants:
- *exact* (P1);
- *shifted* (P2, P3);
- *deformed* (every lesion has a partner, but none identical; P4, P5);
- *partial* (some lesions unpaired; P6);
- *likely chance*.

### 9.3 The result

*Sources: `results/stanford_brainmets/e7/planted/review.txt`, `correctness.csv`,
`deletion_summary.csv`; `results/amibekk_v1/e7/reports/review.txt`, `taxonomy.csv`,
`deletion_summary.csv`.*

**On planted copies:**
- *Correctness passes every criterion:*
  - source patient 100% for P1, P2, P4, 99.0% P6-add, 91.9% P6-remove;
  - right lesions 99.0% (P6-add) and 81.1% (P6-remove), against ≥ 80%.
- *Faithfulness passes:*
  - deleting the top contributor lowers the whole-mask score by 0.33 against 0.15 (Wilcoxon
    p < 0.001);
  - it makes the mask lose its flag or source in 23 cases against 6 (McNemar p 0.001);
  - ΔC barely differs (0.43 vs 0.41), as expected: in an exact copy every pair scores about 1.

**On S:**
- *Taxonomy:* **no flagged S mask is an exact lesion-level copy.**
  - Multi-lesion flags: 50% deformed (against 13% of H-eval chance flags) and 50% partial (against
    87%).
  - Single-lesion flags look like chance: deformed 65% vs 66%, partial 33% vs 30%.
- *Deletion (224 multi-lesion flags):*
  - ΔC 0.29 vs 0.16 and Δ whole-mask 0.16 vs −0.04 (both p < 0.001);
  - the re-match measure is at its ceiling: almost every mask loses its flag whichever lesion is
    deleted (224 vs 203 of 224). Most such masks have exactly two informative lesions, so either
    deletion leaves a single-lesion mask, judged by the stricter single-lesion thresholds;
  - the McNemar p is small (21 discordant pairs, all one way), but the measure says little here.

**Patient profiles** (descriptive, added after the S run): the matches fall along the long
metastases of the most exposed patients.
- `Mets_230`: one 18-slice metastasis, matched on almost every slice.
- `Mets_058`: 17 metastases. The matches concentrate on three that span 6–9 slices, while most short
  ones get none. Two lesions equally long and large (D6 and D14) get none vs all the matches, so
  position matters too.

### 9.4 Reading the patient profile

*Report version: `results/3-6_report50_profile` (report 50 and `Mets_230`'s flags per slice); the full profiles are appendix E4 (`Mets_058`).*

`results/amibekk_v1/e7/reports/patients/Mets_230.png`:
- *Top:* each 3D lesion as a grey bar over the slices it spans. A red dot marks an S mask matching
  that lesion on that slice, and its size shows the count.
- *Middle:* flags naming each slice (blue = E1, orange = E2). Most fall on z94 (24 E1 flags), with
  matches spread over the whole length.
- *Bottom:* the most-matched slice with its zoomed lesion, next to four S masks that match it. They
  have the same blob in the same place, each drawn slightly differently. "Deformed", not copied.

### 9.5 How to say it

"The match reports were built and checked on planted copies before they were applied to S. They named
the true source patient for 92–100% of planted copies and the correct lesions for 81–99% of partial
copies, meeting all criteria. In a deletion test, removing the lesion the report ranked first lowered
the whole-mask score more than removing the lowest-ranked lesion (0.33 vs 0.15 on planted copies,
0.16 vs −0.04 on S; p < 0.001), so the explanations are faithful. Applied to S, no flagged mask was an
exact lesion-level copy: multi-lesion flags resembled a training slice as a whole, with every lesion
paired but none identical."

### 9.6 Questions you may get

- *Isn't the deletion test circular? C is a sum of lesion scores.* For ΔC, almost yes, which is why two
  measures independent of C were added. The whole-mask Dice and a full re-match do not use the lesion
  scores, and both favour the top contributor.
- *No human rater: how do you know the reports make sense?* Correctness is measured against known
  answers, which is stronger evidence than a rater's impression for the claims made. A blinded human
  check of usefulness is planned for the master thesis and stated as a limitation.
- *Why show both NN and strongest pair?* They can name different patients (E4: 4 P6 masks). Showing
  only one would hide that the evidence is split.
- *What does "deformed 50% vs 13%" mean?* S's multi-lesion flags more often have a partner for every
  lesion than chance flags do (which usually have unpaired lesions). Their layout resembles a training
  slice, but none of the lesions is reproduced exactly. It is resemblance, not reproduction.

---

## 10. E8: The paired-MRI check (exploratory)

*Script: `e8_paired.py` (select, rehearse, paired, check). Plan: `docs/e8-plan.md` (closed). Check:
17/17.*

### 10.1 The question

**RQ4 / H4:** stage 2 was trained with each slice's own real mask. If a synthetic mask closely
resembles a training mask, does stage 2 draw that training patient's *tissue*, as if the mask were a
"key"? Concretely: is the synthetic MRI of a strongly matched mask closer to the source slice than
the MRI of a novel mask is to its nearest training slice? This is observational: it can show an
association, not cause.

### 10.2 What was done

1. **Group A** (decision AA): the 56 strongest S resemblances (E7's near-exact, beyond-held-out and
   most-exposed cases; one per chained group; 41 source patients). Each is compared with its
   decision-J source slice.
2. **Group B** (decision AB): for each A case, 5 novel S masks matched on:
   - stratum and number of informative lesions;
   - size bin of the largest lesion;
   - synthetic brain-area tertile.

   Each is compared with its nearest training slice. 280 controls.
3. **Reference:** the same design on H-eval (real flagged vs real novel slices), where no copying is
   possible.
4. **Measure** (decision AC), per pair:
   1. brain masks of both images (as E0);
   2. the synthetic image moved by the E1 shift (≤ 3 px), with no rescaling;
   3. each image z-scored by its own brain's mean and SD;
   4. a local SSIM map (Gaussian window σ = 1.5 px, about 11×11 px; data range fixed at 10);
   5. the map averaged over a region inside *both* brains:
      - the **lesion neighbourhood** (informative lesions grown by 8 px; primary);
      - the whole brain (secondary);
      - the ring (the neighbourhood without lesion pixels; sensitivity).
5. **Test** (decision AD): AUC of A vs B, judged against H-eval's AUC of flagged vs novel. Within A:
   SSIM to the source vs to the best slice of the runner-up patient, and a profile over z ± 5.
6. **Rehearsal** (decision AE), before S's MRIs were read:
   - an H-cal null;
   - a positive mock: planted copies whose "MRI" is the base patient's BRAVO two slices away.

### 10.3 The maths

*Z-score:* z = (x − mean of brain) / SD of brain, over the whole image. It removes the brightness and
contrast differences that come from per-volume min–max scaling.

*SSIM* compares two image patches on brightness, contrast and structure:

SSIM = [(2μ_xμ_y + C1)(2σ_xy + C2)] / [(μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2)]

- μ are local means, σ² local variances and σ_xy the local covariance, all Gaussian-weighted.
- C1 = (0.01·L)² and C2 = (0.03·L)², with L the data range (10).
- 1 = identical patches, about 0 = unrelated.

*The reading:*
- **supported** if the A-vs-B AUC interval is above 0.5 *and* the difference to H-eval's AUC is
  above 0;
- **explained by mask resemblance** if the AUC is above 0.5 but not above H-eval's;
- **not supported** otherwise.

### 10.4 The decisions (and the brain-size question)

- **Why the lesion neighbourhood is primary:** A's synthetic brains are a median 0.51× the area of
  their source slices' brains. Whole-brain SSIM would mostly compare brain outlines.
- **Why no rescaling:** stage 1 keeps the training mask's pixel positions, and stage 2 conditions on
  the mask at those positions. A reproduction of the source tissue would therefore sit at those pixel
  positions, and rescaling the brain would move it away.
- **How the size mismatch is handled:**
  - only pixels inside both brains count, so tissue is never compared with background;
  - B is matched on brain size, so A and B carry the same mismatch;
  - a split by brain-size half is reported.
- **Why the H-eval reference:** where two masks overlap, both images have a bright lesion in the same
  place, so the lesion region looks alike *without* any memorization. H-eval measures how big that
  effect is (AUC 0.53: small).
- **Data range 10** (set by the user, 2026-11-16): about the median whole-image range of z-scored T
  BRAVO (9.80), fixed for every pair. Rejected: a per-pair max − min, which changes with each lesion's
  brightness.
- **Rejected:** A vs B without a reference; whole-brain SSIM as primary; a 2×-diameter crop (about
  12 px for the median lesion, barely larger than the SSIM window).

### 10.5 The result

*Sources: `results/amibekk_v1/e8/paired/review.txt`, `tests.csv`, `within_a.csv`;
`results/stanford_brainmets/e8/rehearse/review.txt`.*

| Set | Lesion-neighbourhood AUC | Reading |
| --- | --- | --- |
| Rehearsal null (H-cal) | 0.48 [0.39, 0.56] | not supported ✓ |
| Rehearsal positive (same patient, z + 2) | 0.985 [0.97, 1.00] | supported ✓ |
| H-eval reference (flagged vs novel) | 0.53 [0.47, 0.60] | — |
| **S, group A vs B** | **0.48 [0.40, 0.57]** (difference to H-eval −0.05 [−0.16, +0.05]) | **not supported** |

- Whole brain (0.48) and ring (0.47) agree.
- **Within A:** SSIM to the source minus SSIM to the runner-up patient's best slice is +0.002
  [−0.018, +0.023] (p 0.70). The source patient's tissue is no closer than another patient's with a
  similar mask.
- **Scale:**
  - the same patient two slices away ≈ 0.46;
  - different patients ≈ 0.05;
  - typical A cases ≈ 0.03;
  - the strongest A case 0.26 is within the spread of unrelated H-eval pairs.

### 10.6 Reading the figures

*Report versions: `results/3-7_ssim_lesion` (the box plot with a typical and the strongest S case and the rehearsal positive), appendix E5, E6.*

**`results/amibekk_v1/e8/paired/examples.png`**, one row per case:
- the aligned synthetic image;
- the source BRAVO;
- the SSIM map, with the lesion neighbourhood outlined in orange on all three.

The background in the map is yellow (both flat, so SSIM ≈ 1), but it lies outside the outline and is
never counted. Values inside the outline are 0.00–0.07 for typical cases.

**`ssim_groups.png`:** three panels (lesion, brain, ring) with four boxes each: S group A (red),
S group B (grey), H-eval flagged (red), H-eval novel (grey).
- The panel title gives the A-vs-B AUC with its interval, H-eval's AUC and the difference.
- *What to see:* the A box is not higher than B (if anything lower: medians 0.031 vs 0.049). S boxes
  sit lower than H-eval's overall, because synthetic images with smaller brains are less similar to
  any real slice. That is why A is compared only with B, and H-eval serves as a reference for the
  difference, not for absolute levels.

**`within_a.png`:**
- *Left:* mean SSIM of each A image to the source patient's slices at offsets −5…+5 (0 = the source).
  The lesion-region line peaks at about 0.055 near 0 and falls to about 0.025 at ±5; the whole-brain
  line is flat. Mask resemblance alone predicts this, since the source lesion is largest near the
  source slice. It is descriptive, with no held-out reference.
- *Right:* source minus runner-up per region. Every interval covers 0.

### 10.7 How to say it

"Synthetic MRIs of the 56 strongest mask resemblances were no closer to their source slice than MRIs
of matched novel masks were to their nearest training slice (lesion-neighbourhood SSIM AUC 0.48
[0.40, 0.57], against 0.53 for held-out flagged vs novel slices). They were also no closer to the
source than to another patient's slice with a similar mask. The pre-stated reading is 'not
supported'. The analysis is observational; whether a training mask acts as a key into stage 2
requires an interventional test with the model (master thesis, RQ-M1)."

### 10.8 Questions you may get

- *Your positive mock was trivially easy (AUC 0.985). Does that show the test is sensitive?* No. It
  shows the analysis is wired correctly (alignment, regions, test), not how faint a resemblance it
  could detect. A graded positive mock (z ± 1, 2, 5, 10) is planned for the master thesis. Stated as a
  limitation.
- *The brains have different sizes. Isn't SSIM meaningless then?* Only pixels inside both brains
  count, the region is local around the lesion, and B is matched on brain size, so A and B share the
  mismatch. The comparison is between groups, not about absolute SSIM.
- *Why did you run E8 if E5's near-exact count had no baseline?* The decision to run it was taken
  explicitly as lean and exploratory: group A is the *strongest* resemblances available, which is the
  best chance to see an effect if there is one. The raw count was not used as the justification.

---

## 11. How the results fit together

**The chain of evidence:**
1. **E5:** S's multi-lesion masks come closer to training masks than held-out masks do. The best
   *single* lesion match is weaker (AUC 0.38), and most of the whole-mask excess reflects S's
   lesion-count mix (exploratory, 7.7).
2. **E6:** the matches fit their best patient *less* uniquely than chance and point to typical slices:
   **typical placement**. Exposure is unrelated to rarity (no H3). But lesions duplicated across many
   training slices are matched more (H1).
3. **E7:** no exact lesion-level copy; multi-lesion flags are *deformed* resemblances of a whole
   layout. The explanations are correct and faithful.
4. **E8:** even the strongest resemblances bring no sign of the source patient's tissue.

**The one-sentence synthesis.** Stage 1 learned *where* metastases tend to be and roughly what layouts
look like, so its multi-lesion masks resemble training layouts more than chance. It does not
reproduce individual patients' lesions. Duplication of long lesions across slices nudges matching
towards them, the known mechanism of memorization, at a level that produces resemblance but no
verbatim copies that the validated detectors can see.

**Why "multi-lesion only" makes sense.**
- A single lesion of 10–40 px is nearly featureless at 128×128: 73–93% of held-out lesions find a
  training lookalike. So single-lesion "copies" cannot be told from chance.
- Layouts of several lesions are specific, so chance resemblance is rarer and an excess becomes
  visible.
- But part of what E1 calls "multi" is one lesion plus small extras (7.7), so the robust multi signal
  is in masks with two or more informative lesions: about +3 to +5 points, by E2 and by the
  exploratory split.

**What each hypothesis came to:**

| | Hypothesis | Outcome | Why |
| --- | --- | --- | --- |
| H1 | copy rate exceeds held-out, concentrated on long (duplicated) lesions | first part: yes for multi-lesion masks (placement); second part: supported | E5 excess; E6 ρ_S − ρ_H +0.086 [+0.041, +0.163] |
| H2 | lesion level flags more than whole mask at equal false-positive rate | not supported on S (supported on planted partial copies) | S has no incomplete exact copies |
| H3 | memorization concentrates in rare configurations | not supported | ρ ≈ 0 |
| RQ3b | correctness ≥ 90% / 80%, faithfulness p < 0.05 | passes | E7 |
| H4 | paired MRIs of flagged masks closer to source | not supported | E8 AUC 0.48 |

---

## 12. Limitations and how to defend them

For each: what it is, the direction in which it biases the result, and the answer.

| Limitation | Bias | Answer |
| --- | --- | --- |
| **Detection limit: 1 px boundary noise** on small lesions (P5: 0% at 12–19 px, 85% at ≥ 80 px) | towards *missing* copies | Measured, not assumed. No output-only method calibrated to chance can do better, since a 1 px change moves a small lesion further than chance matches lie. Results are a lower bound for such copies. |
| **Lesions < 12 px** carry no lesion-level evidence | towards missing small-lesion copies | They match by chance too often to be evidence; E1 still scores those masks. |
| **E2's centroid rounding at 1 px shifts** (half-to-even; 3.9% of T lesions lose shape overlap) | towards missing | Seen in S: 2 of 4 masks identical to a training mask after a ≤ 3 px shift are called novel by E2, but E1 catches all 4. The detectors cover each other. |
| **E1's candidate shortlist** (40 candidates for the shift search) | towards missing (slightly conservative) | Same procedure for every set; E4 recall on 2 px shifts is 100%. |
| **About 25 patients per held-out half**; flags cluster in patients | uncertain chance rates (± a few points) | Bootstrap-by-patient intervals everywhere; S compared with the *measured* H-eval rate (7%), not 5%. |
| **The E4 gate was accepted after the run** (false-positive criteria exceeded by 0.1–0.4 points) | none on recall; makes S's baseline stricter | Stated openly; the alternative (tuning on H-eval) would destroy its independence. |
| **C redefined after a look at H-cal** | none on S (before S) | The planned C could not yield a threshold at all; the class rule stayed the same. |
| **Near-exact count and the E8 entry rule have no chance baseline** | overstates "strong" evidence if taken raw | Found and stated; not used as evidence; E8 run explicitly as exploratory. |
| **E1 "multi" mixes lesion-count compositions** (exploratory, 7.7) | overstates the E1 multi excess | The pre-registered result stands; the composition split shows the robust part (about +5 points in 2+ informative masks) agrees with E2. |
| **Placement mocks recognise placement only weakly** | "undetermined" may hide placement | The rule never misreads placement as copying, which is the error that matters. |
| **H1's held-out reference is sparse** (58 credited lesions) | effect size uncertain | The pre-stated test passes and no mock shows it; report the interval. |
| **E8 observational, brain-size mismatch, mock shows correctness not sensitivity** | towards not finding | Stated; RQ-M1 tests the key effect by intervention. |
| **The repo is not exactly the code that ran; no seeds; sampling filters only partly hold** | uncertainty about S's provenance | Where data and repo disagree, the data wins (decision E). The master thesis retrains with seeds. |
| **Stage-1 slice selection only bounded** (if ≤ 2 px slices were dropped, 10 of 3,823 T slices were never seen) | negligible | Bounded at 0.3% of T. |
| **No human rater** of the reports | unknown usefulness to people | Correctness is measured against known answers; a blinded check is planned for the master thesis. |
| **One analyst made every decision** | possible bias | Pre-registration, locks, mock rehearsals and check stages are designed to remove it. |
| *Resolved:* stage-1 validation set | — | Confirmed 2026-11-16: the whole train set was used, nothing held out, test patients never seen. |

---

## 13. Decision index (A–AE)

The letters S and T are skipped because they name data sets. The full table, with values and config
keys, is in `docs/report-synthesis.md` section 5.

| Letters | Experiment | Walkthrough section |
| --- | --- | --- |
| A (brain), B (min lesion 4 px), C (attach gap 2 px), D (noisy > 9 specks), E (sampling filters) | E0 | 2.2 |
| F (pixel frame) | E2 | 4.4 |
| G (bound 12 px), H (stratified thresholds) | E3 | 5.4 |
| I (planting design), J (caught), K (go/no-go) | E4 | 6.2, 6.3 |
| L (sets), M (tests), N (chained groups), O (sensitivity), P (H2), Q (near-exact) | E5 | 7.4 |
| R (exposure), U (copying vs placement), V (rarity, H3), W (H1) | E6 | 8.3, 8.4 |
| X (reports), Y (deletion test), Z (correctness, taxonomy) | E7 | 9.2 |
| AA (group A), AB (group B), AC (measure), AD (test), AE (rehearsal) | E8 | 10.2, 10.4 |

**Deviations from the plan, and why each was acceptable:**

| Deviation | When | Why acceptable |
| --- | --- | --- |
| Pixel instead of brain-relative positions (F) | before any score | stage 1 never sees a brain; brain-relative positions would measure stage 2 |
| C redefined | on H-cal, before S | the planned C could not be calibrated (0 for 95% of H-cal) |
| P5 reported only | before any detector ran | it cannot pass at chance-calibrated thresholds, by the r² argument |
| E4 false-positive criteria failed, accepted | after the run, stated | tiny excess within intervals; the fix would leak H-eval; the baseline was made stricter |
| Null model dropped | E3 planning | held-out patients are the more credible baseline; it served an out-of-scope question |
| Rule U amended | after the placement mock, before S | margins below chance are a direct sign of placement |
| Near-exact lacks a baseline | after the E5 run | recorded, not repaired, not used as evidence |
| One source per flag in exposure (not every pair) | E6 planning | π_p is defined by nearest neighbours; every-pair crediting used where adjacency matters (H1) |

---

## 14. From here to the master thesis

**What the preliminary project hands over:**
- `memtk`, a toolkit that runs on any (mask, MRI) pool with a config entry:
  - preprocessing and QC;
  - both detectors with calibration;
  - planted-copy validation;
  - the audit statistics;
  - exposure;
  - match reports;
  - the paired-MRI check.
- The finding that stage 1 shows placement-level resemblance and a duplication signal, but no verbatim
  copies.
- A list of what the output-only audit cannot see.

**The master thesis** (Branch B: retrain both models with fixed seeds and known splits):
- **RQ-M1, the mask as a key, by intervention:**
  - give stage 2 training, held-out and synthetic masks with fixed noise;
  - run perturbation series (shift, erode, remove a lesion);
  - add a graded positive mock to measure sensitivity.

  E8 found no observational sign. RQ-M1 asks the causal question.
- **RQ-M2:** does the full pipeline leak even when each stage passes on its own?
- **RQ-M3:** which slices and patients get memorized and why?
  - training-data attribution (e.g. D-TRAK) for `Mets_230` and `Mets_058`;
  - a position-only null generator to settle placement vs copying definitively.
- **RQ-M4:** explain stage 2's reliance on the mask by gradient attribution on the mask channel and
  condition sensitivity against an empty mask (Wen et al. style).
- **Also:**
  - a blinded human check of the match reports;
  - recombinations of lesions across patients;
  - duplication-aware training (does deduplicating adjacent slices reduce H1's effect?). This is a
    suggestion added here, not yet in `docs/thesis-plan.md`.
