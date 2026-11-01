# Experiment Plan — Preliminary Project: Mask Generator Memorization

Oct 22, 2026 · @Mattias Lund Dahlin

## Overview

There are nine experiments in a fixed order. Build two detectors, calibrate them on held-out patients, and prove they work on planted copies. Only then score the synthetic masks, once. Everything after that (patient exposure, match reports and the optional MRI check) analyses those frozen results.

&#91;embedded content: experiment sequence · E0 to E8, one validation gate\]

The dashed line marks where thresholds are frozen. Nothing above it ever sees the synthetic masks, so the detectors can't be tuned toward a desired result.

| Experiment | Answers | Needs |
| --- | --- | --- |
| E0 Data audit | Is the data what we think? | Real and synthetic masks, training code |
| E1 Whole-mask detector | Baseline for RQ2 | E0 |
| E2 Lesion-level detector | Core method for RQ1–RQ3b | E0, the size bound from E3 |
| E3 Chance calibration | What counts as a copy; which lesions count | E1, E2, H-cal |
| E4 Planted-copy validation | Do the detectors work? (gate) | E3, H-eval |
| E5 Main audit | RQ1, RQ2 | E4 passed |
| E6 Patient exposure | RQ1 (H1 duplication), RQ3a (H3 rarity) | E5 |
| E7 Match reports | RQ3b, the xAI contribution | E5, E6 |
| E8 Paired MRI check (optional) | RQ4 (H4), exploratory | E5, about 20 or more flagged masks |

## Shared setup

Every experiment uses the same four data sets, one preprocessing pipeline and fixed definitions. All thresholds are fixed in E3–E4, before anything is run on the synthetic masks in E5. That rule is what makes the result credible: you can't tune the detector until the synthetic data "looks interesting".

**Data sets**

| Name | Content | Role |
| --- | --- | --- |
| T (training reference) | About 3,800 lesion-containing mask slices of the 105 training patients, each with patient ID, slice index and paired real BRAVO slice | What stage 1 could have memorized |
| H-cal (held-out, calibration) | Lesion slices from 25 of the 51 test patients (992 slices), chosen by the seeded split below | Sets thresholds (E3) |
| H-eval (held-out, evaluation) | Lesion slices from the other 26 test patients (1,094 slices) | Negatives for validation (E4) and the baseline in E5 |
| S (synthetic) | About 37,000 synthetic masks from stage 1, each with the synthetic MRI it conditioned. After quality control (E0), the clean and minor masks. | What is being audited |

**Only slices with metastases are compared.**

- T contains only the training slices that hold at least one lesion, about 3,800. Lesion-free slices were never seen by the mask generator and are never used as references.
- Every synthetic mask contains lesions too, since stage 1 was trained only on lesion slices and quality control removes failed samples.
- H-cal and H-eval are restricted the same way, to lesion-containing slices.

All comparisons in E1–E7 are therefore between metastasis-containing slices.

Split the test patients into H-cal and H-eval by patient, with a fixed seed, and keep the split for the whole project. Using the same held-out masks both to set thresholds and to measure false positives would be circular.

*Implemented (E0a):* a matched-pair split, so both halves cover the same range of lesion burden. Sort the 51 test patients by their number of lesion slices, cut the list into consecutive pairs, and send one patient of each pair at random to H-cal and the other to H-eval (seed 2026). The odd patient out goes to H-eval. Result: H-cal 25 patients (median 35 lesion slices, range 3–110), H-eval 26 (median 36, range 3–119). The split is stored in `splits/stanford_brainmets_seed2026.csv`.

**Preprocessing (identical for all four sets)**

1. Bring real data into model space with the repo's own transform. For each volume, apply ScaleIntensity (min–max to \[0, 1\]), then MONAI Resize((128, 128, −1)), whose default interpolation is area averaging. Then binarize the masks at 0.01. The original masks are binary (check that they hold only 0 and 1), but area averaging makes edge pixels fractional, so re-binarizing enlarges each lesion slightly. Put the BRAVO volumes through the same transform. Compare everything in this space, because it is what the models saw and could memorize.
2. Synthetic masks are already binary in the saved files: the sampling script min–max scaled each mask and thresholded it at 0.01. Use them as saved.
3. Label lesions as connected components (8-connectivity). Then apply attach-or-drop to every component smaller than a minimum area. If it lies within 2 pixels of a real-sized lesion, attach it to that lesion as a stray pixel; otherwise drop it as a speck. Two real-sized lesions are never merged. The minimum is learned in E0 from the smallest component sizes that occur in converted real masks, because at 128×128 a genuinely small metastasis covers only a few pixels.
4. Compute brain-relative coordinates. Take the brain bounding box from the paired MRI (Otsu threshold, largest component) and express each lesion's centroid as (x, y) in \[0, 1\] inside that box. Real masks use the real BRAVO slice; synthetic masks use their paired synthetic MRI.
5. Stage 1 was trained without augmentation, so the detectors need no flip or rotation tolerance. A small shift tolerance stays, to absorb sampling noise.

**What the repo confirms** ([amibekk/Thesis](https://github.com/amibekk/Thesis), `Generation/`):

- **Aspect ratio is not kept.** `Resize((128, 128, -1))` scales each in-plane axis independently, so a 160×176 slice is squeezed by factors of 0.80 and 0.73. The distortion differs per patient, which is why positions use brain-relative coordinates and all comparisons happen in model space.
- **Depth is not resampled** (the −1), so slice indices and adjacency are preserved.
- **Masks are area-averaged, then thresholded at 0.01.** Any pixel with more than 1% lesion coverage becomes lesion, which enlarges lesions by about one pixel at the edge. For small lesions that is a large share of their area. Measured in E0a, per patient, the lesion area after conversion is a median of 1.5 times what plain downscaling would give (range 1.08–2.70). E0 step 5 measures it per lesion. No lesion slice is lost or gained by the conversion. Synthetic masks were also thresholded at 0.01.
- **The stage-1 slice selection is not fully known.** The repo's training loop computes an empty-slice filter (`np.sum(image) > 1.0`) but never applies it, and as written it could not train the mask-only model, since it passes the two-channel batch to a one-channel network. The mask channel is binarized at 0.01 in every case, matching our conversion. If the filter actually used dropped slices with at most 1 or 2 lesion pixels, T contains 2 or 10 of its 3,823 slices that stage 1 never saw. That bound is small enough to state as a limitation rather than act on.
- **BRAVO intensity is per-volume min–max scaling to \[0, 1\]** (`ScaleIntensity`), not a histogram method.
- **There is no augmentation** (no flips or rotations), so exact-position copies are plausible.
- **There are no seeds.** The training script in the repo would also not run as written (it is missing the `time` import and calls `merge_data` with the wrong number of arguments). The repo is very close to the code that was run; differences are probably limited to file paths and small fixes like these.

**How S was produced is not fully known.** The repo's `Sampling_mask_bravo.py` makes 15,000 attempts and keeps a pair only if all three conditions hold:

1. The mask's pixel sum is below 2,000.
2. At least one lesion is smaller than 40 pixels.
3. Every lesion pixel lies inside the generated brain.

Your pool has about 37,000 samples. They were generated in several rounds with a script very close to the repo version, probably differing only in file paths. An earlier check flagged about 2,243 masks as poor quality; the quality control in E0 re-runs that check to verify it. This has two consequences:

- Check each filter on the data itself (for example, how many masks lack a lesion under 40 pixels). Apply to the real comparison sets only the filters the pool actually shows.
- A single noise speck counts as "a lesion under 40 pixels". Re-apply that rule after quality control, to synthetic and real sets alike.

**Definitions**

- **Nearest neighbour (NN):** for a mask m, the training slice in T with the highest similarity score.
- **Match score:** the similarity of m to its NN, under a given detector (whole-mask in E1, lesion-level in E2).
- **Copy:** a match score above the threshold τ fixed in E3. Read it as "closer to a training slice than 95% of held-out patients ever get".
- **Source patient:** the patient that owns the NN slice. All counting is per patient as well as per slice, because adjacent slices of one patient are near-duplicates.

* **Informative lesion:** a lesion at or above the size bound set in E3. Only informative lesions count in lesion-level matching, because blobs of a few pixels match each other by chance. This bound is higher than the speck threshold used in quality control.

**Tools and reproducibility**

- Python with NumPy, SciPy (`ndimage.label`, `optimize.linear_sum_assignment`), scikit-image (`regionprops`, `threshold_otsu`) and PyTorch for batched similarity. Everything runs on a CPU; a GPU only speeds up E1.
- One global seed. Every run writes its config and seed to a results folder.
- Cache per-mask features (lesion table, brain box) once in E0, and reuse them everywhere.

## E0 – Data audit and lesion statistics

**Purpose:** Confirm the data are what you think they are, and describe the lesions. The same statistics feed the informative-lesion bound and the optional null model (E3), and the rarity measure (E6).

**Inputs:** T, H-cal, H-eval, S, and the previous thesis's training code and configs.

**Procedure**

1. **Confirm the split.** The previous author confirmed the standard 105/51 split. Still record the patient IDs in each set from the data folders, since every later step depends on a clean held-out set. The stage-1 checkpoint name (correct\_validation\_data\_16\_Epoch149\_of\_200) suggests a validation set was used. It is assumed to come only from the 105 training patients.
2. **Count:**
   - Training patients.
   - Lesion slices per patient: minimum, median and maximum.
   - Synthetic masks.
   - Native in-plane size per patient, which sets how much each patient's slices were squeezed.
3. **Run the preprocessing pipeline once** on all four sets and cache a lesion table with one row per lesion:
   - Set, patient, slice and lesion ID.
   - Area, equivalent diameter, eccentricity, solidity.
   - Brain-relative centroid.
   - A crop centred on the lesion: 16×16 pixels, enlarged for lesions that don't fit.
4. **Describe the distributions:** lesions per slice, lesion area and centroid positions (a 2D density plot). Compare T, H and S side by side. Large differences between S and T, for example S having far fewer multi-lesion slices, are worth knowing before you interpret matches.
5. **Run sensitivity checks:**
   - Re-run step 3 with the minimum lesion area at the learned value and one pixel either side, and with attach distances of 1 and 2 pixels. Also compare each real lesion's area at native resolution with its area after conversion, to measure how much the conversion enlarges lesions.
   - Pick the setting where lesion counts are stable, and fix it for the rest of the project.
6. **Check adjacency.** For each training patient, compute the Dice between consecutive lesion slices. This measures how near-duplicate neighbouring slices are, which is the inflation risk raised by Akbar et al.

**Mask quality control (part of E0)**

Some synthetic masks are noisy, with many scattered 1–2 pixel specks. A likely cause is the sampling script itself: it min–max scales each raw mask and thresholds at 0.01, which turns faint background noise into "lesion". Speckle isn't a copy of anything, so noisy masks are excluded by a rule fixed before E5. Run the check fresh rather than reusing the earlier count of about 2,243 poor masks. Comparing the two verifies that earlier finding.

1. **Learn real lesion sizes.** Convert the real training masks to model space and record the distribution of component sizes. Set the minimum lesion area from it, for example at the 1st percentile. Components smaller than that count as specks.
2. **Count specks** in each synthetic mask. Separate stray pixels (within 2 pixels of a lesion, which preprocessing attaches) from isolated specks, and count how many isolated specks lie outside the brain of the paired MRI.
3. **Classify each mask:**
   - **Clean:** no isolated specks. Stray pixels attached to a lesion don't count.
   - **Minor:** a few isolated specks, for example 3 or fewer. Keep the mask; preprocessing removes them.
   - **Noisy:** more isolated specks than that. Exclude.

   Choose the cut-off by viewing about 100 random masks spread across the range of speck counts.
4. **Edge check, for real-sized lesions only.** Flag a lesion only if more than half its area lies outside the brain. A pixel or two over the edge is normal, given the 0.01 threshold and how often metastases sit near the cortex. Run the same check on real slices, and loosen it if real slices fail.
5. **Re-apply the "lesion under 40 pixels" rule** after cleaning, to all sets.
6. **Record the counts:** total pool, clean, minor, noisy, how many drop out under step 5, and how many stray pixels were attached. If attachments are common, say so in the report: it is another effect of the per-mask scaling and 0.01 threshold.

Also keep the masks exactly as saved: stage 2 was conditioned on them, specks included. E8 and the master thesis use that version.

**Outputs:**

- A data description table.
- Three distribution figures (lesions per slice, area, position).
- The lesion-table cache.
- The fixed preprocessing parameters.
- The median adjacent-slice Dice.

**Check before moving on:** The split is confirmed clean. Every synthetic mask has a quality class (clean, minor or noisy), the counts are recorded, and the cut-off was fixed before any detector touched S.

## E1 – Whole-mask nearest-neighbour detector

**Purpose:** Build the standard detector: for every mask, find the most similar training mask by overlap. It is the baseline that RQ2 compares against, and it mirrors what existing medical memorization studies do.

**Inputs:** Binarized masks from S, H-cal, H-eval and T (from E0).

**Procedure**

1. **Compute plain Dice to every training mask.** Flatten each binary mask into a vector. For a query set Q and the training set T, intersections are one matrix product: Q·Tᵀ. Dice is 2·|q∩t| / (|q| + |t|). This is fast even for thousands of masks on a CPU.
2. **Choose candidates.** Keep, for each query mask, the top 20 training masks by plain Dice, plus the top 20 by centroid distance of the largest lesion. The second list catches copies that are shifted by a few pixels and so have low plain Dice.
3. **Allow shifts and flips.** For each candidate, recompute Dice over shifts of ±3 pixels in x and y (49 positions, about 2% of the 128-pixel field of view). No flips are needed, since training used no augmentation. The candidate's score is the maximum over these.
4. **Record for each query mask:**
   - The NN slice and its source patient.
   - The score.
   - The shift that achieved it.
   - The margin: the best score minus the best score from any other patient. A large margin means a match to one specific patient; a small one means several patients look alike.
5. **Run on all three query sets** (S, H-cal, H-eval) with identical code.

**Outputs:** A results table per set (query ID, NN slice, source patient, score, shift, margin), and histograms of NN scores for S against H-eval.

**Pitfalls:**

- Two small round lesions at a common location can reach a high Dice by coincidence. The held-out calibration in E3 handles this, so don't interpret raw scores.
- Report the top-k cut-off. If a planted shifted copy in E4 is missed, raise k before changing anything else.

## E2 – Lesion-level matching detector

**Purpose:** Build the detector that is new in this project. It compares each synthetic slice with each training slice, lesion by lesion, so it can see partial copies that whole-mask Dice blurs together. Its score is built from parts you can read, which is what makes the match reports explainable (E7).

**Inputs:** The lesion table from E0, covering every lesion in S, H-cal, H-eval and T, and the informative-lesion bound from E3. Run the same code on S, H-cal, H-eval and the planted copies (E4).

**Procedure**

1. **Keep informative lesions only.** Lesions below the size bound set in E3 are neither matched nor counted. A blob of a few pixels has almost no shape, so its matches are mostly chance. Slices with no informative lesion leave the lesion-level analysis, but the whole-mask detector (E1) still scores them.
2. **Score two lesions.** For a lesion a (synthetic) and a lesion b (training), multiply three terms, each between 0 and 1:
   - **Shape:** IoU of the two lesion crops from E0, centred on their centroids.
   - **Position:** exp(−d² / 2σ²), where d is the distance between brain-relative centroids. Start with σ = 0.05, i.e. 5% of the brain box.
   - **Size:** smaller area divided by larger area.

   The lesion score s(a, b) is the product. Store all three factors, not just the product: they are the explanation.
3. **Score a slice pair.** For a synthetic slice q and a training slice t:
   - Pair their lesions one-to-one with the Hungarian algorithm (`linear_sum_assignment` on cost 1 − s).
   - Keep the pairs with s ≥ τ\_L.
   - C(q, t) = (sum of kept pair scores) / max(lesions in q, lesions in t). Extra or missing lesions lower the score, and C = 1 means the same lesions in the same places with the same shapes.

   Example: q has lesions A and B, t has X, Y and Z, and the best pairing is A–X (0.9) and B–Z (0.2). Only A–X passes τ\_L, so C = 0.9 / 3 = 0.30, a partial copy.
4. **Compute only the pairs that can score above zero.** About 35,000 × 3,800 slice pairs is too many to check one by one. Two shortcuts give exactly the same result:
   - **Position prefilter:** skip lesion pairs whose centroids are more than 0.2 of the brain box apart. With σ = 0.05, their position factor is exp(−8), about 0.0003, so s can never reach τ\_L. Confirm this once τ\_L is fixed (E3).
   - **Candidate slices:** compute C(q, t) only for training slices that share at least one lesion pair with q scoring s ≥ 0.1. For every other training slice, C is zero by definition.
5. **Classify each synthetic slice** from its nearest neighbour (NN), the training slice with the highest C:
   - **Full copy:** C(q, NN) ≥ τ\_C.
   - **Partial copy:** C(q, NN) < τ\_C, but at least one lesion pair in the NN slice passes τ\_L.
   - **Novel:** no candidate slice has a lesion pair that passes τ\_L.

   Also record the strongest single lesion-pair score across all candidate slices, and the slice it is in. This shows whether a stronger single-lesion match ever sits outside the NN slice.

**Outputs:** three tables, stored once and used by every later experiment.

| Table | One row per | Contents | Used by |
| --- | --- | --- | --- |
| Slice pairs | Candidate synthetic–training slice pair | Synthetic ID, training slice, patient, slice index, C, lesion counts in each slice, number of matched lesions, strongest single lesion score | Copy classes (E5), exposure and H1 (E6) |
| Lesion pairs | Hungarian-matched lesion pair in each candidate slice pair | Both lesion IDs; the shape, position and size factors; s; whether it passed τ\_L | Match reports and deletion test (E7) |
| Synthetic summary | Synthetic slice | NN slice and C, class, strongest single lesion match and its slice | Rates (E5), report selection (E7) |

Store every candidate pair, not only those above the thresholds. Thresholds are then applied at analysis time, so sensitivity analyses need no recomputation. It also means a lesion copied across several adjacent training slices credits all of them, which E6 needs for H1.

**Optional: small-lesion agreement.** Lesions below the informative-lesion bound stay out of C, but they can support a match that has already been found. Two small blobs coinciding somewhere is likely by chance. Once the informative lesions of q match t, it is much less likely that the small lesions also sit in the same places.

- For each candidate slice pair, count the small lesions in q that have a small lesion in t within a few pixels, for example 3. Use centroid distance only, since their shape carries no information.
- Store that count, and the number of small lesions in each slice, as extra columns in the slice-pairs table.
- Run the same check on H-cal matches to get a chance baseline. Small lesions are common in S because of the sampling filter, so agreement only means something above that baseline.

The check never changes C or the copy class. It is supporting evidence for the match reports (E7).

**Pitfalls:**

- Report how many lesions and slices the informative-lesion bound removes from S and T.
- Report results separately for single-lesion and multi-lesion slices.
- Recombinations (lesions matched to different patients) are out of scope. They are noisy, easily produced by chance, and need a null model; they move to the master thesis as an optional idea.

## E3 – Chance calibration

**Purpose:** Turn raw scores into a meaningful "copy" decision, and decide which lesions are large enough to count as evidence. The reference is real patients the model never saw (H-cal), which shows how close a mask gets to the training set when it was *not* copied. A null model is an optional extra.

**Inputs:** H-cal, the lesion table from E0, and the detectors from E1 and E2.

**Procedure**

1. **Set the informative-lesion bound.**
   - For each H-cal lesion, find its best lesion score against T. Compute the share above a provisional τ\_L, binned by lesion area.
   - Choose the smallest area at which this chance-match rate drops below 5–10%. Expect roughly 10–20 pixels at 128×128.
   - Before fixing it, check how many S and T slices would have no lesion above the bound. The synthetic pool favours masks with a lesion under 40 pixels, so a high bound could remove much of S.
   - This bound is separate from, and higher than, the speck threshold used in quality control. Decide it during coding, together with the E0 quality-control re-run, by checking how each candidate bound affects the synthetic pool.
2. **Set the held-out thresholds.**
   - Run E1 and E2 with H-cal as the query set, using only informative lesions.
   - Set each threshold at the 95th percentile of H-cal: τ\_D for whole-mask Dice, τ\_C for the configuration score, and τ\_L for the best lesion-pair score.
   - Compute them separately for single-lesion and multi-lesion slices.
   - Steps 1 and 2 depend on each other through τ\_L: set a provisional τ\_L, then the bound, then the final τ\_L. One round is enough.
3. **Confirm the prefilter is exact.** Check that at the prefilter distance (0.2 of the brain box), s stays below τ\_L for every lesion size.
4. **Check sensitivity with one alternative per setting:** σ of 0.08 instead of 0.05, the 99th percentile instead of the 95th, and a bound one step lower.
5. **Freeze.** Write all thresholds and settings to a config file under version control, before E4 and well before E5.
6. **Optional, if time allows: a null model.**
   - Build as many fake masks as there are in S from H-cal lesions placed at independent positions (drawn from a 2D density of H-cal centroids, inside the brain box, without overlap).
   - Match the lesion counts to S's distribution.
   - Run both detectors on them. This gives stable chance rates for partial copies, especially in multi-lesion slices where H-cal is sparse.

**Outputs:**

- A frozen threshold file: τ\_D, τ\_C and τ\_L (stratified by lesion count), the informative-lesion bound, and the prefilter distance.
- Chance rates from H-cal, and from the null model if it was built.
- How many lesions and slices the bound excludes.
- A short sensitivity table.

**Check before moving on:** The thresholds are frozen, and the share of S excluded by the bound is acceptable. If it is large, agree with your supervisor before continuing.

## E4 – Detector validation with planted copies

**Purpose:** Prove the detectors find copies when copies exist, and stay quiet when they don't. Without this, a result of "no memorization" in E5 could just mean a blind detector. This is the go/no-go gate before the main audit.

**Inputs:** T, H-eval, and the frozen thresholds from E3.

**Procedure**

1. **Choose the base masks.** With the fixed seed, pick 20 of the 105 training patients and 5 lesion slices from each, giving about 100 base masks. Include both single-lesion and multi-lesion slices, each with at least one informative lesion.
2. **Create planted variants** from each base mask:

   | Variant | How it is made | What it tests |
   | --- | --- | --- |
   | P1 Exact | Unchanged copy | Sanity check |
   | P2 Small shift | Shift by 2 px | Shift tolerance |
   | P3 Large shift | Shift by 5 px | Tolerance limit |
   | P4 Rotation | Rotate by ±5° | Robustness to small geometric change |
   | P5 Morphology | Erode or dilate by 1 px | Robustness to boundary noise from sampling |
   | P6 Partial | Remove one lesion, or add one lesion from H-eval | Partial-copy detection |
3. **Assemble the negatives.** Use all H-eval masks. They are real and were never trained on, so every flag on them is a false positive.
4. **Run both detectors** on positives and negatives, with the frozen thresholds.
5. **Measure:**
   - Recall per variant: the share of planted masks given the right class.
   - Source-patient accuracy: the share where the flagged source is the true patient.
   - False-positive rate on H-eval.

   Present this as one table, with the whole-mask and lesion-level detectors side by side.

**Go/no-go criteria (suggested; to be decided during the work, but fixed before E4 runs):**

- P1 recall about 100%.
- P2, P4 and P5 recall at least 90% for at least one detector.
- The lesion-level detector catches P6 in at least 80% of cases.
- False-positive rate on H-eval at most 7% (95% is the calibrated target, so about 5% is expected).

**If it fails:** Change detector settings (top-k, shift range, σ), re-freeze thresholds on H-cal (E3), and re-run E4. Never adjust anything by looking at S.

**Outputs:** The validation table, and the planted copies themselves, which E7 reuses to check whether explanations are correct. It also doubles as the first evidence for RQ2: the cases where whole-mask Dice misses P6 or P7 and the lesion-level detector catches them.

## E5 – Main audit (RQ1, RQ2)

**Purpose:** Answer whether the mask generator copies more than chance (RQ1), and whether lesion-level matching finds copies that whole-mask matching misses (RQ2). This is the first time S is scored against frozen thresholds.

**Inputs:** S, H-eval, the null masks from E3 if they were built, the frozen thresholds, and both validated detectors.

**Procedure**

1. **Score the synthetic masks.** Run both detectors on S once, with the frozen config. Save the raw tables before doing any analysis.
2. **RQ1, population level.**
   - Compare the NN score distribution of S with that of H-eval, using a one-sided Mann–Whitney U test. Report the effect size as AUC: the probability that a random synthetic mask is closer to the training set than a random held-out mask.
   - Do this for both detectors, split into single-lesion and multi-lesion masks.
3. **RQ1, copy rate (first part of H1).**
   - Compute the share of S flagged as full copies, with a 95% bootstrap confidence interval. Synthetic samples are independent, so resample them directly.
   - Compare with the H-eval rate (bootstrap by patient, since its slices are clustered) and, if it was built, with the null rate.
   - Memorization beyond chance means the S rate sits clearly above both. Report rates on the clean and minor masks. Then repeat with the noisy masks included, using the whole-mask detector only, to show that the exclusion doesn't hide copies. Also re-run the lesion-level detector with drop-only (no attaching), to show that the stray-pixel rule doesn't change the conclusions.
4. **Remove adjacent-slice duplicates.** Group flagged synthetic masks by source patient and slice window (±2 slices), and count each group once. Report both the raw count and the grouped count. The gap shows how much near-duplicate adjacent slices inflate the raw number.
5. **Count near-exact copies separately** (whole-mask Dice > 0.9). These are unambiguous and make strong figure examples.
6. **RQ2, missed by whole-mask.**
   - Cross-tabulate the whole-mask flag (yes or no) against the lesion-level class (full, partial, novel). Use only slices with at least one informative lesion, so both detectors are compared on the same subset.
   - The headline number: the share of lesion-level flags that whole-mask Dice does not flag. To test H2, compare the two detectors at the same false-positive rate on H-eval.
   - Compare the partial-copy rate in S with the H-eval rate, and with the null rate if it was built. Partial copies only matter if S produces them more often than chance.

**Outputs:**

- A main results table: rates with confidence intervals for S, H-eval and, if built, the null, for each detector and class.
- A score-distribution figure (S, H-eval and null overlaid).
- The cross-tabulation for RQ2.
- A list of flagged synthetic masks, which feeds E6 and E7.

**How to read the outcomes:**

| Outcome | Meaning |
| --- | --- |
| S rate clearly above H-eval and null | Memorization beyond chance; go on to E6 and E7 as planned |
| S rate near H-eval and null, but a few near-exact copies | Rare but real memorization; E6 and E7 focus on those cases |
| S rate at or below chance, no near-exact copies | A validated negative result (E4 shows the detector works); the master thesis centres on RQ-M1 and RQ-M2 |

## E6 – Patient exposure, rarity and duplication (RQ1 H1, RQ3a H3)

**Purpose:** Move from "how many samples are copies" to "which patients are exposed, and why". Then test two hypotheses together. H3 says memorization concentrates on patients with unusual metastasis configurations, as Feldman and Zhang's long-tail argument predicts. H1 says copies concentrate on lesions that appear in many adjacent training slices.

**Inputs:**

- Flagged copies from E5, grouped by slice window.
- NN assignments for H-eval, and for the null masks if they were built.
- The lesion table from E0.

**Procedure**

1. **Observed exposure.** For each training patient p, count the grouped copies whose source patient is p. Count full copies and partial copies separately. Use every stored slice pair above threshold, not only nearest neighbours, so a lesion copied across adjacent training slices credits all of them.
2. **Expected exposure.**
   - Some patients attract matches simply because they have many slices or common-looking lesions.
   - Estimate this from the H-eval masks, and from the null masks if they were built: each gets an NN patient even when not flagged, so their shares give the expected share π\_p for each patient.
   - Exposure ratio = observed copies of p / (π\_p × total copies). A ratio well above 1 means over-exposed.
3. **Rarity per slice.** For every training slice, compute 1 minus its highest configuration score C against slices from *other* training patients. High rarity means no other patient has anything similar. This uses the E2 detector's own similarity scale, so rarity and copying are measured on the same scale.
4. **Rarity per patient.**
   - Describe each patient by lesion count, total lesion area, median lesion size, the most lesions on one slice, and the spread of lesion positions.
   - Standardize these, and define rarity as the mean distance to the 5 most similar other patients.
5. **Test H3.**
   - **Slice level:** compare the rarity of training slices that are the source of a copy against all other training slices (Mann–Whitney, with AUC as effect size).
   - **Patient level:** compute the Spearman correlation between patient rarity and exposure ratio. Use a 10,000-permutation test for the p-value and a bootstrap over patients for the confidence interval.
   - With 105 patients the power is limited, so report effect sizes first and p-values second.
6. **Describe the top 10 most-exposed patients:** lesion count, sizes, locations and number of slices, under pseudonymous IDs only.

**Duplication test (H1), run alongside step 5**

1. **Label lesions in 3D.** In each training volume, after the same 128×128 conversion, label lesions with 26-connectivity, so a lesion touching across slices counts as one.
2. **Record each 3D lesion's slice span:** how many 2D training slices contain it.
3. **Count copies per 3D lesion.** Map each 2D training lesion to its 3D lesion. Using all stored slice pairs above threshold, count how many synthetic copies (full or partial) matched one of its slices.
4. **Test whether copies rise with slice span** with a Spearman correlation between slice span and copy count across 3D lesions. Because bigger lesions span more slices, repeat it within lesion-size bins, so size alone doesn't drive the result.
5. **Account for the sampling filter.** The synthetic pool favours masks with at least one lesion under 40 pixels, which in turn favours small lesions spanning few slices. Count only training slices that would pass the same filter, and state this in the report.

**Outputs:**

- A sorted exposure plot showing how concentrated copies are: whether a few patients carry most of them.
- A rarity-versus-exposure scatter with the correlation and confidence interval.
- A table of the most-exposed patients, and a plot of copies against 3D slice span (H1).

**Pitfall:** If E5 found very few copies, E6 becomes descriptive. Report the cases you have rather than forcing a statistical test.

## E7 – Match reports and their evaluation (RQ3b, xAI)

**Purpose:** Explain every flagged copy in terms a clinician or data steward can check: which patient, which slice, which lesion, and why. Then test two properties. Faithfulness: the lesions a report names are the ones that actually drive the score. Correctness: on planted copies with a known origin, the report names the true source patient, slice and lesions. That evaluation is what makes this an xAI contribution rather than just a visualization.

**Inputs:** The flagged list from E5, the per-lesion tables from E2, the planted copies from E4, and the E6 results.

**Procedure**

1. **Generate a match report** for each flagged synthetic mask, or for the top 50 by score if there are many. Each report has four parts:
   - **Panel A:** the synthetic mask and its NN training mask side by side, plus an overlay coloured by overlap, synthetic-only and training-only.
   - **Panel B:** lines connecting matched lesion pairs, and a small table with each lesion's shape, position and size factors and its contribution to C.
   - **Panel C:** context. Source-patient pseudonym and slice index, adjacent slices that also match, the margin to the second-best patient, the patient's exposure ratio from E6, and the chance rate for this lesion count from E3.
   - **Verdict line:** generated from rules, for example: "2 of 3 lesions match patient P017, slice 84; the third lesion is novel → partial copy."

   You are authorized to use the data, so reports showing real masks can go in the written report. Use pseudonymous patient IDs.
2. **Sort the flags into a taxonomy.** Review all flagged cases (or the top 50) and assign each to one of: exact, shifted, deformed, partial, or likely false alarm (a very common configuration). Count each category. This is the qualitative finding.
3. **Test faithfulness by deletion.**
   - For each flagged multi-lesion mask, remove the lesion the report names as the largest contributor and recompute C against the source slice.
   - Then remove instead the lesion named as the smallest contributor.
   - A faithful explanation shows a much larger drop for the first deletion. Report the mean drop in each case, with a paired Wilcoxon test.
4. **Check correctness on planted copies.** Flagged synthetic samples have no known source, so correctness is measured on the planted copies from E4, where the true source is known.
   - Generate match reports for all planted copies.
   - For each, record whether the report names the correct source patient, the correct source slice (or one within ±2 slices of it), and, for partial copies (P6), the correct lesions.
   - Report the share explained correctly for each variant.
   - Suggested criteria: the correct source patient for at least 90% of planted copies, and the correct lesions for at least 80% of P6.

**Optional: small-lesion agreement in the reports.** If the E2 small-lesion check was run, add it to Panel C and the verdict line, for example: "the large lesion matches patient P017, slice 84, and both small lesions also coincide in position (H-cal chance rate: x%)". It strengthens or weakens the case for a copy, but it does not change the class, the taxonomy or the deletion test.

**Outputs:**

- A match-report generator in code.
- Three to four anonymized example reports for the written report.
- Taxonomy counts.
- Deletion-test results.
- Correctness on planted copies, by variant.

**Limitation:** A blinded human check of the reports was not feasible in the preliminary project and moves to the master thesis. State this in the report.

## E8 – Paired synthetic MRI check (RQ4, H4, optional)

**Purpose:** An optional, observational first look at the master thesis's "mask as key" question. When a synthetic mask copies training slice t, is the paired synthetic MRI also unusually close to t's real BRAVO slice?

It is the last experiment and depends on E5. Run it only if E5 yields about 20 or more flagged masks and time remains before 24 November. Otherwise it moves to the master thesis as a phase-1 pilot.

**Inputs:** Flagged and novel synthetic masks from E5, their paired synthetic MRIs, and the real BRAVO slices of T. When relating a mask to its MRI, use the mask exactly as saved, specks included, since stage 2 was conditioned on that version.

**Procedure**

1. **Form two groups:**
   - **Group A:** synthetic pairs whose mask is a full or partial copy of training slice t.
   - **Group B:** synthetic pairs with novel masks, matched to A by the number and size of informative lesions.
2. **Pick each pair's comparison slice.** For A, use the source slice t. For B, use the NN training slice: the highest C, or the highest whole-mask Dice if C is zero.
3. **Normalize the images.** Put the real BRAVO slice through the training preprocessing (resize to 128×128, \[0, 1\] normalization). Then z-score intensities inside the brain region of both images, and align them with the shift found in E1.
4. **Measure two similarities** between the synthetic MRI and its comparison slice:
   - **Whole-brain:** SSIM over the brain region.
   - **Lesion region:** SSIM in a crop around the matched lesions, about 2× the lesion diameter, which includes the surrounding tissue.
5. **Compare and control.**
   - Compare A against B with a Mann–Whitney test, for both measures.
   - Within A, also compare similarity to t against similarity to other slices of the same patient (±5 slices), and against other patients' slices with similar masks.

**Output:** A short table and one example figure, reported as preliminary and observational.

**Limits to state in the report:**

- If A is more similar than B, it suggests the mask acts as a key, but it doesn't prove it. Both stages may have memorized the same rare patient independently, and common configurations may simply produce similar tissue.
- Only the interventional experiment in the master thesis (RQ-M1) can separate these explanations.
- One detail helps: the sampling script reused a single starting-noise tensor for every synthetic MRI, so differences between MRIs come from the mask and the per-step sampling noise, not from the starting point.

## Schedule, deliverables and decisions

The full draft goes to your supervisor by Nov 30, 2026, and the final report is due Dec 9, 2026. The plan uses milestone dates rather than fixed weeks, so heavy and light weeks can even out. Each milestone only needs to be met by its date. The critical path is E0 → E3 → E4 → E5; E6 and E7 can overlap once E5 is done.

| Complete by | Experiments | Done when | Priority |
| --- | --- | --- | --- |
| 31 Oct | E0, including mask quality control | Split confirmed; preprocessing and quality-control rules fixed; lesion table cached | Must |
| 9 Nov | E1, E2, E3 | Both detectors run on H-cal; informative-lesion bound and thresholds frozen | Must |
| 9 Nov, if time | E3 null model | Stable chance rates for partial copies | Could |
| 13 Nov | E4 (gate) | Validation passed | Must |
| 17 Nov | E5 | Main audit tables for RQ1 and RQ2 | Must |
| 24 Nov | E6; E7 match reports, deletion test and correctness on planted copies | Results for RQ3a, H1, H3 and RQ3b | Must |
| 24 Nov, if time | E8 | Optional RQ4 (H4); otherwise moved to the master thesis | Could |
| 30 Nov | Writing | Full draft to supervisor | Must |
| 1–8 Dec | Revision after feedback | Final report | Must |

**Write while you run.** Draft the data and methods sections from early November, while E1–E4 are running, so only results and discussion remain after 24 November.

**If you fall behind, cut in this order:**

1. E8 (optional RQ4). Move it to the master thesis.
2. The optional small-lesion agreement in E2 and E7.
3. The optional null model in E3, if it hasn't been built yet.
4. The E7 taxonomy. Keep the deletion test and the correctness check on planted copies.
5. The lesion-level part of the E7 correctness check. Keep source-patient correctness.

Never cut E4: without validation, a negative result means nothing.

**Deliverables for the written report**

- **Tables:** data description (E0); validation (E4); main rates with confidence intervals (E5); RQ2 cross-tabulation (E5); most-exposed patients (E6).
- **Figures:** lesion distributions (E0); score distributions for S, H-eval and the null (E5); exposure concentration and rarity scatter (E6); 3–4 anonymized match reports (E7).
- **Code:** a reusable mask-memorization toolkit (preprocessing, both detectors, calibration, report generator). The master thesis runs it again on retrained or original models.

**Decisions still open**

- [ ] Go/no-go criteria for E4: decided during the work, but fixed before E4 runs.
- [ ] The informative-lesion bound: decided during coding, by looking at how candidate bounds affect the synthetic pool, together with the quality-control re-run in E0.
