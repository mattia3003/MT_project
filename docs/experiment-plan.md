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
4. Compute brain-relative coordinates (used for anatomical questions such as E6 rarity; copy detection in E2 uses pixel coordinates, decision F, 2026-11-03). Take the brain bounding box from the paired MRI (decided in E0b, 2026-11-01: pixels above a low threshold, every region ≥ 30 px, holes filled; plain Otsu with the largest component trimmed dim tissue and lost separate brain pieces) and express each lesion's centroid as (x, y) in \[0, 1\] inside that box. Real masks use the real BRAVO slice; synthetic masks use their paired synthetic MRI.
5. Stage 1 was trained without augmentation, so the detectors need no flip or rotation tolerance. A small shift tolerance stays, to absorb sampling noise.

**What the repo confirms** ([amibekk/Thesis](https://github.com/amibekk/Thesis), `Generation/`):

- **Aspect ratio is not kept.** `Resize((128, 128, -1))` scales each in-plane axis independently, so a 160×176 slice is squeezed by factors of 0.80 and 0.73. The distortion differs per patient, which is why all comparisons happen in model space (and why brain-relative coordinates were first planned for positions; E2 uses pixel coordinates, decision F, because stage 1 copies would keep the pixel positions of the training mask in this same space).
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

**Purpose:** Confirm the data are what you think they are, and describe the lesions. The same statistics feed the informative-lesion bound (E3) and the rarity measure (E6).

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

*Status (2026-11-03): done.* Implemented as `scripts/e0a_prepare_data.py` and `scripts/e0b_audit.py` (stages, decisions and status in `docs/e0b-plan.md`); the check passes (E0a 32/32, E0b 43/43), and a full re-run from scratch gave byte-identical outputs. Fixed parameters are in `configs/default.yaml` (minimum lesion area 4 px, attach gap 2 px, noisy above 9 isolated specks, brain mask `low_threshold_all` at 0.02); the environment is pinned in `requirements-lock.txt`. Outputs: data description table, distribution figures, adjacent-slice Dice and native-vs-model areas in `results/stanford_brainmets/e0b/stats/`; the lesion table in `processed_data/model_space/<dataset|pool>/lesions.csv`. Deviations from the procedure above, with reasons in `docs/report-notes.md`:
- Minor masks have up to 9 isolated specks, not 3: the counts show a break at 9 (decision D).
- A failed synthetic MRI does not exclude its mask (separate `mask_class` and `mri_failed` columns); edge flags only flag.
- The "lesion under 40 pixels" rule is a flag column, not a filter: on the data it fails for 10.7% of the masks as saved, so it is not applied to the real sets (decision E; only the pixel-sum filter is).
- The lesion table stores the centroid in three frames (pixels, image fraction, brain-relative); E2 uses pixel coordinates for position (decision F, 2026-11-03), brain-relative coordinates stay for anatomical questions.
- The 16×16 lesion crop (step 3) is not cached; it is built in E2, where it is used.
- Besides the plan's items, E0 adds a bound-impact table (input for the informative-lesion bound in E3) and brain size by slice height.

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
   - **Position:** exp(−d² / 2σ²), where d is the distance between the centroids in pixel (model-space) coordinates. Start with σ = 0.05 of the 128-pixel field, i.e. 6.4 px. *(Decision F, 2026-11-03: pixel coordinates instead of the brain-relative coordinates first planned; reasons under "Carried over from E0" below.)*
   - **Size:** smaller area divided by larger area.

   The lesion score s(a, b) is the product. Store all three factors, not just the product: they are the explanation.
3. **Score a slice pair.** For a synthetic slice q and a training slice t:
   - Pair their lesions one-to-one with the Hungarian algorithm (`linear_sum_assignment` on cost 1 − s).
   - Keep the pairs with s ≥ 0.1 (the candidate cut of step 4).
   - C(q, t) = (sum of kept pair scores) / max(lesions in q, lesions in t). Extra or missing lesions lower the score, and C = 1 means the same lesions in the same places with the same shapes. C measures how alike the whole configuration is; it does not depend on τ\_L. Every lesion counts equally, whatever its size (kept after a discussion on 2026-11-03: size weighting would make C behave like whole-mask Dice and blur the contrast RQ2 tests, and an added small lesion, as in E4's P6, would barely lower C; how much lesion area is copied is reported separately, E5 step 7).

   Example: q has lesions A and B, t has X, Y and Z, and the best pairing is A–X (0.9) and B–Z (0.2). C = (0.9 + 0.2) / 3 = 0.37. Only A–X passes τ\_L; if 0.37 stays below τ\_C, q is a partial copy.

   *Changed 2026-11-03 (E3):* C first kept only the pairs with s ≥ τ\_L. Since τ\_L is set so that only about 5% of held-out slices have such a pair, C was 0 for about 95% of H-cal slices and τ\_C (its 95th percentile) came out at about 0, so full and partial copies could not be told apart. Counting every pair from the candidate cut makes C a real configuration score (H-cal median 0.54 for single-, 0.29 for multi-lesion slices); the shortcuts stay exact.
4. **Compute only the pairs that can score above zero.** About 35,000 × 3,800 slice pairs is too many to check one by one. Two shortcuts give exactly the same result:
   - **Position prefilter:** skip lesion pairs whose centroids are more than 0.2 of the field apart (25.6 px). With σ = 0.05 (6.4 px), their position factor is exp(−8), about 0.0003, so s can never reach τ\_L. Confirm this once τ\_L is fixed (E3).
   - **Candidate slices:** compute C(q, t) only for training slices that share at least one lesion pair with q scoring s ≥ 0.1. For every other training slice, C is zero by definition.
5. **Classify each synthetic slice** from its nearest neighbour (NN), the training slice with the highest C:
   - **Full copy:** C(q, NN) ≥ τ\_C. The configuration as a whole is closer than chance, even if no single lesion passes τ\_L (several moderately similar lesions in the same places).
   - **Partial copy:** C(q, NN) < τ\_C, but at least one lesion pair over all candidate slices passes τ\_L (the strongest single pair, as τ\_L is calibrated).
   - **Novel:** neither.

   Also stored for every slice: how many lesions of the NN slice pass τ\_L (`n_kept`) and whether all of them do (`all_copied`), which separate one copied lesion from several in the reports. On H-cal (all chance, stratified thresholds) the rule gives full 5.1% / 5.2%, partial 2.3% / 3.5% (single / multi), so 7–9% of held-out slices are flagged at all; E5 compares S with H-eval per class. (Between the τ\_C problem and this fix, a structural rule, "every lesion paired above τ\_L", was used for one run on 2026-11-03; it is now only the descriptive `all_copied`.)

   Also record the strongest single lesion-pair score across all candidate slices, and the slice it is in. This shows whether a stronger single-lesion match ever sits outside the NN slice.

**Outputs:** three tables, stored once and used by every later experiment.

| Table | One row per | Contents | Used by |
| --- | --- | --- | --- |
| Slice pairs | Candidate synthetic–training slice pair | Synthetic ID, training slice, patient, slice index, C, lesion counts in each slice, number of matched lesions, strongest single lesion score | Copy classes (E5), exposure and H1 (E6) |
| Lesion pairs | Hungarian-matched lesion pair in each candidate slice pair | Both lesion IDs; the shape, position and size factors; s; whether it passed τ\_L | Match reports and deletion test (E7) |
| Synthetic summary | Synthetic slice | NN slice and C, class, strongest single lesion match and its slice | Rates (E5), report selection (E7) |

Store every candidate pair, not only those above the thresholds. Thresholds are then applied at analysis time, so sensitivity analyses need no recomputation. It also means a lesion copied across several adjacent training slices credits all of them, which E6 needs for H1.

**Carried over from E0 (2026-11-03); E2 is planned in `docs/e1-e3-plan.md`**

- **Position frame: decided 2026-11-03 (decision F): pixel coordinates.** The reasoning, as first recorded: step 2 originally measured position relative to the brain box of the paired MRI. But stage 1 generates masks without seeing a brain, so a memorized mask reproduces its training mask at the same *pixel* positions. The synthetic brain box comes from stage 2, and synthetic brains are smaller (E0: median brain area 3,009 vs 7,765 px), so even a pixel-perfect copy can get shifted brain-relative coordinates. E0 also found synthetic lesions compact in pixels but with a real-like spread relative to the brain box, which suggests stage 2 fits the brain around the lesions. E4 cannot reveal the problem, because planted copies are real masks with their own real MRI. Chosen: pixel (model-space) coordinates for copy detection in E2/E4/E5 (σ = 0.05 × 128 ≈ 6.4 px; the prefilter's 0.2 becomes ≈ 26 px), consistent with E1's pixel-wise Dice; brain-relative coordinates for anatomical questions (density figure, E6 rarity). The E0 lesion table stores all three frames (pixels, image fraction, brain-relative), so nothing needs recomputing either way. Evidence: `docs/report-notes.md`, section 3 ("Synthetic lesion positions") and section 5.
- **Lesion crop.** E0 did not cache the crops of step 2 (16×16 pixels centred on the centroid, larger for lesions that don't fit); E2 builds them from the E0 label maps (`labels.npz`) where they are used.
- **Table formats.** CSV for tables people read (up to a few hundred thousand rows: lesion and slice tables, summaries, everything in `results/`). Parquet (pandas + `pyarrow`, to be added to `pyproject.toml` and the lock file) for the large machine tables, i.e. the slice-pair and lesion-pair tables with all candidate pairs (likely several million rows), stored in `processed_data/`, never in git. They carry both the readable `lesion_id` and the integer `lesion_idx` used in computation.

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

**Purpose:** Turn raw scores into a meaningful "copy" decision, and decide which lesions are large enough to count as evidence. The reference is real patients the model never saw (H-cal), which shows how close a mask gets to the training set when it was *not* copied. The optional null model was dropped (see step 6).

**Inputs:** H-cal, the lesion table from E0, and the detectors from E1 and E2.

**Procedure**

1. **Set the informative-lesion bound.**
   - For each H-cal lesion, find its best lesion score against T. Compute the share above a provisional τ\_L, binned by lesion area.
   - Choose the smallest area at which this chance-match rate drops below 5–10%. Expect roughly 10–20 pixels at 128×128.
   - Before fixing it, check how many S and T slices would have no lesion above the bound. The synthetic pool favours masks with a lesion under 40 pixels, so a high bound could remove much of S.
   - This bound is separate from, and higher than, the speck threshold used in quality control. Decide it during coding, together with the E0 quality-control re-run, by checking how each candidate bound affects the synthetic pool.
2. **Set the held-out thresholds.**
   - Run E1 and E2 with H-cal as the query set, using only informative lesions.
   - Set each threshold at the 95th percentile of H-cal: τ\_D for whole-mask Dice, τ\_C for the configuration score C(q, NN) (C as redefined in E2 step 3), and τ\_L for the best lesion-pair score.
   - Compute them separately for single-lesion and multi-lesion slices.
   - Steps 1 and 2 depend on each other through τ\_L: set a provisional τ\_L, then the bound, then the final τ\_L. One round is enough.
3. **Confirm the prefilter is exact.** Check that at the prefilter distance (0.2 of the field, 25.6 px), s stays below τ\_L for every lesion size.
4. **Check sensitivity with one alternative per setting:** σ of 0.08 instead of 0.05, the 99th percentile instead of the 95th, and a bound one step lower.
5. **Freeze.** Write all thresholds and settings to a config file under version control, before E4 and well before E5.
6. **Null model: dropped (decision 2026-11-01).** It was planned as an optional set of fake masks built from H-cal lesions at independent positions, to give stable chance rates for partial copies. Its main use was the recombination question, which is out of scope, and held-out real patients are the more credible baseline. The cost: chance rates for multi-lesion slices rest on H-cal alone, which has 303 multi-lesion slices from 17 patients (≥2 lesions of ≥4 px), or 192 from 12 patients (≥2 lesions of ≥15 px). To handle this, report a bootstrap-by-patient confidence interval for every threshold, and pool the single- and multi-lesion thresholds if the multi-lesion one is unstable.

**Outputs:**

- A frozen threshold file: τ\_D, τ\_C and τ\_L (stratified by lesion count, decision H), the informative-lesion bound, and the prefilter distance.
- Chance rates from H-cal, with bootstrap-by-patient confidence intervals for each threshold.
- How many lesions and slices the bound excludes.
- A short sensitivity table.

**Check before moving on:** The thresholds are frozen, and the share of S excluded by the bound is acceptable. If it is large, weigh it explicitly and record the reason with the decision (parameter choices are the author's; no supervisor sign-off needed, 2026-11-03).

*Status (2026-11-03): done.* Implemented as `scripts/e3_calibrate.py` (stages `bound-review`, `calibrate`, `freeze`, `check`; details and decisions in `docs/e1-e3-plan.md`). Frozen in `configs/frozen/e3_stanford_brainmets.yaml`: informative-lesion bound 12 px (decision G; S main keeps an informative lesion in 72% of slices), thresholds stratified by lesion count (decision H): τ\_D 0.905 / 0.729, τ\_L 0.739 / 0.816, τ\_C 0.719 / 0.406 (single / multi), with bootstrap-by-patient intervals. The E3 check passes (16/16). Deviation: C counts every paired lesion from the candidate cut, not only those passing τ\_L (E2 step 3). The sensitivity table (σ 0.08, 99th percentile, bound 10 and 25 px) is in `results/stanford_brainmets/e3/calibrate/`.

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
- False-positive rate on H-eval at most 7% (95% is the calibrated target, so about 5% is expected). *Note 2026-11-03:* state this per class. Each of τ\_D, τ\_L and τ\_C is calibrated at 5%, but the lesion-level classes combine two of them: on H-cal, full copies are 5.1% / 5.2% and full or partial together 7.4% / 8.7% (single / multi).

**If it fails:** Change detector settings (top-k, shift range, σ), re-freeze thresholds on H-cal (E3), and re-run E4. Never adjust anything by looking at S.

**Carried over from E1–E3 (2026-11-03), for planning E4**

- **Tools in place.** The frozen thresholds are in `configs/frozen/e3_stanford_brainmets.yaml`; read them with `memtk.calibration.read_frozen` and give each slice its threshold with `frozen_threshold` (stratum by lesions ≥ 4 px for E1, by informative lesions ≥ 12 px for E2). H-eval with decision E: `memtk.e3.common.load_set(cfg, dataset, "H-eval")`. Both detectors take label maps (`wholemask.match`, `lesionmatch.match`), so planted masks first go through `lesions.attach_or_drop` with decisions B, C (4 px, gap 2), exactly like the real masks.
- **What "full" means under the frozen class rule, and P6.** Full = C(q, NN) ≥ τ\_C, the configuration closer than chance; it does not require every lesion to be copied (that is the descriptive `all_copied`). Because τ\_C is low for multi-lesion slices (0.406), most P6 variants come out *full*: adding one H-eval lesion to a single-lesion base gives C = 1.0 / 2 = 0.5; removing one lesion from a three-lesion base gives C ≈ 2/3. Decide before the go/no-go criteria are fixed: either P6 counts as caught when flagged (full or partial) and the partial-copy check uses `all_copied` / `n_kept`, or the expected classes per variant are restated in terms of the frozen rule.
- **P3 sits at the edge by construction.** A single lesion shifted 5 px keeps shape and size but its position factor drops to exp(−25 / (2 · 6.4²)) ≈ 0.737, just under τ\_L (0.739) and above τ\_C (0.719); E1's shift search only reaches 3 px. P3 stays "reported only", as planned.
- **The false-positive criterion per class** (see the note under the criteria above); H-cal chance rates at the frozen thresholds are in the frozen file (`hcal_chance_rates`).

**Outputs:** The validation table, and the planted copies themselves, which E7 reuses to check whether explanations are correct. It also doubles as the first evidence for RQ2: the cases where whole-mask Dice misses P6 and the lesion-level detector catches them.

*Status (2026-11-04): planned in `docs/e4-plan.md`.* Decided there: decision I (planting design: 10 + 10 base patients so ~35–40 bases are multi-lesion; P4 about the lesion-pixel centroid; P5 as erode and dilate sub-variants with the 3×3 cross, gated only on masks that keep every base lesion at ≥ the bound; P6-remove and P6-add as sub-variants) and decision J (caught = flagged, i.e. E1 copy or E2 full or partial, *and* source patient = base patient; whether every lesion was copied shows in `all_copied`). This resolves the P6 question above. *Criteria decided 2026-11-04 (decision K, before any detector ran on the planted set):* P1 100% for both detectors; P2 and P4 ≥ 90% for at least one detector; P6 (remove and add) ≥ 80% for the lesion-level detector; H-eval false positives per stratum E1 copy ≤ 7%, E2 full ≤ 7%, E2 flagged ≤ 12%. **Deviation from the suggested criteria: P5 is reported only, like P3.** From the score definitions, a 1 px erosion or dilation to area ratio r gives a lesion score of about r² (median r 0.43 eroded, 1.7 dilated), below the single-lesion thresholds even for lesions ≥ 80 px, so no detector setting could make P5 pass; the detection limit is stated as a limitation. A planted mask counts in the gate only if every informative base lesion is still ≥ the bound (all variants but P6-remove; decision I amended). Details: `docs/e4-plan.md`, decision K.

*Status (2026-11-04): done; plan closed (`docs/e4-plan.md`), E4 check 16/16 (planted set and gate reproduced from the seed and the stored tables; S never read).* *Result (`e4_validate.py validate`): gate ACCEPTED with a recorded deviation.* Every recall criterion passes: P1 100%, P2 100%, P4 98.9%, P6 97.1% (caught as in decision J). On H-eval, E1 copy 7.2% / 7.4% and E2 full single 7.1% exceed the 7% limit by 0.1–0.4 points (H-cal 5.1–5.3%; every bootstrap-by-patient interval contains both; flags cluster in few patients), while E2 full multi (2.9%) and E2 flagged (10.1% / 8.5%) pass. Changing a setting and re-calibrating on H-cal cannot fix a difference between two samples of 25–26 patients, and fitting thresholds to H-eval would leak it, so the user accepted the three as a recorded deviation (`validation.accepted_deviation`; reasons in `docs/e4-plan.md`). Reported only: P3 caught 92% by the lesion-level detector, 13% by whole-mask Dice; P5 0% at 12–19 px, about 85% at ≥ 80 px (the detection limit). First RQ2 evidence: the lesion-level detector alone catches 79 of 100 P3, 19 of 37 P6-remove and 33 of 100 P6-add masks.

## E5 – Main audit (RQ1, RQ2)

**Purpose:** Answer whether the mask generator copies more than chance (RQ1), and whether lesion-level matching finds copies that whole-mask matching misses (RQ2). This is the first time S is scored against frozen thresholds.

**Inputs:** S, H-eval, the frozen thresholds, and both validated detectors.

**Procedure**

1. **Score the synthetic masks.** Run both detectors on S once, with the frozen config. Save the raw tables before doing any analysis.
2. **RQ1, population level.**
   - Compare the NN score distribution of S with that of H-eval, using a one-sided Mann–Whitney U test. Report the effect size as AUC: the probability that a random synthetic mask is closer to the training set than a random held-out mask.
   - Do this for both detectors, split into single-lesion and multi-lesion masks.
3. **RQ1, copy rate (first part of H1).**
   - Compute the share of S flagged as full copies, with a 95% bootstrap confidence interval. Synthetic samples are independent, so resample them directly.
   - Compare with the H-eval rate (bootstrap by patient, since its slices are clustered).
   - Memorization beyond chance means the S rate sits clearly above it. Report rates on the clean and minor masks (`mask_class` in `qc.csv` from E0b: a failed MRI does not exclude its mask, decided 2026-11-02). Then repeat with the noisy masks included, using the whole-mask detector only, to show that the exclusion doesn't hide copies. Also re-run the lesion-level detector with drop-only (no attaching), to show that the stray-pixel rule doesn't change the conclusions.
4. **Remove adjacent-slice duplicates.** Group flagged synthetic masks by source patient and slice window (±2 slices), and count each group once. Report both the raw count and the grouped count. The gap shows how much near-duplicate adjacent slices inflate the raw number.
5. **Count near-exact copies separately** (whole-mask Dice > 0.9). These are unambiguous and make strong figure examples.
6. **RQ2, missed by whole-mask.**
   - Cross-tabulate the whole-mask flag (yes or no) against the lesion-level class (full, partial, novel). Use only slices with at least one informative lesion, so both detectors are compared on the same subset.
   - The headline number: the share of lesion-level flags that whole-mask Dice does not flag. To test H2, compare the two detectors at the same false-positive rate on H-eval.
   - Compare the partial-copy rate in S with the H-eval rate. Partial copies only matter if S produces them more often than chance.
7. **Describe how much of each flagged slice is copied** (added 2026-11-03). C weights every lesion equally, so one copied large lesion next to unmatched small ones gives a partial copy even when it dominates the lesion area. For every flagged slice, report alongside the class: `n_kept` of the informative lesions (pairs ≥ τ\_L), the strongest pair, and the **copied-area share** (area of the paired lesions with s ≥ τ\_L divided by the total informative lesion area of the two slices). Optional sensitivity: an area-weighted C with its own τ\_C, calibrated on H-cal in the same way, to show whether any conclusion depends on the weighting. The main definition stays equal-weighted (see E2 step 3).

**Outputs:**

- A main results table: rates with confidence intervals for S and H-eval, for each detector and class.
- A score-distribution figure (S and H-eval overlaid).
- The cross-tabulation for RQ2.
- A list of flagged synthetic masks, which feeds E6 and E7.

**How to read the outcomes:**

| Outcome | Meaning |
| --- | --- |
| S rate clearly above H-eval | Memorization beyond chance; go on to E6 and E7 as planned |
| S rate near H-eval, but a few near-exact copies | Rare but real memorization; E6 and E7 focus on those cases |
| S rate at or below chance, no near-exact copies | A validated negative result (E4 shows the detector works); the master thesis centres on RQ-M1 and RQ-M2 |

**Carried over from E1–E3 (2026-11-03), for planning E5**

- **Run time on S main (35,920 masks), measured on T queries:** E1 about 4.4 ms per query (≈ 3 min), E2 about 10 ms per query slice (≈ 6–10 min; S lesions are more concentrated, so more candidate pairs). The E2 tables for S will hold several million lesion pairs: store them as Parquet in `processed_data/`, as for H-cal.
- **The drop-only re-run (step 3) needs its own label maps and thresholds.** The frozen thresholds were calibrated with attach-or-drop at gap 2. A drop-only re-run needs drop-only label maps (gap 0) for S, T and H-cal, and τ\_L and τ\_C recalibrated on H-cal with them, or it compares unlike scores. Decide when E5 is planned whether to recalibrate or to report the drop-only re-run as descriptive only.
- **The E3 sensitivity settings** (σ 0.08, 99th percentile, bound 10 and 25 px) have threshold values in `results/stanford_brainmets/e3/calibrate/sensitivity.csv`. Whether S is also scored under them (as sensitivity analyses of the main rates) is decided when E5 is planned.
- **Noisy masks (step 3) and strata:** the E1-only run on noisy masks uses the same stratified τ\_D, with the stratum counted on the cleaned mask, as for the main audit.

**Carried over from E4 (2026-11-04), for planning E5**

- **The gate:** E5 runs only if `memtk.e4.common.gate_status` returns PASS or ACCEPTED (now ACCEPTED, a recorded deviation on three false-positive criteria; see E4 result).
- **The H-eval baseline is already scored:** `validate` stored E1 and E2 tables for H-eval in `processed_data/model_space/<dataset>/e4/` and one row per slice with both detectors' classes in `heval_classes.csv`; E5 reuses them instead of re-running the detectors on H-eval.
- **Compare with the measured H-eval rates, not with 5%:** at the frozen thresholds H-eval gives E1 copy 7.2% / 7.4%, E2 full 7.1% / 2.9%, E2 flagged 10.1% / 8.5% (single / multi; `results/<dataset>/e4/validate/fp_heval.csv`), with flags clustered in few patients; the bootstrap by patient (`calibration.bootstrap_share`) is needed for every H-eval rate.
- **Detection limit:** copies whose small lesions carry about 1 px of boundary noise are not detected (E4 P5: 0% at 12–19 px); the S rates are lower bounds for such copies.

## E6 – Patient exposure, rarity and duplication (RQ1 H1, RQ3a H3)

**Purpose:** Move from "how many samples are copies" to "which patients are exposed, and why". Then test two hypotheses together. H3 says memorization concentrates on patients with unusual metastasis configurations, as Feldman and Zhang's long-tail argument predicts. H1 says copies concentrate on lesions that appear in many adjacent training slices.

**Inputs:**

- Flagged copies from E5, grouped by slice window.
- NN assignments for H-eval.
- The lesion table from E0.

**Procedure**

1. **Observed exposure.** For each training patient p, count the grouped copies whose source patient is p. Count full copies and partial copies separately. Use every stored slice pair above threshold, not only nearest neighbours, so a lesion copied across adjacent training slices credits all of them.
2. **Expected exposure.**
   - Some patients attract matches simply because they have many slices or common-looking lesions.
   - Estimate this from the H-eval masks: each gets an NN patient even when not flagged, so their shares give the expected share π\_p for each patient.
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

1. **Label lesions in 3D.** In each training volume, after the same 128×128 conversion, label lesions with 26-connectivity, so a lesion touching across slices counts as one. *Carried over from E0 (2026-11-02):* small real components (1–3 px) are mostly the tail or start of a metastasis seen in neighbouring slices (manual check of Mets_120 in ITK-SNAP), and real masks have no speck noise. So the 3D linking and slice span should probably use all components of the real mask as saved (`seg.nii.gz`), not the cleaned 2D labels; otherwise every lesion loses its thin end slices. The 2D lesion IDs stay unchanged; the 3D linking adds a `lesion3d_id` column. *Carried over from E3 (2026-11-03): neighbouring metastases merge in 3D.* 2D lesions are labelled per slice, and their L-numbers follow size within each slice, so L1 in one slice need not be L1 in the next; two metastases that touch in one slice are one 2D lesion there and two in the next (e.g. Mets_266, Mets_120). With 26-connectivity on the masks as saved, 38 of the 1,428 3D lesions of T (2.7%) hold two or more separate 2D lesions in some slice; they are large and long and contain 630 of the 6,101 2D lesions (10.3%), in 19 of 105 patients (most: Mets_120 11, Mets_024 4, Mets_266 3). Such a 3D lesion joins separate metastases: its slice span is too long, and copies of different metastases are credited to it, which biases H1 (copies against slice span). Decide when E6 is planned: (a) link 2D lesions slice to slice by overlap, one-to-one (Hungarian on the overlap between consecutive slices), so a merge or split keeps separate identities; or (b) keep 26-connectivity and run H1 again without the merged 3D lesions as a sensitivity; at least report the share affected. Not an issue for E1–E5, which compare single slices.
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
   - **Panel B:** lines connecting matched lesion pairs, and a small table with each lesion's shape, position and size factors and its contribution to C. Also the copied-area share (E5 step 7), so a report shows when one large copied lesion carries most of the lesion area although C counts it as one lesion among several.
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

**Carried over from E4 (2026-11-04), for planning E7**

- **Nearest neighbour and strongest pair can name different patients.** In E4, 4 P6 masks were flagged as full copies of another patient's slice (highest C, e.g. 0.91), while their strongest lesion pair (s = 1.00) pointed to the true base slice. A match report should show both, and the correctness check should say which one it scores. E4's `planted_outcomes.csv` holds both for every planted mask.
- **Planted copies for the correctness check:** `planted.csv` and `planted_lesions.csv` (the source lesion of every planted lesion) in `processed_data/model_space/<dataset>/e4/`.

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
| 31 Oct | E0, including mask quality control | Split confirmed; preprocessing and quality-control rules fixed; lesion table cached | Must (done 3 Nov) |
| 9 Nov | E1, E2, E3 | Both detectors run on H-cal; informative-lesion bound and thresholds frozen | Must (done 3 Nov) |
| — | E3 null model | Dropped (2026-11-01); see E3 step 6 | — |
| 13 Nov | E4 (gate) | Validation passed | Must (done 4 Nov: ACCEPTED with a recorded deviation) |
| 17 Nov | E5 | Main audit tables for RQ1 and RQ2 | Must |
| 24 Nov | E6; E7 match reports, deletion test and correctness on planted copies | Results for RQ3a, H1, H3 and RQ3b | Must |
| 24 Nov, if time | E8 | Optional RQ4 (H4); otherwise moved to the master thesis | Could |
| 30 Nov | Writing | Full draft to supervisor | Must |
| 1–8 Dec | Revision after feedback | Final report | Must |

**Write while you run.** Draft the data and methods sections from early November, while E1–E4 are running, so only results and discussion remain after 24 November.

**Plan documents for E5–E8 (suggested 2026-11-04, after E4; the user decides when each is planned).** As E1–E3 shared one plan, the rest group into three:

| Plan | Experiments | Why |
| --- | --- | --- |
| `docs/e5-plan.md` | E5 alone | The only step that scores S, once: every analysis choice is fixed in the plan before the run (sensitivity runs on S, the drop-only re-run, the adjacent-slice window, the tests, the noisy masks), as E4's criteria were fixed before `validate`. Its outputs are frozen evidence that E6–E8 only read. Last exit item: the number of flagged masks, which decides whether E8 runs (about 20 or more). |
| `docs/e6-e7-plan.md` | E6 and E7 together | Same frozen E5 pair tables, per-patient counting and deadline (24 Nov); shared decisions made once: 3D lesion identity (E6 H1 and E7's adjacent slices in Panel C) and nearest neighbour vs strongest pair (report-notes section 5); E7's Panel C shows E6's exposure ratio. Cuttable steps (taxonomy, lesion-level correctness) marked optional. Order inside: build and check the match-report generator on the E4 planted copies first (known source; needs no S, can start while E5 runs), then apply it to the S flags, so report rules are never tuned on S. |
| `docs/e8-plan.md` | E8 alone | Different data and methods (paired synthetic MRI, BRAVO, SSIM); optional, last, own go/no-go after E5; first to cut without touching anything else. |

Rough effort: E5 4–6 working days, E6 + E7 7–9, E8 3–4 if run.

**If you fall behind, cut in this order:**

1. E8 (optional RQ4). Move it to the master thesis.
2. The optional small-lesion agreement in E2 and E7.
3. The optional null model in E3. (Already dropped, 2026-11-01.)
4. The E7 taxonomy. Keep the deletion test and the correctness check on planted copies.
5. The lesion-level part of the E7 correctness check. Keep source-patient correctness.

Never cut E4: without validation, a negative result means nothing.

**Deliverables for the written report**

- **Tables:** data description (E0); validation (E4); main rates with confidence intervals (E5); RQ2 cross-tabulation (E5); most-exposed patients (E6).
- **Figures:** lesion distributions (E0); score distributions for S and H-eval (E5); exposure concentration and rarity scatter (E6); 3–4 anonymized match reports (E7).
- **Code:** a reusable mask-memorization toolkit (preprocessing, both detectors, calibration, report generator). The master thesis runs it again on retrained or original models.

**Decisions still open**

- [x] Go/no-go criteria for E4: decision K, 2026-11-04, fixed before `validate` ran (P5 reported only); three false-positive criteria failed and were accepted as a recorded deviation (see E4 result).
- [x] Position frame for copy detection in E2: pixel coordinates (decision F, 2026-11-03); see "Carried over from E0" in E2.
- [ ] The informative-lesion bound: decided during coding, by looking at how candidate bounds affect the synthetic pool, together with the quality-control re-run in E0.
