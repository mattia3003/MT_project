# Report notes: preliminary project

A running log of decisions, findings, limitations and open questions, gathered while the
experiments run. It is raw material for the written report, not report text. Sections follow
the report structure.

Entry format: **claim or number**, then why or how, then *source* and date. Sources are files
in git (`splits/`, `results/`, `docs/`) or pipeline stages that regenerate the data
(`processed_data/` is not in git). Mark an entry *superseded* rather than deleting it.

---

## 1. Data and preprocessing

**Real dataset**
- **156 patients: 105 train / 51 test**, the standard split, confirmed by the previous author.
  Raw volumes: BRAVO float64; segmentation float32 with values exactly {0, 1} in all patients;
  identity affine (RAS, 1 mm). *Source: `e0a check` (split checks); inspection, 2026-11-01.*
- **In-plane sizes vary** (most common 144×176 and 160×176; also 160×192, 144×192 and a few
  others). Depth is 144 or 160 slices. *Source: `patients.csv` (`real` stage), 2026-11-01.*
- **Lesion slices:** train 3,823 (per patient min 2 / median 27 / max 128); test 2,086
  (min 3 / median 35 / max 119). *Source: `splits/stanford_brainmets_seed2026.csv`, 2026-11-01.*
- **Very small lesion slices in model space:** T has 2 slices with 1 lesion pixel, 10 with
  ≤2 and 85 with ≤5; H-cal 15 and H-eval 20 with ≤5. *Source: `lesion_slices.csv`, 2026-11-01.*

**Conversion to model space (128×128)**
- **The original training transform is reproduced exactly** (LoadImage → EnsureChannelFirst →
  ToTensor → ScaleIntensity(0, 1) per volume → Resize((128, 128, −1)) with area interpolation;
  mask > 0.01). Verified in three ways: a fresh MONAI run is bit-identical; an independent
  nibabel + PyTorch re-implementation differs by exactly 0; and all 32 consistency checks pass.
  *Source: `results/stanford_brainmets/e0a/check_report_amibekk_v1.txt`, 2026-11-01.*
- **The conversion enlarges lesions.** Per patient, the lesion area after conversion is a median
  of 1.5× what plain downscaling would give (range 1.08–2.70), strongest for small lesions,
  because any pixel with >1% lesion coverage becomes lesion. Lesion pixel totals, native →
  model: T 364,829 → 280,550; H-cal 79,759 → 62,713; H-eval 90,417 → 69,883. E0b measures it
  per lesion. *Source: `patients.csv`, 2026-11-01.*
- **No lesion slice is lost or gained by the conversion** (5,909 lesion slices in total).
  *Source: check report, 2026-11-01.*
- **Real and synthetic images share the same array orientation** (checked visually: midline
  horizontal in both). *Source: `results/stanford_brainmets/e0a/check_grid_amibekk_v1.png`, 2026-11-01.*

**Synthetic pool**
- **37,624 files in six naming schemes**, most likely one per generation round: `<n>` 8,284;
  `file_<n>` 5,960; `file_2_<n>` 5,990; `file_3_<n>` 6,005; `file_4_<n>` 5,674;
  `file_5_<n>` 5,711. The sample ID is the full file stem (`1033` and `file_1033` are different
  samples). *Source: `synthetic_index.csv` (`synthetic` stage), 2026-11-01.*
- **Every sample is 128×128 with a binary, non-empty mask.** Mask size 2–731 px (median 27);
  8-connected components 1–529 (median 2). *Source: `synthetic_index.csv`, check report, 2026-11-01.*
- **The rounds look alike on every index statistic:** old-QC exclusion 5.6–6.3%, median mask
  27–28 px, masks with >10 components 4.4–4.7%. So far nothing suggests one round was produced
  differently. *Source: `synthetic_index.csv`, 2026-11-01.*
- **187 synthetic MRIs are nearly blank** (maximum intensity < 0.3; lowest 0.055). Synthetic
  intensities go slightly negative (minimum −0.047); real model-space data lies in [0, 1].
  *Source: `synthetic_index.csv`, 2026-11-01.* Maximum intensity later turned out not to
  identify failed images; see "failed synthetic MRI" in section 2.
- **Old quality-control list: 2,243 masks** (1,583 "excessive noise fragments", 660 "pixels
  outside brain", the latter triggered by a single pixel). All entries match a file.
  *Source: `docs/corrupted_synthetic_masks_10_0.txt`, check report, 2026-11-01.*
- *Preliminary, verify in E0b:* in a sample of 1,600 masks, about 11% have no component under
  40 px, so the sampling script's "<40 px" filter did not hold for the whole pool.
  *Source: inspection sample, 2026-11-01.*

## 2. Methods decisions

- **All comparisons happen in model space, converted from the 3D volumes.** Why: ScaleIntensity
  uses each volume's min and max, so the model space cannot be derived from single slices. The
  earlier per-slice native-resolution files were therefore not reusable. 2026-11-01.
- **Brain mask: fixed threshold 0.02, keep every region ≥ 30 px, fill holes** (`low_threshold_all`;
  decision A, set in the config 2026-11-01). Six candidates were compared on all 6,101 T lesions
  (≥ 4 px) and 2,000 random synthetic samples. Share of lesions more than half outside the
  brain / even partly outside:

  | Method | T >half | T partly | S >half |
  | --- | --- | --- | --- |
  | Otsu, largest region | 74 (1.21%) | 9.2% | 3.7% |
  | Otsu + 2 px dilation | 66 (1.08%) | 1.2% | 3.3% |
  | Otsu, all regions | 13 (0.21%) | 8.5% | 2.6% |
  | 0.02, largest region | 33 (0.54%) | 2.0% | 2.5% |
  | 0.02, all regions | 1 (0.02%) | 1.5% | 1.6% |

  Plain Otsu fails in two independent ways:
  - It keeps only the largest region. Low slices show the brain as separate pieces (lower lobes,
    cerebellum), and lesions in the discarded pieces look outside. Keeping all regions ≥ 30 px
    fixes this under either threshold (74 → 13; 33 → 1).
  - It trims the dim brain edge. The Otsu threshold (~0.10) lands at the intensity of dim
    tissue (~0.09), so lesions at the edge are cut. A fixed 0.02 works because the images are
    skull-stripped: background lies near 0 (99.9th percentile ~0.001). Dilation is a blunt fix:
    it pushes the edge out everywhere but does not recover lost pieces (74 → 66).

  Rejected: Otsu variants. Without dilation the edge stays trimmed (8.5% partly outside even with
  all regions); with dilation the separate pieces are still lost (66 more than half outside).
  A dilated Otsu with all regions was not tested, as the fixed threshold already solves both.
  *Source: `results/stanford_brainmets/e0b/brain_review/review.txt` and `methods_real.png`;
  intensity levels from a read-only sample of 600 T and 600 synthetic slices, 2026-11-01.*
- **The brain threshold (0.02) is a config value, checked per dataset, and holds here.**
  `brain.low_threshold` depends on the data (it assumes skull-stripped images with a near-zero
  background), so it is set in the config, and `brain-review` reports whether it sits in an
  empty gap between background and tissue. On this data, moving it to 0.01 / 0.04 changes the
  brain area by a median of +0.6% / −1.1% on T (95th percentile 1.7% / 3.1%) and +0.9% / −1.3% on
  S (2.0% / 3.2%). Background noise (99.9th percentile per slice, away from brain and mask) is
  0.000 / 0.001 for the median slice, and 0.006 (T) / 0.021 (S) for the near-worst 1%. The
  synthetic near-worst is at the threshold, but such bright background pixels are isolated and
  fall below the 30 px region minimum, which the small area changes confirm.
  *Source: `results/stanford_brainmets/e0b/brain_review/review.txt` and
  `threshold_sensitivity_amibekk_v1.csv` (all 3,823 T lesion slices, 2,000 random synthetic
  samples), 2026-11-02.*
- **A failed synthetic MRI is judged by its brain tissue area, not its maximum intensity**
  (`brain.min_tissue_px` = 20 px, decision A, set 2026-11-01: below 20 px every viewed sample has
  no brain at all, and the smallest real T lesion slice has 31 px). Tissue = pixels of the brain
  mask above. Why: a bright
  lesion on an otherwise blank image gives a high maximum, while some normal but dim brains have
  a low one. Among the synthetic samples, 82 with maximum ≥ 0.3 have < 100 px of tissue, and 38
  with maximum < 0.3 have ≥ 400 px. A small brain alone is never a failure (see section 3).
  *Source: read-only check on the full pool, 2026-11-01; tissue counts in
  `results/stanford_brainmets/e0b/brain_review/tissue_counts_amibekk_v1.csv`.*
- **A failed synthetic MRI means no brain region at all; a stricter rule was rejected.** Every
  kept tissue region has ≥ 30 px, so `min_tissue_px` = 20 fails exactly the images without any
  brain region (152 samples, 0.40%). The `brain` outline sheet showed small images that pass:
  a lesion with a thin rim of tissue, or a small faint patch elsewhere, on black.
  - *Tested:* failing images with < 20 px of tissue farther than 5 px from the lesions (mask
    components ≥ 4 px). This flagged 243 samples (+91) and no T slice.
  - *Rejected, for three reasons:*
    - It did not rank images by plausibility: it scored compact small slices with a central
      lesion (`file_5622`, `file_5_4695`: 142–158 px of tissue, 12 px away) below fragmented
      images (`12212`, `file_5412`: 112–120 px, 80–83 px away).
    - A genuine top-of-head slice with a metastasis also has most of its tissue near the
      lesion: T has Mets_314/z137 with 31 px of tissue.
    - It would add two pool-specific thresholds (margin, speck size), which a new pool would
      have to retune.
  - *Also found:* measured from the raw mask, the specks of noisy masks, grown by the margin,
    covered normal brains; ignoring components < 4 px fixed that.
  - The code was removed after the decision.

  *Source: manual inspection and a temporary `brain-review` extension (since removed), its counts
  and sheet `near_blank_away_amibekk_v1.png`; `results/amibekk_v1/e0b/brain/outlines.png`,
  2026-11-02.*
- **H-cal/H-eval split by matched pairs, seed 2026.** Sort the 51 test patients by number of
  lesion slices, pair neighbours, and assign one of each pair at random to each set; the odd
  patient out goes to H-eval. Why: lesion burden ranges from 3 to 119 slices, and a plain random
  half could put the heavy patients mostly in one set. Result: H-cal 25 patients / 992 slices
  (median 35), H-eval 26 / 1,094 (median 36). Rejected: simple random split.
  *Source: `splits/stanford_brainmets_seed2026.csv`, 2026-11-01.*
- **The mask before thresholding (`seg_soft`) is stored as well**, so the 0.01 threshold can be
  varied in sensitivity analyses without reconverting. 2026-11-01.
- **The old QC list does not define the audited set S.** Why: its outside-brain rule drops a mask
  for a single pixel, while the plan flags only lesions with >50% of their area outside the
  brain. It is kept only to compare with the fresh QC in E0b. 2026-11-01.
- **Null model dropped.** The optional null model (fake masks from H-cal lesions at independent
  positions) mainly served the recombination question, which is out of scope. Held-out real
  patients are the more credible chance baseline. Decided to manage workload. Chance rates come
  from H-cal only; see the limitation in section 4. 2026-11-01.
- **Reproducibility setup:** one config file (`configs/default.yaml`) and one global seed (2026).
  Every stage writes a run log (seed, full config, package versions). Stages refuse to overwrite
  existing outputs, and the split is fixed for the whole project. The earlier segmentation work
  is preserved under the git tag `segmentation-final`. 2026-11-01.

## 3. Findings

*(Experiment results go here from E0b onwards.)*

- *Observation, to quantify in E0b:* synthetic brains often look smaller or off-centre compared
  with real lesion slices. Positions are compared in brain-relative coordinates, so this matters
  mainly for the distribution comparison. *Source: check grid, 2026-11-01.*
- *Observation:* some synthetic masks are dominated by scattered specks across the background
  (e.g. sample `2750`). *Source: check grid, 2026-11-01.*
- *Preliminary, to quantify in E0b `stats`:* **synthetic lesions are spatially concentrated.**
  Lesion centroids (lesions ≥4 px) spread less than in T: row SD 14.9 vs 24.1 px, column SD 18.6
  vs 25.3 px, around the same centre. Synthetic brain boxes are smaller (median 63×66 vs
  103×95 px; Otsu brain area median ~3,000 vs ~7,200 px). Only 3.8% of synthetic lesions lie more
  than half outside the synthetic brain (T: 1.4%). So stage 2 draws smaller brains around
  concentrated lesions, rather than too-small brains around normally spread ones. One possible
  reading is that synthetic samples resemble slices from near the top of the head. The centroid
  spread uses the masks only, so it does not depend on the brain-mask method.
  *Source: read-only sample, all T lesion slices vs 3,000 synthetic samples (old-QC and
  near-blank excluded), 2026-11-01.* With the proposed brain mask (`low_threshold_all`), 1.6% of
  synthetic lesions lie more than half outside the brain (T: 0.02%). *Source: brain-review
  `review.txt`, 2000 random samples, 2026-11-01.*
- **Stage 2 paints a bright dot at mask specks, even far outside the brain.** Of 2,000 random
  synthetic samples, 120 have background pixels above 0.02 (99.9th percentile of the area
  outside the brain). 93% of those bright pixels (13,070 of 14,037) lie on or next to a mask
  pixel, and these samples have noisy masks (median 22.5 mask components vs 2 for the rest).
  So stage 2 follows the mask channel even where it draws no brain. Relevant to RQ-M1 (is the
  mask a key into stage 2?) and E8. *Source: read-only check, 2,000 random samples (seed 7),
  2026-11-02.*
- **The smaller synthetic brains look like genuine anatomy**, not a brain-mask artefact: compact
  slices resembling the top or bottom of the head, outlined well by every method. Median brain
  area 2,975 px (S) vs 7,765 px (T lesion slices). *Source:
  `results/stanford_brainmets/e0b/brain_review/methods_synthetic_amibekk_v1.png`, `review.txt`,
  2026-11-01.* `stats` compares them with real slices by height.
- **Some synthetic MRIs contain no brain at all.** 152 of 37,624 samples have < 20 px of brain
  tissue; the 32 lowest have 0 px (a black image with only a lesion or a faint smudge). Between
  ~50 and ~400 px the images show small brain fragments, similar to the smallest real T lesion
  slices (lowest: 31 px, then 183 px and 236 px; 8 T slices have < 400 px). *Source:
  `results/stanford_brainmets/e0b/brain_review/near_blank_amibekk_v1.png`,
  `tissue_counts_amibekk_v1.csv`, 2026-11-01.*

## 4. Limitations and assumptions

- **Stage-1 validation set unknown.** The checkpoint name (`correct_validation_data_16_...`)
  suggests a validation set, but which data it held out is unknown, and the original code used
  no seeds. Scenarios:
  - A. Some training patients held out → T contains patients stage 1 never saw.
  - B. Random slices held out → negligible; neighbouring slices are near-duplicates.
  - C. Test patients used only for monitoring or checkpoint choice → H-cal/H-eval stay valid,
    because they were not trained on.
  - D. No validation set → no effect.

  Only A matters. It biases results towards *no memorization*: copies can only come from
  trained patients, while unseen patients add chance similarity for S and H-eval alike. So any
  copy found is real, and rates may be slightly underestimated. E6 correlations (H1, H3) are
  diluted towards zero. Planned mitigation: in E6, re-run the H3 correlation with a random
  10–20% of training patients removed, many times. Do not infer the held-out patients from the
  results (circular). The master thesis (Branch B, retrained with known splits) removes this
  limitation. 2026-11-01.
- **Stage-1 slice selection is not fully known.** The repo's training loop computes an
  empty-slice filter (`np.sum(image) > 1.0`) but never applies it, and as written it could not
  train the mask-only model, since it passes a two-channel batch to a one-channel network. The
  mask is binarized at 0.01 in every case, matching our conversion. Bound: if slices with ≤1 or
  ≤2 lesion pixels were dropped, T contains 2 or 10 of its 3,823 slices that stage 1 never saw.
  *Source: amibekk/Thesis `Generation/diffusion_training.py`, `lesion_slices.csv`, 2026-11-01.*
- **The original models cannot be reproduced** (no seeds anywhere in the original code).
  From CLAUDE.md.
- **How the synthetic pool was produced is only partly known.** The repo's sampling script makes
  15,000 attempts with three filters; the pool shows six naming schemes and partly violates the
  <40 px filter (see section 1). E0b checks each filter on the data.
- **Chance rates for multi-lesion slices rest on few patients** (null model dropped). H-cal has
  303 multi-lesion slices from 17 patients (≥2 lesions ≥4 px), or 192 from 12 patients (≥2
  lesions ≥15 px). Slices of one patient are near-duplicates, so the patient count is the
  effective sample size. Mitigation in E3: bootstrap-by-patient CIs for each threshold, and pool
  single- and multi-lesion thresholds if the multi-lesion one is unstable.
  *Source: `lesion_slices.csv` + seg volumes, 2026-11-01.*
- **The orientation check is specific to this dataset.** All raw affines are the identity, so
  the independent check cannot detect re-orientation problems here. A new dataset with
  non-identity affines needs its own orientation check. 2026-11-01.
- **Brain-mask borders are not measured against a ground truth.** No reference brain masks
  exist. The method was chosen from three things:
  - lesion containment on T: annotated lesions must lie inside the brain, which detects masks
    that are too small, at lesion locations only;
  - the intensity gap of skull-stripped images (see the threshold entry in section 2);
  - visual inspection.

  Border accuracy away from lesions, and on synthetic images, is not measured. The brain mask is
  used only for coarse purposes (edge check > 50% outside, speck location, brain box), where
  1–2 px border errors have little effect. A direct check on real data is possible if needed:
  the native skull-stripped non-zero region, pushed through the model-space transform, is a
  reference mask. *Source: brain-review, 2026-11-01/02.*
- **Some synthetic MRIs show only a small piece of tissue around the lesion.** About 0.2–1% of
  the pool, depending on the measure, shows a lesion with a thin rim of tissue or a small faint
  patch on black. At 128×128 it cannot be decided visually whether they imitate top-of-head
  slices or are failed generations; even lesion and tissue are hard to tell apart beyond
  trusting the mask. They are kept, not flagged as failed. Analyses that use the brain (edge
  check, brain-relative positions, E8) should report their sensitivity to brain size, using
  `brain_px` / `tissue_px` in `brain.csv`. *Source: as the failed-MRI entry in section 2,
  2026-11-02.*

## 5. Open questions

**For the previous author** (status: open, 2026-11-01)
- [ ] Was there a validation set for stage 1 (and stage 2)? What was it used for (loss
      monitoring, checkpoint choice)?
- [ ] Was it drawn from the 105 training patients or the 51 test patients? By patient or by
      slice, and what fraction? Is there any patient list, folder listing, log or script?
- [ ] Which slice filter was actually used for stage-1 training, and were masks binarized for it?
- [ ] Were all six sampling rounds made with the same checkpoints and the same filters?
- [ ] Which MONAI version was used for training?

**Project decisions** (from the experiment plan)
- [ ] Informative-lesion bound: decided during E0b/E3 coding, by its effect on the synthetic pool.
- [ ] E4 go/no-go criteria: fixed before E4 runs.
- [ ] **Does a failed synthetic MRI exclude its mask from the mask audit?** Decide at the E0b `qc`
      stage. The mask comes from stage 1 whatever stage 2 does, so a blank or failed MRI does not
      make its mask invalid for E1–E5, which use masks only. A failed MRI matters where the
      brain is used: the edge check, brain-relative positions, and E8. Option: `mri_failed` is
      a flag that excludes a sample from those analyses only, rather than a QC class that
      removes the mask from S. The experiment plan does not say. The failed images come in at
      least two kinds: a lesion with a thin rim of tissue painted around it, and a small faint
      patch away from the lesion, each on an otherwise black image. *Source: `brain` stage
      outline sheet `results/amibekk_v1/e0b/brain/outlines.png` (row 3) and a read-only check of
      its 8 samples, 2026-11-02.*
- [ ] **Position frame for copy detection (E2), a deviation from the plan to decide when E2 is
      planned.** The plan measures lesion position relative to the brain box of the paired MRI.
      But stage 1 generates masks without seeing a brain, so a memorized mask reproduces its
      training mask at the same *pixel* positions. The synthetic brain box comes from stage 2,
      and synthetic brains are clearly smaller (section 3), so even a pixel-perfect copy can get
      shifted brain-relative coordinates. E4 cannot reveal this: planted copies are real masks
      with their own real MRI. Recommendation: absolute model-space coordinates for copy
      detection (consistent with E1's pixel-wise Dice); brain-relative coordinates for
      anatomical questions (density figure, E6 rarity). E0b stores all three frames, so nothing
      is lost either way. 2026-11-01.

## 6. Figure and table candidates

- **Data overview figure:** `results/stanford_brainmets/e0a/check_grid_amibekk_v1.png`. Random
  lesion slices of T, H-cal and H-eval and random synthetic samples, all in model space.
- **Data description table:** patients and lesion slices per set (section 1 and the split CSV).
- **Synthetic pool table:** samples per naming scheme, with old-QC rate and mask statistics
  (section 1).
- **Native vs model-space figure:** one slice before and after conversion, with the soft mask
  and the 0.01 threshold. It illustrates lesion enlargement. A draft exists outside the repo;
  produce it properly in E0b.
- **Brain-mask method figure:** `results/stanford_brainmets/e0b/brain_review/methods_real.png`
  (rows 1–4: where plain Otsu fails; six methods side by side). A reduced version (Otsu vs
  `low_threshold_all`, 3 rows) would fit a methods appendix.
