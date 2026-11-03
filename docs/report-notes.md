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
- **Per lesion, the enlargement depends strongly on size.** On 3,937 real slices with exactly
  one component in both spaces, model-space area divided by the native area scaled to 128×128
  (plain downscaling = 1), median by native lesion area: 1–5 px 2.64×, 6–10 px 2.15×, 11–20 px
  1.81×, 21–50 px 1.51×, 51–100 px 1.33×, 101–200 px 1.24×, 201–500 px 1.15×, > 500 px 1.10×.
  The 0.01 threshold adds about a one-pixel rim, a large share of a tiny lesion and a small
  share of a big one. Refines the per-patient median of 1.5× above. Real and synthetic masks
  share this effect (both were thresholded at 0.01), so it does not explain why synthetic
  lesions are smaller than real ones in model space. *Source:
  `results/stanford_brainmets/e0b/stats/native_vs_model_stanford_brainmets.csv` and `.png`,
  2026-11-03.*
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
- **Minimum lesion area 4 px** (`lesions.min_lesion_area`, decision B, set 2026-11-02). Smaller
  8-connected components are specks.
  - *Rule:* the largest m with at most 1% of T components smaller than m: 61 of 6,162 (0.99%)
    are under 4 px; a 5 px minimum would make 2.7% of real components specks.
  - *Clear size break in both T and S:* real components are rare below 4 px, while 62% of all
    synthetic components are 1 px (120,756 of 194,716), with a sharp drop at 3 px; from 4 px up
    the size curves are similar.
  - *Sensitivity (3 / 4 / 5 px):* T lesions 6,134 / 6,101 / 5,994; T slices without a lesion
    10 / 19 / 53; synthetic masks without a lesion 106 / 463 / 1,258 (1.2% at 4 px).

  *Source: `results/stanford_brainmets/e0b/lesions_review/review.txt`,
  `component_sizes_amibekk_v1.csv`, `attach_or_drop_amibekk_v1.csv`, 2026-11-02.*
- **Attach gap 2 px** (`lesions.attach_gap`, decision C, set 2026-11-02). A speck with at most 2
  background px between it and a lesion is attached as a stray; other specks are dropped.
  Nothing is moved or deleted (strays keep their pixels; dropped specks are left out of the
  label map), and two lesions are never merged.
  - *Evidence:* synthetic specks by gap to the nearest lesion: 11,162 at gap 1, 5,939 at gap 2,
    1,740 at gap 3, then a flat level of about 1,150 per gap value at gaps 4–10 (random noise
    near a lesion). Gaps 1–2 are a halo around lesions; gap 3 is close to the noise level.
  - *Cost:* about 20% of gap-2 attachments are probably chance (noise level / 5,939), adding
    1 px each, mostly in noisy masks that QC excludes anyway. Only 4 T specks are attached.

  The plan's default (2) was thus confirmed by the data. *Source: `lesions_review/review.txt`,
  `speck_gaps_amibekk_v1.csv`, `specks_by_gap_amibekk_v1.png`, 2026-11-02.*
- **Noisy masks: more than 9 isolated specks** (`qc.max_isolated_specks`, decision D, set
  2026-11-02). Masks with 1–9 isolated specks are *minor* (kept; specks are not lesions), more
  than 9 *noisy* (left out of the main audit; results also reported with them included).
  - *Evidence:* masks per exact speck count fall steadily from 1 to about 9 (2,614, 585, 294,
    215, 146, 94, 85, 74, 56), then turn into a flat tail of about 25–30 masks per count from
    10 to 20, plus 928 masks with more than 20: two populations, occasional specks and noise
    fields. The stratified sheet shows the same change: up to 9 specks the masks look like
    clean ones with a few dots, from 10 the dots spread over the image.
  - *Effect:* 1,242 noisy masks (3.3%); main audit on 36,382 clean and minor masks. The plan's
    suggested cut of 3 would have excluded 670 more masks (4–9 specks) that look and behave
    like clean ones: lesions mostly outside the brain stay at the clean-mask level (1–2%) up to
    19 specks, and rise to 7.8% only at 20 or more.
  - *Cost:* more stray growth in the 4–9 group (14–18% of lesions grown > 25% by strays, against
    11% at 3 specks); covered by E5's drop-only re-run and by reporting with and without
    noisy masks.
  - *Chosen from the data's own break* rather than a fixed number, so the same rule carries over
    to a new pool.

  *Source: `results/amibekk_v1/e0b/qc_review/review.txt`, `speck_counts_amibekk_v1.csv`,
  `speck_strata_amibekk_v1.csv` and `.png`, 2026-11-02.*
- **QC classes, with the mask kept apart from the MRI** (`qc` stage, 2026-11-02). Every synthetic
  sample gets a `mask_class` from its mask alone (`no_lesion` / `noisy` / `minor` / `clean`), a
  separate `mri_failed` flag, and a combined `qc_class` with the plan's precedence (failed MRI
  first) for reporting:

  | QC class | Masks | Share | Edge-flagged | No lesion < 40 px |
  | --- | --- | --- | --- | --- |
  | `mri_failed` | 152 | 0.4% | 82 | 70 |
  | `no_lesion` | 393 | 1.0% | 0 | 393 |
  | `noisy` | 1,241 | 3.3% | 153 | 191 |
  | `minor` | 3,701 | 9.8% | 89 | 488 |
  | `clean` | 32,137 | 85.4% | 829 | 4,360 |

  - *Why apart:* the mask comes from stage 1 whatever stage 2 does. 82 of the 152 failed MRIs
    carry a clean mask (70 have no lesion); they stay available to the mask audit and are left
    out only where the brain is used (edge check, brain-relative positions, E8).
  - *Main audit (E1–E5): `mask_class` clean or minor*, 35,920 masks. Clean and minor masks have
    trustworthy lesions (their few isolated specks are dropped; lesions mostly outside the brain
    stay at 1–2%). Noisy masks are where noise blobs become "lesions" (7.8% outside the brain at
    20 or more specks) and strays distort shapes; results are also reported with them, as
    planned. `no_lesion` masks have nothing to match.
  - *Edge flags only flag.* The rule fires on 2 of 5,909 real slices (T 1, H-eval 1, H-cal 0),
    so it needs no loosening. Flagged synthetic lesions in clean masks look like real lesions
    placed where stage 2 drew no brain, and a genuine copy of a lesion near a real cortex can
    land outside a smaller synthetic brain, so excluding flagged masks could remove real copies.
  - Matches the estimates from planning (about 86% clean, 1% no lesion); noisy is 3.3% with the
    cut at 9.

  *Source: `results/amibekk_v1/e0b/qc/summary.txt`, `qc_classes_amibekk_v1.csv`,
  `edge_check_real_stanford_brainmets.csv`, `qc_classes_amibekk_v1.png`, 2026-11-02.*
- **Two frames for lesion positions, and why they differ (explanation for the report).** Every
  lesion's centre is stored in three frames (`lesions.csv`); two matter for interpretation:
  - *Pixels:* the position on the shared 128×128 grid, the same for every image. Answers
    "where in the image is the lesion?"
  - *Relative to the brain box:* the position inside the bounding box of the slice's own brain,
    0 at its first row/column and 1 at its last; every brain box is in effect stretched to the
    same square. Answers "where inside its own brain is the lesion?"

  The same pixel can mean different brain-relative positions, because brains differ in size. A
  lesion at pixel row 85 lies at (85 − 15) / 98 ≈ 0.71 in a real brain spanning rows 15–113,
  but at (85 − 40) / 50 = 0.90, near the bottom edge, in a small synthetic brain spanning rows
  40–90. Analogy: a GPS position on one shared map against "how far across its own town" a
  point lies; a small and a big town can both have a point halfway across, at very different
  GPS positions.

  *Why it matters:* stage 1 makes the mask without seeing a brain, so a memorized copy of a
  training mask reproduces its pixel positions. A synthetic lesion's brain-relative position
  also depends on how large stage 2 drew the brain afterwards, which says nothing about copying.
  Hence the recommendation of pixel coordinates for copy detection (open question in section 5)
  and brain-relative coordinates for anatomical questions. The centroid-density figure shows
  both frames: compact synthetic positions in pixels, a real-like spread relative to the brain
  (section 3). *Source: the explanation and the figure
  `results/stanford_brainmets/e0b/stats/centroid_density_amibekk_v1.png`, 2026-11-03.*
- **Input for the informative-lesion bound (E3): what each candidate bound costs.** Share of
  slices left without an informative lesion, and H-cal patients that keep at least one
  informative slice:

  | Bound | T | H-cal | H-eval | S main | H-cal patients (of 25) |
  | --- | --- | --- | --- | --- | --- |
  | 10 px | 11% | 10% | 11% | 20% | 24 |
  | 15 px | 27% | 24% | 25% | 37% | 24 |
  | 20 px | 39% | 35% | 36% | 50% | 23 |
  | 30 px | 54% | 51% | 57% | 67% | 23 |
  | 40 px | 63% | 63% | 69% | 77% | 18 |

  Because synthetic lesions are smaller, every bound costs S more than T. At 20 px half the main
  audit leaves the lesion-level analysis (those masks are still scored by whole-mask Dice).
  Above 30 px H-cal starts losing patients, weakening calibration. Expected E3 range 10–20 px,
  unless the chance-match rates force it higher (see the shape-and-size finding in section 3:
  lesions under about 40 px are hard to tell apart by shape and size). Not decided here.
  *Source: `results/stanford_brainmets/e0b/stats/bound_impact_amibekk_v1.csv`, 2026-11-03.*
- **Sampling filters applied to the held-out real sets: only the pixel-sum filter**
  (`filters.real_sets: [max_mask_px]`, decision E, set 2026-11-03). The plan applies to the real
  comparison sets only the sampling filters the pool actually shows, so that the chance
  baseline (H-cal in E3, H-eval in E4/E5) describes the same kind of masks as S. T is never
  filtered: it is the reference copies are matched against.
  - *Pixel sum < 2,000:* holds for every synthetic mask (largest 731 px) and removes no real
    slice. Applied; in practice H-cal (992 slices, 25 patients) and H-eval (1,094 slices, 26
    patients) stay complete.
  - *A lesion < 40 px:* does not hold; 10.7% of the masks as saved (14.6% after cleaning) have
    none, although the repo labels lesions as we do. Applying it would remove about a quarter
    of H-cal and H-eval (252 and 263 slices) and make H less like S: H-cal would have only
    small-lesion masks, against 87% in the main audit. Not applied.
  - *Every lesion pixel inside the brain:* not shown with a meaningful brain definition (85% of
    the pool passes with our brain mask; the repo's check only required mask pixels to be
    brighter than 0.005 in the synthetic MRI, which painted lesions always are). Real slices
    pass at 98–99% anyway, and the `qc` edge check flags the cases that matter. Not applied.
  - *Remaining difference, for E3:* the main audit has a lesion under 40 px more often than
    H-cal (87% against 75%). If chance-match scores depend on it, E3 reports results stratified
    by "has a lesion under 40 px".

  *Source: `results/stanford_brainmets/e0b/filters_review/review.txt`, `filter_pass_amibekk_v1.csv`,
  `filter_cost_stanford_brainmets.csv`; repo `Generation/Sampling_mask_bravo.py`, 2026-11-03.*
- **The lesion definition and the informative-lesion bound are separate settings.**
  `min_lesion_area` defines what a lesion is (object vs noise) everywhere: QC speck counts,
  `no_lesion`, the lesion table and IDs, cleaned masks for E1, lesion counts and rarity. The
  informative-lesion bound (E3, from chance matches on H-cal) defines which lesions count as
  evidence in lesion-level matching. One setting cannot do both: a 15 px lesion definition would
  turn real small metastases into noise; a 4 px matching bound would flood E2 with chance
  matches between featureless blobs. 2026-11-02.

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
  2026-11-01.* `stats` compares them with real slices by height: see the next entry.
- **Synthetic brains are the size of real end-of-head slices, and far more often.** Over all
  20,829 real slices with brain, brain area rises from the ends of the head to a median of
  about 9,900 px in the middle; real lesion slices follow the same curve. The median synthetic
  brain (3,009 px; main audit, failed MRIs left out) matches real slices only in the outer
  10–15% at each end of the head (relative positions ≤ 0.15 and ≥ 0.90).
  - Brain area at most 3,009 px: 50.0% of synthetic samples against 11.7% of real lesion slices
    (T 12.4%, H-cal 8.9%, H-eval 12.0%).
  - The synthetic sizes are bimodal, with peaks around 1,300 and 3,800 px.

  So the small synthetic brains look like genuine end-of-head slices, but stage 2 draws them far
  more often than lesion slices occur there in real data. That fits stage 1's compact, central
  lesion placement (section 3, lesion positions) with stage 2 drawing a brain around it.
  *Source: `results/stanford_brainmets/e0b/stats/brain_by_height_amibekk_v1.png`,
  `brain_by_height_stanford_brainmets.csv`, `summary.txt`, 2026-11-03.*
- **Neighbouring lesion slices of a patient are near-duplicates.** Dice between consecutive
  lesion slices (z, z + 1) on the cleaned masks:

  | | Patients | Slice pairs | Median of per-patient medians | Pairs with Dice ≥ 0.5 |
  | --- | --- | --- | --- | --- |
  | T | 105 | 3,351 | 0.80 | 86% |
  | H-cal | 25 | 872 | 0.78 | 88% |
  | H-eval | 26 | 975 | 0.79 | 87% |

  About 4% of pairs have Dice 0 (a lesion appears or vanishes between slices). A copied lesion
  can therefore match several neighbouring training slices, which is the inflation risk the
  plan raises (Akbar et al.): E5 must group matches by source patient and a ±2-slice window,
  and count per patient and per lesion, not per slice. *Source:
  `results/stanford_brainmets/e0b/stats/adjacency_dice_stanford_brainmets.csv` and `.png`,
  `summary.txt`, 2026-11-03.*
- **Some synthetic MRIs contain no brain at all.** 152 of 37,624 samples have < 20 px of brain
  tissue; the 32 lowest have 0 px (a black image with only a lesion or a faint smudge). Between
  ~50 and ~400 px the images show small brain fragments, similar to the smallest real T lesion
  slices (lowest: 31 px, then 183 px and 236 px; 8 T slices have < 400 px). *Source:
  `results/stanford_brainmets/e0b/brain_review/near_blank_amibekk_v1.png`,
  `tissue_counts_amibekk_v1.csv`, 2026-11-01.*
- **Below about 40 px, real lesions are near-indistinguishable by shape and size alone.** For
  random pairs of T lesions from different patients within the same size range, the median of
  shape overlap (IoU, centroids aligned) × size ratio is:

  | Size | 4–5 px | 6–9 | 10–14 | 15–19 | 20–39 | ≥ 40 |
  | --- | --- | --- | --- | --- | --- | --- |
  | Median shape × size | 0.40 | 0.49 | 0.52 | 0.57 | 0.49 | 0.27 |
  | Pairs ≥ 0.5 | 28% | 49% | 57% | 72% | 47% | 21% |

  Similarity peaks at 15–19 px; 4–5 px blobs differ more because of pixel rounding. Only large
  lesions have distinctive shapes. Lesion-level matching must therefore rely on position and
  multi-lesion configurations. The informative-lesion bound (E3) protects against chance
  matches; it may land at the upper end of the expected 10–20 px, or above. *Source: read-only
  check, 6,101 T lesions (attach-or-drop 4 px, gap 2), 3,000 pairs per size range, seed 2026,
  2026-11-02. Alignment on the rounded centroid; E2's exact alignment may shift the numbers
  slightly.*
- **Small real components are mostly the tail or start of a metastasis.** Most 1–5 px
  components in T sit where a larger metastasis appears in neighbouring slices, so they are
  lesion edges seen in one slice rather than annotation noise. *Source: manual ITK-SNAP
  inspection of Mets_120 at 128×128 (a patient with many metastases), using
  `lesions_review/small_components_T.csv`, 2026-11-02; other patients not inspected.*
- **Synthetic masks: smaller lesions, more two-lesion masks, no large lesions.** Real lesion
  slices against the main audit (S main = `mask_class` clean or minor, 35,920 masks):

  | | T | H-cal | H-eval | S main |
  | --- | --- | --- | --- | --- |
  | Lesion area, median (IQR), px | 19 (11–45) | 23 (13–45) | 17 (11–28) | 13 (8–26) |
  | Single-lesion slices | 68% | 69% | 64% | 53% |
  | Two-lesion slices | 18% | 18% | 17% | 34% |
  | Most lesions in one slice | 13 | 7 | 13 | 7 |

  The synthetic area distribution lies below every real one over the whole range, and the
  largest synthetic lesions stop near 330 px (T: about 700). Lesions per slice are similar on
  average (1.63 against 1.60), but synthetic masks have two lesions about twice as often and
  four or more rarely. So S does not have fewer multi-lesion slices than T (a concern raised in
  the experiment plan); it has more two-lesion slices and a shorter tail. *Source:
  `results/stanford_brainmets/e0b/stats/description_amibekk_v1.csv`, `lesion_area_amibekk_v1.png`,
  `lesions_per_slice_amibekk_v1.png`, 2026-11-03.*
- **Synthetic lesion positions: compact in pixels, real-like relative to the brain box.**
  - *In pixels:* synthetic lesion centroids fill a compact central rectangle (roughly rows
    40–85, columns 55–105) with fairly sharp edges; real centroids spread over the whole round
    brain region. Centroid SD row/column 15.0/18.8 px in S main against 24.1/25.3 in T
    (H-cal 25.0/23.9, H-eval 24.9/26.4).
  - *Relative to the brain box:* the spread is the same as in real data (SD 0.24/0.27 against
    0.24/0.25 in T), smoother and more box-like; real lesions follow the round brain shape and
    sit slightly right of centre (column mean 0.57 against 0.50).
  - *Reading:* stage 1 places lesions in a narrow central pixel area, and stage 2 draws a brain
    fitted around them, so brain-relative positions of synthetic lesions mostly reflect stage
    2's brain drawing, not stage 1's placement. Pixel-exact copies of real lesions far from the
    centre are therefore unlikely to exist in S.

  Confirms and quantifies the earlier read-only sample (section 3, 2026-11-01) on the full
  pool. *Source: `results/stanford_brainmets/e0b/stats/centroid_density_amibekk_v1.png`,
  `description_amibekk_v1.csv`, 2026-11-03.*
- **Synthetic lesions are smaller than real ones.** Median lesion area 13 px in S against 19 px
  in T (H-cal 23, H-eval 17); 5th percentile 4 px against 6 px. 85% of synthetic masks have a
  lesion under 40 px (T 74%, H-cal 75%, H-eval 76%), so the sampling script's "at least one lesion
  under 40 px" filter shows in the pool, but 15% of the pool has no such lesion. Lesions per
  slice are similar (S 1.64, T 1.60). *Source: `results/stanford_brainmets/e0b/lesions/summary.txt`
  and `results/amibekk_v1/e0b/lesions/summary.txt` (min lesion area 4 px, attach gap 2), 2026-11-02.*
- **2.0% of synthetic lesions lie more than half outside the brain** (1,213 of 61,635), against
  1 of 6,101 in T (H-cal 0, H-eval 1). Part of this comes from the 152 failed MRIs without any
  brain, where every lesion counts as outside; the `qc` edge check separates the two. 82% of the
  isolated specks lie outside the brain (T: none). *Source: as above, 2026-11-02.* Broken down
  by QC group in the next entry.
- **The old QC list (2,243 masks) evidently cut at about 10 fragments.** It excluded every mask
  with 10 or more isolated specks, and fewer below that (79% at 6–9, 49% at 4–5, 29% at 3, 5% at
  1, 0.8% at 0). That explains its count, and it agrees with the break used for decision D.
  *Source: `results/amibekk_v1/e0b/qc_review/speck_strata_amibekk_v1.csv` (column
  `old_qc_excluded`), 2026-11-02.* The full cross-table follows in the next entry.
- **The old QC list is fully explained by two rules.**
  - *"Excessive noise fragments" (1,583 masks):* the smallest count in its reasons is "11
    distinct blobs", so the rule was more than 10 components, lesions and strays included. That
    catches all 1,241 of our noisy masks, plus 583 minor and 60 clean masks with many separate
    lesions and strays (and 1 no-lesion mask).
  - *"Pixels outside brain" (660 masks):* fires on a single pixel outside the brain (281 reasons
    read "Pixels outside brain: 1"). So 204 clean and 151 minor masks were dropped for a pixel;
    the plan replaced this rule with the edge check (> 50% of a real-sized lesion outside).

  Cross-table, old reason by QC class: noise fragments 939 noisy / 583 minor / 60 clean / 1
  no-lesion; pixels outside brain 302 noisy / 151 minor / 204 clean / 3 failed MRI. Verified:
  the earlier count of 2,243 was right, but its rules were too coarse to define S.
  *Source: `results/amibekk_v1/e0b/qc/old_qc_crosstab_amibekk_v1.csv`, `synthetic_index.csv`
  (`legacy_reason`), 2026-11-02.*
- **Most synthetic lesions outside the brain sit in clean masks, not noisy ones.** Lesions more
  than half outside the synthetic brain, by provisional QC group (cut at 3 isolated specks;
  decision D pending):

  | Group | Masks | Lesions | Mostly outside | Share | Masks affected |
  | --- | --- | --- | --- | --- | --- |
  | Clean (0 specks) | 32,137 | 51,816 | 838 | 1.6% | 829 |
  | Minor (1–3) | 3,423 | 5,410 | 71 | 1.3% | 70 |
  | Noisy (> 3) | 1,912 | 4,325 | 220 | 5.1% | 172 |
  | Failed MRI | 152 | 84 | 84 | 100% | 82 |

  They are small (median 5–7 px in every group). Likely cause: stage 1 places lesions without
  seeing a brain, and stage 2 draws smaller brains than real ones (median 3,009 vs 7,765 px), so
  a lesion where a real cortex would be can fall outside the synthetic brain.
  - *Recommendation for `qc`:* flag rather than exclude. A genuine copy of a training lesion near
    a real cortex lands at the same pixel position, which can lie outside a smaller synthetic
    brain; excluding edge-flagged masks could remove real copies.
  - Most are below the expected informative-lesion bound, so they barely affect lesion-level
    matching.

  *Source: `lesions.csv`, `slices.csv` and `brain.csv` of `amibekk_v1` (min lesion area 4 px,
  attach gap 2), read-only breakdown, 2026-11-02.*
- **In noisy synthetic masks, some "lesions" are noise blobs.** The synthetic masks with the most
  lesions are all noisy masks (15–444 isolated specks), and several of their lesions of 4 px or
  more lie in the background outside the brain (e.g. `file_2593`: L6, L7, L9 of 5–7 px;
  `file_2_5453`: L6, L7, L9 of 4–5 px). Lesion counts per mask are therefore only meaningful
  after QC, which excludes noisy masks from the main audit. *Source:
  `results/amibekk_v1/e0b/lesions/overview.png` (row 3), views of `file_2593`, read-only check,
  2026-11-02.*
- **Attached strays barely change lesions in clean masks; the distortion sits in noisy masks.**
  At 4 px / gap 2, by provisional QC group (cut at 3 isolated specks; decision D pending):

  | Group | Masks | Lesions with any stray | Median growth, if any | Lesions grown > 25% | Stray px / lesion px |
  | --- | --- | --- | --- | --- | --- |
  | Clean (0 isolated) | 32,219 | 5.9% | +7.7% | 0.4% | 0.4% |
  | Minor (1–3) | 3,038 | 36% | +9.1% | 4.8% | 3.2% |
  | Noisy (> 3) | 1,904 | 78% | +16% | 23% | 9.3% |

  In clean masks a stray at gap 1–2 is most likely the lesion's own faint fringe, cut off by
  the 0.01 threshold. Noisy masks, where the halo is partly noise, are excluded from the main
  audit by QC; E5's drop-only re-run covers the rest. Example: sample `401` has 93 components,
  2 lesions, 14 attached strays (lesions +29% and +73%) and 77 dropped specks. *Source:
  read-only check on all 37,624 synthetic masks, 2026-11-02.*
- **Synthetic masks are dominated by 1–2 px specks.** 62% of all synthetic mask components are
  1 px and 5% are 2 px (T: 0.08% and 0.4%). They gather in a halo 1–2 px around lesions (17,101
  specks at gap 1–2) and are spread over the background (105,468 specks at gap ≥ 11). Fits the
  suspected cause: per-mask min–max scaling and the 0.01 threshold in the sampling script.
  *Source: `lesions_review/component_sizes_amibekk_v1.csv`, `speck_gaps_amibekk_v1.csv`,
  2026-11-02.*

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
  - *The repo is not the code that ran* (previous author, 2026-11-03): the code was written and
    run on a university cloud system, often in notebooks, and the GitHub repo was assembled at
    the end to store it. Choices and code that ran may be missing from it. Where the data and
    the repo disagree, the data wins and we make reasoned assumptions, stated as such.
  - *What the repo version does* (`Generation/Sampling_mask_bravo.py`): mask min–max scaled and
    thresholded at 0.01; kept if the pixel sum < 2000; if at least one component (skimage
    `label`, default = 8-connected, like ours) is under 40 px; and if every mask pixel has
    intensity > 0.005 in the synthetic BRAVO (pixel-wise, no brain region). It saves as
    `file_2_<n>`, one of the six naming schemes.
  - *On the data* (`filters-review`): the pixel-sum filter holds for every mask; the small-lesion
    filter fails for 10.7% of the masks as saved, although the labelling matches; so it was not
    applied in that form to the whole pool. *Source: repo file above; `results/stanford_brainmets/
    e0b/filters_review/review.txt`, 2026-11-03.*
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

- **19 T lesion slices have no lesion of at least 4 px** (only 1–3 px components), from the
  minimum lesion area. Their cleaned masks are empty, so neither detector can match a synthetic
  mask to them. If these tiny components are real lesion edges (see section 3), copies of such
  slices would be missed; they are 0.5% of T slices. *Source: `lesions_review/review.txt`,
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
- [x] **Does a failed synthetic MRI exclude its mask from the mask audit?** *Resolved
      2026-11-02 at the `qc` stage: no.* `qc.csv` keeps `mask_class` and `mri_failed` as separate
      columns; the main audit uses `mask_class` clean or minor, and `mri_failed` only excludes a
      sample where the brain is used. See "QC classes" in section 2. Original question: decide
      at the E0b `qc` stage. The mask comes from stage 1 whatever stage 2 does, so a blank or failed MRI does not
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
      *Further evidence for pixel coordinates (2026-11-03):* synthetic lesions are compact in
      pixels but have a real-like spread relative to the brain box, which suggests stage 2 fits
      the brain around the lesions. Brain-relative positions of synthetic lesions would then
      measure stage 2's drawing rather than stage 1's placement, while a copy is something stage
      1 produces. See "Synthetic lesion positions" in section 3. *Source:
      `results/stanford_brainmets/e0b/stats/centroid_density_amibekk_v1.png`.*

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
