# Report notes: preliminary project

A running log of decisions, findings, limitations and open questions, gathered while the
experiments run. It is raw material for the written report, not report text. Sections follow
the report structure. Only what the report may use goes here: decisions with their evidence and
rejected alternatives, findings, limitations, open questions, figure candidates. How the work
was done (re-runs, verifications, refactors, bugs fixed, replaced numbers, preliminary
observations later measured in full) goes in `docs/process-log.md`.

Entry format: **claim or number**, then why or how, then *source* and date. Sources are files
in git (`splits/`, `results/`, `docs/`) or pipeline stages that regenerate the data
(`processed_data/` is not in git). When numbers are replaced, keep the current ones here and
move the old ones, with the reason, to the process log.

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

## 2. Methods decisions

- **All comparisons happen in model space, converted from the 3D volumes.** Why: ScaleIntensity
  uses each volume's min and max, so the model space cannot be derived from single slices.
  2026-11-01.
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
  S (2.0% / 3.2%). Background noise (99.9th percentile per slice, more than 3 px from the brain
  and more than 1 px from the mask, in Chebyshev distance) is 0.000 / 0.001 for the median
  slice, and 0.0004 (T) / 0.013 (S) for the near-worst 1%, well below the threshold; isolated
  bright background pixels would in any case fall below the 30 px region minimum, which the
  small area changes confirm. *Source: `results/stanford_brainmets/e0b/brain_review/review.txt`
  and `threshold_sensitivity_amibekk_v1.csv` (all 3,823 T lesion slices, 2,000 random synthetic
  samples), re-run 2026-11-03.* Earlier numbers, measured differently: process log, 2026-11-03.
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
  existing outputs, and the split is fixed for the whole project. 2026-11-01.
  **The pipeline is deterministic:** re-running E0a and E0b from scratch gives byte-identical
  tables, caches and model-space volumes. *Source: full re-run, process log 2026-11-03.*
- **Software:** Python 3.11.9, MONAI 1.6.0, PyTorch 2.6.0, NumPy 2.4.6, SciPy 1.17.1,
  scikit-image 0.26.0, nibabel 5.4.2, pandas 2.3.3, matplotlib 3.11.1. The environment is pinned
  in `requirements-lock.txt`, and a fresh install from it reproduces E0 byte for byte.
  *Source: `requirements-lock.txt`; fresh-venv run, process log 2026-11-03.*
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
- **Position frame for copy detection: pixel coordinates** (decision F, set 2026-11-03; a
  deviation from the experiment plan, which first measured position relative to the brain box of
  the paired MRI). Stage 1 generates masks without seeing a brain, so a copy keeps its training
  mask's pixel positions. Synthetic brains are smaller (median 3,009 vs 7,765 px), and synthetic
  lesions are compact in pixels but have a real-like spread relative to the brain box, so
  brain-relative positions would mostly measure how stage 2 drew the brain, not what stage 1
  copied. Pixel coordinates are also consistent with E1's pixel-wise Dice. Settings: σ = 0.05 ×
  128 = 6.4 px; prefilter 0.2 × 128 = 25.6 px. Brain-relative coordinates are kept for anatomical
  questions (density figures, E6 rarity). Not testable by E4, whose planted copies are real masks
  with their own real MRI. *Source: `results/stanford_brainmets/e0b/stats/centroid_density_amibekk_v1.png`,
  `description_amibekk_v1.csv`, `brain_by_height_amibekk_v1.png`; `docs/experiment-plan.md` E2;
  `docs/e1-e3-plan.md`, 2026-11-03.*
- **E2's speed-ups are exact.** The position prefilter (lesion pairs more than 25.6 px apart are
  skipped) and the candidate-slice cut (C computed only for training slices sharing a lesion
  pair with s ≥ 0.1) give identical results to scoring every lesion pair. Checked on random toy
  label maps and on 30 real T slices against all of T: 5,608 candidate slice pairs, 0
  differences in C. Exact for any τ_L ≥ 0.1, which the code enforces. *Source:
  `tests/test_lesionmatch.py` (brute-force test); sanity check on real T, 2026-11-03.*
- **Informative-lesion bound 12 px** (decision G, set 2026-11-03; provisional τ_L 0.77,
  chance-rate cut 10%). The provisional τ_L is the 95th percentile of the best E2 scores of the
  1,360 H-cal lesions ≥ 10 px. At that τ_L every area bin from 12 px up stays below 10% (12–14
  px: 8.4%). The 5% cut would give 25 px, but the 15–19 and 20–24 px bins sit at 5.1% and 5.8%
  (about ±1.7 points each, from 196 and 155 lesions), so where that cut lands is decided by noise.
  A 25 px bound would also leave 59% of S main without an informative lesion (E1 still scores
  them), against 28% at 12 px; H-cal keeps 24 of 25 patients at 12 px. The bound only decides
  which lesions may count as evidence: the false-positive rate is set afterwards by τ_D, τ_L and τ_C
  at the H-cal 95th percentile, per slice. 12 px lies in the experiment plan's expected 10–20 px.
  25 px is kept as a sensitivity in `calibrate`. *Source:
  `results/stanford_brainmets/e3/bound_review/review.txt`, `chance_rates.csv`;
  `results/stanford_brainmets/e0b/stats/bound_impact_amibekk_v1.csv`, 2026-11-03.*
- **The configuration score C was redefined, because the planned τ_C could not be calibrated on
  H-cal** (2026-11-03, deviation from the experiment plan; the class rule itself is the plan's).
  - *Planned:* C(q, t) kept only the Hungarian lesion pairs with s ≥ τ_L, summed their s and
    divided by max(lesions in q, lesions in t); τ_C was the 95th percentile of C(q, NN) on H-cal.
  - *Why that failed, from the data:* τ_L is itself the 95th percentile of each H-cal slice's
    strongest lesion pair, so by construction only ~5% of H-cal slices have any pair reaching
    τ_L. For the other ~95%, C was exactly 0. The 95th percentile of a score that is 0 for 95% of
    the slices sits at the edge of that zero mass: τ_C came out at 0.020 (bootstrap interval
    [0.000, 0.202]; 0.064 / 0.062 per stratum). A full copy (C ≥ 0.02) then meant the same as "any
    pair reaches τ_L"; τ_C and τ_L flagged exactly the same H-cal slices (5.0% pooled; 2.1% /
    12.6% per stratum). No usable τ_C could be found from H-cal with that C: the zero mass is a
    consequence of the definitions, not of sample size, so more H-cal data would not help.
  - *Redefined:* C counts every Hungarian pair with s ≥ 0.1 (the candidate cut, so the shortcuts
    stay exact), independent of τ_L. It then measures how alike the whole lesion configuration
    is: on H-cal its median is 0.54 (single-lesion slices) and 0.29 (multi), and its 95th
    percentile gives τ_C = 0.719 [0.709, 0.726] and 0.406 [0.391, 0.432].
  - *Classes (the plan's rule):* full copy = C(q, NN) ≥ τ_C (the configuration is closer than
    chance, even when no single lesion passes τ_L: of the 12 multi-lesion H-cal slices above τ_C,
    8 have no pair ≥ τ_L); partial = not full but the strongest lesion pair ≥ τ_L; novel
    otherwise. Stored as well: how many lesions pass τ_L, and whether all do. On H-cal (chance),
    stratified: full 5.1% / 5.2%, partial 2.3% / 3.5%, flagged at all 7.4% / 8.7% (single /
    multi); false-positive criteria must therefore be stated per class.
  - *Considered and set aside:* a structural rule (full copy = every lesion of both slices
    paired with s ≥ τ_L; chance 2.5% / 0%) was used for one run; it gives no calibrated
    configuration score and is kept only as the descriptive "all lesions copied". A nested rule
    (full also needs a pair ≥ τ_L) would keep the overall false-positive rate at 5% but never
    flag a configuration of moderately similar lesions.

  *Source: `results/stanford_brainmets/e3/calibrate/review.txt`, `thresholds.csv`,
  `classes_hcal.csv`, `score_histograms.png`; first run's numbers in the process log, 2026-11-03.*
- **C weights every lesion equally, not by size** (2026-11-03, after discussion). Considered:
  weighting each lesion pair by area (or by how rarely lesions of that size match by chance), so
  that a copied large lesion next to unmatched small ones would count as a full copy. Kept equal
  because:
  - such a slice is not missed: whole-mask Dice flags it (e.g. 0.93 for one 200 px lesion
    copied with two extra 15 px lesions), E2 calls it a partial copy with its copied lesion
    and score, and E5 counts near-exact whole-mask copies separately;
  - "partial" is the honest label: a memorized slice would be reproduced with all its lesions,
    and E4's planted variant P6 (one lesion added or removed) expects exactly that class, which
    an area-weighted C would miss for small added lesions (*corrected 2026-11-04:* under the
    frozen rule A most P6 variants come out *full*, since C stays above τ_C multi; E4 therefore
    counts P6 as caught when flagged and shows incompleteness with `all_copied`, decision J
    below; the point about area weighting stands, as an added small lesion would barely lower
    an area-weighted C);
  - area weighting turns the lesion-level score into a variant of whole-mask Dice and blurs the
    contrast RQ2 tests;
  - weak small-lesion evidence is already handled by the informative-lesion bound and the
    calibrated τ_L; chance-based weights would add a tuned component that is harder to explain
    in a match report.

  What is reported instead: the copied-area share per flagged slice (E5, E7); an area-weighted C
  with its own τ_C is an optional E5 sensitivity. *Source: `docs/experiment-plan.md` E2 step 3,
  E5 step 7; E3 chance rates by size (`bound_review/chance_rates.csv`), 2026-11-03.*
- **Thresholds stratified by lesion count, for both detectors** (decision H, 2026-11-03). Single-
  and multi-lesion slices reach very different chance scores, so one pooled threshold would flag
  the strata unequally:

  | Threshold | Single | Multi | Pooled | Pooled flags (single / multi) |
  | --- | --- | --- | --- | --- |
  | τ_D (E1) | 0.905 [0.896, 0.909] | 0.729 [0.696, 0.769] | 0.898 | 7.3% / 0% |
  | τ_L (E2) | 0.739 [0.730, 0.761] | 0.816 [0.781, 0.839] | 0.771 | 2.1% / 12.6% |
  | τ_C (E2) | 0.719 [0.709, 0.726] | 0.406 [0.391, 0.432] | 0.710 | 6.9% / 0% |

  95th percentiles of H-cal (989 slices for E1, 837 with an informative lesion for E2; 25 / 24
  patients), 2,000-draw bootstrap-by-patient intervals. The multi stratum rests on 17 patients
  with intervals tight enough to use. Strata: lesions ≥ 4 px for E1, informative lesions for E2.
  *Source: `results/stanford_brainmets/e3/calibrate/thresholds.csv`, 2026-11-03.*
- **Thresholds frozen before any detector touched S** (2026-11-03). Every E3 stage read only T
  and H-cal, and its run log records this. The frozen values (with the class rule, bound, detector
  settings and their chance rates on H-cal: E1 copies 5.1% / 5.3%, E2 full copies 5.1% / 5.2%,
  partial 2.3% / 3.5%, single / multi) were verified to be reproducible from the stored H-cal
  tables, and the E2 shortcuts were verified exact at the frozen thresholds, including against
  brute force on 20 random H-cal slices (2,428 candidate slice pairs). *Source:
  `configs/frozen/e3_stanford_brainmets.yaml`, `results/stanford_brainmets/e3/check_report.txt`.*
- **E4 planting design** (decision I, 2026-11-04, set by the user while planning E4).
  - *Base masks:* 10 T patients drawn from those with ≥ 3 multi-lesion slices (3 multi- and 2
    single-lesion slices each) and 10 from the other eligible patients (5 slices each); slices of
    one patient at least 2 apart in z. Eligible: ≥ 5 slices with an informative lesion (92 of
    the 103 T patients that have one; 45 of the 92 have ≥ 3 multi-lesion slices). A plain random
    draw would give about 25 multi-lesion bases (T: 805 of 3,172 informative slices are
    multi-lesion, 25%), too few for P6-remove, which needs two informative lesions. Drawing half
    multi-lesion overall was set aside as over-representing them against T.
  - *P4* rotates by ±5° about the centroid of all lesion pixels. Rotating about the image centre
    also moves peripheral lesions (about 4.4 px at 50 px from the centre, beyond E1's 3 px shift
    search) and would mix rotation with the shift that P2 and P3 already test.
  - *P5* is split into erode and dilate (1 px, 3×3 cross). Half of the informative T lesions are
    ≤ 28 px (quartiles 17 / 64 px), so erosion can destroy a lesion and dilation roughly doubles
    it. The gate counts only planted masks that keep every base lesion at ≥ the bound; destroyed
    ones are reported separately and recall is shown by lesion size.
  - *P6* is split into remove (one random informative lesion, bases with ≥ 2) and add (one
    informative H-eval lesion at its own pixel position, inside the base's brain mask, ≥ 3
    background px from the base lesions).

  *Source: read-only query of `processed_data/model_space/stanford_brainmets/lesions.csv` and
  `slices.csv`, 2026-11-04; `docs/e4-plan.md`.*
- **What counts as caught in E4** (decision J, 2026-11-04, set by the user). A planted copy is
  caught when it is flagged (E1: copy; E2: full or partial) *and* the source patient is the base
  patient (E2: the nearest neighbour's patient for full copies, the strongest pair's patient for
  partial ones). Under the frozen rule A, "full" means the configuration is closer than chance,
  not that every lesion is copied, so most P6 variants come out full (one lesion added to a
  one-lesion base: C = 1.0 / 2 = 0.5 ≥ τ_C multi 0.406). That a copy is incomplete is shown by
  `all_copied` = False, not by the class. Considered and set aside: "flagged, any patient" (a
  flag pointing at the wrong patient is not a detection) and an expected class per variant
  (P6's class depends on C, so the expectation would be arbitrary). The go/no-go criteria
  themselves (decision K) are set after the planted set is built and before any detector scores
  it. *Source: `docs/e4-plan.md`; frozen thresholds in `configs/frozen/e3_stanford_brainmets.yaml`.*
- **Which planted copies count in the gate** (decision I amended, 2026-11-04, with decision K).
  A planted mask counts only if every informative base lesion is still a lesion ≥ the 12 px
  bound, for every variant except P6-remove (which removes a lesion on purpose). First planned
  for P5 only; extended because 6 of 100 rotated masks (P4) had a 12–19 px lesion fall just below
  12 px (area ratio down to 0.85), so they test the bound, not robustness to rotation. Counted:
  P1–P3 and P6-add 100 of 100, P4 94, P5-erode 50, P5-dilate 98; all masks are still reported.
  *Source: `results/stanford_brainmets/e4/plant/variant_summary.csv`, `area_change.csv`.*
- **E4 go/no-go criteria** (decision K, 2026-11-04, set before any detector ran on the planted
  set; caught as in decision J):

  | Criterion | Value |
  | --- | --- |
  | P1 exact | 100% caught by both detectors (a miss is a bug, not a setting to tune) |
  | P2 shift 2 px, P4 rotate 5° | ≥ 90% caught by at least one detector |
  | P6 one lesion removed or added (pooled) | ≥ 80% caught by the lesion-level detector |
  | H-eval false positives, per stratum | E1 copy ≤ 7%, E2 full ≤ 7%, E2 flagged (full or partial) ≤ 12% |
  | P3 shift 5 px, P5 erode / dilate 1 px | reported only |

  The false-positive limits sit 2–3 points above the H-cal chance rates (5.1 / 5.3%, 5.1 / 5.2%,
  7.4 / 8.7%, single / multi); a single 7% limit on "flagged" would fail at chance.
  *Deviation from the experiment plan's suggestion (P5 ≥ 90%): P5 is reported only.* From the
  score definitions, an eroded lesion with area ratio r has shape factor ≈ r and size factor r,
  so s ≈ r², and Dice 2r / (1 + r); dilation the same with 1 / r. The planted set gives r
  0.29–0.86 after erosion (median 0.43) and 1.15–3.1 after dilation (median 1.7); even for
  lesions ≥ 80 px, s ≈ 0.52 / 0.59 and Dice ≈ 0.84 / 0.87 stay below the single-lesion
  thresholds (τ_C 0.719, τ_D 0.905). The thresholds are set by chance, so no detector setting
  could make P5 pass: a P5 criterion would fail with no allowed remedy. Set aside: P5 gated at
  ≥ 90%, and a P5 criterion per size bin (single-lesion masks fail even at ≥ 80 px). This was
  decided from the definitions and the planted geometry, before any score existed.
  *Source: `configs/default.yaml` (`validation.criteria`), `docs/e4-plan.md` (decision K),
  `results/stanford_brainmets/e4/plant/area_change.csv`.*
- **E4 gate accepted with a recorded deviation** (2026-11-04, decided by the user *after* the
  first `validate` run; stated as such in the report). Every recall criterion passed, but three
  of six false-positive criteria failed on H-eval by 0.1–0.4 points: E1 copy 7.2% [4.5, 9.7]
  (single) and 7.4% [4.9, 11.3] (multi), E2 full single 7.1% [4.8, 9.6], against a limit of 7%
  (H-cal at the same thresholds 5.1–5.3%). Reasons for accepting: every bootstrap-by-patient
  interval contains both the limit and the H-cal rate; the flags cluster in few patients (the 3
  patients with most flags give 33–62% of a cell's flags), so with 25–26 patients per held-out
  half the gap is patient sampling; it goes both ways (E2 full multi 2.9% on H-eval vs 5.2% on
  H-cal), as chance does. The planned remedy (change a detector setting, re-calibrate on H-cal)
  cannot help: re-calibration gives the same thresholds, and choosing thresholds so that H-eval
  passes would leak it into calibration. Rejected: stopping at FAIL (blocks E5 with no allowed
  remedy) and re-calibrating on H-cal and H-eval pooled (no independent check and no clean E5
  baseline). Consequence: E5 compares S with the measured H-eval rates, not with 5%. The
  possibility was flagged before the run (the multi stratum rests on few patients), but the
  criterion was written as a point estimate. *Source: `configs/default.yaml`
  (`validation.accepted_deviation`), `results/stanford_brainmets/e4/validate/fp_heval.csv`,
  `gate.csv`; patient clustering from a read-only check of `heval_classes.csv` (process log).*

## 3. Findings

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
- **Some T patients have identical cleaned masks on neighbouring slices** (e.g. Mets_323 z73/z74
  and z104/z105, Mets_314 z45/z48). Found by E1's self-match check: 4 of 300 random T slices
  had an exact duplicate in the same patient. They add to the near-duplicate neighbouring slices
  above and support counting copies per patient and slice window, not per slice. *Source: E1
  self-match sanity check (`memtk.wholemask.match`, 300 random T slices against T, seed 2026),
  2026-11-03.*
- **Single small lesions match by chance very often.** For each of the 1,548 H-cal lesions
  (never trained on), its best E2 match among T's 6,101 lesions reaches s ≥ 0.5 for 73–93% of
  the lesions in every area bin from 5 to 39 px, and for 52% at ≥ 40 px; the 95th percentile of
  these chance scores (lesions ≥ 10 px) is 0.77. T's lesions cover the brain so densely that
  nearly every held-out lesion has a similar blob within a few pixels (position factor often
  ≥ 0.9). A single small lesion is therefore weak evidence of copying; lesion-level evidence has
  to come from larger lesions and multi-lesion configurations, consistent with the E0 finding
  that lesions under about 40 px are hard to tell apart by shape and size. *Source:
  `results/stanford_brainmets/e3/bound_review/chance_rates.csv`, `examples.png`, 2026-11-03.*
- **Chance whole-mask matches are close for single-lesion slices.** 5% of held-out
  single-lesion slices have a training mask with Dice ≥ 0.905 after a shift of up to ±3 px; a
  single small lesion easily finds a near-identical training mask. Multi-lesion slices reach only
  0.729, since a whole layout of several lesions rarely overlaps by chance. Whole-mask Dice
  therefore needs a very high bar for single-lesion slices. *Source:
  `results/stanford_brainmets/e3/calibrate/thresholds.csv`, `score_histograms.png`, 2026-11-03.*
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
- **Neighbouring metastases merge when lesions are linked in 3D.** 2D lesions are labelled per
  slice and numbered by size within each slice, so lesion identities are not stable across
  slices: two metastases that touch in one slice are one lesion there and two in the next (seen
  in the overview sheet for Mets_266). Linking the masks as saved in 3D with 26-connectivity,
  38 of T's 1,428 3D lesions (2.7%) hold two or more separate 2D lesions in some slice; being
  large and long, they contain 630 of the 6,101 2D lesions (10.3%), in 19 of 105 patients
  (Mets_120: 11; Mets_024: 4; Mets_266: 3). Irrelevant for the slice-wise detectors (E1–E5); it
  affects the slice span of H1 and any 3D-level explanation (E6, E7). *Source: read-only check,
  T label maps and `volumes/<pid>/seg.nii.gz`, scipy 26-connected labelling, 2026-11-03;
  `results/stanford_brainmets/e0b/lesions/overview.png`.*
- **Synthetic masks: smaller lesions, more two-lesion masks, no large lesions.** Real lesion
  slices against the main audit (S main = `mask_class` clean or minor, 35,920 masks):

  | | T | H-cal | H-eval | S main |
  | --- | --- | --- | --- | --- |
  | Lesion area, median (IQR), px | 19 (11–45) | 23 (13–45) | 17 (11–28) | 13 (8–26) |
  | Slices with a lesion under 40 px | 74% | 75% | 76% | 87% |
  | Single-lesion slices | 68% | 69% | 64% | 53% |
  | Two-lesion slices | 18% | 18% | 17% | 34% |
  | Most lesions in one slice | 13 | 7 | 13 | 7 |

  The synthetic area distribution lies below every real one over the whole range (5th
  percentile 4 px in all of S against 6 px in T), and the largest synthetic lesions stop near
  330 px (T: about 700). The sampling script's "at least one lesion under 40 px" filter shows in
  the pool, but 13.5% of S main (14.6% of all of S) has no such lesion; see decision E in
  section 2. Lesions per slice are similar on average (1.63 against 1.60), but synthetic masks
  have two lesions about twice as often and four or more rarely. So S does not have fewer
  multi-lesion slices than T (a concern raised in the experiment plan); it has more two-lesion
  slices and a shorter tail. *Source:
  `results/stanford_brainmets/e0b/stats/description_amibekk_v1.csv`, `lesion_area_amibekk_v1.png`,
  `lesions_per_slice_amibekk_v1.png`, 2026-11-03; 5th percentiles from
  `results/<dataset|pool>/e0b/lesions/summary.txt` (`area_p5`), 2026-11-02.*
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

  Confirms on the full pool an earlier read-only sample of 3,000 masks (process log,
  2026-11-01). *Source: `results/stanford_brainmets/e0b/stats/centroid_density_amibekk_v1.png`,
  `description_amibekk_v1.csv`, 2026-11-03.*
- **2.0% of synthetic lesions lie more than half outside the brain** (1,213 of 61,635), against
  1 of 6,101 in T (H-cal 0, H-eval 1). Part of this comes from the 152 failed MRIs without any
  brain, where every lesion counts as outside; the `qc` edge check separates the two. 82% of the
  isolated specks lie outside the brain (T: none). *Source:
  `results/amibekk_v1/e0b/lesions/summary.txt` and `results/stanford_brainmets/e0b/lesions/summary.txt`
  (min lesion area 4 px, attach gap 2), 2026-11-02.* Broken down by QC class in "Most synthetic
  lesions outside the brain sit in clean masks" below.
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
  than half outside the synthetic brain, by QC class (`qc_class`: failed MRI first, cut at 9
  isolated specks):

  | QC class | Masks | Lesions | Mostly outside | Share | Masks affected |
  | --- | --- | --- | --- | --- | --- |
  | Clean (0 specks) | 32,137 | 51,816 | 838 | 1.6% | 829 |
  | Minor (1–9) | 3,701 | 6,757 | 93 | 1.4% | 89 |
  | Noisy (> 9) | 1,241 | 2,978 | 198 | 6.6% | 153 |
  | No lesion | 393 | 0 | – | – | – |
  | Failed MRI | 152 | 84 | 84 | 100% | 82 |

  They are small (median 5–7 px in every class). Likely cause: stage 1 places lesions without
  seeing a brain, and stage 2 draws smaller brains than real ones (median 3,009 vs 7,765 px), so
  a lesion where a real cortex would be can fall outside the synthetic brain.
  - *Hence edge flags only flag* (decided at the `qc` stage, see "QC classes" in section 2): a
    genuine copy of a training lesion near a real cortex lands at the same pixel position, which
    can lie outside a smaller synthetic brain; excluding edge-flagged masks could remove real
    copies.
  - Most are below the expected informative-lesion bound, so they barely affect lesion-level
    matching.

  *Source: read-only breakdown of `lesions.csv` (`frac_outside_brain` > 0.5) and `qc.csv` of
  `amibekk_v1`, 2026-11-03; "masks affected" equals the edge-flagged column of
  `results/amibekk_v1/e0b/qc/qc_classes_amibekk_v1.csv`. First computed with a provisional cut
  at 3 specks (process log, 2026-11-03).*
- **In noisy synthetic masks, some "lesions" are noise blobs.** The synthetic masks with the most
  lesions are all noisy masks (15–444 isolated specks), and several of their lesions of 4 px or
  more lie in the background outside the brain (e.g. `file_2593`: L6, L7, L9 of 5–7 px;
  `file_2_5453`: L6, L7, L9 of 4–5 px). Lesion counts per mask are therefore only meaningful
  after QC, which excludes noisy masks from the main audit. *Source:
  `results/amibekk_v1/e0b/lesions/overview.png` (row 3), views of `file_2593`, read-only check,
  2026-11-02.*
- **Attached strays barely change lesions in clean masks; the distortion sits in noisy masks.**
  At 4 px / gap 2, by mask class (`mask_class`, cut at 9 isolated specks; failed MRIs keep their
  mask class, masks without a lesion have no strays):

  | Mask class | Masks | Lesions | Lesions with any stray | Median growth, if any | Lesions grown > 25% | Stray px / lesion px |
  | --- | --- | --- | --- | --- | --- | --- |
  | Clean (0 isolated) | 32,219 | 51,900 | 5.9% | +7.7% | 0.4% | 0.4% |
  | Minor (1–9) | 3,701 | 6,757 | 43% | +10% | 6.9% | 4.1% |
  | Noisy (> 9) | 1,241 | 2,978 | 80% | +18% | 27% | 10.5% |

  Growth = stray px / lesion px without strays; the last column is stray px over lesion px
  including strays. In clean masks a stray at gap 1–2 is most likely the lesion's own faint
  fringe, cut off by the 0.01 threshold. Noisy masks, where the halo is partly noise, are
  excluded from the main audit by QC; minor masks, kept in it, carry a moderate share (7% of
  their lesions grown > 25%), which E5's drop-only re-run covers. Example: sample `401` has 93
  components, 2 lesions, 14 attached strays (lesions +29% and +73%) and 77 dropped specks.
  *Source: read-only breakdown of `lesions.csv` (`area`, `stray_px`) and `qc.csv` of
  `amibekk_v1`, all 37,624 masks, 2026-11-03. First computed with a provisional cut at 3 specks
  (process log, 2026-11-03).*
- **Synthetic masks are dominated by 1–2 px specks.** 62% of all synthetic mask components are
  1 px and 5% are 2 px (T: 0.08% and 0.4%). They gather in a halo 1–2 px around lesions (17,101
  specks at gap 1–2) and are spread over the background (105,468 specks at gap ≥ 11). Fits the
  suspected cause: per-mask min–max scaling and the 0.01 threshold in the sampling script.
  *Source: `lesions_review/component_sizes_amibekk_v1.csv`, `speck_gaps_amibekk_v1.csv`,
  2026-11-02.*
- **What 1 px of boundary change does to small lesions** (E4 `plant`, 2026-11-04). Of the 147
  informative lesions of the 100 planted-copy bases, a 1 px erosion (3×3 cross) leaves none of
  the 12–19 px lesions at ≥ 12 px and half of the 20–39 px ones (median area ratio 0.43); 12 of
  the 100 eroded masks have no lesion ≥ 4 px left. A 1 px dilation roughly doubles 12–19 px
  lesions (median ratio 2.07) and merges 2 lesions into a neighbour. A ±5° rotation about the
  lesion centroid pushes 6 lesions of 12–19 px just below 12 px (smallest ratio 0.85). At
  128×128 the small lesions that dominate the data are fragile to any boundary noise, so P5
  tests robustness only for lesions of about 20 px and more (50 of 100 eroded masks keep every
  lesion and enter the gate). *Source: `results/stanford_brainmets/e4/plant/area_change.csv`,
  `variant_summary.csv`, `area_change.png`.*
- **E4: both detectors find planted copies** (2026-11-04). Share caught (flagged with the base
  patient as source, decision J) of the gated planted masks, with bootstrap-by-patient intervals:

  | Variant | Whole-mask (E1) | Lesion-level (E2) | Either |
  | --- | --- | --- | --- |
  | P1 exact (100) | 100% | 100% | 100% |
  | P2 shift 2 px (100) | 100% | 100% | 100% |
  | P3 shift 5 px (100, reported only) | 13% [5, 23] | 92% [87, 97] | 92% |
  | P4 rotate 5° (94) | 78% [64, 89] | 98% [95, 100] | 99% |
  | P6-remove (37) | 41% [23, 59] | 92% [83, 100] | 92% |
  | P6-add (100) | 67% [59, 75] | 99% [97, 100] | 100% |

  P6 at 97% (133 of 137) was expected from the definitions: the remaining lesions are pixel-exact,
  so P6 tests whether an incomplete copy is seen, not robustness to noise. Among masks E2 caught,
  the nearest neighbour is the base slice itself for 99–100% of P1–P4 and P6-add, but for 41% of
  P6-remove and none of P5-erode (11% of P5-dilate): there it is another slice of the base
  patient. An eroded lesion resembles the neighbouring slice, where the same lesion is smaller,
  more than its own slice; adjacent slices are near-duplicates, which is why sources are counted
  per patient and E6 credits every pair above the threshold. *Source:
  `results/stanford_brainmets/e4/validate/validation.csv` (`nn_is_base`), `review.txt`.*
- **First RQ2 evidence: lesion-level matching catches planted copies that whole-mask Dice misses**
  (E4, 2026-11-04). Of the gated masks, E2 alone catches 79 of 100 shifted by 5 px (P3; E1's
  shift search reaches 3 px, so only large lesions keep overlap: E1 catches 44% at ≥ 80 px, 0% below
  40 px), 19 of 37 with one lesion removed (P6-remove) and 33 of 100 with one held-out lesion added
  (P6-add; an added lesion swamps the Dice of a small base: E1 30% at 12–19 px, 96% at ≥ 80 px).
  E1 alone catches at most 1 mask per variant (P4, P6-add) and 8 P5-dilate masks. This is evidence
  on constructed copies; whether S contains such copies is E5's question. *Source:
  `results/stanford_brainmets/e4/validate/rq2_planted.csv`, `recall_by_group.csv`,
  `recall_by_size.png`.*
- **The detection limit for boundary noise, measured** (E4, 2026-11-04). Recall of all P5 masks
  (either detector) by the size of the base's largest lesion: erosion 0% at 12–19 px, 12% at
  20–39 px, 47% at 40–79 px, 85% at ≥ 80 px; dilation 13%, 30%, 59%, 85%. The lesion score of a
  planted lesion against its own base lesion has a median of about 0.36 (shape and size factors
  both about 0.6), as the r² reasoning behind decision K predicted. *Source:
  `results/stanford_brainmets/e4/validate/recall_by_group.csv`, `factors.png`.*
- **False positives on H-eval at the frozen thresholds** (E4, 2026-11-04): E1 copy 7.2% / 7.4%,
  E2 full 7.1% / 2.9%, E2 partial 3.0% / 5.5%, E2 flagged 10.1% / 8.5% (single / multi; 1,089
  slices of 26 patients, 906 with an informative lesion), against 5.1% / 5.3%, 5.1% / 5.2%,
  2.3% / 3.5% and 7.4% / 8.7% on H-cal. Flags cluster in patients: E1 copy single 50 flags from
  15 of 26 patients (44% from the top 3), E2 full multi 8 flags from 6 of 14 (62% from the top
  3). *Source: `results/stanford_brainmets/e4/validate/fp_heval.csv`; clustering from a
  read-only check of `heval_classes.csv`.*
- **Small lesions can match exactly by chance** (E4, 2026-11-04). Two H-eval slices match a
  training slice of another patient with whole-mask Dice 1.00 after a shift of 1–2 px: single
  lesions of 8 and 4 px (`Mets_021/z96` with `Mets_051/z88`, `Mets_053/z75` with
  `Mets_167/z65`). Not leakage (held-out patients, unrelated slices): a few-pixel blob has
  almost no shape. E1 counts every lesion ≥ 4 px, so such matches enter its chance tail; E2
  ignores lesions below the 12 px bound. *Source: `results/stanford_brainmets/e4/validate/
  heval_flags.png`; read-only check of `e1_heval.csv`.*
- **The nearest neighbour and the strongest lesion pair can name different patients** (E4,
  2026-11-04). The 4 missed P6 masks were all flagged as full copies, but of another patient's
  slice (highest C, e.g. 0.91 for `Mets_107/z56` with one lesion removed, against
  `Mets_271/z53`), while their strongest lesion pair (s = 1.00) still pointed to the base. A
  single remaining small lesion often fits another patient's slice as a whole better than its
  own (incomplete) slice does. The one P4 miss is different: both detectors rank the base slice
  first, but its 13 px lesion shrank to 12 px with shape overlap 0.67 (s 0.61), not closer than
  chance. *Source: `results/stanford_brainmets/e4/validate/misses.png`, `planted_outcomes.csv`.*

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
  <40 px filter (see decision E in section 2). E0b checks each filter on the data.
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

- **Detection limit: copies with boundary noise are not detected** (2026-11-04, decision K).
  Both detectors are calibrated to chance, and at 128×128 a 1 px change of a lesion's boundary
  moves it further from its original than chance matches lie: lesion score ≈ r² for area ratio r
  (median r 0.43 after a 1 px erosion, 1.7 after a 1 px dilation), below the single-lesion
  thresholds even for lesions ≥ 80 px; multi-lesion configurations of lesions of about 40 px and
  more can still pass. So the audit finds copies that reproduce the lesion boundaries (with
  shifts up to a few pixels and small rotations, tested by P2–P4), not copies whose small
  lesions are redrawn with about 1 px of boundary noise; a generator that memorized
  configurations but not exact boundaries would be undercounted. P5's recall by lesion size
  (E4 `validate`) quantifies this: 0% at 12–19 px, 85% at ≥ 80 px (section 3). *Source:
  `docs/e4-plan.md` (decision K), `results/stanford_brainmets/e4/plant/area_change.csv`.*

- **The chance rate at the frozen thresholds is about 7%, not 5%, on H-eval** (E4, 2026-11-04).
  The thresholds are 95th percentiles of 25 H-cal patients; on the other 26 held-out patients
  they flag 7.1–7.4% in three cells (E1 copy, E2 full single). E5 therefore compares S with the
  measured H-eval rates per class and stratum, never with the nominal 5%, and every chance rate
  is reported with a bootstrap-by-patient interval. With so few held-out patients, chance rates
  are uncertain by a few points. *Source: `results/stanford_brainmets/e4/validate/fp_heval.csv`.*
- **Combined perturbations were not tested** (decided 2026-11-04 with the user). E4 varies one
  thing at a time (shift, rotation, boundary, one lesion removed or added). A copy that both
  drops a lesion and moves the others is not planted: its result follows from the parts, since C
  averages the lesion scores and each lesion score is shape × position × size (e.g. one of two
  lesions removed and the other shifted 2 px: caught as partial, s 0.95; shifted 5 px: missed,
  s 0.737 < τ_L 0.739). A copy altered so much that it falls below the chance level cannot be told
  from a novel mask by any output-only method; whether S contains such "drifted" partial copies
  is described in E5/E7 from the stored lesion pairs, not validated. *Source: discussion,
  2026-11-04; frozen thresholds.*

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
- [x] E4 go/no-go criteria (decision K in `docs/e4-plan.md`): fixed after the `plant` stage and
      before `validate` runs. *Resolved 2026-11-04: see "E4 go/no-go criteria" in section 2
      (P5 reported only; detection limit in section 4).* The lesion-level false-positive criterion must be stated per class
      (full copies ~5% at chance; full or partial together 7–9%); proposal: E1 copy ≤ 7%, E2
      full ≤ 7%, E2 flagged ≤ 12% per stratum. Also weigh P5-dilate on small lesions (may be
      missed by both detectors by construction). 2026-11-03, updated 2026-11-04.
      *P6's expected class: resolved 2026-11-04 by decision J* (caught = flagged and the right
      source patient; incompleteness shown with `all_copied`; see section 2).
- [ ] **3D lesion identity for H1 (E6).** 26-connected 3D linking merges neighbouring metastases
      for 2.7% of T's 3D lesions, holding 10.3% of the 2D lesions (section 3). Options: link 2D
      lesions slice to slice by overlap (one-to-one), or keep 26-connectivity and re-run H1
      without the merged 3D lesions as a sensitivity. Decide when E6 is planned; noted in
      `docs/experiment-plan.md`, E6 duplication test step 1. 2026-11-03.
- [x] **Does a failed synthetic MRI exclude its mask from the mask audit?** *Resolved
      2026-11-02 at the `qc` stage: no.* `qc.csv` keeps `mask_class` and `mri_failed` as separate
      columns; the main audit uses `mask_class` clean or minor, and `mri_failed` only excludes a
      sample where the brain is used. See "QC classes" in section 2; the question as first
      recorded is in the process log (2026-11-02).
- [x] **Position frame for copy detection (E2), a deviation from the plan to decide when E2 is
      planned.** *Resolved 2026-11-03: pixel coordinates (decision F); see "Position frame for
      copy detection" in section 2.* The question as recorded: The plan measures lesion position relative to the brain box of the paired MRI.
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

- [ ] **How a match report presents a flag whose nearest neighbour and strongest lesion pair name
      different patients (E7).** Seen in E4 for 4 P6 masks (section 3). Options: show both in
      Panel A/C and let the verdict line name the strongest pair's patient when it differs; and
      decide which one the E7 correctness check scores. Decide when E7 is planned; noted in
      `docs/experiment-plan.md`, E7 "Carried over from E4". 2026-11-04.

## 6. Figure and table candidates

- **E4 planted variants** (methods): rows picked by hand from
  `results/stanford_brainmets/e4/plant/variants_grid.png` or the `variant_<P>.png` sheets; with
  `area_change.png` for what 1 px of boundary change does to small lesions.
- **E4 recall by lesion size** (results, RQ2 and the detection limit):
  `results/stanford_brainmets/e4/validate/recall_by_size.png`; the recall table in section 3.
- **What a chance match looks like** (methods or discussion):
  `results/stanford_brainmets/e4/validate/heval_flags.png`.
- **Where each variant loses its score** (methods): `results/stanford_brainmets/e4/validate/factors.png`.

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
