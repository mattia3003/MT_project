# Project context: memorization in a mask→MRI diffusion cascade

Preliminary project (autumn 2026) leading into a master thesis (spring 2027).
Topic: memorization in synthetic 2D brain MRI with metastases, produced by a two-stage
diffusion pipeline from a previous master thesis. Key theme: explainable (xAI) detection
and investigation of memorization.

Reference docs (read the relevant one before starting work):
- `docs/thesis-plan.md`: research questions, hypotheses, roadmap, risks.
- `docs/experiment-plan.md`: procedures for E0–E8, shared setup and definitions, schedule, open decisions.
  This is the main reference when implementing an experiment.
- `docs/literature-timeline.md`: literature timeline and notes.
- `docs/report-notes.md`: running log of decisions, findings, limitations and open questions,
  organized by report section. Raw material for the written report.
- `docs/architecture.md`: pipeline flow chart, module diagram (Mermaid) and data dictionary
  (every file the pipeline reads or writes, with its columns).
This file is a summary. If it disagrees with the docs, the docs win; flag the mismatch.

## Data
- Stanford brain metastasis dataset (likely BrainMetShare, Grøvik et al.): 156 patients,
  3D MRI, split 105 train / 51 test by patient. Each patient folder has an ID.
- Previous thesis code: https://github.com/amibekk/Thesis (`Generation/diffusion_training.py`,
  `Generation/Sampling_mask_bravo.py`, `Generation/Metrics/helper_functions.py`). Very close to
  what was run (differences probably file paths; training script misses `import time`, wrong
  arg count).
- Real data: skull-stripped Stanford volumes, in-plane sizes vary (e.g. 160x176, 144x192).
  Training transform per volume: LoadImage -> EnsureChannelFirst -> ToTensor ->
  ScaleIntensity(0, 1) (per-volume min-max) -> Resize((128, 128, -1)) (MONAI default
  interpolation = area). Aspect ratio NOT kept (axes squeezed independently); depth NOT
  resampled. Masks binarized after resize at threshold 0.01 (enlarges lesions by ~1 px at the edge;
  measured lesion area per patient = median 1.5x plain downscaling, range 1.08-2.70).
- No augmentation, no seeds.
- Stage 1 trained on slices whose mask sum > thres (lesion slices); stage 2 on all non-empty slices.
- All synthetic data (masks and MRI) is 128x128. ALL comparisons are done in this 128x128
  "model space": real data must go through exactly the same transform before comparison.
- Real masks are binary (0/1) at native resolution; area-resize + 0.01 threshold enlarges them
  in model space (see above; strongest for small lesions). No lesion slice is lost or gained.
  Learn the minimum real lesion size from converted real masks.
- Split (E0a, fixed): `splits/stanford_brainmets_seed2026.csv`. T = 105 train patients (3,823
  lesion slices); test patients split by matched pairs on lesion-slice count, seed 2026:
  H-cal 25 (992 slices), H-eval 26 (1,094 slices).
- Stage-1 slice selection is not fully known (repo's empty-slice filter is never applied; its
  mask-only branch could not run as written). Bounded impact: if slices with <=1 or <=2 lesion px
  were dropped, 2 or 10 of the 3,823 T slices were never seen. State as a limitation.
- Synthetic pool in hand: 37,624 files in six naming schemes (`<n>`, `file_<n>`, `file_2_<n>` ...
  `file_5_<n>`, most likely one per generation round; they look alike on all index statistics).
  The sample ID is the full file stem, never a parsed number (`1033` and `file_1033` differ).
  Generated in several rounds with a script very close to the repo version (repo shows 15,000 attempts per run). Repo filters: mask min-max scaled then
  thresholded at 0.01; kept only if pixel sum < 2000, at least one lesion < 40 px, all lesion
  pixels inside synthetic brain. An earlier check flagged ~2,243 poor masks; RE-RUN quality
  control fresh in E0 to verify that count. Verify which filters actually hold on the data and
  apply only those to real comparison sets.
- Mask quality control (E0): many noisy masks have scattered 1-2 px specks (likely from the
  per-mask min-max + 0.01 threshold). Classify masks clean / minor (<= ~3 specks, keep, specks
  removed) / noisy (exclude); edge check only for real-sized lesions (>50% outside brain);
  re-apply the <40 px rule after cleaning to all sets; report results with and without noisy
  masks.
- Lesion labelling: 8-connected components, then "attach-or-drop" for components below the
  learned minimum lesion area: attach to a real-sized lesion if within 2 px (stray pixel),
  otherwise drop (speck). Never merge two real-sized lesions. Apply identically to real and
  synthetic masks. QC counts isolated specks only; stray pixels are counted separately.
  Sensitivity: attach distance 1 vs 2 px, and drop-only. Keep masks as saved (with specks) for any mask<->MRI analysis (stage 2 saw them).
- Saved synthetic files: NIfTI of shape (128, 128, 2) = [synthetic BRAVO, synthetic mask].
- All synthetic MRIs started from ONE shared initial noise tensor (noise_mr created once).
- Checkpoints used for sampling (epoch 149 of 200): stage 1
  `correct_validation_data_16_Epoch149_of_200`, stage 2 `conditional_syn_26mars_16_Epoch149_of_200`.
  The stage-1 name implies a validation set; assume it came only from the 105 training patients.
- Stage 1 (mask generator) was trained only on lesion-containing (non-empty) mask slices.
- Stage 2 (image generator) was trained on (nearly) all BRAVO slices, each concatenated with
  its own real annotation mask (empty masks for lesion-free slices).
- Synthetic samples have NO link to training patients (no ID, no slice index). Nearest-neighbour
  matching against training data is what assigns a synthetic sample to a patient.
- CONFIRMED by the previous author: the standard 105/51 split was used unchanged (no test
  patients in training).
- Data use is authorized: all results, including match reports showing real masks, may go in
  the written report (use pseudonymous patient IDs).
- Decided during the work (not yet fixed): the informative-lesion bound (choose while coding,
  by checking how candidate bounds affect the synthetic pool, alongside the QC re-run), and the
  E4 go/no-go criteria (must be fixed before E4 runs; never tune on S).

## Models (MONAI `DiffusionModelUNet`, same architecture for both stages)
```
DiffusionModelUNet(spatial_dims=2, in_channels=model_input, out_channels=1,
                   num_channels=(128, 256, 256), attention_levels=(False, True, True),
                   num_res_blocks=1, num_head_channels=256)
```
- Both models are PIXEL-SPACE DDPMs, NOT latent diffusion models: no autoencoder anywhere,
  `DiffusionInferer` (not `LatentDiffusionInferer`), noise sampled at (1, 128, 128) and saved
  directly as images. Don't call them LDMs.
- Stage 1: `in_channels=1` (mask only, unconditional).
- Stage 2: `in_channels=2` → mask is CONCATENATED as an input channel. Only the image channel
  is noised; the mask is passed unchanged at every timestep (training: real mask; sampling:
  synthetic mask from stage 1).
- `with_conditioning` is off → attention layers are SELF-attention only. There is NO
  cross-attention in either model, despite what the original project description says.
  Cross-attention methods (Bright Ending, NeMo) do not apply as published.
- No seeds were used in the original codebase → original models cannot be reproduced exactly.

## Preliminary project (output-only, no weights needed)
Experiment flow (text version of the diagram in the plans):
E0 data audit + mask QC -> E1 whole-mask detector + E2 lesion-level detector (in parallel)
-> E3 calibration on H-cal [FREEZE all thresholds; S never touched before this point]
-> E4 planted-copy validation = GATE (fail -> adjust on H-cal, re-freeze, re-run E4)
-> E5 main audit on S (scored once) -> E6 patient exposure + E7 match reports (in parallel)
-> E8 paired-MRI check (optional, last). Critical path: E0 -> E3 -> E4 -> E5.
Roadmap: preliminary audit (no weights) -> master thesis on retrained models (Branch B, planned;
original weights = Branch A fallback); the preliminary toolkit is reused in the master thesis.

Timeline: full draft to supervisor by 30 Nov 2026, submission 9 Dec 2026. Milestones:
E0 + mask QC by 31 Oct; E1-E3 (bound + thresholds frozen) by 9 Nov; E4 gate by 13 Nov;
E5 by 17 Nov; E6/E7 by 24 Nov; E8 (optional RQ4) last, if time. If behind, cut: E8 (moves to
the master thesis), then optional small-lesion agreement (E2/E7) (null model already dropped),
then E7 taxonomy, then the lesion-level part of
the E7 correctness check (keep source-patient correctness + deletion test). Never cut E4.
ALL comparisons are between slices WITH metastases: T = the ~3,800 lesion-containing training
slices only; every synthetic mask has lesions; H-cal/H-eval are lesion slices only.

Overarching: To what extent does the mask generator in a two-stage 2D denoising diffusion (DDPM)
pipeline for brain-metastasis MRI synthesis memorize patient-specific lesion configurations, and
can this be detected and explained using generated outputs alone?
- RQ1: To what extent do synthetic brain-metastasis segmentation masks match training masks
  more closely than held-out real masks do? Calibrate against held-out test patients (H-cal,
  95th-percentile thresholds, as in Dar et al. 2024). Null model DROPPED (2026-11-01): chance
  rates come from H-cal only; multi-lesion strata rest on 12-17 H-cal patients, so report
  bootstrap-by-patient CIs for thresholds and pool single/multi strata if unstable.
- RQ2: Does lesion-level matching reveal memorization (partial copies) that whole-mask Dice
  misses? Recombinations across patients are OUT OF SCOPE (optional master-thesis idea).
- RQ3a: Which patients are most exposed, and is exposure linked to how unusual their
  metastasis configuration is?
- RQ3b (xAI): Can each flagged match be explained at patient and lesion level, such that the
  lesions named in the explanation are the ones that drive the match score (faithfulness,
  deletion test) and the explanation identifies the true source on planted copies with known
  origin (correctness)? Via match reports. Suggested criteria: paired Wilcoxon p < 0.05 for top
  vs smallest contributor deletion; correct source patient for >= 90% of planted copies and
  correct lesions for >= 80% of planted partial copies (P6). No human rater (master thesis).
- Hypotheses: H1 (RQ1) copy rate exceeds the held-out rate, and copies concentrate on lesions
  spanning many adjacent training slices (3D lesion labelling, 26-connectivity; Spearman of
  copies vs slice span, also within lesion-size bins). H2 (RQ2) at equal false-positive rate
  on held-out masks, lesion-level matching flags more than whole-mask Dice. H3 (RQ3a)
  memorization concentrates in rare configurations (Spearman of exposure vs rarity).
- RQ4 (optional, exploratory, E8, only if >= ~20 flagged masks and time remains): is the paired
  synthetic MRI of a flagged mask closer to its source slice than paired MRIs of novel masks are
  to their nearest training slice? H4: yes (observational; causal test is RQ-M1). Use masks as
  saved; whole-brain and lesion-region SSIM; Mann-Whitney A vs B.

### Lesion-level detector (E2), as decided
- Informative-lesion bound: only lesions at/above a size bound count in lesion matching. Set in
  E3 from H-cal: smallest area where the chance lesion-match rate drops below ~5-10% (expect
  ~10-20 px at 128x128). Separate from (higher than) the QC speck threshold. Slices with no
  informative lesion are excluded from lesion-level analysis but still scored by whole-mask
  Dice (E1). Report exclusion counts. For RQ2, compare both detectors on the same subset.
- Optional small-lesion agreement: sub-bound lesions never enter C or the copy class. For each
  candidate slice pair, count small lesions in q with a small lesion in t within ~3 px (centroid
  distance only); store as extra slice-pair columns; chance baseline from the same check on H-cal.
  Used only as supporting evidence in E7 match reports (Panel C + verdict line).
- Lesion score s(a, b) = shape (IoU of centred crops) x position exp(-d^2/2 sigma^2), sigma =
  0.05 brain box) x size (smaller/larger area). Store all three factors.
- Slice-pair score C(q, t): Hungarian one-to-one pairing on cost 1 - s, keep pairs with
  s >= tau_L, C = sum(kept s) / max(n_lesions_q, n_lesions_t). NN = training slice with max C.
- Exact shortcuts (no approximation): position prefilter (skip lesion pairs > 0.2 brain box
  apart; confirm s < tau_L there) and candidate slices (only compute C for training slices
  sharing a lesion pair with s >= 0.1; all others have C = 0).
- Classes: full copy C(q, NN) >= tau_C; partial copy C < tau_C but >= 1 lesion pair in NN slice
  passes tau_L; novel otherwise. Also record the strongest single lesion-pair score across all
  candidate slices (and its slice).
- Outputs (store ALL candidate pairs, apply thresholds at analysis time):
  1. slice-pair table (synthetic id, train slice, patient, slice index, C, lesion counts,
     n matched, strongest single lesion score),
  2. lesion-pair table (Hungarian-matched pairs: lesion ids, shape/position/size factors, s,
     passed tau_L),
  3. synthetic summary (NN slice + C, class, strongest single lesion match + its slice).
  Exposure (E6) uses all stored pairs above threshold, not just NN, so a lesion copied across
  adjacent training slices credits all of them.
- E4 planted copies: 20 of the 105 training patients x 5 lesion slices; variants P1 exact,
  P2 shift 2 px, P3 shift 5 px (tolerance limit, reported only), P4 rotate 5 deg, P5 erode/
  dilate 1 px, P6 remove one lesion or add one from H-eval. Negatives: all H-eval masks.
- Count per patient and per lesion, not per slice (adjacent slices are near-duplicates).

## Master thesis (plan: retrain both models with fixed seeds, Branch B; compute is available)
Not aimed at publication for now. Original weights (Branch A) only as a fallback if they turn up.
- Phase 1 pilot, only if E8 was not done in the preliminary project: paired-MRI check, observational, on existing outputs:
  are paired MRIs of flagged masks closer to their source slice than those of novel masks are
  to their nearest training slice? Use masks exactly as saved (stage 2 saw them with specks).
- RQ-M1: Is a training mask a "key" into stage 2? Feed training vs held-out vs synthetic masks
  with fixed noise seeds; mask-perturbation series (shift/erode/remove lesion).
- RQ-M2 (fallback): does the full pipeline leak even when each stage passes on its own?
- RQ-M3: which slices/patients get memorized and why (training-data attribution, e.g. D-TRAK).
- RQ-M4: explain reliance on the mask via gradient attribution on the mask channel, mask
  perturbation, and condition sensitivity vs an empty mask (Wen et al. style; empty mask is
  in-distribution for stage 2). Self-attention maps are optional, indirect evidence.
- Optional: recombinations of lesions from different patients (needs a null model); blinded
  human check of the match reports.

## Conventions
- Always split and aggregate by patient ID, never by slice.
- Never let test-patient data leak into any training or threshold-fitting step.
- Set and log random seeds for every training/sampling run.

## Code layout and working rules
- `configs/default.yaml`: datasets, synthetic pools, paths, seed (2026) and parameters. New data
  is a new config entry, not new code. Relative paths resolve from the repo root.
- `src/memtk/`: reusable library (installed editable: `pip install -e .`). Building blocks such
  as transforms, IO and the E1/E2 detectors live here, with no hardcoded paths.
- `scripts/`: one entry script per experiment, named so that alphabetical order is run order
  (`e0a_prepare_data.py`, `e0b_audit.py`, `e3_calibrate.py`, ...). Steps are subcommands
  (`e0a_prepare_data.py split|real|synthetic|check`). `scripts/README.md` has the run order,
  inputs and outputs. E1/E2 have no scripts; they run inside E3/E4/E5.
- Outputs: processed data in `processed_data/model_space/<dataset|pool>/` (not in git); results
  in `results/<dataset|pool>/<experiment>/` (in git); every stage writes a run_info JSON (seed,
  config, versions). Stages refuse to overwrite without `--overwrite`.
- Slice index `z` is the 0-based array index. Model-space volumes use an identity affine.
- The user stages and commits all git changes manually; work in small steps, each one
  reviewed before the next.
- At the end of each step, and whenever a decision, finding, limitation or open question comes
  up in discussion, suggest additions to `docs/report-notes.md` (dated, with their source) for
  the user to review. Never record a number without saying where it comes from.
- When a stage or module is added or changed, update `docs/architecture.md` in the same change
  (flow chart, module diagram from the actual imports, data dictionary), together with the
  status table in `scripts/README.md`. Earlier segmentation work is under the git tag `segmentation-final`.
