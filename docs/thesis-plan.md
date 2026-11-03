# Thesis Plan — Memorization in a Mask→MRI Diffusion Cascade

Oct 20, 2026 · @Mattias Lund Dahlin

## Summary

For the preliminary project, I recommend a lesion-level audit of the mask generator that needs only its outputs. It would ask:

- Does the mask generator reproduce patients' metastasis configurations more often than chance?
- Which patients are most exposed?
- Do whole-mask metrics miss lesion-level copies?

The xAI component is an interpretable matching method that names the training patient and the lesion behind each match. All of this works without network weights.

The master thesis then moves to the cascade. It asks whether a copied mask acts as a key into the image model, whether the full pipeline leaks even when each stage passes on its own, and which slices get memorized. The first two are answered with experiments that need weights, either the originals or retrained models. Your fallback question about pipeline leakage stays in as a research question that holds up whatever the preliminary project finds.

Your three preliminary ideas hold up, with one change of emphasis:

- **Mask memorization and patient-specific lesion configurations:** keep.
- **xAI:** keep, but aim it at the measurement, not at the generator, since you have no weights.
- **Local versus whole-image memorization:** fold into the mask audit as lesion-level versus whole-mask. The image-side version belongs to the master thesis.

## Roadmap

&#91;embedded content: roadmap · preliminary project, weights gate, master thesis\]

The preliminary project needs no weights. The weights question only decides how the master thesis starts, and the current plan is to retrain (Branch B).

## Has it been done?

As far as my searches reach, no one has audited a mask (label) generator for memorization, and no one has measured medical memorization per patient or per lesion. Local memorization has been studied, but only in text-to-image models of natural images, and those methods read the model's cross-attention. That rules them out for the preliminary project, because the mask generator has no cross-attention and you have no weights for either model. Your angle is the medical, lesion-level, output-only version of it. Neither model has cross-attention. Both are MONAI UNets with self-attention only, and stage 2 takes the mask as a second input channel. So in the master thesis, mask-channel attribution replaces these attention-based methods (RQ-M4).

| Your idea | Closest prior work | What has not been done | Verdict |
| --- | --- | --- | --- |
| The mask generator memorizes lesion shapes and positions | Cascades such as [brainSPADE](https://arxiv.org/abs/2209.08256) and [MedSegFactory](https://arxiv.org/abs/2504.06897) judge masks on realism and downstream use only. [Edirisooriya et al. 2026](https://arxiv.org/abs/2603.04340) tests privacy on the image stage only. | Any memorization audit of a mask or label generator | New |
| Patient-specific metastasis patterns | Re-identification has been shown from chest X-rays ([Packhäuser et al. 2022](https://www.nature.com/articles/s41598-022-19045-3)) and histopathology slides ([2024](https://arxiv.org/abs/2403.12816)). Medical memorization studies count samples, not patients ([Dar et al. 2024](https://arxiv.org/abs/2402.01054)). | Metastasis configuration as an identifier; memorization reported per patient and linked to how rare the patient is | New angle |
| Lesion-region versus whole-image memorization | Natural images: [Bright Ending attention](https://arxiv.org/abs/2410.21665) (ICLR 2025), [FB-Mem](https://arxiv.org/abs/2508.12148) (2025), [curvature differences](https://arxiv.org/abs/2605.26756) (May 2026). Medical work uses whole-image metrics: [DeepSSIM](https://arxiv.org/abs/2509.16582) and [DeepSSIM++](https://arxiv.org/abs/2609.03615) (Sep 2026). | Lesion-level memorization in medical data, and any method that works without model internals | Idea exists; the application is new. Cite all three natural-image papers. |
| xAI to investigate memorization | [Wen et al.](https://arxiv.org/abs/2407.21720) and [NeMo](https://arxiv.org/abs/2406.02366) explain memorization from model internals. | Explanations built only from outputs and training data, at the level of patients and lesions | Modest but defensible; frame it as an "explainable audit" |

Two cautions:

- The field moves fast. DeepSSIM++ is from September 2026, and the same group is working on brain MRI memorization. Re-run the search before you submit, especially MICCAI 2026, MIDL 2026 and their workshops.
- "New" here means not found in arXiv and web searches. Ask your supervisor whether the group knows of unpublished or in-review work.

## Preliminary project: research questions

There are four core research questions (RQ1, RQ2, RQ3a and RQ3b) and one optional, exploratory question (RQ4). All can be answered from the synthetic masks and their paired MRIs, the real training masks, and held-out real masks. No model weights are needed. All comparisons are between slices that contain metastases: every synthetic mask has lesions, and the training reference is the about 3,800 lesion-containing training slices.

**Overarching question:** To what extent does the mask generator in a two-stage 2D denoising diffusion (DDPM) pipeline for brain-metastasis MRI synthesis memorize patient-specific lesion configurations, and can this be detected and explained using generated outputs alone?

- *Details:* Stage 1 is an unconditional DDPM that generates lesion masks. Stage 2 is a DDPM that generates the matching BRAVO slice, conditioned on the mask by channel concatenation. Both work in pixel space at 128×128.

**Research questions**

- **RQ1 – Memorization beyond chance:** To what extent do synthetic brain-metastasis segmentation masks from a 2D unconditional diffusion model match training masks more closely than held-out real masks do?
  - *Details:* Metastases are small, often single, and cluster in common locations, so some matches will happen by chance. Calibrate against held-out masks. (A null model was considered and dropped on 2026-11-01; see the Experiment Plan, E3.)
- **RQ2 – Lesion-level versus whole-mask:** Does matching individual lesions reveal memorization that whole-mask similarity misses?
  - *Details:* The case to look for is the partial copy: some lesions match one training slice, and the others are new. Only lesions above an informative size bound count, since blobs of a few pixels match each other by chance.
- **RQ3a – Patient exposure:** Which patients are most exposed, and is exposure linked to how unusual their metastasis configuration is (lesion count, size, location, slice position)?
- **RQ3b – Explanation:** Can each flagged match be explained at the level of patient and lesion, such that the lesions named in the explanation are the ones that drive the match score (faithfulness), and the explanation identifies the true source on planted copies with a known origin (correctness)?
  - *Details:* The question tests two measurable properties, named with standard xAI terms:
    - **Faithfulness:** removing the lesion the explanation names as the top contributor lowers the match score clearly more than removing the smallest contributor (deletion test).
    - **Correctness:** on planted copies with a known origin (from E4), the explanation names the true source patient, slice and lesions.

    Both are measured in E7. This is the xAI contribution.
- **RQ4 – Mask–image association (optional, exploratory):** For masks flagged as copies, is the paired synthetic MRI closer to the source slice than the paired MRIs of novel masks are to their nearest training slice?
  - *Details:* This uses existing outputs only, so it can show an association, not that the mask caused the similarity. It is the last experiment (E8) and runs only if E5 yields about 20 or more flagged masks and time remains. Otherwise it moves to the master thesis as a phase-1 pilot for RQ-M1.

**Hypotheses** (numbered to follow the research questions)

- **H1 (RQ1):** The copy rate among synthetic masks exceeds the held-out rate, and copies concentrate on lesions that appear in many adjacent training slices.
  - *Rationale:* Slicing a 3D lesion into 2D creates many near-identical training examples, and duplication is known to drive copying (Somepalli et al.).
- **H2 (RQ2):** At the same false-positive rate on held-out masks, lesion-level matching flags more memorization than whole-mask Dice.
- **H3 (RQ3a):** Memorization concentrates in rare configurations, as Feldman and Zhang's long-tail argument predicts.
  - H1 is about duplication within a patient, and H3 about rarity across patients. Both are tested together in E6.
- **RQ3b success criteria** (no directional hypothesis; thresholds to agree with your supervisor):
  - **Faithfulness:** the mean score drop for the top contributor exceeds that for the smallest contributor (paired Wilcoxon test, p < 0.05).
  - **Correctness:** the report names the correct source patient for at least 90% of planted copies, and the correct lesions for at least 80% of planted partial copies. A blinded human check is left to the master thesis and stated as a limitation.
- **H4 (RQ4, optional):** Paired MRIs of flagged masks are more similar to their source slice than paired MRIs of novel masks are to their nearest training slice.
  - *Rationale:* Stage 2 was trained with each slice's own real mask, so a near-copy mask resembles a cue it has seen. The causal version is tested in the master thesis (RQ-M1).

| Question | Hypothesis | Experiments |
| --- | --- | --- |
| RQ1 | H1 | E5, E6 |
| RQ2 | H2 | E4, E5 |
| RQ3a | H3 | E6 |
| RQ3b | Success criteria | E7 |
| RQ4 (optional) | H4 | E8 |

**Deliberately out of scope:**

- Training or retraining models.
- Membership-inference attacks.
- xAI that reads model internals.
- Image-side analyses beyond the optional RQ4.
- Recombinations of lesions from different patients (an optional idea for the master thesis).
- Mitigation.

## Preliminary project: methods and work plan

The method has six steps, and each output is reused in the master thesis. The plan runs to a full draft for your supervisor by 30 November and submission on 9 December. Milestones are dates rather than fixed weeks, so heavy and light weeks can even out. The Experiment Plan has the detailed schedule and what to cut if you fall behind.

**Methods**

1. **Data audit (the blocker).**
   - Confirmed by the previous author: both models were trained only on the 105 training patients of the Stanford dataset (likely BrainMetShare, from Grøvik et al.), and the 51 test patients were never used. This clean held-out set is what lets you tell memorization apart from mere similarity.
   - Use the lesion slices of the 51 test patients as the held-out baseline.
   - Record the mask format (binary or multi-class, slice index, whether slices are registered to a common space), the number of synthetic masks, and how many lesion slices the mask generator was trained on (only non-empty masks were used; about 3,800 slices, which form the reference set T). Every comparison is between slices with metastases. Real volumes vary in size (for example 160×176) and were scaled to 128×128 for training, with masks re-binarized at about 0.01. Compare everything in that 128×128 model space. The synthetic pool (about 37,000 samples) may not match the repo's sampling filters, and some masks are noisy. Verify the filters on the data, run the mask quality control in the Experiment Plan, and apply the same rules to every comparison set.
2. **Whole-mask similarity.** For each synthetic mask, find its nearest training mask by Dice or IoU. Allow small shifts. No flips are needed, since training used no augmentation. Stage 1 was trained only on non-empty masks, so empty synthetic masks should be rare; report any separately.
3. **Lesion-level matching.**
   - Split each mask into lesions (connected components). Attach stray pixels within 2 pixels of a lesion to that lesion, and drop isolated specks.
   - Describe each lesion by its centroid relative to the brain (the bounding box from the paired MRI slice works), its area, and its shape (centred IoU, eccentricity, solidity).
   - Keep only informative lesions, above a size bound set from held-out data, since blobs of a few pixels match each other by chance.
   - For each synthetic–training slice pair, match lesions one-to-one with the Hungarian algorithm, and score the configuration by the matched lesion scores divided by the larger lesion count. This detects full and partial copies. Only pairs that can score above zero are computed (a position prefilter and candidate slices), which keeps about 35,000 × 3,800 slice pairs affordable.
4. **Chance calibration.**
   - Set the copy threshold at the 95th percentile of held-out-to-training nearest-neighbour similarity, as in Dar et al. 2024.
   - A null model (fake configurations with lesions drawn independently from the empirical distribution of positions and sizes) was considered and dropped on 2026-11-01. It mainly served the out-of-scope recombination question, and held-out patients are the more credible baseline. Thresholds get bootstrap-by-patient confidence intervals instead.
5. **Detector validation.** Plant known copies before trusting any result: real training masks, some shifted, rotated or eroded, mixed into the synthetic set. Report the detector's recall and precision on them. This is cheap and answers the first question a reviewer will ask.
6. **Patient level and explanation.**
   - Aggregate matches per patient, normalized by the patient's number of training slices. Synthetic samples carry no patient ID or slice index, so the nearest training match is what assigns each one to a patient.
   - Measure each patient's rarity as the distance to the nearest other patient in lesion-feature space.
   - Test H3 with a rank correlation between exposure and rarity. Test H1 by labelling lesions in 3D in each training volume, counting how many slices each spans, and relating that count to how often its slices are matched.
   - Produce a match report for each flagged sample: the synthetic and training masks overlaid, the matched lesions highlighted, and each lesion's contribution to the score. Because the score adds up from individual lesions, every match explains itself.

**Schedule**

| Complete by | Work | Output |
| --- | --- | --- |
| 31 Oct | Data audit and mask quality control; read Dar et al. 2024 and DeepSSIM | Data description and confirmed training split |
| 9 Nov | Whole-mask and lesion-level detectors; informative-lesion bound; held-out thresholds | Frozen thresholds |
| 17 Nov | Planted-copy validation (gate); main audit | Results for RQ1 and RQ2 |
| 24 Nov | Patient exposure, duplication and rarity tests; match reports, deletion test and correctness on planted copies; optional RQ4 (E8) if time | Results for RQ3a, RQ3b, H1 and H3 |
| 30 Nov | Writing, started in early November | Full draft to supervisor |
| 9 Dec | Revision after feedback | Submitted report and slides |

## Master thesis: research questions

The master thesis asks how memorization travels through the cascade. Every question still holds if the preliminary project finds little memorization in the masks.

**Overarching question:** How does memorization propagate through a two-stage mask→MRI diffusion pipeline, and can xAI explain where and why it happens?

1. **RQ-M1 – Is the mask a key?**
   - Feed the image model three kinds of mask: training masks, held-out real masks, and novel synthetic masks. Does a training mask make it more likely to reproduce that patient's training image?
   - As the mask is perturbed further (shifted, eroded, lesions removed), at what point does the output stop being a copy?
   - This is Somepalli et al.'s caption-as-key finding as a controlled medical experiment, and the causal follow-up to the observational paired-MRI check (preliminary RQ4). It needs only real masks as input, so it runs even if stage 1 never copies. Stage 2 was trained with each slice's own real mask concatenated to its input, so a training mask is exactly the condition it saw beside that patient's slice. The key effect is plausible for lesion slices; an empty mask carries no key.
2. **RQ-M2 – Does the pipeline leak when each stage passes?** This is your fallback. Can a synthetic mask and MRI together point to one patient when each falls below its own copy threshold? Define a joint match (mask match plus image match to the same patient) and compare it with the two single-stage rates.
3. **RQ-M3 – Which slices and patients get memorized, and why?**
   - Relate memorization to slice position, near-duplicate adjacent slices (Akbar et al.), lesion burden, and the patient rarity measured in the preliminary project.
   - Use training-data attribution (D-TRAK or Journey-TRAK) to trace flagged samples back to the training slices and patients that shaped them.
4. **RQ-M4 – Can xAI localize memorization inside the image model?**
   - Adapt Wen et al.'s condition-sensitivity signal to spatial mask conditioning: the gap between the noise predicted with the mask and with a null mask, mapped over the image.
   - Test whether these maps light up the regions that are actually copied, using the lesion-level copies from the preliminary toolkit as ground truth. An empty mask was a normal training input for stage 2, so it is an in-distribution null for these maps. Stage 2 has no cross-attention: it is a MONAI UNet with self-attention only, with the mask concatenated as a second input channel. Cross-attention localizers (Bright Ending, NeMo) therefore don't apply as published. Use gradient attribution on the mask channel and the mask-perturbation series instead; self-attention maps are optional and only indirect evidence.

**Optional RQ-M5, image-side local memorization:** Is the lesion region of a synthetic MRI copied more often than the rest of the slice?

**Optional, carried over from the preliminary project:** a blinded human check of the match reports. A rater who can't see the detector's output judges whether pairs are copies and whether each explanation matches what they see. This completes the RQ3b evaluation.

**Possibly carried over from the preliminary project:**

- **The paired-MRI check** (preliminary RQ4 and H4), if it wasn't done in the preliminary project. It is observational and uses existing outputs, so it runs in phase 1 as a pilot for RQ-M1.
- **Recombinations** (optional): synthetic masks whose lesions match lesions from different patients. They need a null model to separate them from chance.

**If the preliminary project finds no mask memorization,** M1 and M2 become the core of the thesis, and the preliminary result is reported as a negative finding.

## Master thesis: methods and phases

The plan is Branch B: retrain both models with the existing codebase. Compute is available for this. Branch A remains a fallback if the original weights turn up. Retraining is slower, but it gives you something the originals cannot: ground truth for membership.

**Branch A – original weights obtained.** Run RQ-M1 to M4 on the exact models that produced your data, so the preliminary results carry over directly. Training-data attribution (RQ-M3) also needs the training data the models saw and enough GPU memory to compute per-sample gradients.

**Branch B – retrain with the codebase (planned).**

- Fix seeds and log every configuration, which solves the reproducibility gap you found.
- Design the training split yourself: keep a known set of patients out so membership is ground truth, and train one deliberately memorization-prone model (small data, long training) as a positive control.
- Compute is available. Time is the constraint, so estimate training time from the codebase early, and consider lower resolution or fewer patients if needed.
- Regenerate the synthetic data and re-run the preliminary toolkit on it.

**Phases**, assuming a January–June semester:

| Phase | Weeks | Work | Feeds |
| --- | --- | --- | --- |
| 1 Setup | 1–4 | Get the weights (A) or retrain with seeds (B); reproduce the baseline outputs; re-run the preliminary toolkit; paired-MRI check, if not done in the preliminary project | All |
| 2 Key experiment | 5–8 | Real, held-out and synthetic masks into stage 2; mask-perturbation series | RQ-M1 |
| 3 Pipeline leak | 9–11 | Define the joint patient match; compare it with single-stage rates | RQ-M2 |
| 4 Explanation | 12–16 | Training-data attribution; condition-sensitivity maps; validation against lesion-level copies | RQ-M3, RQ-M4 |
| 5 Writing | 17–20 | Thesis; optionally a workshop paper | – |

## Reading order

Read the first group now: it shapes the preliminary-project methods. The rest can wait until the master thesis.

| When | Paper | Why |
| --- | --- | --- |
| Now, by 31 Oct | [Dar et al. 2024](https://arxiv.org/abs/2402.01054) (methods section) | Copy threshold and held-out baseline |
| Now, by 31 Oct | [DeepSSIM](https://arxiv.org/abs/2509.16582) and [DeepSSIM++](https://arxiv.org/abs/2609.03615) | Latest brain MRI memorization metrics; your closest competitors |
| By 9 Nov | [FB-Mem](https://arxiv.org/abs/2508.12148) and [Bright Ending](https://arxiv.org/abs/2410.21665) | Local memorization; must be cited for RQ2 |
| By 9 Nov | [Edirisooriya et al. 2026](https://arxiv.org/abs/2603.04340) | The only mask→image privacy study; defines your novelty |
| By 17 Nov | [Histopathology re-identification](https://arxiv.org/abs/2403.12816) | Patient-level privacy framing for RQ3a |
| Master, phase 1 | [Gu et al.](https://arxiv.org/abs/2310.02664), [Jain et al. (CFG)](https://arxiv.org/abs/2411.16738), [Dutt 2025](https://arxiv.org/abs/2502.07516) | Why conditioning drives memorization (RQ-M1) |
| Master, phase 4 | [Wen et al.](https://arxiv.org/abs/2407.21720), [curvature differences](https://arxiv.org/abs/2605.26756), [NeMo](https://arxiv.org/abs/2406.02366) | Model-internal xAI (RQ-M4) |
| Master, phase 4 | [Journey-TRAK](https://arxiv.org/abs/2312.06205), [D-TRAK](https://arxiv.org/abs/2311.00500) | Training-data attribution (RQ-M3) |
| Master, if MIA is used | [Dubiński et al.](https://arxiv.org/abs/2306.12983) | Evaluating MIA fairly |
| Master, if mitigating | [MemControl](https://arxiv.org/abs/2405.19458) | Medical mitigation baseline |

Four new papers since the last list are DeepSSIM++, Bright Ending, the curvature-difference paper, and the histopathology re-identification study.

## Risks and open questions

The two earlier blockers are resolved. The previous author confirmed the standard 105/51 split, and the master thesis will retrain the models (Branch B), so the original weights are no longer needed. Data use is authorized, so all results, including match reports with real masks, can go in the written report. One item remains open, the informative-lesion bound, and it will be decided during coding together with the quality-control re-run.

| Risk | Effect | Fallback |
| --- | --- | --- |
| Few lesion slices, and many near-duplicate adjacent slices | Chance matches and inflated slice-level counts | Count matches per patient and per lesion, not per slice; rely on the held-out calibration |
| Slices are not registered to a common space | Lesion positions can't be compared across patients | Copy detection compares positions in the 128×128 model space the mask generator worked in, since a copy keeps its training mask's pixel positions (decision F, 2026-11-03); brain-relative coordinates from the paired MRI's bounding box are used for anatomical questions |
| No mask memorization found | RQ1 gives a negative result | Report it as an audited negative with measured detector recall; the master thesis centres on RQ-M1 and M2 |
| Retraining takes longer than planned (Branch B) | Phase 1 overruns | Compute is available, so the limit is time: estimate training time early; if needed, train at lower resolution or on fewer patients, and keep the positive-control model small |

**To settle with your supervisor:**

- [x] Report length: none is fixed; it depends on the work done. Deadlines: draft by 30 November, submission 9 December.
- [x] Training split: the previous author confirmed the standard 105/51 split, with no test patients in training.
- [ ] The informative-lesion bound: decided during coding, by checking how candidate bounds affect the synthetic pool, together with the quality-control re-run that verifies the earlier count of about 2,243 poor masks.
- [x] Compute for retraining next semester: available. The plan is to retrain (Branch B).
- [x] Publication: the master thesis is not aimed at a publication for now.
