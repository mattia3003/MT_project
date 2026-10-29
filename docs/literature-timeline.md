# Memorization in Synthetic Brain MRI — Literature Timeline & Gap Analysis

Oct 17, 2026 · @Mattias Lund Dahlin

## Summary

Your cascade idea holds up. I found one medical study with a two-stage mask→image pipeline (cardiac MRI, March 2026), and it checks memorization only in the image stage. I found nothing that traces memorization across the two stages or explains it with xAI.

Your roughly 30 sources run from 2013 to 2025 and fall into five eras:

1. Generative foundations (2013–2017).
2. Diffusion becomes competitive (2020–2021).
3. Diffusion reaches medicine and copying is first measured (2022).
4. A wave of memorization and membership-inference work (2023).
5. Memorization measured in medical latent diffusion, plus work on how conditioning drives it (2024 onward).

Most of your notes are accurate. Three need fixing (see Notes check): LiRA, SecMI, and the 2023 Dar paper.

## Timeline

Your sources ordered by first public release, usually the first arXiv version. Journal versions often appear a year later.

&#91;embedded content: evolution of the field · 5 eras, 2013 to 2026\]

The medical memorization line starts only in 2023, about eight months after diffusion reached medical imaging in 2022. Conditioned medical models are the latest step.

| First release | Source | What it added |
| --- | --- | --- |
| Dec 2013 | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | VAEs: fast and diverse, but blurry |
| Jun 2014 | [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | GANs: fast and sharp, but weak mode coverage |
| Mar 2015 | [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) | Introduces diffusion probabilistic models |
| May 2015 | [U-Net](https://arxiv.org/abs/1505.04597) | The backbone later diffusion denoisers use |
| Jun 2017 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Transformers; cross-attention later carries conditions |
| Nov 2018 | [CheXNeXt](https://stanfordmlgroup.github.io/projects/chexnext/) | Context: deep learning reads chest X-rays at radiologist level |
| 2019 | [Grøvik et al., brain metastasis detection](https://onlinelibrary.wiley.com/doi/abs/10.1002/jmri.26766) | Detects and segments metastases on multisequence MRI |
| Jan 2020 | [Yi et al., robustness to false-negative annotations](https://arxiv.org/abs/2001.09501) | Metastasis segmentation that tolerates missing labels |
| Feb 2020 | [SimCLR](https://arxiv.org/abs/2002.05709) | Contrastive embeddings, the basis of later copy detectors |
| Jun 2020 | [DDPM](https://arxiv.org/abs/2006.11239) | Diffusion reaches GAN-level sample quality |
| Aug 2020 | [Feldman and Zhang](https://arxiv.org/abs/2008.03703) | Memorizing long-tail examples helps accuracy |
| Nov 2020 | [Score SDE](https://arxiv.org/abs/2011.13456) | Unifies diffusion and score-based models |
| Jun 2021 | [ISC 2021 dataset](https://arxiv.org/abs/2106.09672) | A benchmark for image copy detection |
| Dec 2021 | [LiRA](https://arxiv.org/abs/2112.03570) | Membership inference judged by TPR at low FPR |
| Dec 2021 | [Latent Diffusion](https://arxiv.org/abs/2112.10752) | Diffusion in an autoencoder's latent space |
| Feb 2022 | [SSCD](https://arxiv.org/abs/2202.10261) | Copy-detection descriptor, later used by Somepalli et al. |
| Sep 2022 | [Pinaya et al., brain LDM](https://arxiv.org/abs/2209.07162) | Conditioned 3D brain LDM; synthetic brains released |
| 2022 | [Packhäuser et al., patient re-identification](https://www.nature.com/articles/s41598-022-19045-3) | Networks re-identify patients from chest X-rays |
| Nov 2022 | [Khader et al., Medical Diffusion](https://arxiv.org/abs/2211.03364) | 3D latent DDPM for MRI and CT |
| Nov 2022 | [Kazerouni et al., survey](https://arxiv.org/abs/2211.07804) | Survey of diffusion models in medical imaging |
| Dec 2022 | [Somepalli et al., Diffusion Art or Digital Forgery?](https://arxiv.org/abs/2212.03860) | First large study of replication |
| Jan 2023 | [Hu and Pang, MIA on diffusion models](https://arxiv.org/abs/2301.09956) | White-box loss and likelihood attacks |
| Jan 2023 | [Carlini et al., Extracting Training Data](https://arxiv.org/abs/2301.13188) | Extracts training images; diffusion leaks more than GANs |
| Feb 2023 | [SecMI](https://arxiv.org/abs/2302.01316) | Query-based MIA on DDPMs and LDMs |
| May 2023 | [Akbar et al., Beware of diffusion models](https://arxiv.org/abs/2305.07644) | Medical diffusion memorizes more than StyleGAN; 2D slices are a risk |
| May 2023 | [Somepalli et al., Understanding and Mitigating Copying](https://arxiv.org/abs/2305.20086) | Duplication and specific captions drive copying |
| Jun 2023 | [Privacy Distillation](https://arxiv.org/abs/2306.01322) | Filter out copies, then retrain on the filtered data |
| Jul 2023 | [Dar et al., 3D LDM memorization](https://arxiv.org/abs/2307.01148) | Contrastive copy detection in 3D medical LDMs |
| Feb 2024 | [Dar et al., unconditional LDMs memorize](https://arxiv.org/abs/2402.01054) | CT, MR and X-ray; effects of augmentation, data size and over-training |
| 2024 | [Training-epoch effect (BVM 2024)](https://link.springer.com/chapter/10.1007/978-3-658-44037-4_27) | Over-training and memorization (you had no access) |

The MNI-spaces note is reference material, not a dated source, so it is left out of the table.

## Notes check

Most of your summaries are accurate. The table lists only the notes that need a fix or an addition; all other entries are correct as written.

| Source | Verdict | Fix or addition |
| --- | --- | --- |
| [LiRA, Carlini et al. 2021](https://arxiv.org/abs/2112.03570) | Fix | The paper does not say attacks are only viable at low FPR. It says attacks should be *evaluated* by TPR at low FPR (e.g. 0.1%), because averages such as accuracy or AUC hide whether any single example is confidently exposed. LiRA also does not assign a point to whichever group is "closest": it fits a Gaussian to the in-model and out-model scores for that example and thresholds the likelihood ratio. |
| [SecMI, Duan et al. 2023](https://arxiv.org/abs/2302.01316) | Fix | SecMI's critique is aimed at earlier GAN and VAE attacks, which it calls inapplicable to diffusion models; it does not criticise small member sets. The critique of unrealistic evaluation setups comes later, from [Dubiński et al. 2023](https://arxiv.org/abs/2306.12983). SecMI also reads the model's intermediate denoising outputs, so it is gray-box rather than pure black-box. |
| [Dar et al. 2023, 3D LDM](https://arxiv.org/abs/2307.01148) | Check | The abstract mentions no augmentation experiment. The augmentation result, with numbers, is in the 2024 follow-up, so check you haven't merged the two papers. The data are 3D coronary CTA and knee MRI. |
| [Dar et al. 2024, unconditional LDMs](https://arxiv.org/abs/2402.01054) | Correct; add | Your note on dataset size is right: on fastMRI, memorized samples rose from 143 to 368 as training data grew from 1k to 5k images, while the percentage fell. Add three points. It was published in Nature Biomedical Engineering (August 2025). It includes a 2D chest X-ray LDM, which is relevant to your 2D setting. A "copy" is defined as a nearest-neighbour embedding correlation above the 95th percentile of train-to-validation correlations, a threshold you could reuse. |
| [Carlini et al. 2023](https://arxiv.org/abs/2301.13188) | Correct; add | The headline result is the extraction of over 1,000 training images from Stable Diffusion and Imagen with a generate-and-filter attack. The GAN comparison was run on CIFAR-10. |
| [Somepalli et al. 2023](https://arxiv.org/abs/2305.20086) | Correct; add | The finding that specific captions act as retrieval keys is your strongest argument that a mask condition could do the same. |
| [Privacy Distillation](https://arxiv.org/abs/2306.01322) | Correct; add | The filtering step uses a re-identification network, which links it to your Packhäuser et al. (2022) entry. |
| Theory section (trilemma) | Add a source; soften one claim | Cite the trilemma to [Xiao et al. 2021](https://arxiv.org/abs/2112.07804), which defines it as quality, mode coverage and fast sampling. Don't describe LDMs as solving all three: they make each step cheaper, but sampling still takes many steps. |
| No access: Kazerouni survey | Free copy | The same survey is free on [arXiv](https://arxiv.org/abs/2211.07804). |
| No access: training-epoch paper (BVM 2024) | Covered elsewhere | Its over-training finding is reproduced in Dar et al. 2024 (over-training increases memorization), so you can cite that paper instead. |

## Suggested additional reading

If you read only three, make them Gu et al., Wen et al. and Edirisooriya et al. They cover conditioning, explanation, and the closest existing cascade study.

| Theme | Paper | First release | Why it matters for your thesis |
| --- | --- | --- | --- |
| Conditioning drives memorization | [Gu et al., On Memorization in Diffusion Models](https://arxiv.org/abs/2310.02664) (TMLR 2025) | Oct 2023 | Class-conditional models memorize more than unconditional ones, even with random labels; a unique label per sample pushed CIFAR-10 memorization above 65%. Your mask is a near-unique condition per slice. |
| Conditioning drives memorization | [Jain et al., Classifier-free guidance inside the attraction basin may cause memorization](https://arxiv.org/abs/2411.16738) | Nov 2024 | Relevant if your image model samples with classifier-free guidance. It shows guidance can pull samples toward a specific training image. |
| Conditioning drives memorization (medical) | [Dutt, De-identification traces enhance memorization in synthetic chest X-ray](https://arxiv.org/abs/2502.07516) | Feb 2025 | Medical evidence that specific parts of the condition (prompt tokens) drive memorization. It is the text-conditioned counterpart to your mask question. |
| Closest cascade precedent | [Edirisooriya et al., Fidelity, utility and privacy in synthetic cardiac MRI](https://arxiv.org/abs/2603.04340) | Mar 2026 | Uses a two-stage pipeline: masks are generated, then condition image synthesis. Privacy is tested only on the image stage (nearest neighbours, MIA AUC around 0.6), not on the mask stage or the interaction between stages. |
| Cascade pipelines in brain MRI | [Fernandez et al., brainSPADE](https://arxiv.org/abs/2209.08256) | Sep 2022 | An early brain MRI cascade: a diffusion label generator followed by a semantic image generator. Use it as a design precedent; it does not study memorization. |
| xAI: explain memorization | [Wen et al., Detecting, Explaining, and Mitigating Memorization](https://arxiv.org/abs/2407.21720) (ICLR 2024) | 2024 | Detects memorization from the size of the condition-driven noise prediction in a single step, and attributes it to individual prompt tokens. This can be translated to mask regions. |
| xAI: localize memorization | [Hintersdorf et al., Finding NeMo](https://arxiv.org/abs/2406.02366) (NeurIPS 2024) | Jun 2024 | Traces memorization of individual samples to specific neurons, often a single neuron per sample. |
| xAI: local memorization | [Di et al., Foreground–Background Memorization](https://arxiv.org/abs/2508.12148) | Aug 2025 | A segmentation-based metric for memorization confined to image regions. It maps directly onto lesion versus healthy-tissue regions. |
| xAI: data attribution | [Georgiev et al., The Journey, Not the Destination](https://arxiv.org/abs/2312.06205) | Dec 2023 | Attributes a generated image to the training images that shaped it, step by step through sampling. |
| xAI: data attribution | [Zheng et al., Data attribution on diffusion models (D-TRAK)](https://arxiv.org/abs/2311.00500) | Nov 2023 | A practical attribution method and evaluation protocol for diffusion models. |
| Medical memorization metrics | [Scardace et al., DeepSSIM](https://arxiv.org/abs/2509.16582) | Sep 2025 | A memorization metric built for brain MRI LDMs; it learns to predict SSIM under anatomy-preserving augmentation. Use it as a baseline against your contrastive or SSCD detector. |
| Medical memorization mitigation | [Dutt et al., MemControl](https://arxiv.org/abs/2405.19458) | May 2024 | Mitigates memorization in medical diffusion by fine-tuning few parameters. Useful if your thesis proposes a fix. |
| MIA evaluation realism | [Dubiński et al., Towards more realistic MIAs on large diffusion models](https://arxiv.org/abs/2306.12983) | Jun 2023 | Shows earlier MIA evaluations were too generous. Read it before you report any MIA numbers. |
| Joint image and mask generation | [Akbar et al., Brain tumor segmentation using synthetic MR images](https://arxiv.org/abs/2306.02986) (Sci Data 2024) | Jun 2023 | The companion to the "Beware" paper. It generates BRATS images and masks jointly as one sample, which contrasts with your sequential setup. |
| Theory section source | [Xiao et al., Denoising Diffusion GANs](https://arxiv.org/abs/2112.07804) | Dec 2021 | The original source of the generative-learning trilemma. |

## Is the cascade gap real?

Mostly yes. The two neighbouring topics are both studied, but where they meet is not:

- **Medical memorization** has been studied in unconditional models (Dar et al. 2023, 2024; Akbar et al.) and text-conditioned models (Dutt 2025).
- **Conditioning and memorization** has been studied on natural images: captions (Somepalli et al. 2023), class labels (Gu et al.) and classifier-free guidance (Jain et al.).
- **Cascades in medical imaging** exist, from brainSPADE (2022) to MedSegFactory (ICCV 2025), but they are evaluated for realism and downstream segmentation, not privacy.
- **The one exception I found**, Edirisooriya et al. (cardiac MRI, March 2026), tests memorization only in the mask-conditioned image stage. It does not test the mask generator, and it does not test whether memorization carries over from one stage to the next.

Four questions appear to be open:

1. Does the mask generator memorize lesion shapes and positions? A metastasis map is patient-identifying in its own right.
2. Does a near-copy of a training mask act as a key that retrieves the matching training image, as captions do in Somepalli et al. 2023?
3. Does each stage on its own pass a memorization check while the full pipeline leaks?
4. Does memorization sit in the lesion region rather than the whole slice? Whole-image similarity would then miss it (see Di et al. 2025).

One caveat: searches can't cover the 2026 MICCAI and MIDL proceedings exhaustively. Do a last check there before you claim novelty in writing.

## Where you could contribute

The strongest thesis is a stage-wise memorization audit of the mask→image cascade, with xAI used to explain each copy it finds. The three studies below build on each other. The first needs only the synthetic data; the second and third need the trained model weights.

1. **Stage-wise audit.** This works with outputs only.
   - Compare synthetic masks with training masks: nearest-neighbour Dice, plus lesion count and position.
   - Compare synthetic images with training images using a contrastive or SSCD copy detector, thresholded as in Dar et al. 2024.
   - Report how often an image is a copy when its mask is a copy, versus when its mask is novel.
   - Aggregate per patient, not per slice: adjacent slices make slice-level counts misleading (Akbar et al.).
2. **Is the mask a key?** This needs the image model. Give stage 2 three kinds of mask: real training masks, held-out real masks, and synthetic masks. Then measure copy rates for each. Then perturb a memorized mask (shift, erode or remove a lesion) and find the point where the output stops being a copy. That turns Somepalli et al.'s caption-as-key finding into a controlled medical experiment.
3. **Explain the copies.** Each method below explains a different aspect of a copy:
   - **Why the model copied.** Condition-sensitivity maps adapt Wen et al.: the gap between the noise predicted with and without the mask, mapped over the image.
   - **Which regions are copied.** Attribution maps (Integrated Gradients or Grad-CAM) on the copy detector's similarity score show where a synthetic image matches its nearest training image, for example lesion versus background (compare FB-Mem).
   - **Which training slices it came from.** Training-data attribution (D-TRAK, Journey-TRAK) tests whether those slices cluster within one patient.
   - **Where in the network it happens.** Neuron localization (NeMo) has to be adapted, because mask conditioning usually enters by concatenation or ControlNet rather than cross-attention. That adaptation is itself new.

&#91;embedded content: proposed audit of the mask→image cascade · 2 stages, 3 checks, 1 xAI layer\]

Checks 1 and 2 test each stage on its own. Check 3, the new question, tests whether memorization carries from the mask stage into the image stage.

## Further reading

Four papers found while planning the preliminary project. They are not yet placed in the timeline.

| Paper | First release | Why it matters for your thesis |
| --- | --- | --- |
| [Scardace et al., DeepSSIM++: scalable memorization detection](https://arxiv.org/abs/2609.03615) | Sep 2026 | The newest medical memorization metric, from the same group as DeepSSIM. It is the closest competitor to your detectors and the paper to cite as the current state of the art. |
| [Chen et al., Local memorization via Bright Ending attention](https://arxiv.org/abs/2410.21665) (ICLR 2025) | Oct 2024 | Defines local memorization, where only parts of an image are copied. It must be cited for the lesion-level versus whole-mask question (RQ2). It relies on cross-attention, so it does not apply to your models directly. |
| [Kim and Lee, Localizing memorized regions via curvature differences](https://arxiv.org/abs/2605.26756) | May 2026 | Another way to locate memorized regions, comparing against an unconditional or under-trained baseline. It could be adapted with an empty mask as the baseline in the master thesis. |
| [Re-identification from histopathology images](https://arxiv.org/abs/2403.12816) | Mar 2024 | Shows patients can be re-identified from medical images. It supports framing memorization per patient rather than per sample (RQ3). |
