# Process log: preliminary project

A dated log of how the work was done: stages built, re-runs and verifications, refactors, bugs
fixed, numbers that were replaced and why, and things tried and removed. It is the history
behind `docs/report-notes.md`, which keeps only what the report may use (decisions with their
evidence, findings, limitations, open questions). Per-experiment plans and stage status live in
`docs/e0b-plan.md` and its successors.

Entry format: date heading, newest at the bottom; each entry says **what was done or changed**,
why, and its *source* (commit, file in git, or pipeline stage). Replaced numbers are kept here
in full, with what replaced them.

---

## 2026-09-22 to 2026-10-29: earlier segmentation work

- **Segmentation experiments before this project** (ResNet50 metrics, synthetic-data statistics
  with and without corrupt files). Removed from the working tree on 2026-11-01 and kept under the
  git tag `segmentation-final`. *Source: commits 175b919 to 2270376, a80cbd6.*

## 2026-11-01: project setup and E0a

- **Library and config set up** (`src/memtk/`, `configs/default.yaml`, one seed 2026, run logs per
  stage). *Source: commits 694137c, 3f735e7.*
- **E0a built and checked:** `split`, `real`, `synthetic`, `check`; 32/32 checks pass. A code
  review fixed naming issues for new datasets. *Source: commits 3f735e7 to 3001571;
  `results/stanford_brainmets/e0a/check_report_amibekk_v1.txt`.*
- **Earlier per-slice native-resolution files were not reused.** The model space must be
  converted from the 3D volumes (per-volume min–max), so the old per-slice files could not
  serve. See "all comparisons happen in model space" in `report-notes.md` section 2.
- **Docs added:** `report-notes.md`, `architecture.md`, `e0b-plan.md`; null model dropped from the
  plans. *Source: commits 20ef4fc, d478e76, ac30d45, 04341cf.*
- **Preliminary observations, later measured on the full data** (moved from `report-notes.md`
  on 2026-11-03; the final numbers are in its sections 2 and 3):
  - In a sample of 1,600 synthetic masks, about 11% had no component under 40 px, so the
    sampling script's "< 40 px" filter did not hold for the whole pool. *Source: inspection
    sample.* Final: 10.7% of the masks as saved (decision E, `filters-review`, 2026-11-03).
  - Synthetic brains often looked smaller or off-centre than real lesion slices; some synthetic
    masks were dominated by scattered specks (e.g. sample `2750`). *Source: check grid.*
    Final: brain size by slice height (`stats`), speck counts (`lesions-review`, `qc-review`).
  - Synthetic lesions looked spatially concentrated: centroid spread (lesions ≥ 4 px) row SD
    14.9 vs 24.1 px, column SD 18.6 vs 25.3 px, around the same centre; brain boxes median 63×66
    vs 103×95 px; Otsu brain area median ~3,000 vs ~7,200 px; 3.8% of synthetic lesions more
    than half outside the Otsu brain (T 1.4%), 1.6% with the proposed `low_threshold_all` mask
    (T 0.02%). *Source: read-only sample, all T lesion slices vs 3,000 synthetic samples (old-QC
    and near-blank excluded); brain-review `review.txt`, 2,000 random samples.* Final: full-pool
    centroid spread 15.0/18.8 px (`stats`), 2.0% of lesions outside (`lesions`, 2026-11-02).

## 2026-11-02: E0b stages `brain` to `qc`, decisions A to D

- **Stages built:** `brain-review` → decision A, `brain`, `lesions-review` → decisions B, C,
  `lesions` + `view`, `qc-review` → decision D, `qc`. Pyright set up in standard mode.
  *Source: commits 5a49d50 to 0c30421; `docs/e0b-plan.md` status table.*
- **A stricter failed-MRI rule was tried and removed.** Implementation notes from the test (the
  rule and its rejection are in `report-notes.md` section 2, "a failed synthetic MRI"):
  - measured from the raw mask, the specks of noisy masks, grown by the 5 px margin, covered
    normal brains; ignoring components < 4 px fixed that;
  - the code (a temporary `brain-review` extension) was removed after the decision.
- **Open question resolved at the `qc` stage: does a failed MRI exclude its mask?** Answer: no
  (separate `mask_class` and `mri_failed` columns; see "QC classes" in `report-notes.md`
  section 2). The question as first recorded: the mask comes from stage 1 whatever stage 2
  does, so a blank or failed MRI does not make its mask invalid for E1–E5, which use masks
  only. A failed MRI matters where the brain is used: the edge check, brain-relative positions,
  and E8. Option: `mri_failed` as a flag that excludes a sample from those analyses only,
  rather than a QC class that removes the mask from S. The experiment plan did not say. The
  failed images come in at least two kinds: a lesion with a thin rim of tissue painted around
  it, and a small faint patch away from the lesion, each on an otherwise black image. *Source:
  `results/amibekk_v1/e0b/brain/outlines.png` (row 3) and a read-only check of its 8 samples.*

## 2026-11-03: E0b `stats`, `filters-review`, `check`, restructuring, full re-run

- **Stages built:** `stats` (items 1–6), `filters-review` → decision E, `check`. *Source:
  commits 444fcc6 to f785804.*
- **Brain-threshold noise levels re-measured.** "More than 3 px from the brain" was first
  measured in the four straight directions only; it now grows diagonally too (Chebyshev), like
  every other distance in the project. Old near-worst background levels 0.006 (T) / 0.021 (S),
  new 0.0004 / 0.013. The area changes and the conclusion did not change. *Source:
  `results/stanford_brainmets/e0b/brain_review/review.txt`, `threshold_sensitivity_amibekk_v1.csv`.*
- **Code restructured (E0b step 8b):** the stages moved out of `audit.py` (split up and removed)
  into the stage package `src/memtk/e0b/` (one module per topic); `scripts/e0b_audit.py` is only
  the command line; the code conventions were written into `CLAUDE.md`. *Source: commit 6850543.*
- **Full re-run of E0a and E0b from scratch (E0b step 9), verifying the restructuring.**
  - All 13 stages run in order with `--overwrite` (about 20 min); every stage exits 0. E0a check
    32/32, E0b check 43/43, pytest 60 passed.
  - `processed_data/model_space/`: all 15 tables and caches and all 624 volume files (156
    patients × `bravo`, `seg_soft`, `seg`, `labels`) are byte-identical to the run before the
    restructuring (compared against a copy and file hashes taken just before the re-run).
  - `results/` (`git diff`): no table changed. Run logs changed (time, full config; `script`
    now names the stage module, e.g. `src\memtk\e0b\stats.py`, the command is in `argv`). The
    E0b check report changed only in timestamps. Four figures changed with the same examples
    in the same order: the E0a check grid (layout from `plotting.save`, last drawn 2026-11-01),
    the QC class sheet and both lesion overview sheets (tile subtitles in the singular,
    outlines on the lesion numbers). All other figures were rewritten byte-identical, as they
    had already been drawn after the title fix of 2026-11-02. All figures checked by eye: no
    issues.
  - Not re-run: the `view` helper pictures in `results/*/e0b/views/` (on demand, not a stage).
- **Docs split:** process entries moved from `report-notes.md` into this log.
- **Two report-notes tables recomputed with the final QC classes** (they were made on
  2026-11-02 with a provisional cut at 3 isolated specks, before decision D set 9). A read-only
  script first reproduced the old numbers with the provisional cut (confirming the
  definitions), then recomputed with `qc.csv`. Old numbers, by provisional group:
  - *Lesions more than half outside the brain* (masks / lesions / mostly outside / share /
    masks affected): clean 32,137 / 51,816 / 838 / 1.6% / 829; minor (1–3 specks) 3,423 / 5,410
    / 71 / 1.3% / 70; noisy (> 3) 1,912 / 4,325 / 220 / 5.1% / 172; failed MRI 152 / 84 / 84 /
    100% / 82. With the final classes, clean and failed MRI are unchanged; minor 1.4%, noisy 6.6%.
  - *Stray growth* (masks / lesions with a stray / median growth / grown > 25% / stray px over
    lesion px): clean 32,219 / 5.9% / +7.7% / 0.4% / 0.4%; minor 3,038 / 36% / +9.1% / 4.8% /
    3.2%; noisy 1,904 / 78% / +16% / 23% / 9.3%. The old mask counts left out masks without a
    lesion (463); the final `mask_class` does the same. With the final classes, clean is
    unchanged; minor 43% / +10% / 6.9% / 4.1%, noisy 80% / +18% / 27% / 10.5%.

  Conclusions unchanged. *Source: `lesions.csv`, `qc.csv` of `amibekk_v1`; script kept outside
  the repo (read-only).*
- **Duplicate entry merged:** the 2026-11-02 report-notes entry "Synthetic lesions are smaller
  than real ones" (from the `lesions` summaries, all of S) was merged into the `stats` entry
  "Synthetic masks: smaller lesions, more two-lesion masks, no large lesions" (S main). It added
  the 5th percentile (4 vs 6 px) and the share with a lesion under 40 px, now given for S main
  (86.5%) in the table and for all of S (85.4%) in the text.
