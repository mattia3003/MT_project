"""Quality classes of synthetic samples (E0b `qc`).

Two views of a sample, kept apart because the mask comes from stage 1 and the MRI from stage 2:
- `mask_class`, from the mask alone (after attach-or-drop): `no_lesion` (no lesion at all),
  `noisy` (more isolated specks than the cut-off), `minor` (1..cut-off), `clean` (none).
  The mask experiments (E1-E5) use this; a failed MRI does not make its mask invalid.
- `mri_failed`: the paired MRI has no brain region. Analyses that need the brain use this.
`qc_class` combines both with the plan's precedence (first match wins: mri_failed, no_lesion,
noisy, minor, clean), for reporting one class per sample.

Flags are columns, not classes: `edge` (a lesion more than the edge fraction outside the brain)
and `has_lesion_lt40` (the sampling filter's "a lesion under 40 px").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MASK_CLASSES = ("no_lesion", "noisy", "minor", "clean")
QC_CLASSES = ("mri_failed", *MASK_CLASSES)


def mask_class(n_lesions: pd.Series, n_specks: pd.Series, max_specks: int) -> pd.Series:
    """Class of each mask from its lesion and isolated-speck counts."""
    labels = np.select([n_lesions == 0, n_specks > max_specks, n_specks > 0], list(MASK_CLASSES[:3]),
                       default="clean")
    return pd.Series(labels, index=n_lesions.index, name="mask_class")


def qc_class(mri_failed: pd.Series, mask_classes: pd.Series) -> pd.Series:
    """One class per sample with the plan's precedence: a failed MRI first, else the mask class."""
    return mask_classes.where(~mri_failed.astype(bool), "mri_failed").rename("qc_class")


def edge_lesions(lesions: pd.DataFrame, outside_frac: float) -> pd.Series:
    """Per slice key: number of lesions with more than `outside_frac` of their area outside the
    brain (keys without lesions are absent; treat them as 0)."""
    edge = lesions[lesions["frac_outside_brain"] > outside_frac]
    return edge.groupby("key").size().rename("n_edge_lesions")
