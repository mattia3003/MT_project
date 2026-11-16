"""Unit tests for the E8 analysis (memtk.e8.paired, decisions AC, AD) on small hand-built inputs: the
three branches of the reading, AUC against a reference set, measurement of a pair whose query is the
comparison image moved by a shift, and the within-A table."""

import numpy as np
import pandas as pd
import pytest

from memtk.e8.paired import READINGS, Settings, auc_tests, measure_pairs, reading, within_table

S = Settings("low_threshold_all", 0.02, bound=12, grow_px=8, data_range=10.0, sigma=1.5,
             regions=("lesion", "brain", "ring"))


def toy_slice(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A textured brain disc with a 4 x 4 bright lesion: (image, mask as saved, label map)."""
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[:64, :64]
    brain = (rr - 32) ** 2 + (cc - 32) ** 2 <= 24 ** 2
    image = np.where(brain, 0.3 + 0.3 * rng.random((64, 64)), 0.0)
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[28:32, 30:34] = 1
    image[labels > 0] = 0.95
    return image, labels > 0, labels


def scored(a: list[float], b: list[float]) -> pd.DataFrame:
    """A measured table with cases a and controls b (each its own cluster) in every region."""
    n = len(a) + len(b)
    d = pd.DataFrame({"kind": "t", "group": ["A"] * len(a) + ["B"] * len(b), "cluster": [f"c{i}" for i in range(n)]})
    for r in S.regions:
        d[f"ssim_{r}"] = a + b
    return d


def test_reading_has_three_branches():
    assert reading({"auc_low": 0.6, "delta_low": 0.05}, 0.5) == READINGS[0]
    assert reading({"auc_low": 0.6, "delta_low": -0.05}, 0.5) == READINGS[1]
    assert reading({"auc_low": 0.45, "delta_low": 0.2}, 0.5) == READINGS[2]


def test_auc_against_a_reference_set():
    query = scored([0.9, 0.8, 0.85, 0.95], [0.1, 0.2, 0.15, 0.05])
    reference = scored([0.1, 0.3, 0.2, 0.4], [0.2, 0.1, 0.4, 0.3])
    t = auc_tests(query, reference, S, draws=200, rng=np.random.default_rng(0))
    row = t[t["region"] == "lesion"].iloc[0]
    assert row["auc"] == 1.0 and row["auc_low"] == 1.0
    assert row["auc_ref"] == pytest.approx(0.5)
    assert row["delta"] == pytest.approx(0.5) and row["delta_low"] > 0
    assert reading(row, 0.5) == READINGS[0]


def test_a_shifted_copy_of_the_comparison_measures_one():
    image, saved, labels = toy_slice(0)
    other = toy_slice(1)[0]
    dr, dc = 2, -1
    moved = (np.roll(image, (-dr, -dc), axis=(0, 1)), np.roll(saved, (-dr, -dc), axis=(0, 1)),
             np.roll(labels, (-dr, -dc), axis=(0, 1)))
    pairs = pd.DataFrame({"query": ["q", "q"], "comparison": ["t_same", "t_other"], "shift_row": [dr, dr],
                          "shift_col": [dc, dc]})
    real = {"t_same": (image, saved), "t_other": (other, saved)}
    m = measure_pairs(pairs, lambda _: moved, real, S)
    same, other_row = m.iloc[0], m.iloc[1]
    assert float(same["ssim_lesion"]) == pytest.approx(1.0) and float(same["ssim_brain"]) == pytest.approx(1.0)
    assert float(other_row["ssim_lesion"]) < 0.9
    assert int(same["px_ring"]) == int(same["px_lesion"]) - 16      # the ring leaves out the 16 lesion pixels


def test_within_a_compares_t_with_the_runner_up_and_profiles_z():
    rows = []
    for i in range(6):
        rows.append({"group": "A", "kind": "t", "dz": 0, "case": f"a{i}", "cluster": f"p{i}", "v": 0.6 + 0.01 * i})
        rows.append({"group": "A", "kind": "runner_up", "dz": 0, "case": f"a{i}", "cluster": f"p{i}", "v": 0.2})
        for dz in (-1, 1):
            rows.append({"group": "A", "kind": "dz", "dz": dz, "case": f"a{i}", "cluster": f"p{i}", "v": 0.4})
    m = pd.DataFrame(rows)
    for r in S.regions:
        m[f"ssim_{r}"] = m["v"]
    w = within_table(m, S, draws=200, rng=np.random.default_rng(0))
    ru = w[(w["measure"] == "t minus runner-up") & (w["region"] == "lesion")].iloc[0]
    assert ru["mean"] == pytest.approx(0.425) and ru["low"] > 0 and ru["wilcoxon_p"] < 0.05
    prof = w[(w["measure"] == "mean SSIM by z offset") & (w["region"] == "lesion")].set_index("dz")["mean"]
    assert prof.to_dict() == pytest.approx({-1: 0.4, 0: 0.625, 1: 0.4})
    assert within_table(m[m["kind"] == "t"], S, draws=10, rng=np.random.default_rng(0)).empty
