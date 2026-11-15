"""Unit tests for the E5 additions to memtk.sets: a synthetic pool by mask class, and label maps made
again with other attach-or-drop settings (drop-only)."""

import numpy as np
import pandas as pd

from memtk.io import LabelStackWriter
from memtk.sets import load_pool, relabel


def test_load_pool_keeps_the_requested_mask_classes(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    qc = pd.DataFrame({"key": ["1033", "file_1033", "x", "y"],
                       "mask_class": ["clean", "minor", "noisy", "no_lesion"], "n_lesions": [1, 2, 1, 0]})
    qc.to_csv(pool / "qc.csv", index=False)
    writer = LabelStackWriter((8, 8))
    for i, key in enumerate(qc["key"]):
        m = np.zeros((8, 8), dtype=np.int32)
        m[i, :4] = 1 if key != "y" else 0
        writer.add(key, m)
    writer.save(pool / "labels.npz")

    s = load_pool({"output_root": str(tmp_path)}, "pool", ["clean", "minor"])
    assert s.keys == ["1033", "file_1033"]                    # sample IDs stay text
    assert s.patients.tolist() == s.keys                      # each sample its own cluster
    assert s.labels.dtype == np.uint8 and s.labels.shape == (2, 8, 8)
    assert s.labels[1, 1, :4].tolist() == [1, 1, 1, 1]
    assert s.left_out == {"mask_class no_lesion": 1, "mask_class noisy": 1,
                          "no lesion of at least the minimum area": 0}


def test_relabel_drop_only_drops_a_stray_that_gap_2_attaches():
    mask = np.zeros((1, 12, 12), dtype=bool)
    mask[0, 2:5, 2:5] = True          # a 9 px lesion
    mask[0, 2, 7] = True              # a 1 px speck, 2 background px away
    attached = relabel(mask, min_area=4, gap=2)
    dropped = relabel(mask, min_area=4, gap=0)
    assert attached[0, 2, 7] == 1 and (attached[0] > 0).sum() == 10
    assert dropped[0, 2, 7] == 0 and (dropped[0] > 0).sum() == 9
    assert dropped.dtype == np.uint8
