"""Tests of the report-figure helpers: centred lesion crops, repository-relative source paths, the
temporary save resolution, and the source-path filter of the check stage."""

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.image import imread

from memtk import config, plotting
from memtk.report_figures.check import is_repo_path
from memtk.report_figures.common import index_row
from memtk.report_figures.method import PAIR_CROP, centred_crop


def test_centred_crop_puts_the_rounded_centroid_in_the_middle():
    lesion = np.zeros((40, 40), dtype=bool)
    lesion[5:8, 30:35] = True                      # centroid (6, 32): near the border, so the crop is padded
    crop = centred_crop(lesion)
    assert crop.shape == (2 * PAIR_CROP + 1, 2 * PAIR_CROP + 1)
    assert crop.sum() == lesion.sum()
    assert np.allclose(np.argwhere(crop).mean(axis=0), (PAIR_CROP, PAIR_CROP))


def test_index_row_makes_paths_repo_relative():
    row = index_row("x", "3.1", "content", [config.REPO_ROOT / "results" / "a.csv", "a description"])
    assert row["sources"] == "results/a.csv; a description"


def test_resolution_applies_inside_the_block_and_is_restored(tmp_path: Path):
    before = plotting.SAVE_DPI
    with plotting.resolution(150):
        plotting.save(Figure(figsize=(1, 1)), tmp_path / "a.png")
        assert plotting.SAVE_DPI == 150
    assert plotting.SAVE_DPI == before
    plotting.save(Figure(figsize=(1, 1)), tmp_path / "b.png")
    assert imread(tmp_path / "a.png").shape[:2] == (150, 150)
    assert imread(tmp_path / "b.png").shape[:2] == (before, before)
    assert not (tmp_path / "a.pdf").exists()
    with plotting.resolution(150, for_print=True):
        plotting.save(Figure(figsize=(1, 1)), tmp_path / "c.png")
    assert (tmp_path / "c.pdf").exists() and not plotting.SAVE_FOR_PRINT


def test_is_repo_path_skips_descriptions_and_placeholders():
    assert is_repo_path("results/amibekk_v1/e5/analyze/rates.csv")
    assert not is_repo_path("processed_data/model_space/<pool>/lesions.csv")
    assert not is_repo_path("T BRAVO")
