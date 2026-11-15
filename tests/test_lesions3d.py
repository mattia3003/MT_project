"""Unit tests for memtk.lesions3d on small hand-built volumes (E6 decision W): slice-to-slice linking
never joins two metastases that touch in one slice, while 26-connectivity does; linking needs
overlap between slices; the span counts thin end slices; 2D lesions map to 3D lesions by majority."""

import numpy as np

from memtk.lesions3d import components_26, lesion3d_table, link_slices, map_2d_to_3d


def bridged() -> np.ndarray:
    """Two metastases (A 12 px, B 9 px) separate in slices 0 and 2, touching (one component) in
    slice 1."""
    seg = np.zeros((8, 12, 3), dtype=np.uint8)
    for z in (0, 2):
        seg[2:5, 1:5, z] = 1    # A
        seg[2:5, 7:10, z] = 1   # B
    seg[2:5, 1:10, 1] = 1       # A and B joined
    return seg


def test_linking_keeps_touching_metastases_apart_26_connectivity_joins_them():
    linked = lesion3d_table(link_slices(bridged()), "P")
    assert (linked["max_parts"] == 1).all()                    # no chain holds two pieces in a slice
    assert linked["span"].tolist() == [3, 1, 1]                # A runs through (larger IoU); B ends, B restarts
    assert linked["z_first"].tolist() == [0, 0, 2]
    joined = lesion3d_table(components_26(bridged()), "P")
    assert len(joined) == 1 and joined.loc[0, "max_parts"] == 2 and joined.loc[0, "span"] == 3


def test_linking_needs_overlap_26_connectivity_does_not():
    seg = np.zeros((8, 8, 2), dtype=np.uint8)
    seg[2:4, 2:4, 0] = 1
    seg[4:6, 4:6, 1] = 1       # touches the first only diagonally across slices
    assert link_slices(seg).max() == 2
    assert components_26(seg).max() == 1


def test_numbering_follows_first_appearance():
    seg = np.zeros((8, 8, 3), dtype=np.uint8)
    seg[1:3, 1:3, 1:3] = 1     # appears in slice 1
    seg[5:7, 5:7, 0] = 1       # appears in slice 0
    for labels in (link_slices(seg), components_26(seg)):
        assert labels[5, 5, 0] == 1 and labels[1, 1, 1] == 2


def test_span_counts_thin_end_slices():
    seg = np.zeros((8, 8, 3), dtype=np.uint8)
    seg[3, 3, 0] = 1
    seg[2:5, 2:5, 1] = 1
    seg[4, 4, 2] = 1
    table = lesion3d_table(link_slices(seg), "Mets_001")
    assert table[["lesion3d_id", "span", "volume_px", "largest_area_px"]].values.tolist() == [["Mets_001/D1", 3, 11, 9]]


def test_map_2d_to_3d_by_majority_of_pixels():
    labels3d = np.zeros((6, 6), dtype=np.int32)
    labels3d[0, 0:5] = 2
    labels3d[1, 0:2] = 3
    labels2d = np.zeros((6, 6), dtype=np.int32)
    labels2d[0, 0:5] = 1
    labels2d[1, 0:2] = 1       # lesion 1: 5 px in 3D lesion 2, 2 px in 3D lesion 3
    labels2d[4, 4] = 2         # lesion 2: in no 3D lesion
    assert map_2d_to_3d(labels2d, labels3d) == {1: 2, 2: 0}
