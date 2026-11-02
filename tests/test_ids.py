"""Slice keys and lesion IDs: building, parsing and refusing what would break them."""

import pytest

from memtk.ids import lesion_id, parse_lesion_id, parse_slice_key, sample_key, slice_key


def test_real_slice_key_and_lesion_id_round_trip():
    key = slice_key("Mets_005", 127)
    assert key == "Mets_005/z127"
    assert parse_slice_key(key) == ("Mets_005", 127)
    lid = lesion_id(key, 1)
    assert lid == "Mets_005/z127/L1"
    assert parse_lesion_id(lid) == ("Mets_005/z127", 1)


def test_synthetic_ids_keep_the_full_sample_id():
    for sid in ("1033", "file_1033", "file_2_1033"):
        lid = lesion_id(sample_key(sid), 12)
        assert lid == f"{sid}/L12"
        assert parse_lesion_id(lid) == (sid, 12)


def test_ids_with_letters_resembling_the_separators():
    # 'L' and 'z' inside a patient or sample ID must not confuse parsing
    assert parse_lesion_id(lesion_id(slice_key("Lz_L3", 0), 10)) == ("Lz_L3/z0", 10)
    assert parse_lesion_id(lesion_id(sample_key("zL7"), 2)) == ("zL7", 2)


@pytest.mark.parametrize("bad", ["", "a/b"])
def test_slashes_and_empty_ids_are_refused(bad):
    with pytest.raises(ValueError):
        sample_key(bad)
    with pytest.raises(ValueError):
        slice_key(bad, 3)


def test_invalid_numbers_and_strings_are_refused():
    with pytest.raises(ValueError):
        slice_key("Mets_005", -1)
    with pytest.raises(ValueError):
        lesion_id("Mets_005/z1", 0)
    for bad in ("Mets_005/z127", "Mets_005/z127/Lx", "/L3"):
        with pytest.raises(ValueError):
            parse_lesion_id(bad)
    with pytest.raises(ValueError):
        parse_slice_key("file_2_1033")
