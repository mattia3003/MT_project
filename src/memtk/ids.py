"""Identifiers of slices and lesions, shared by every experiment.

- Slice key: real slices `<patient_id>/z<z>` (z = 0-based slice index; ITK-SNAP shows z + 1),
  synthetic samples their sample ID (the full file stem), e.g. `Mets_005/z127`, `file_2_1033`.
- Lesion ID: `<slice key>/L<n>`, n = the lesion's number within its slice (L1 = largest), e.g.
  `Mets_005/z127/L1`, `file_2_1033/L2`.

IDs are unique within one source (a dataset or pool); every table also has a `source` column, so
(source, lesion_id) is unique across all data. IDs are for reading and joining, not for sorting:
`z13` sorts before `z2` as text, so tables carry integer columns (`z`, `label`) for that. Patient
and sample IDs must not contain '/', which separates the parts.
"""

from __future__ import annotations


def _check_part(part: str, what: str) -> str:
    if not part or "/" in part:
        raise ValueError(f"Invalid {what} '{part}': must be non-empty and contain no '/'")
    return part


def slice_key(patient_id: str, z: int) -> str:
    """Key of a real slice: `<patient_id>/z<z>`."""
    if int(z) < 0:
        raise ValueError(f"Slice index must be >= 0, got {z}")
    return f"{_check_part(str(patient_id), 'patient ID')}/z{int(z)}"


def sample_key(sample_id: str) -> str:
    """Key of a synthetic sample: its sample ID, checked."""
    return _check_part(str(sample_id), "sample ID")


def parse_slice_key(key: str) -> tuple[str, int]:
    """(patient_id, z) of a real slice key."""
    patient_id, sep, z = key.rpartition("/z")
    if not sep or not z.isdigit():
        raise ValueError(f"Not a real slice key: '{key}'")
    return _check_part(patient_id, "patient ID"), int(z)


def lesion_id(key: str, label: int) -> str:
    """ID of lesion number `label` (1 = largest) of a slice: `<slice key>/L<label>`."""
    if int(label) < 1:
        raise ValueError(f"Lesion numbers start at 1, got {label}")
    return f"{key}/L{int(label)}"


def parse_lesion_id(lid: str) -> tuple[str, int]:
    """(slice key, label) of a lesion ID."""
    key, sep, label = lid.rpartition("/L")
    if not sep or not key or not label.isdigit():
        raise ValueError(f"Not a lesion ID: '{lid}'")
    return key, int(label)
