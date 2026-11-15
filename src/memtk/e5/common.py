"""Helpers shared by the E5 stages: the `audit:` config section (decisions L-Q) checked and hashed,
the sensitivity variants of decision O, and the detector settings and thresholds of each variant
(from the frozen E3 file and the E5 file that `prepare` writes). E5 takes every setting and
threshold from the frozen files, never from the live detector config.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from memtk import config
from memtk.audit import CLASSES
from memtk.calibration import STRATA
from memtk.io import LabelStack
from memtk.sets import SliceSet
from memtk.stages import decided, require

REVIEW = "docs/e5-plan.md (decisions L-Q)"
# Variant kinds of decision O and what each one re-runs: re-thresholding needs no detector run;
# sigma and bound change only E2; drop-only changes the label maps, so both detectors run again
RERUN = {"percentile": (), "sigma": ("lesion",), "bound": ("lesion",), "drop_only": ("whole_mask", "lesion")}


def audit_config(cfg: dict) -> dict:
    """The `audit:` section with every value set (decisions L-Q) and only implemented choices;
    stops with a message otherwise."""
    audit = {key: decided(cfg, f"audit.{key}", REVIEW) for key in
             ("sets", "primary", "size_bins_px", "window", "sensitivity", "h2", "near_exact", "e8_min_strong")}
    primary = audit["primary"]
    checks = {
        "audit.sets needs 'main' and 'noisy'": set(audit["sets"]) == {"main", "noisy"},
        f"audit.primary.classes must be among {CLASSES}": set(primary["classes"]) <= set(CLASSES),
        f"audit.primary.strata must be among {STRATA}": set(primary["strata"]) <= set(STRATA),
        "audit.primary.adjust: only 'holm' is implemented": primary["adjust"] == "holm",
        "audit.window.kind: only 'chained' is implemented": audit["window"]["kind"] == "chained",
        "audit.h2: only 'matched_on_hcal' is implemented": audit["h2"] == "matched_on_hcal",
        "audit.near_exact.beyond: only 'heldout_max' is implemented": audit["near_exact"]["beyond"] == "heldout_max",
    }
    failed = [msg for msg, ok in checks.items() if not ok]
    if failed:
        raise SystemExit("; ".join(failed))
    for name in audit["sensitivity"]:
        parse_variant(name)
    return audit


def audit_hash(cfg: dict) -> str:
    """SHA-256 of the `audit:` section (canonical JSON): recorded by `prepare` and `score`, so a
    later stage can refuse to run on an analysis that changed (the lock)."""
    return hashlib.sha256(json.dumps(cfg["audit"], sort_keys=True).encode()).hexdigest()


def parse_variant(name: str) -> tuple[str, float]:
    """(kind, value) of a sensitivity variant name: `percentile_99`, `sigma_0.08` (fraction of
    the field), `bound_10` (px) or `drop_only` (value 0, unused)."""
    if name == "drop_only":
        return "drop_only", 0.0
    kind, _, value = name.partition("_")
    try:
        number = float(value)
    except ValueError:
        number = None
    if kind not in RERUN or number is None:
        raise SystemExit(f"audit.sensitivity: unknown variant '{name}' (percentile_<p>, sigma_<fraction>, "
                         f"bound_<px> or drop_only)")
    return kind, number


def check_prepared(cfg: dict, e5: dict, e3_hash: str) -> None:
    """Stop unless the E5 file was made with the current `audit:` section and frozen E3 file."""
    if e5["audit_sha256"] != audit_hash(cfg):
        raise SystemExit("The audit: section changed since 'prepare' ran; run 'e5_audit.py prepare --overwrite' first "
                         "(only possible before S is scored).")
    if e5["e3_frozen_sha256"] != e3_hash:
        raise SystemExit("The frozen E3 file changed since 'prepare' ran; run 'e5_audit.py prepare --overwrite' first "
                         "(only possible before S is scored).")


def drop_only_t(cfg: dict, dataset: str, t: SliceSet) -> SliceSet:
    """T with its drop-only label maps (written by `prepare`): the reference of the drop-only variant."""
    path = config.output_dir(cfg, dataset) / "e5" / "labels_drop_only_T.npz"
    require(path, "e5_audit.py prepare")
    stack = LabelStack(path)
    return SliceSet("T", t.keys, t.patients, np.stack([stack[k] for k in t.keys]), dict(t.left_out))


def score_marker(cfg: dict, pool: str) -> Path:
    """The file `score` writes when S has been scored completely; once it exists, nothing fixed
    before S (the E5 file, `audit:`) may change."""
    return config.output_dir(cfg, pool) / "e5" / "score.complete.json"


def refuse_after_score(cfg: dict, pool: str, stage: str) -> None:
    """Stop a pre-S stage once S has been scored (even with --overwrite)."""
    if score_marker(cfg, pool).exists():
        raise SystemExit(f"S has already been scored ({score_marker(cfg, pool)}); '{stage}' would change values "
                         f"fixed before S. See docs/e5-plan.md (the lock).")


def variant_frozen(e3: dict, variant: dict) -> dict:
    """The frozen E3 dict with a variant's thresholds and detector settings (`variant`: one entry
    of `variants` in the E5 file), so `calibration.run_frozen`, `e1_classes` and `e2_classes`
    apply it unchanged."""
    out = copy.deepcopy(e3)
    out["thresholds"] = copy.deepcopy(variant["thresholds"])
    s = variant["settings"]
    out["informative_bound_px"] = int(s["informative_bound_px"])
    out["detectors"]["lesion"].update({"sigma_px": float(s["sigma_px"]), "prefilter_px": float(s["prefilter_px"]),
                                       "candidate_min_s": float(s["candidate_min_s"])})
    return out
