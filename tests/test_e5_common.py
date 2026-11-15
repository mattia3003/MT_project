"""Unit tests for the E5 helpers in memtk.e5.common: the audit config, its hash, sensitivity
variant names, and a variant's thresholds and settings applied to the frozen E3 dict."""

import copy

import pytest

from memtk.e5.common import audit_config, audit_hash, parse_variant, variant_frozen

AUDIT = {"sets": {"main": ["clean", "minor"], "noisy": ["noisy"]},
         "primary": {"classes": ["e1_copy", "e2_full"], "strata": ["single", "multi"], "adjust": "holm", "alpha": 0.05},
         "size_bins_px": [12, 20], "window": {"kind": "chained", "max_gap": 2},
         "sensitivity": ["percentile_99", "sigma_0.08", "bound_10", "drop_only"], "h2": "matched_on_hcal",
         "near_exact": {"min_score": 0.9, "beyond": "heldout_max", "size_bins_px": [4, 12]}, "e8_min_strong": 20}
E3 = {"thresholds": {"whole_mask": {"tau_D": {"single": 0.9, "multi": 0.7}},
                     "lesion": {"tau_L": {"single": 0.74, "multi": 0.82}, "tau_C": {"single": 0.72, "multi": 0.41}}},
      "informative_bound_px": 12,
      "detectors": {"whole_mask": {"top_k_dice": 20, "top_k_centroid": 20, "max_shift_px": 3},
                    "lesion": {"position_frame": "pixel", "sigma_px": 6.4, "prefilter_px": 25.6, "candidate_min_s": 0.1}}}


def cfg_with(audit: dict) -> dict:
    return {"audit": audit, "_config_path": "test.yaml"}


def test_parse_variant_names():
    assert parse_variant("percentile_99") == ("percentile", 99.0)
    assert parse_variant("sigma_0.08") == ("sigma", 0.08)
    assert parse_variant("bound_25") == ("bound", 25.0)
    assert parse_variant("drop_only") == ("drop_only", 0.0)
    for bad in ("sigma", "bound_x", "shift_2"):
        with pytest.raises(SystemExit):
            parse_variant(bad)


def test_audit_config_accepts_the_decided_values_and_refuses_null_or_unknown():
    assert audit_config(cfg_with(AUDIT))["window"]["max_gap"] == 2
    unset = copy.deepcopy(AUDIT)
    unset["h2"] = None
    with pytest.raises(SystemExit, match="audit.h2"):
        audit_config(cfg_with(unset))
    unknown = copy.deepcopy(AUDIT)
    unknown["window"]["kind"] = "fixed"
    with pytest.raises(SystemExit, match="chained"):
        audit_config(cfg_with(unknown))


def test_audit_hash_changes_with_any_value_but_not_with_key_order():
    reordered = dict(reversed(list(AUDIT.items())))
    assert audit_hash(cfg_with(AUDIT)) == audit_hash(cfg_with(reordered))
    changed = copy.deepcopy(AUDIT)
    changed["window"]["max_gap"] = 3
    assert audit_hash(cfg_with(AUDIT)) != audit_hash(cfg_with(changed))


def test_variant_frozen_applies_thresholds_and_settings_without_touching_e3():
    variant = {"thresholds": {"whole_mask": {"tau_D": {"single": 0.95, "multi": 0.8}},
                              "lesion": {"tau_L": {"single": 0.8, "multi": 0.86}, "tau_C": {"single": 0.77, "multi": 0.47}}},
               "settings": {"informative_bound_px": 25, "sigma_px": 10.24, "prefilter_px": 25.6, "candidate_min_s": 0.1}}
    vf = variant_frozen(E3, variant)
    assert vf["thresholds"]["lesion"]["tau_C"]["multi"] == 0.47 and vf["informative_bound_px"] == 25
    assert vf["detectors"]["lesion"]["sigma_px"] == 10.24
    assert E3["informative_bound_px"] == 12 and E3["detectors"]["lesion"]["sigma_px"] == 6.4   # a copy, not E3 itself
