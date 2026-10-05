import json

import pytest

from scripts.run_safe_r2_final_lockbox_pipeline import (
    apply_profile_to_view,
    evaluate_gate,
    validate_frozen_profile,
)
from steel_defect.safe_r2 import load_profile


def _metric(score, map50=0.5, recall=0.9, predictions=10):
    return {
        "score": score,
        "map50": map50,
        "r_micro": recall,
        "predictions": predictions,
    }


def test_profile_fails_closed_without_oof_authorization(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "metadata": {
            "method": "SAFE-R2",
            "selected_from_group_oof": "A3_full_rerank",
            "deployment_allowed": False,
            "gate": "STOP",
            "training_view": "original",
            "test_labels_used": False,
        }
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="not authorized"):
        validate_frozen_profile(path)


def test_profile_accepts_explicit_passed_group_oof(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "metadata": {
            "method": "SAFE-R2",
            "selected_from_group_oof": "A3_full_rerank",
            "deployment_allowed": True,
            "gate": "PASS",
            "training_view": "original",
            "test_labels_used": False,
        }
    }), encoding="utf-8")
    assert validate_frozen_profile(path)["metadata"]["gate"] == "PASS"


def test_lockbox_gate_is_fixed_and_strict():
    baseline = {"original": _metric(60.0), "grid_crops": _metric(60.0)}
    passing = {"original": _metric(60.2, map50=0.501), "grid_crops": _metric(60.2, map50=0.501)}
    assert evaluate_gate(baseline, passing)["gate"] == "PASS"
    failing = {"original": _metric(60.3, map50=0.501), "grid_crops": _metric(59.99, map50=0.501)}
    decision = evaluate_gate(baseline, failing)
    assert decision["gate"] == "STOP"
    assert "grid_crops:score_decreased" in decision["failures"]


def test_apply_profile_preserves_membership_with_synthetic_cells():
    row = {
        "image_id": "x.jpg",
        "source_id": "x.jpg",
        "image_size": [100, 100],
        "ground_truth": {},
        "candidates": [
            {"bbox": [10, 10, 30, 30], "score": 0.5, "class_id": 0,
             "is_global": True, "touches_internal_edge": False},
        ],
    }
    cell = {
        "rows": [row],
        "ground_truth": {"x.jpg": {}},
        "sizes": {"x.jpg": (100, 100)},
        "signature": {},
    }
    baseline = [{
        "image_id": "x.jpg", "category_name": "jieba",
        "bbox": [10, 10, 30, 30], "score": 0.5,
    }]
    output, diagnostic = apply_profile_to_view(
        cell, cell, baseline, load_profile(None)
    )
    assert len(output) == len(baseline) == diagnostic["predictions"]
    assert output[0]["image_id"] == "x.jpg"
    assert output[0]["category_name"] == "jieba"
