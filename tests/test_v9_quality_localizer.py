from __future__ import annotations

from scripts.run_v9_quality_localizer_experiment import MODEL_B_SHA256, decision
from steel_defect.runtime import load_yaml


def _view(score: float, map50: float, recall: float) -> dict:
    return {
        "score_proxy_20_60_20": score,
        "map50_macro": map50,
        "recall_micro": recall,
    }


def test_decision_requires_both_views_and_weighted_gate() -> None:
    baseline = {
        "sha256": MODEL_B_SHA256,
        "weighted_score": 68.0,
        "views": {"original": _view(68.0, 0.50, 0.97), "grid_crops": _view(68.0, 0.50, 0.97)},
    }
    passing = {
        "sha256": "new",
        "weighted_score": 68.40,
        "views": {"original": _view(68.4, 0.515, 0.969), "grid_crops": _view(68.4, 0.515, 0.969)},
    }
    result = decision({"candidates": [baseline, passing]})
    assert result["status"] == "PROMOTE_TO_FUSION_STUDY"
    assert result["test_or_submission_authorized"] is False


def test_v9_arms_are_matched_except_quality_power_and_name() -> None:
    p1 = load_yaml("configs/train/semifinal_v9_model_b_quality_p1_4090.yaml")
    p2 = load_yaml("configs/train/semifinal_v9_model_b_quality_p2_4090.yaml")
    assert p1["quality_power"] == 1.0
    assert p2["quality_power"] == 2.0
    for config in (p1, p2):
        config.pop("quality_power")
        config.pop("name")
    assert p1 == p2
