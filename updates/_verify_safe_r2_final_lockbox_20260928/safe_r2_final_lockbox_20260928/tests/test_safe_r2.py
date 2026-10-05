import json
from pathlib import Path

import pytest

from steel_defect.classes import CLASS_NAMES
from steel_defect.safe_r2 import (
    apply_safe_r2,
    select_provenance_representatives,
)


CLASS_ID = CLASS_NAMES.index("jieba")


def raw(box, score, *, global_=False, edge=False, class_id=CLASS_ID):
    return {
        "bbox": box,
        "score": score,
        "class_id": class_id,
        "touches_internal_edge": edge,
        "is_global": global_,
    }


def prediction(box=(10, 10, 30, 30), score=0.2):
    return {
        "image_id": "sample.jpg",
        "category_name": "jieba",
        "bbox": list(box),
        "score": score,
    }


def refinement_profile(**updates):
    refinement = {
        "enabled": True,
        "max_dispersion": 0.55,
        "minimum_anchor_iou": 0.50,
        "minimum_donors": 2,
        "require_non_edge_local": True,
        "score_power": 1.0,
        "anchor_iou_power": 0.0,
    }
    refinement.update(updates)
    return {"support_iou": 0.30, "score_blend": 0.0, "refinement": refinement}


def test_provenance_is_capped_at_one_representative_per_bucket():
    anchor = prediction()
    candidates_a = [
        raw([10, 10, 30, 30], 0.4),
        raw([11, 11, 31, 31], 0.99),  # same bucket cannot add support
        raw([10, 10, 30, 30], 0.3, global_=True),
    ]
    candidates_b = [
        raw([10, 10, 30, 30], 0.2),
        raw([10, 10, 30, 30], 0.1, global_=True),
        raw([10, 10, 30, 30], 1.0, class_id=CLASS_ID + 1),
    ]
    reps = select_provenance_representatives(anchor, candidates_a, candidates_b, 0.5)
    assert set(reps) == {"a_local", "a_global", "b_local", "b_global"}
    assert len(reps) == 4
    assert reps["a_local"]["bbox"] == [10, 10, 30, 30]


def test_no_support_is_lossless_fallback():
    original = prediction()
    result = apply_safe_r2(
        [original],
        [raw([60, 60, 80, 80], 0.9)],
        [],
        (100, 100),
        refinement_profile(),
        include_diagnostics=True,
    )
    assert len(result) == 1
    assert result[0]["bbox"] == original["bbox"]
    assert result[0]["score"] == original["score"]
    assert result[0]["safe_r2"]["refinement_reason"] == "missing_independent_model_support"


def test_json_profile_reranks_without_deleting_predictions(tmp_path: Path):
    items = [prediction(score=0.8), prediction(box=(50, 50, 70, 70), score=0.2)]
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "support_iou": 0.5,
        "score_blend": 1.0,
        "linear": {
            "intercept": -2.0,
            "coefficients": {"ab_presence": 6.0},
        },
    }), encoding="utf-8")
    candidates_a = [raw([50, 50, 70, 70], 0.7)]
    candidates_b = [raw([50, 50, 70, 70], 0.6)]
    result = apply_safe_r2(items, candidates_a, candidates_b, (100, 100), profile)
    assert len(result) == len(items)
    assert result[0]["bbox"] == [50, 50, 70, 70]
    assert {tuple(item["bbox"]) for item in result} == {
        (10, 10, 30, 30), (50, 50, 70, 70)
    }


def test_ug_rbv_refines_with_independent_low_dispersion_support():
    anchor = prediction(box=(10, 10, 30, 30))
    candidates_a = [
        raw([12, 10, 32, 30], 0.4, edge=False),
        raw([12, 10, 32, 30], 0.4, global_=True),
    ]
    candidates_b = [
        raw([14, 10, 34, 30], 0.9, edge=False),
        raw([14, 10, 34, 30], 0.9, global_=True),
    ]
    result = apply_safe_r2(
        [anchor], candidates_a, candidates_b, (100, 100),
        refinement_profile(), include_diagnostics=True,
    )
    assert result[0]["bbox"] == pytest.approx([14, 10, 34, 30])
    assert result[0]["safe_r2"]["refined"] is True


def test_ug_rbv_requires_non_edge_local_donor():
    anchor = prediction()
    candidates_a = [raw([12, 10, 32, 30], 0.8, edge=True)]
    candidates_b = [raw([13, 10, 33, 30], 0.9, global_=True)]
    result = apply_safe_r2(
        [anchor], candidates_a, candidates_b, (100, 100),
        refinement_profile(), include_diagnostics=True,
    )
    assert result[0]["bbox"] == anchor["bbox"]
    assert result[0]["safe_r2"]["refinement_reason"] == "no_non_edge_local_donor"


def test_ug_rbv_trust_region_falls_back_to_anchor():
    anchor = prediction(box=(10, 10, 30, 30))
    # IoU 0.43 clears the support gate but the voted box fails a strict trust region.
    candidates_a = [raw([18, 10, 38, 30], 0.9, edge=False)]
    candidates_b = [raw([18, 10, 38, 30], 0.8, global_=True)]
    result = apply_safe_r2(
        [anchor], candidates_a, candidates_b, (100, 100),
        refinement_profile(minimum_anchor_iou=0.8), include_diagnostics=True,
    )
    assert result[0]["bbox"] == anchor["bbox"]
    assert result[0]["safe_r2"]["refinement_reason"] == "outside_anchor_trust_region"


def test_output_is_deterministic_and_default_preserves_zero_score():
    items = [prediction(score=0.0), prediction(box=(40, 40, 60, 60), score=0.2)]
    first = apply_safe_r2(items, [], [], (100, 100))
    second = apply_safe_r2(items, [], [], (100, 100))
    assert first == second
    by_box = {tuple(item["bbox"]): item for item in first}
    assert by_box[(10, 10, 30, 30)]["score"] == 0.0
