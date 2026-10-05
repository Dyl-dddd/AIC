from scripts.analyze_ensemble_v4_results import fuse_predictions, weighted_summary


def prediction(image_id, category, box, score):
    return {
        "image_id": image_id,
        "category_name": category,
        "bbox": box,
        "score": score,
    }


def test_fusion_scales_model_b_and_suppresses_overlap():
    model_a = [prediction("x.jpg", "jieba", [0, 0, 10, 10], 0.60)]
    model_b = [prediction("x.jpg", "jieba", [0, 0, 10, 10], 0.90)]
    fused = fuse_predictions(model_a, model_b, model_b_scale=0.50, nms_iou=0.55)
    assert len(fused) == 1
    assert fused[0]["score"] == 0.60


def test_fusion_never_merges_different_classes():
    model_a = [prediction("x.jpg", "jieba", [0, 0, 10, 10], 0.60)]
    model_b = [prediction("x.jpg", "jiaza", [0, 0, 10, 10], 0.90)]
    fused = fuse_predictions(model_a, model_b, model_b_scale=1.0, nms_iou=0.55)
    assert {item["category_name"] for item in fused} == {"jieba", "jiaza"}


def test_weighted_summary_uses_frozen_view_weights():
    result = weighted_summary(
        {
            "original": {"score": 70.0},
            "grid_crops": {"score": 60.0},
        }
    )
    assert abs(result["weighted_score"] - 66.19289340101522) < 1e-12
    assert result["robust_min_score"] == 60.0
