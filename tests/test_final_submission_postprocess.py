import numpy as np

from scripts.analyze_final_submission_postprocess import _vote_arrays, postprocess
from scripts.analyze_source_aware_fusion import source_aware_fuse


def prediction(box, score):
    return {
        "image_id": "sample.jpg",
        "category_name": "jieba",
        "bbox": box,
        "score": score,
    }


def test_extra_nms_removes_lower_scored_overlap():
    rows = [prediction([0, 0, 20, 20], 0.9), prediction([1, 1, 21, 21], 0.8)]
    output = postprocess(rows, "nms", 0.6)
    assert len(output) == 1
    assert output[0]["bbox"] == [0, 0, 20, 20]
    assert output[0]["score"] == 0.9


def test_box_vote_changes_only_coordinates_and_keeps_anchor_score():
    boxes = np.asarray([[0, 0, 20, 20], [2, 2, 22, 22]], dtype=np.float32)
    scores = np.asarray([0.9, 0.6], dtype=np.float32)
    classes = np.asarray([0, 0], dtype=np.int64)
    voted_boxes, voted_scores, voted_classes = _vote_arrays(boxes, scores, classes, 0.6)
    assert voted_boxes.shape == (1, 4)
    assert np.isclose(voted_scores[0], 0.9)
    assert voted_classes.tolist() == [0]
    assert np.allclose(voted_boxes[0], [0.8, 0.8, 20.8, 20.8])


def test_source_aware_fusion_boosts_cross_model_consensus():
    model_a = [prediction([0, 0, 20, 20], 0.4)]
    model_b = [prediction([1, 1, 21, 21], 0.8)]
    output = source_aware_fuse(
        model_a, model_b, match_iou=0.5, boost=1.5,
        coord_mode="anchor", unmatched_b_factor=1.0,
    )
    assert len(output) == 1
    assert output[0]["bbox"] == [0, 0, 20, 20]
    assert np.isclose(output[0]["score"], 0.6)
