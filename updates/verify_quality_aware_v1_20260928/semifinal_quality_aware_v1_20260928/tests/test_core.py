from pathlib import Path
import sys
import gzip
import json

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.deblur import adaptive_deblur
from steel_defect.focal_trainer import ElementwiseFocalBCE
from steel_defect.geometry import (
    Tile,
    classwise_nms,
    classwise_weighted_box_fusion,
    clip_box_to_tile,
    generate_grid_tiles,
    generate_tiles,
    xyxy_to_yolo,
)
from steel_defect.image_io import imread, imwrite
from steel_defect.inference import InferenceOptions, merge_candidates
from steel_defect.metrics import evaluate_predictions, tune_class_thresholds
from steel_defect.recall_booster import (
    build_action_arrays,
    iter_candidate_records,
    matched_ground_truth,
    scale_boxes,
)
from scripts.infer import select_split_images
from steel_defect.voc import Annotation, VocRecord
from scripts.prepare_data import split_records, split_train_calibration_val


def test_tiles_cover_edges():
    tiles = generate_tiles(4096, 3000, 1024, 0.25)
    assert tiles[0].x == 0 and tiles[0].y == 0
    assert max(t.x + t.width for t in tiles) == 4096
    assert max(t.y + t.height for t in tiles) == 3000


def test_semifinal_grid_matches_observed_crop_geometry():
    tiles = generate_grid_tiles(4096, 3000, 1387, 1516, rows=2, cols=3)
    assert [(tile.x, tile.y) for tile in tiles] == [
        (0, 0), (1354, 0), (2709, 0),
        (0, 1484), (1354, 1484), (2709, 1484),
    ]
    assert all((tile.width, tile.height) == (1387, 1516) for tile in tiles)


def test_clip_and_yolo_conversion():
    tile = Tile(100, 200, 100, 100, 300, 400)
    clipped = clip_box_to_tile((90, 190, 160, 260), tile, min_visibility=0.2)
    assert clipped == (0.0, 0.0, 60.0, 60.0)
    cx, cy, w, h = xyxy_to_yolo(clipped, 100, 100)
    assert np.allclose([cx, cy, w, h], [0.3, 0.3, 0.6, 0.6])


def test_long_box_fragments_are_not_converted_to_background():
    tile = Tile(0, 0, 1024, 1024, 4096, 3000)
    # Only about one third of this full-height defect is visible, but it spans
    # the tile vertically and must remain a positive training fragment.
    clipped = clip_box_to_tile((100, 0, 140, 3000), tile, min_visibility=0.35)
    assert clipped == (100.0, 0.0, 140.0, 1024.0)


def test_classwise_nms_does_not_mix_classes():
    boxes = np.asarray([[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.asarray([0, 0, 1], dtype=np.int64)
    keep = classwise_nms(boxes, scores, classes, 0.5)
    assert keep == [0, 2]


def test_deblur_gate_preserves_sharp_image_shape():
    image = np.zeros((64, 64), dtype=np.uint8)
    image[:, 32:] = 255
    result = adaptive_deblur(image, threshold=1.0)
    assert not result.applied
    assert result.image.shape == image.shape


def test_unicode_image_io(tmp_path):
    path = tmp_path / "中文目录" / "钢板.png"
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert imwrite(path, image)
    restored = imread(path, 0)
    assert np.array_equal(restored, image)


def test_elementwise_focal_loss_shape():
    pred = torch.zeros((2, 5, 9))
    target = torch.zeros_like(pred)
    target[0, 0, 2] = 1.0
    loss = ElementwiseFocalBCE(gamma=2.0, alpha=0.25)(pred, target)
    assert loss.shape == pred.shape
    assert torch.isfinite(loss).all()


def test_weighted_box_fusion_keeps_classes_separate():
    boxes = np.asarray([[0, 0, 10, 10], [1, 1, 11, 11], [1, 1, 11, 11]], dtype=np.float32)
    scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)
    classes = np.asarray([0, 0, 1], dtype=np.int64)
    fused_boxes, fused_scores, fused_classes = classwise_weighted_box_fusion(boxes, scores, classes, 0.5)
    assert len(fused_boxes) == 2
    assert set(fused_classes.tolist()) == {0, 1}
    assert np.isclose(fused_scores.max(), 0.9)


def test_cached_candidates_allow_postprocess_replay():
    candidates = [
        {"bbox": [0, 0, 10, 10], "score": 0.9, "class_id": 0, "touches_internal_edge": False},
        {"bbox": [1, 1, 11, 11], "score": 0.8, "class_id": 0, "touches_internal_edge": True},
    ]
    neutral = InferenceOptions(device="cpu", half=False, edge_penalty=1.0)
    penalized = InferenceOptions(device="cpu", half=False, edge_penalty=0.5)
    neutral_output = merge_candidates(candidates, (20, 20), ["defect"], neutral)
    penalized_output = merge_candidates(candidates, (20, 20), ["defect"], penalized)
    assert neutral_output[0]["score"] == penalized_output[0]["score"] == 0.9
    assert candidates[1]["score"] == 0.8


def test_recall_action_scaling_and_matching():
    cache = {
        "image_size": [100, 100],
        "candidates": [
            {"bbox": [10, 20, 30, 40], "score": 0.9, "class_id": 0},
        ],
    }
    actions = [{"source_class": "a", "target_class": "b", "scale": [1.0, 2.0]}]
    boxes, scores, classes, size = build_action_arrays(
        [cache], ["a", "b"], actions, pre_nms_iou=0.5, score_floor=0.001
    )
    assert size == (100, 100)
    assert np.allclose(boxes, [[10, 10, 30, 50]])
    assert np.allclose(scores, [0.9])
    assert classes.tolist() == [1]
    hit, total, per_class = matched_ground_truth(
        boxes, scores, classes, {"b": [[10, 10, 30, 50]]}, ["a", "b"]
    )
    assert (hit, total) == (1, 1)
    assert per_class["b"] == (1, 1)


def test_recall_candidate_score_threshold_filters_before_actions():
    cache = {
        "image_size": [100, 100],
        "candidates": [
            {"bbox": [10, 10, 20, 20], "score": 0.01, "class_id": 0},
            {"bbox": [40, 40, 50, 50], "score": 0.2, "class_id": 0},
        ],
    }
    actions = [{"source_class": "a", "target_class": "a", "scale": [1.0, 1.0]}]
    boxes, scores, classes, _ = build_action_arrays(
        [cache], ["a"], actions, pre_nms_iou=0.5, score_floor=0.001,
        candidate_score_threshold=0.05,
    )
    assert boxes.tolist() == [[40.0, 40.0, 50.0, 50.0]]
    assert np.allclose(scores, [0.2])
    assert classes.tolist() == [0]


def test_scale_boxes_clips_and_marks_invalid():
    boxes = np.asarray([[0, 0, 2, 2], [90, 90, 100, 100]], dtype=np.float32)
    scaled, valid = scale_boxes(boxes, (100, 100), 0.1, 0.1)
    assert valid.tolist() == [False, False]
    assert scaled.shape == (0, 4)


def test_scale_boxes_supports_center_shift():
    boxes = np.asarray([[10, 20, 30, 40]], dtype=np.float32)
    shifted, valid = scale_boxes(boxes, (100, 100), 1.0, 1.0, 0.5, -0.5)
    assert valid.tolist() == [True]
    assert np.allclose(shifted, [[20, 10, 40, 30]])


def test_streaming_candidate_cache_reader(tmp_path):
    path = tmp_path / "candidates.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "candidate_cache_header", "schema": 3}) + "\n")
        handle.write(json.dumps({"image_id": "a.jpg", "image_size": [10, 10], "candidates": []}) + "\n")
    records = list(iter_candidate_records(path))
    assert len(records) == 1
    assert records[0]["image_id"] == "a.jpg"


def test_infer_select_split_images(tmp_path):
    source = tmp_path / "images"
    source.mkdir()
    first = source / "a.jpg"
    second = source / "b.jpg"
    first.touch()
    second.touch()
    manifest = tmp_path / "splits.json"
    manifest.write_text(json.dumps({"splits": {"dev": ["b.jpg"]}}), encoding="utf-8")
    selected = select_split_images([first, second], source, manifest, "dev")
    assert selected == [second]


def test_inference_options_reject_invalid_overlap():
    try:
        InferenceOptions(overlap=1.0)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("invalid overlap was accepted")


def test_real_gt_metrics_and_threshold_tuning():
    ground_truth = {"image.jpg": {"defect": [[0, 0, 10, 10]]}}
    predictions = [
        {"image_id": "image.jpg", "category_name": "defect", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": "image.jpg", "category_name": "defect", "bbox": [20, 20, 30, 30], "score": 0.1},
    ]
    metrics = evaluate_predictions(predictions, ground_truth, ["defect"], operating_threshold=0.5)
    assert metrics["per_class"]["defect"]["ap50"] == 1.0
    assert metrics["per_class"]["defect"]["ap75"] == 1.0
    assert metrics["per_class"]["defect"]["precision"] == 1.0
    assert metrics["per_class"]["defect"]["recall"] == 1.0
    strict = evaluate_predictions(
        predictions, ground_truth, ["defect"], class_thresholds={"defect": 0.95}
    )
    assert strict["per_class"]["defect"]["ap50"] == 1.0
    assert strict["per_class"]["defect"]["recall"] == 0.0
    thresholds = tune_class_thresholds(predictions, ground_truth, ["defect"], beta=2.0)
    assert 0.01 <= thresholds["defect"] <= 0.5


def test_group_stratified_split_has_no_group_leakage(tmp_path):
    records = []
    for group_index in range(8):
        class_id = group_index % 2
        path = tmp_path / f"group{group_index}-Raw00-f_00001.jpg"
        annotation = Annotation(str(class_id), class_id, (1, 1, 5, 5))
        records.append(VocRecord(path, path.with_suffix(".xml"), 10, 10, (annotation,)))
    splits = split_records(records, val_ratio=0.25, seed=42, trials=500)
    train_groups = {record.group_id for record in splits["train"]}
    val_groups = {record.group_id for record in splits["val"]}
    assert train_groups.isdisjoint(val_groups)
    assert {ann.class_id for record in splits["train"] for ann in record.annotations} == {0, 1}
    assert {ann.class_id for record in splits["val"] for ann in record.annotations} == {0, 1}


def test_three_way_group_split_is_disjoint(tmp_path):
    records = []
    for group_index in range(30):
        path = tmp_path / f"group{group_index}-Raw00-f_00001.jpg"
        annotations = tuple(
            Annotation(str(class_id), class_id, (1, 1, 5, 5)) for class_id in range(3)
        )
        records.append(VocRecord(path, path.with_suffix(".xml"), 10, 10, annotations))
    splits = split_train_calibration_val(records, 0.2, 0.1, 42, trials=500)
    group_sets = [{record.group_id for record in splits[name]} for name in ("train", "calibration", "val")]
    assert all(group_sets)
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])


def test_gc10_adjacent_frames_share_group(tmp_path):
    first = tmp_path / "img_06_3436814000_00005.jpg"
    second = tmp_path / "img_06_3436814000_00693.jpg"
    annotations = (Annotation("x", 0, (1, 1, 5, 5)),)
    a = VocRecord(first, first.with_suffix(".xml"), 10, 10, annotations)
    b = VocRecord(second, second.with_suffix(".xml"), 10, 10, annotations)
    assert a.group_id == b.group_id == "img_06_3436814000"
