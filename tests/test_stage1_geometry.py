from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.run_semifinal_stage1 import add_micro, crop_gt, experiment_jobs, verified_file, score_proxy
from steel_defect.geometry import Tile, generate_tiles
from steel_defect.inference import InferenceOptions, inference_tiles, predict_candidates_array
from steel_defect.voc import Annotation, VocRecord


def test_grid_coordinates_and_crop_single_view():
    options = InferenceOptions(tile_layout="semifinal_grid")
    tiles = inference_tiles(4096, 3000, options)
    assert [(t.x, t.y) for t in tiles] == [(0, 0), (1354, 0), (2709, 0), (0, 1484), (1354, 1484), (2709, 1484)]
    assert len(inference_tiles(1387, 1516, options)) == 1
    assert len(inference_tiles(500, 700, options)) == 1
    with pytest.raises(ValueError):
        inference_tiles(6000, 4000, options)


def test_default_sliding_unchanged_and_unknown_layout_rejected():
    assert inference_tiles(4096, 3000, InferenceOptions()) == generate_tiles(4096, 3000, 1024, 0.25)
    with pytest.raises(ValueError):
        InferenceOptions(tile_layout="typo")


class FakeBoxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf)


class FakeModel:
    def predict(self, source, **kwargs):
        return [SimpleNamespace(boxes=FakeBoxes(
            xyxy=torch.tensor([[10., 20., 110., 120.]]),
            conf=torch.tensor([0.9]), cls=torch.tensor([0.]))) for _ in source]


def test_grid_network_coordinates_are_offset_once():
    pixels = np.zeros((3000, 4096), dtype=np.uint8)
    rows, size = predict_candidates_array(FakeModel(), pixels,
        InferenceOptions(tile_layout="semifinal_grid", batch=2, global_pass=False))
    assert size == (4096, 3000)
    assert len(rows) == 6
    assert rows[1]["bbox"] == [1364., 20., 1464., 120.]
    assert rows[-1]["bbox"] == [2719., 1504., 2819., 1604.]


def test_global_inverse_resize_coordinates():
    pixels = np.zeros((3000, 4096), dtype=np.uint8)
    rows, _ = predict_candidates_array(FakeModel(), pixels,
        InferenceOptions(tile_layout="semifinal_grid", global_pass=True, imgsz=1024))
    assert len(rows) == 7
    assert rows[-1]["is_global"]
    assert rows[-1]["bbox"] == [40., 80., 440., 480.]


def test_crop_labels_clip_and_exclude_qilie():
    record = VocRecord(Path("x.jpg"), Path("x.xml"), 4096, 3000, (
        Annotation("jieba", 0, (1400., 100., 1500., 200.)),
        Annotation("qilie", 2, (1400., 100., 1500., 200.))))
    assert crop_gt(record, Tile(1354, 0, 1387, 1516, 4096, 3000)) == {"jieba": [[46., 100., 146., 200.]]}


def test_hash_guard_and_micro_counts(tmp_path):
    file = tmp_path / "weight.pt"
    file.write_bytes(b"wrong weight")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verified_file(file, "0" * 64)
    result = add_micro({"summary": {"map50_macro": 0.4}, "per_class": {"jieba": {
        "ground_truth": 4, "true_positives": 3, "false_positives": 2}}})
    assert result["summary"]["recall_micro"] == 0.75
    assert result["summary"]["precision_micro"] == 0.6
    assert result["summary"]["false_negatives"] == 1
    assert result["summary"]["score_proxy_20_60_20"] == 65.0


def test_contest_formula_reproduces_reported_score():
    assert score_proxy(0.0001, 0.9877, 0.1044) == pytest.approx(61.352)
    assert score_proxy(0.2, 0.95, 0.45) == pytest.approx(70.0)


def test_stage2_is_bounded_and_single_model_per_job():
    assert experiment_jobs(2) == [
        ("old", "sliding", 1024, 0.0001),
        ("new", "semifinal_grid", 1024, 0.0001),
        ("new", "semifinal_grid", 1280, 0.0001),
    ]
    with pytest.raises(ValueError, match="Unknown experiment stage"):
        experiment_jobs(3)
