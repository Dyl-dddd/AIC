"""Full-height strip views: geometry, inference wiring, and data preparation.

These tests intentionally avoid importing torch: the inference helpers they
exercise only need tensor-like boxes with ``detach``/``cpu``/``numpy``, so a
small numpy-backed double keeps the module collectable on machines without a
full PyTorch installation."""

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from steel_defect.geometry import generate_strip_tiles
from steel_defect.inference import InferenceOptions, predict_candidates_array
from steel_defect.voc import Annotation, VocRecord
from scripts.prepare_data import process_record


def test_strip_tiles_cover_full_width_and_height():
    tiles = generate_strip_tiles(4096, 3000, 810, 0.2)
    assert tiles[0].x == 0
    assert all(tile.y == 0 and tile.height == 3000 for tile in tiles)
    assert tiles[-1].x + tiles[-1].width == 4096
    assert {tile.width for tile in tiles} == {810}
    # stride = round(810 * (1 - 0.2)) = 648; the final strip is edge-aligned.
    assert [tile.x for tile in tiles][:3] == [0, 648, 1296]
    assert tiles[-1].x == 4096 - 810


def test_strip_tiles_single_strip_for_narrow_images():
    tiles = generate_strip_tiles(700, 900, 810, 0.2)
    assert len(tiles) == 1
    assert (tiles[0].x, tiles[0].y, tiles[0].width, tiles[0].height) == (0, 0, 700, 900)


def test_strip_tiles_reject_bad_parameters():
    with pytest.raises(ValueError):
        generate_strip_tiles(100, 100, 0, 0.2)
    with pytest.raises(ValueError):
        generate_strip_tiles(100, 100, 50, 1.0)


def test_inference_options_reject_bad_strip_parameters():
    with pytest.raises(ValueError):
        InferenceOptions(strip_width=0)
    with pytest.raises(ValueError):
        InferenceOptions(strip_overlap=1.0)
    with pytest.raises(ValueError):
        InferenceOptions(strip_imgsz=-1)


class FakeTensor:
    """Minimal tensor double: the inference helpers only call these methods."""

    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def __len__(self):
        return len(self.array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBoxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf)


class RecordingModel:
    """Returns one fixed box per input view and records the calls."""

    def __init__(self):
        self.calls = []

    def predict(self, source, **kwargs):
        self.calls.append({"count": len(source), "imgsz": kwargs.get("imgsz")})
        return [
            SimpleNamespace(boxes=FakeBoxes(
                xyxy=FakeTensor([[10.0, 20.0, 110.0, 120.0]]),
                conf=FakeTensor([0.9]), cls=FakeTensor([0.0])))
            for _ in source
        ]


def test_strip_pass_offsets_coordinates_once():
    pixels = np.zeros((3000, 4096), dtype=np.uint8)
    options = InferenceOptions(batch=2, strip_pass=True, strip_width=810, strip_overlap=0.2)
    rows, size = predict_candidates_array(RecordingModel(), pixels, options)
    assert size == (4096, 3000)
    # 20 square tiles + 7 strips, one fake box per view.
    assert len(rows) == 27
    strip_rows = rows[-7:]
    assert strip_rows[0]["bbox"] == [10.0, 20.0, 110.0, 120.0]
    assert strip_rows[1]["bbox"] == [658.0, 20.0, 758.0, 120.0]
    assert strip_rows[-1]["bbox"] == [3296.0, 20.0, 3396.0, 120.0]


def test_strip_pass_uses_dedicated_imgsz():
    pixels = np.zeros((3000, 4096), dtype=np.uint8)
    model = RecordingModel()
    options = InferenceOptions(
        batch=64, strip_pass=True, strip_width=810, strip_overlap=0.2, strip_imgsz=1536
    )
    predict_candidates_array(model, pixels, options)
    # One flush of the 20 square tiles at base imgsz, one flush of the
    # 7 strips at the dedicated strip imgsz.
    assert [call["imgsz"] for call in model.calls] == [1024, 1536]
    assert [call["count"] for call in model.calls] == [20, 7]


def test_strip_pass_disabled_keeps_view_count():
    pixels = np.zeros((3000, 4096), dtype=np.uint8)
    rows, _ = predict_candidates_array(
        RecordingModel(), pixels, InferenceOptions(batch=2)
    )
    assert len(rows) == 20


def test_prepare_data_strip_layout_keeps_full_defect_extent(tmp_path):
    image_path = tmp_path / "g0001-Raw00-f_00001.jpg"
    image = np.zeros((3000, 4096), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    # A full-height zonglie defect between x=2000 and x=2030.
    annotation = Annotation("zonglie", 1, (2000.0, 100.0, 2030.0, 2900.0))
    record = VocRecord(image_path, image_path.with_suffix(".xml"), 4096, 3000, (annotation,))
    output = tmp_path / "out"
    args = SimpleNamespace(
        view_mode="tiles", tile_layout="strips", output=output,
        tile_size=1024, overlap=0.25, min_visibility=0.35,
        negative_ratio=0.25, seed=42, deblur=False, clahe=False,
        deblur_threshold=85.0, strip_width=810, strip_overlap=0.2,
    )
    metadata, counts = process_record(record, "train", args, ["jieba", "zonglie"])
    strips = [item for item in metadata if item["is_strip"]]
    assert strips
    assert all(item["size"][1] == 3000 for item in strips)
    assert counts["zonglie"] == 2
    # x=1296 and x=1944 strips contain the defect; one negative strip is
    # sampled at negative_ratio=0.25; only the x=1296 strip is verified here.
    positives = sorted(
        (item for item in strips
         if "__x01296_" in item["tile"] or "__x01944_" in item["tile"]),
        key=lambda item: item["tile"],
    )
    assert len(positives) == 2
    label_path = output / "labels" / "train" / f"{positives[0]['tile']}.txt"
    fields = label_path.read_text(encoding="utf-8").split()
    assert fields and len(fields) == 5
    class_id, cx, cy, width, height = fields
    assert class_id == "1"
    assert float(cx) == pytest.approx(719.0 / 810.0)
    assert float(cy) == pytest.approx(1500.0 / 3000.0)
    assert float(width) == pytest.approx(30.0 / 810.0)
    assert float(height) == pytest.approx(2800.0 / 3000.0)
