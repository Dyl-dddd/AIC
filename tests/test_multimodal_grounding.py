from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from multimodal_grounding.data import GroundingDataset, collate_grounding, load_annotations, tokenize_query
from multimodal_grounding.metrics import acc_at_iou, box_iou, sanitize_bbox
from multimodal_grounding.model import TextGuidedBoxRegressor


def _write_tiny_dataset(root: Path) -> Path:
    (root / "Images" / "visible").mkdir(parents=True)
    (root / "Images" / "infrared").mkdir(parents=True)
    (root / "Images" / "depth").mkdir(parents=True)
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[10:18, 20:26] = [200, 220, 240]
    depth = np.zeros((24, 32), dtype=np.uint16)
    depth[10:18, 20:26] = 1200
    Image.fromarray(rgb).save(root / "Images" / "visible" / "000001.png")
    Image.fromarray(rgb[..., ::-1]).save(root / "Images" / "infrared" / "000001.png")
    Image.fromarray(depth).save(root / "Images" / "depth" / "000001.png")
    payload = {
        "000001_001": {
            "visible": "Images/visible/000001.png",
            "infrared": "Images/infrared/000001.png",
            "depth": "Images/depth/000001.png",
            "query": "the bright block",
            "bbox": [0.6, 0.4, 0.8, 0.75],
        }
    }
    path = root / "sample.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_dataset_and_tensor_shape(tmp_path):
    data_path = _write_tiny_dataset(tmp_path)
    samples = load_annotations(data_path, require_bbox=True)
    dataset = GroundingDataset(samples, image_size=32)
    item = dataset[0]
    assert item["image"].shape == (7, 32, 32)
    assert torch.all((item["bbox"] >= 0) & (item["bbox"] <= 1))


def test_tokenize_query_is_fixed_length():
    tokens, mask = tokenize_query("The silver light bulb", max_tokens=6, vocab_size=128)
    assert tokens.shape == (6,)
    assert mask.sum().item() == 4
    assert tokens[0].item() >= 2


def test_model_forward_returns_valid_boxes(tmp_path):
    data_path = _write_tiny_dataset(tmp_path)
    samples = load_annotations(data_path, require_bbox=True)
    dataset = GroundingDataset(samples, image_size=64)
    batch = collate_grounding([dataset[0]])
    model = TextGuidedBoxRegressor(width=8, text_dim=16, vocab_size=8192)
    pred = model(batch["image"], batch["tokens"], batch["mask"])
    assert pred.shape == (1, 4)
    assert torch.all(pred >= 0)
    assert torch.all(pred <= 1)
    assert torch.all(pred[:, 0] <= pred[:, 2])
    assert torch.all(pred[:, 1] <= pred[:, 3])


def test_iou_and_acc():
    pred = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.2, 0.2]])
    target = torch.tensor([[0.0, 0.0, 0.5, 0.5], [0.8, 0.8, 1.0, 1.0]])
    iou = box_iou(pred, target)
    assert torch.allclose(iou, torch.tensor([1.0, 0.0]))
    assert acc_at_iou(pred, target) == 0.5


def test_sanitize_bbox_repairs_invalid_values():
    bbox = sanitize_bbox([float("nan"), 2.0, -1.0, 0.5])
    assert bbox == [0.0, 0.0, 1.0, 1.0]
    bbox = sanitize_bbox([0.8, 0.8, 0.1, 0.2])
    assert 0.0 <= bbox[0] < bbox[2] <= 1.0
    assert 0.0 <= bbox[1] < bbox[3] <= 1.0
