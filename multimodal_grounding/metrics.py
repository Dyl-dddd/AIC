from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def box_iou(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = pred.float()
    target = target.float()
    lt = torch.maximum(pred[:, :2], target[:, :2])
    rb = torch.minimum(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    pred_area = _box_area(pred)
    target_area = _box_area(target)
    union = (pred_area + target_area - intersection).clamp(min=1e-8)
    return intersection / union


def generalized_box_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    lt = torch.maximum(pred[:, :2], target[:, :2])
    rb = torch.minimum(pred[:, 2:], target[:, 2:])
    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, 0] * wh[:, 1]
    pred_area = _box_area(pred)
    target_area = _box_area(target)
    union = (pred_area + target_area - intersection).clamp(min=1e-8)
    iou = intersection / union

    enclose_lt = torch.minimum(pred[:, :2], target[:, :2])
    enclose_rb = torch.maximum(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
    enclose_area = (enclose_wh[:, 0] * enclose_wh[:, 1]).clamp(min=1e-8)
    giou = iou - (enclose_area - union) / enclose_area
    return 1.0 - giou


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    return wh[:, 0] * wh[:, 1]


def acc_at_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    if pred.numel() == 0:
        return 0.0
    return float((box_iou(pred, target) >= threshold).float().mean().item())


def sanitize_bbox(values: Iterable[float], eps: float = 1e-5) -> list[float]:
    bbox = [float(v) for v in values]
    if len(bbox) != 4 or any(math.isnan(v) or math.isinf(v) for v in bbox):
        return [0.0, 0.0, 1.0, 1.0]

    x1, y1, x2, y2 = bbox
    x1, x2 = sorted((max(0.0, min(1.0, x1)), max(0.0, min(1.0, x2))))
    y1, y2 = sorted((max(0.0, min(1.0, y1)), max(0.0, min(1.0, y2))))

    if x2 - x1 < eps:
        center = (x1 + x2) * 0.5
        x1 = max(0.0, center - eps * 0.5)
        x2 = min(1.0, center + eps * 0.5)
        if x2 - x1 < eps:
            x1, x2 = 0.0, eps
    if y2 - y1 < eps:
        center = (y1 + y2) * 0.5
        y1 = max(0.0, center - eps * 0.5)
        y2 = min(1.0, center + eps * 0.5)
        if y2 - y1 < eps:
            y1, y2 = 0.0, eps
    return [x1, y1, x2, y2]


def numpy_iou(pred: Iterable[float], target: Iterable[float]) -> float:
    pred_array = np.asarray(list(pred), dtype=np.float32)[None, :]
    target_array = np.asarray(list(target), dtype=np.float32)[None, :]
    pred_t = torch.from_numpy(pred_array)
    target_t = torch.from_numpy(target_array)
    return float(box_iou(pred_t, target_t).item())
