"""Ultralytics trainer with focal classification loss."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25


def configure_focal(gamma: float, alpha: float) -> None:
    global FOCAL_GAMMA, FOCAL_ALPHA
    FOCAL_GAMMA = float(gamma)
    FOCAL_ALPHA = float(alpha)
    os.environ["STEEL_FOCAL_GAMMA"] = str(FOCAL_GAMMA)
    os.environ["STEEL_FOCAL_ALPHA"] = str(FOCAL_ALPHA)


class FocalDetectionLoss(v8DetectionLoss):
    """Drop-in v8 detection loss with focal classification term."""

    def __init__(self, model, tal_topk: int = 10):
        super().__init__(model, tal_topk=tal_topk)
        gamma = float(os.environ.get("STEEL_FOCAL_GAMMA", FOCAL_GAMMA))
        alpha = float(os.environ.get("STEEL_FOCAL_ALPHA", FOCAL_ALPHA))
        self.bce = ElementwiseFocalBCE(gamma=gamma, alpha=alpha)


class ElementwiseFocalBCE(nn.Module):
    """Focal BCE with the same unreduced shape expected by v8DetectionLoss."""

    def __init__(self, gamma: float, alpha: float):
        super().__init__()
        if gamma < 0:
            raise ValueError("focal gamma must be non-negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("focal alpha must be in [0, 1]")
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        probability = pred.sigmoid()
        p_t = target * probability + (1.0 - target) * (1.0 - probability)
        alpha_factor = target * self.alpha + (1.0 - target) * (1.0 - self.alpha)
        return loss * ((1.0 - p_t) ** self.gamma) * alpha_factor


class FocalDetectionModel(DetectionModel):
    def init_criterion(self):
        return FocalDetectionLoss(self)


class FocalDetectionTrainer(DetectionTrainer):
    def get_model(
        self,
        cfg: Optional[str] = None,
        weights: Optional[str] = None,
        verbose: bool = True,
    ):
        model = FocalDetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)
        if weights:
            model.load(weights)
        return model
