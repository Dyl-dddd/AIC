"""Inference-compatible task-aligned quality training for Ultralytics YOLO11.

Ultralytics' task-aligned assigner already emits soft positive targets whose
magnitude contains localization quality.  The stock detector trains those
targets with plain BCE.  This module changes only the classification loss to
an elementwise Varifocal form, so exported checkpoints retain the ordinary
YOLO11 Detect tensor layout and remain compatible with the existing inference
and ensemble code.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import ultralytics
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.loss import v8DetectionLoss


SUPPORTED_ULTRALYTICS = "8.3.169"
VARIFOCAL_GAMMA = 2.0
VARIFOCAL_ALPHA = 0.75
QUALITY_POWER = 1.0


def ensure_ultralytics_compatibility(version: str | None = None) -> None:
    """Fail closed because this extension relies on v8DetectionLoss internals."""
    actual = ultralytics.__version__ if version is None else str(version)
    if actual != SUPPORTED_ULTRALYTICS:
        raise RuntimeError(
            "quality-aware training requires ultralytics "
            f"{SUPPORTED_ULTRALYTICS}, found {actual}"
        )


def configure_quality_loss(gamma: float, alpha: float, quality_power: float = 1.0) -> None:
    """Configure the process-local loss before the trainer constructs a model."""
    global VARIFOCAL_GAMMA, VARIFOCAL_ALPHA, QUALITY_POWER
    gamma, alpha, quality_power = float(gamma), float(alpha), float(quality_power)
    if gamma < 0.0:
        raise ValueError("varifocal gamma must be non-negative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("varifocal alpha must be in [0, 1]")
    if quality_power <= 0.0:
        raise ValueError("quality power must be positive")
    VARIFOCAL_GAMMA = gamma
    VARIFOCAL_ALPHA = alpha
    QUALITY_POWER = quality_power
    os.environ["STEEL_VARIFOCAL_GAMMA"] = str(gamma)
    os.environ["STEEL_VARIFOCAL_ALPHA"] = str(alpha)
    os.environ["STEEL_QUALITY_POWER"] = str(quality_power)


class ElementwiseVarifocalBCE(nn.Module):
    """Varifocal loss with the unreduced shape expected by v8DetectionLoss.

    A positive target is the task-aligned soft quality target, optionally
    transformed by ``quality_power``.  Negatives receive the standard
    ``alpha * p**gamma`` hard-negative weight.  The returned tensor is not
    reduced because the parent YOLO loss normalizes its sum by target quality.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75,
                 quality_power: float = 1.0):
        super().__init__()
        if gamma < 0.0:
            raise ValueError("varifocal gamma must be non-negative")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("varifocal alpha must be in [0, 1]")
        if quality_power <= 0.0:
            raise ValueError("quality power must be positive")
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.quality_power = float(quality_power)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError("prediction and quality target shapes must match")
        if not torch.is_floating_point(pred) or not torch.is_floating_point(target):
            raise TypeError("prediction and quality target must be floating point")
        quality = target.clamp(0.0, 1.0).pow(self.quality_power)
        positive = target.gt(0.0).to(dtype=pred.dtype)
        probability = pred.sigmoid()
        weight = (
            self.alpha * probability.pow(self.gamma) * (1.0 - positive)
            + quality.to(dtype=pred.dtype) * positive
        )
        return F.binary_cross_entropy_with_logits(
            pred, quality.to(dtype=pred.dtype), reduction="none"
        ) * weight


class QualityAwareDetectionLoss(v8DetectionLoss):
    """Drop-in YOLO11 loss reweighting task-aligned soft class targets."""

    def __init__(self, model, tal_topk: int = 10):
        ensure_ultralytics_compatibility()
        super().__init__(model, tal_topk=tal_topk)
        gamma = float(os.environ.get("STEEL_VARIFOCAL_GAMMA", VARIFOCAL_GAMMA))
        alpha = float(os.environ.get("STEEL_VARIFOCAL_ALPHA", VARIFOCAL_ALPHA))
        power = float(os.environ.get("STEEL_QUALITY_POWER", QUALITY_POWER))
        self.bce = ElementwiseVarifocalBCE(gamma, alpha, power)


class QualityAwareDetectionTrainer(DetectionTrainer):
    """Trainer that keeps EMA/checkpoints as stock ``DetectionModel`` objects.

    The custom criterion is injected only after ``DetectionTrainer`` has
    created its stock-model EMA copy. Ultralytics serializes the EMA model, so
    saved ``.pt`` files do not pickle a project-specific model subclass.
    """

    def get_model(
        self,
        cfg: Optional[str] = None,
        weights: Optional[str] = None,
        verbose: bool = True,
    ):
        ensure_ultralytics_compatibility()
        model = DetectionModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose
        )
        if weights:
            model.load(weights)
        return model

    def _setup_train(self, world_size):
        super()._setup_train(world_size)
        training_model = de_parallel(self.model)
        training_model.criterion = QualityAwareDetectionLoss(training_model)


__all__ = [
    "ElementwiseVarifocalBCE",
    "QualityAwareDetectionLoss",
    "QualityAwareDetectionTrainer",
    "SUPPORTED_ULTRALYTICS",
    "configure_quality_loss",
    "ensure_ultralytics_compatibility",
]
