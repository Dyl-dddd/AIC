import math

import pytest
import torch
import torch.nn.functional as F

from steel_defect.quality_trainer import (
    ElementwiseVarifocalBCE,
    SUPPORTED_ULTRALYTICS,
    configure_quality_loss,
    ensure_ultralytics_compatibility,
    QualityAwareDetectionTrainer,
)
from ultralytics.nn.tasks import DetectionModel


def test_negative_varifocal_weight_matches_definition():
    pred = torch.tensor([[0.0]], requires_grad=True)
    target = torch.zeros_like(pred)
    loss = ElementwiseVarifocalBCE(gamma=2.0, alpha=0.75)(pred, target)
    expected = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * 0.75 * 0.5**2
    assert torch.allclose(loss, expected)
    loss.sum().backward()
    assert torch.isfinite(pred.grad).all()


def test_positive_target_is_weighted_by_localization_quality():
    pred = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[0.25, 0.81]])
    loss = ElementwiseVarifocalBCE(gamma=2.0, alpha=0.75)(pred, target)
    expected = F.binary_cross_entropy_with_logits(pred, target, reduction="none") * target
    assert torch.allclose(loss, expected)
    assert loss[0, 1] > loss[0, 0]


def test_quality_power_transforms_target_and_weight_consistently():
    pred = torch.tensor([[0.3]])
    target = torch.tensor([[0.25]])
    loss = ElementwiseVarifocalBCE(quality_power=2.0)(pred, target)
    quality = target.square()
    expected = F.binary_cross_entropy_with_logits(pred, quality, reduction="none") * quality
    assert torch.allclose(loss, expected)


@pytest.mark.parametrize("kwargs", [
    {"gamma": -1.0}, {"alpha": -0.1}, {"alpha": 1.1}, {"quality_power": 0.0},
])
def test_invalid_quality_loss_parameters_fail(kwargs):
    with pytest.raises(ValueError):
        ElementwiseVarifocalBCE(**kwargs)


def test_shape_and_dtype_contracts_fail_closed():
    loss = ElementwiseVarifocalBCE()
    with pytest.raises(ValueError):
        loss(torch.zeros(2), torch.zeros(3))
    with pytest.raises(TypeError):
        loss(torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64))


def test_ultralytics_version_is_pinned():
    ensure_ultralytics_compatibility(SUPPORTED_ULTRALYTICS)
    with pytest.raises(RuntimeError):
        ensure_ultralytics_compatibility("999.0.0")


def test_process_configuration_is_validated(monkeypatch):
    configure_quality_loss(2.0, 0.75, 1.0)
    with pytest.raises(ValueError):
        configure_quality_loss(-0.1, 0.75, 1.0)


def test_quality_trainer_constructs_stock_detection_model():
    trainer = object.__new__(QualityAwareDetectionTrainer)
    trainer.data = {"nc": 2, "channels": 3}
    model = trainer.get_model(cfg="yolo11n.yaml", verbose=False)
    assert type(model) is DetectionModel

