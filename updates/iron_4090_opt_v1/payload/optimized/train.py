"""Training compatibility wrapper adding deferred tiled validation.

The cloud queue evaluates saved checkpoints on complete original images, so
per-epoch tiled validation can be skipped without changing model updates.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

import scripts.train as original


class DeferredValidationDetectionTrainer(DetectionTrainer):
    def validate(self):
        fitness = 0.0 if self.fitness is None else float(self.fitness)
        metrics = {
            "metrics/precision(B)": 0.0,
            "metrics/recall(B)": 0.0,
            "metrics/mAP50(B)": 0.0,
            "metrics/mAP50-95(B)": 0.0,
            "val/box_loss": 0.0,
            "val/cls_loss": 0.0,
            "val/dfl_loss": 0.0,
        }
        return metrics, fitness

    def final_eval(self):
        return None


ORIGINAL_YOLO_TRAIN = YOLO.train


def train_with_deferred_validation(self, *args, **kwargs):
    defer_validation = bool(kwargs.pop("defer_validation", False))
    if defer_validation:
        if kwargs.get("trainer") is not None:
            raise ValueError("Deferred validation is only supported for the stock BCE trainer")
        kwargs["trainer"] = DeferredValidationDetectionTrainer
    return ORIGINAL_YOLO_TRAIN(self, *args, **kwargs)


YOLO.train = train_with_deferred_validation


if __name__ == "__main__":
    original.main()
