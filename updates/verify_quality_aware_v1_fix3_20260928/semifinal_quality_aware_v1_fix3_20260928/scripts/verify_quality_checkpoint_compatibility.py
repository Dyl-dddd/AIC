"""Load a trained checkpoint in a fresh process and verify stock YOLO inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steel_defect.classes import CLASS_NAMES
from steel_defect.runtime import sha256_file


def tensor_leaves(value):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensor_leaves(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_leaves(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-epochs", type=int, required=True)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    completed_epochs = int(payload.get("epoch", -2)) + 1
    if completed_epochs != args.expected_epochs:
        raise ValueError(
            f"terminal checkpoint epoch mismatch: {completed_epochs} != {args.expected_epochs}"
        )
    model = YOLO(str(args.checkpoint))
    if type(model.model) is not DetectionModel:
        raise TypeError(
            "checkpoint is not a stock Ultralytics DetectionModel: "
            f"{type(model.model).__module__}.{type(model.model).__qualname__}"
        )
    names = [model.names[index] for index in sorted(model.names)]
    if names != CLASS_NAMES:
        raise ValueError(f"checkpoint class order mismatch: {names}")
    network = model.model.float().cpu().eval()
    with torch.no_grad():
        output = network(torch.zeros(1, 3, 128, 128, dtype=torch.float32))
    leaves = list(tensor_leaves(output))
    if not leaves or not all(torch.isfinite(item).all() for item in leaves):
        raise FloatingPointError("fresh-process stock-model forward is empty or non-finite")
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "sha256": sha256_file(args.checkpoint),
        "model_class": f"{type(model.model).__module__}.{type(model.model).__qualname__}",
        "classes": names,
        "tensor_leaves": len(leaves),
        "finite": True,
        "input_shape": [1, 3, 128, 128],
        "completed_epochs": completed_epochs,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

