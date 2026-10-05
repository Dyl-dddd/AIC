from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import GroundingDataset, GroundingSample, collate_grounding
from .metrics import acc_at_iou, box_iou, generalized_box_iou_loss
from .model import TextGuidedBoxRegressor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_by_visible_path(
    samples: list[GroundingSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[GroundingSample], list[GroundingSample]]:
    if len(samples) < 2 or val_ratio <= 0:
        return samples, samples

    groups: dict[str, list[GroundingSample]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.visible)].append(sample)

    keys = list(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)
    if len(keys) < 2:
        return samples, samples

    val_count = max(1, min(len(keys) - 1, round(len(keys) * val_ratio)))
    val_keys = set(keys[:val_count])
    train_samples = [sample for key in keys if key not in val_keys for sample in groups[key]]
    val_samples = [sample for key in keys if key in val_keys for sample in groups[key]]
    return train_samples, val_samples


def make_loader(
    samples: list[GroundingSample],
    image_size: int,
    max_tokens: int,
    vocab_size: int,
    batch_size: int,
    workers: int,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = GroundingDataset(
        samples=samples,
        image_size=image_size,
        max_tokens=max_tokens,
        vocab_size=vocab_size,
        augment=augment,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_grounding,
        pin_memory=torch.cuda.is_available(),
    )


def grounding_loss(pred: torch.Tensor, target: torch.Tensor, l1_weight: float, giou_weight: float) -> torch.Tensor:
    l1 = nn.functional.smooth_l1_loss(pred, target)
    giou = generalized_box_iou_loss(pred, target).mean()
    return l1_weight * l1 + giou_weight * giou


@torch.no_grad()
def evaluate(
    model: TextGuidedBoxRegressor,
    loader: DataLoader,
    device: torch.device,
    l1_weight: float,
    giou_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    total_acc = 0.0
    total = 0
    for batch in loader:
        target = batch["bbox"]
        if target is None:
            continue
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        mask = batch["mask"].to(device)
        target = target.to(device)
        pred = model(images, tokens, mask)
        batch_size = images.shape[0]
        loss = grounding_loss(pred, target, l1_weight, giou_weight)
        iou = box_iou(pred, target)
        total_loss += float(loss.item()) * batch_size
        total_iou += float(iou.sum().item())
        total_acc += acc_at_iou(pred, target) * batch_size
        total += batch_size
    if total == 0:
        return {"loss": 0.0, "iou": 0.0, "acc05": 0.0}
    return {
        "loss": total_loss / total,
        "iou": total_iou / total,
        "acc05": total_acc / total,
    }


def save_checkpoint(
    path: str | Path,
    model: TextGuidedBoxRegressor,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: dict[str, Any],
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": model.config(),
        "metrics": metrics,
        "args": args,
    }
    torch.save(payload, path)


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
