"""Train a genuinely independent crop classifier on train-only steel annotations.

This script never uses development or Test annotations for fitting. The
classifier is an experiment; it cannot be submitted without a separate,
frozen-development candidate-ranking audit.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data/official_v2_20260831b/splits.json"
IMAGES = ROOT / "data/official/train"
OUT = Path(os.environ.get("CROP_VERIFIER_OUT", str(ROOT / "runs/semifinal/v20_crop_verifier_20261002")))
PRETRAINED = Path(r"D:\DevCaches\torch\hub\checkpoints\mobilenet_v3_small-047dcff4.pth")
CLASSES = (
    "background", "jieba", "zonglie", "jiaza", "yiwuyaru", "huashang",
    "mamianmakeng", "yanghuatiepi", "gunyin",
)
SIZE = 160


def gray_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        image = image[:, :, 0] if image.shape[2] == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def crop_patch(image: np.ndarray, bbox: list[float] | tuple[float, ...]) -> np.ndarray:
    height, width = image.shape
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = max(x2 - x1, 2.0), max(y2 - y1, 2.0)
    pad_x, pad_y = max(12.0, 0.22 * bw), max(12.0, 0.22 * bh)
    left = max(0, int(np.floor(x1 - pad_x)))
    top = max(0, int(np.floor(y1 - pad_y)))
    right = min(width, int(np.ceil(x2 + pad_x)))
    bottom = min(height, int(np.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        return np.full((SIZE, SIZE), 127, dtype=np.uint8)
    patch = image[top:bottom, left:right]
    scale = min(SIZE / patch.shape[1], SIZE / patch.shape[0])
    target_w = max(1, min(SIZE, int(round(patch.shape[1] * scale))))
    target_h = max(1, min(SIZE, int(round(patch.shape[0] * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(patch, (target_w, target_h), interpolation=interpolation)
    canvas = np.full((SIZE, SIZE), int(np.median(patch)), dtype=np.uint8)
    offset_x, offset_y = (SIZE - target_w) // 2, (SIZE - target_h) // 2
    canvas[offset_y:offset_y + target_h, offset_x:offset_x + target_w] = resized
    return canvas


def iou_one(box: list[float], others: list[list[float]]) -> float:
    if not others:
        return 0.0
    x1, y1, x2, y2 = box
    best = 0.0
    for a, b, c, d in others:
        iw = max(0.0, min(x2, c) - max(x1, a))
        ih = max(0.0, min(y2, d) - max(y1, b))
        area = (x2 - x1) * (y2 - y1) + (c - a) * (d - b) - iw * ih
        if area > 0:
            best = max(best, iw * ih / area)
    return best


def jitter_positive(box: list[float], rng: random.Random) -> list[float]:
    x1, y1, x2, y2 = box
    w, h = max(x2 - x1, 2), max(y2 - y1, 2)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for _ in range(20):
        nw = w * rng.uniform(0.9, 1.12)
        nh = h * rng.uniform(0.9, 1.12)
        nx = cx + rng.uniform(-0.07, 0.07) * w
        ny = cy + rng.uniform(-0.07, 0.07) * h
        candidate = [nx - nw / 2, ny - nh / 2, nx + nw / 2, ny + nh / 2]
        if iou_one(candidate, [box]) >= 0.6:
            return candidate
    return box


def negative_box(width: int, height: int, sizes: list[tuple[float, float]],
                 ground_truth: list[list[float]], rng: random.Random) -> list[float] | None:
    for _ in range(50):
        if rng.random() < 0.2:
            bw, bh = rng.uniform(20, 240), rng.uniform(20, 240)
        else:
            bw, bh = rng.choice(sizes)
            bw *= rng.uniform(0.75, 1.25)
            bh *= rng.uniform(0.75, 1.25)
        bw, bh = min(max(bw, 12.0), width * 0.7), min(max(bh, 12.0), height * 0.7)
        x1, y1 = rng.uniform(0, width - bw), rng.uniform(0, height - bh)
        candidate = [x1, y1, x1 + bw, y1 + bh]
        if iou_one(candidate, ground_truth) < 0.04:
            return candidate
    return None


def prepare() -> None:
    metadata = json.loads(SPLITS.read_text(encoding="utf-8"))
    rng = random.Random(20261002)
    train_names = metadata["splits"]["train"]
    groups = {name: metadata["records"][name]["group"] for name in train_names}
    dev_groups = {metadata["records"][name]["group"] for name in metadata["splits"]["dev"]}
    if set(groups.values()) & dev_groups:
        raise RuntimeError("train/dev group overlap")
    sizes = [
        (float(a[3]) - float(a[1]), float(a[4]) - float(a[2]))
        for name in train_names for a in metadata["records"][name]["annotations"]
        if a[0] in CLASSES[1:]
    ]
    crop_root = OUT / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    counts = Counter()
    for image_index, name in enumerate(train_names, 1):
        image = gray_image(IMAGES / name)
        height, width = image.shape
        annotations = metadata["records"][name]["annotations"]
        boxes = [list(map(float, row[1:])) for row in annotations]
        group_hash = hashlib.sha256(groups[name].encode()).digest()
        split = "val" if int.from_bytes(group_hash[:4], "big") % 10 == 0 else "train"
        examples: list[tuple[list[float], str]] = []
        for row, box in zip(annotations, boxes, strict=True):
            if row[0] not in CLASSES[1:]:
                continue
            examples.append((box, row[0]))
            examples.append((jitter_positive(box, rng), row[0]))
        for _ in range(8):
            candidate = negative_box(width, height, sizes, boxes, rng)
            if candidate is not None:
                examples.append((candidate, "background"))
        for item_index, (box, label) in enumerate(examples):
            crop = crop_patch(image, box)
            destination = crop_root / f"{image_index:05d}_{item_index:03d}.jpg"
            encoded, buffer = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not encoded:
                raise OSError(destination)
            buffer.tofile(str(destination))
            manifest.append({"file": destination.name, "label": CLASSES.index(label), "split": split, "source": name})
            counts[(split, label)] += 1
        if image_index % 200 == 0:
            print(f"PREPARE {image_index}/{len(train_names)} crops={len(manifest)}", flush=True)
    if not manifest or not any(row["split"] == "val" for row in manifest):
        raise RuntimeError("empty preparation")
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    report = {"source_train_images": len(train_names), "source_dev_images_used": 0,
              "train_dev_group_overlap": 0, "samples": len(manifest),
              "counts": {f"{split}/{name}": value for (split, name), value in sorted(counts.items())}}
    (OUT / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


class Crops(Dataset):
    def __init__(self, manifest: list[dict], split: str, augment: bool):
        self.rows = [row for row in manifest if row["split"] == split]
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image = gray_image(OUT / "crops" / row["file"])
        if self.augment:
            if random.random() < 0.5:
                image = cv2.flip(image, 1)
            if random.random() < 0.5:
                image = cv2.flip(image, 0)
            image = np.clip(image.astype(np.float32) * random.uniform(0.85, 1.15)
                            + random.uniform(-12, 12), 0, 255).astype(np.uint8)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255.0)
        tensor = tensor.unsqueeze(0).expand(3, -1, -1)
        return (tensor - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None], int(row["label"])


def model_for_training() -> nn.Module:
    if not PRETRAINED.is_file():
        raise FileNotFoundError(f"missing pretrained weights: {PRETRAINED}")
    model = mobilenet_v3_small(weights=None)
    state = torch.load(PRETRAINED, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    for parameter in model.features[:5].parameters():
        parameter.requires_grad = False
    return model


def train(epochs: int, batch_size: int) -> None:
    torch.manual_seed(20261002)
    random.seed(20261002)
    np.random.seed(20261002)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    train_data, val_data = Crops(manifest, "train", True), Crops(manifest, "val", False)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_for_training().to(device)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": 2e-5},
        {"params": model.classifier.parameters(), "lr": 1e-4},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_counts = Counter(row["label"] for row in train_data.rows)
    weights = torch.tensor([
        min(4.0, (max(class_counts.values()) / class_counts[index]) ** 0.35)
        for index in range(len(CLASSES))
    ], device=device, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.03)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    history = []
    best = float("inf")
    for epoch in range(epochs):
        model.train()
        running, seen = 0.0, 0
        for batch_index, (inputs, labels) in enumerate(train_loader, 1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = criterion(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(labels)
            seen += len(labels)
            if batch_index % 50 == 0:
                print(f"EPOCH {epoch+1}/{epochs} batch={batch_index}/{len(train_loader)} loss={running/seen:.5f}", flush=True)
        model.eval()
        val_loss, val_correct, val_seen = 0.0, 0, 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                val_loss += float(loss.item()) * len(labels)
                val_correct += int((logits.argmax(1) == labels).sum())
                val_seen += len(labels)
        scheduler.step()
        row = {"epoch": epoch + 1, "train_loss": running / seen,
               "val_loss": val_loss / val_seen, "val_accuracy": val_correct / val_seen}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_loss"] < best:
            best = row["val_loss"]
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "epoch": epoch + 1, "val_loss": best}, OUT / "best.pt")
    (OUT / "train_report.json").write_text(json.dumps({"history": history, "best_val_loss": best,
                                                      "device": str(device)}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "train"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    else:
        train(args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
