"""Score frozen V12 OOF candidates with independent train-only crop classifier."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import mobilenet_v3_small

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.v20_crop_verifier import CLASSES, IMAGES, OUT, SIZE, gray_image, crop_patch


ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
OOF = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
BLENDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.60)


def identity(item: dict) -> tuple:
    return item["image_id"], item["category_name"], tuple(item["bbox"])


def load_model() -> nn.Module:
    checkpoint = torch.load(OUT / "best.pt", map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("checkpoint class mismatch")
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model"])
    return model


def score_candidates(rows: list[dict], image_root: Path, model: nn.Module,
                     device: torch.device, batch_size: int = 256) -> np.ndarray:
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[row["image_id"].split("::", 1)[0]].append(index)
    result = np.zeros(len(rows), dtype=np.float32)
    batch: list[np.ndarray] = []
    indices: list[int] = []
    labels: list[int] = []
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    def flush() -> None:
        if not batch:
            return
        images = torch.from_numpy(np.stack(batch)).to(device, non_blocking=True)
        images = images.float().div_(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        images = (images - mean) / std
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            probabilities = model(images).softmax(dim=1)
        selected = probabilities[torch.arange(len(labels), device=device), torch.tensor(labels, device=device)]
        result[indices] = selected.float().cpu().numpy()
        batch.clear()
        indices.clear()
        labels.clear()

    model.eval().to(device)
    for number, (source, positions) in enumerate(sorted(by_source.items()), 1):
        image = gray_image(image_root / source)
        for position in positions:
            row = rows[position]
            suffix = row["image_id"].split("::", 1)
            view = image
            if len(suffix) == 2:
                ox, oy = (int(value[1:]) for value in suffix[1].split("_"))
                view = image[oy:oy + 1516, ox:ox + 1387]
            batch.append(crop_patch(view, row["bbox"]))
            indices.append(position)
            labels.append(CLASSES.index(row["category_name"]))
            if len(batch) >= batch_size:
                flush()
        if number % 50 == 0:
            print(f"SCORE {number}/{len(by_source)} images; candidates={len(result)}", flush=True)
    flush()
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("invalid probabilities")
    return result


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model()
    result = {"protocol": "independent train-only MobileNetV3 crop verifier; frozen V12 group-OOF dev",
              "checkpoint_sha256": hashlib.sha256((OUT / "best.pt").read_bytes()).hexdigest(),
              "views": {}}
    cells = {}
    oof_rows = {}
    probabilities = {}
    for view in ("original", "grid_crops"):
        cell = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        rows = json.loads((OOF / f"v12_{view}_predictions.json").read_text(encoding="utf-8"))
        if not all(row["image_id"] in cell["ground_truth"] for row in rows):
            raise RuntimeError("OOF image identity mismatch")
        prob_path = OUT / f"{view}_class_probability.npy"
        if prob_path.is_file():
            probability = np.load(prob_path)
            if len(probability) != len(rows):
                raise RuntimeError("probability cache length mismatch")
        else:
            probability = score_candidates(rows, IMAGES, model, device)
            np.save(prob_path, probability)
        cells[view], oof_rows[view], probabilities[view] = cell, rows, probability
        baseline = metrics50(rows, cell["ground_truth"], cell["sizes"])
        result["views"][view] = {"baseline": baseline, "blends": {}}
        print(f"BASELINE {view} score={baseline['score']:.6f} map={baseline['map50']:.6f}", flush=True)
    for alpha in BLENDS:
        for view in ("original", "grid_crops"):
            rows, probability = oof_rows[view], probabilities[view]
            output = []
            for row, extra in zip(rows, probability, strict=True):
                item = dict(row)
                item["score"] = float(max(float(row["score"]), 1e-12) ** (1.0 - alpha)
                                      * max(float(extra), 1e-12) ** alpha)
                output.append(item)
            metric = metrics50(output, cells[view]["ground_truth"], cells[view]["sizes"])
            baseline = result["views"][view]["baseline"]
            result["views"][view]["blends"][str(alpha)] = {
                "score": metric["score"], "score_delta": metric["score"] - baseline["score"],
                "map50": metric["map50"], "map50_delta": metric["map50"] - baseline["map50"],
                "tp": metric["tp"], "fp": metric["fp"], "fn": metric["fn"],
                "class_ap_delta": {name: metric["per_class"][name]["ap50"] - baseline["per_class"][name]["ap50"]
                                   for name in baseline["per_class"]},
            }
            print(f"BLEND alpha={alpha} {view} score_delta={metric['score']-baseline['score']:+.6f} "
                  f"map_delta={metric['map50']-baseline['map50']:+.6f} tp={metric['tp']}", flush=True)
    (OUT / "dev_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
