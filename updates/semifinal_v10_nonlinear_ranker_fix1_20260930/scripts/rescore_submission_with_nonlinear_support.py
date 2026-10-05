"""Membership-preserving XGBoost score transfer from VF and V9 support."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.rescore_submission_with_varifocal import iou_matrix, load_predictions
from steel_defect.classes import CLASS_NAMES
from steel_defect.runtime import sha256_file


PERMITTED = [name for name in CLASS_NAMES if name != "qilie"]
MATCH_FLOOR = 0.50


def identity_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in rows:
        payload = [item["image_id"], item["category_name"], item["bbox"]]
        digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def add_support(features: list[dict], baseline: list[dict], quality: list[dict], prefix: str) -> None:
    base_groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    quality_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for index, item in enumerate(baseline):
        base_groups[(str(item["image_id"]), str(item["category_name"]))].append((index, item))
    for item in quality:
        quality_groups[(str(item["image_id"]), str(item["category_name"]))].append(item)
    for key, base_items in base_groups.items():
        candidates = quality_groups.get(key, [])
        if not candidates:
            continue
        left = np.asarray([item[1]["bbox"] for item in base_items], dtype=np.float32)
        right = np.asarray([item["bbox"] for item in candidates], dtype=np.float32)
        matrix = iou_matrix(left, right)
        best = matrix.argmax(axis=1)
        values = matrix[np.arange(len(left)), best]
        for local, overlap in enumerate(values):
            if float(overlap) < MATCH_FLOOR:
                continue
            target = base_items[local][0]
            base_score = max(float(baseline[target]["score"]), 1e-12)
            score = max(float(candidates[int(best[local])]["score"]), 1e-12)
            features[target].update({
                f"{prefix}_present": 1.0,
                f"{prefix}_iou": float(overlap),
                f"log_{prefix}_score": math.log(score),
                f"{prefix}_log_advantage": math.log(score) - math.log(base_score),
            })


def image_sizes(source: Path, image_ids: set[str]) -> dict[str, tuple[int, int]]:
    paths: dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_file() and path.name in image_ids:
            if path.name in paths:
                raise ValueError(f"duplicate source image name: {path.name}")
            paths[path.name] = path
    missing = image_ids - set(paths)
    if missing:
        raise ValueError(f"missing source images: {sorted(missing)[:5]}")
    output = {}
    for name, path in paths.items():
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot read source image: {path}")
        height, width = image.shape[:2]
        output[name] = (width, height)
    return output


def add_context(features: list[dict], baseline: list[dict], sizes: dict[str, tuple[int, int]]) -> None:
    by_image: dict[str, list[int]] = defaultdict(list)
    by_image_class: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(baseline):
        image_id = str(row["image_id"])
        class_name = str(row["category_name"])
        by_image[image_id].append(index)
        by_image_class[(image_id, class_name)].append(index)
    image_rank, class_rank = {}, {}
    for indices in by_image.values():
        ordered = sorted(indices, key=lambda i: float(baseline[i]["score"]), reverse=True)
        for rank, index in enumerate(ordered):
            image_rank[index] = rank / max(len(ordered) - 1, 1)
    for indices in by_image_class.values():
        ordered = sorted(indices, key=lambda i: float(baseline[i]["score"]), reverse=True)
        for rank, index in enumerate(ordered):
            class_rank[index] = rank / max(len(ordered) - 1, 1)
    for index, row in enumerate(baseline):
        image_id = str(row["image_id"])
        class_name = str(row["category_name"])
        width, height = sizes[image_id]
        x1, y1, x2, y2 = map(float, row["bbox"])
        box_width = max(x2 - x1, 1.0)
        box_height = max(y2 - y1, 1.0)
        width_ratio = box_width / max(width, 1)
        height_ratio = box_height / max(height, 1)
        center_x = (x1 + x2) / (2.0 * max(width, 1))
        center_y = (y1 + y2) / (2.0 * max(height, 1))
        features[index].update({
            "log_width_ratio": math.log(max(width_ratio, 1e-8)),
            "log_height_ratio": math.log(max(height_ratio, 1e-8)),
            "log_area_ratio": math.log(max(width_ratio * height_ratio, 1e-12)),
            "log_aspect": math.log(max(box_width / box_height, 1e-8)),
            "center_x": center_x,
            "center_y": center_y,
            "edge_distance": min(center_x, 1.0 - center_x, center_y, 1.0 - center_y),
            "image_rank": image_rank[index],
            "image_class_rank": class_rank[index],
            "log_image_count": math.log(len(by_image[image_id])),
            "log_image_class_count": math.log(len(by_image_class[(image_id, class_name)])),
        })
        for name in PERMITTED:
            features[index][f"class_{name}"] = float(class_name == name)


def build(
    baseline_path: Path, vf_path: Path, p1_path: Path, p2_path: Path,
    source: Path, model_path: Path, config_path: Path,
    output_path: Path, report_path: Path,
) -> dict:
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite output or report")
    baseline = load_predictions(baseline_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    deployment = config["deployment"]
    names = tuple(deployment["feature_names"])
    blend = float(deployment["blend"])
    if sha256_file(model_path) != deployment["model_sha256"]:
        raise ValueError("deployment model SHA256 mismatch")
    features = []
    for row in baseline:
        base = max(float(row["score"]), 1e-12)
        feature = {"log_base_score": math.log(base)}
        for prefix in ("vf", "p1", "p2"):
            feature.update({
                f"{prefix}_present": 0.0,
                f"{prefix}_iou": 0.0,
                f"log_{prefix}_score": math.log(1e-12),
                f"{prefix}_log_advantage": math.log(1e-12) - math.log(base),
            })
        features.append(feature)
    for path, prefix in ((vf_path, "vf"), (p1_path, "p1"), (p2_path, "p2")):
        add_support(features, baseline, load_predictions(path), prefix)
    sizes = image_sizes(source, {str(row["image_id"]) for row in baseline})
    add_context(features, baseline, sizes)
    matrix = np.asarray([[row[name] for name in names] for row in features], dtype=np.float32)
    model = XGBClassifier()
    model.load_model(model_path)
    probabilities = model.predict_proba(matrix)[:, 1]
    output = []
    for row, probability in zip(baseline, probabilities, strict=True):
        item = dict(row)
        base = max(float(row["score"]), 1e-12)
        calibrated = max(float(probability), 1e-12)
        item["score"] = base ** (1.0 - blend) * calibrated ** blend
        output.append(item)
    before = identity_digest(baseline)
    after = identity_digest(output)
    if before != after:
        raise RuntimeError("rescoring changed membership or geometry")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    report = {
        "predictions": len(output),
        "identity_sha256": before,
        "membership_and_geometry_preserved": True,
        "blend": blend,
        "model_sha256": sha256_file(model_path),
        "config_sha256": sha256_file(config_path),
        "baseline_sha256": sha256_file(baseline_path),
        "vf_sha256": sha256_file(vf_path),
        "p1_sha256": sha256_file(p1_path),
        "p2_sha256": sha256_file(p2_path),
        "output_sha256": sha256_file(output_path),
        "probability": {
            "minimum": float(np.min(probabilities)),
            "median": float(np.median(probabilities)),
            "maximum": float(np.max(probabilities)),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--vf", type=Path, required=True)
    parser.add_argument("--p1", type=Path, required=True)
    parser.add_argument("--p2", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.baseline, args.vf, args.p1, args.p2, args.source,
        args.model, args.config, args.output, args.report,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
