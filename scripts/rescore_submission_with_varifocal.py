"""Rescore a frozen V7 submission with a frozen Varifocal quality profile.

The operation is membership and geometry preserving: only ``score`` changes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from steel_defect.runtime import sha256_file


MATCH_FLOOR = 0.50


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    top_left = np.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    wh = np.maximum(bottom_right - top_left, 0.0)
    intersection = wh[..., 0] * wh[..., 1]
    left_area = np.maximum(left[:, 2] - left[:, 0], 0.0) * np.maximum(
        left[:, 3] - left[:, 1], 0.0
    )
    right_area = np.maximum(right[:, 2] - right[:, 0], 0.0) * np.maximum(
        right[:, 3] - right[:, 1], 0.0
    )
    return intersection / np.maximum(
        left_area[:, None] + right_area[None, :] - intersection, 1e-12
    )


def load_predictions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"submission must be a JSON array: {path}")
    required = {"image_id", "category_name", "bbox", "score"}
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or not required.issubset(item):
            raise ValueError(f"invalid prediction at index {index}: {path}")
        box = item["bbox"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"invalid bbox at index {index}: {path}")
    return payload


def deployment_config(path: Path) -> tuple[dict, float, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    deployment = payload.get("deployment", payload)
    profile = deployment.get("profile", deployment)
    blend = float(deployment.get("blend", 0.75))
    match_floor = float(payload.get("protocol", {}).get("match_floor", MATCH_FLOOR))
    names = profile.get("names")
    expected = [
        "log_base_score", "vf_present", "vf_iou", "log_vf_score",
        "vf_log_advantage",
    ]
    if names != expected:
        raise ValueError(f"unexpected deployment features: {names}")
    if not 0.0 <= blend <= 1.0 or not 0.0 <= match_floor <= 1.0:
        raise ValueError("invalid blend or match floor")
    return profile, blend, match_floor


def probability(features: dict[str, float], profile: dict) -> float:
    vector = np.asarray([features[name] for name in profile["names"]], dtype=np.float64)
    mean = np.asarray(profile["mean"], dtype=np.float64)
    scale = np.asarray(profile["scale"], dtype=np.float64)
    coefficients = np.asarray(profile["coefficients"], dtype=np.float64)
    if not (len(vector) == len(mean) == len(scale) == len(coefficients)):
        raise ValueError("profile vector lengths differ")
    logit = float(profile["intercept"]) + float(((vector - mean) / scale) @ coefficients)
    logit = max(min(logit, 40.0), -40.0)
    return 1.0 / (1.0 + math.exp(-logit))


def identity_digest(rows: Iterable[dict]) -> str:
    digest = hashlib.sha256()
    for item in rows:
        payload = [item["image_id"], item["category_name"], item["bbox"]]
        digest.update(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def rescore(
    baseline: list[dict], quality: list[dict], profile: dict, blend: float,
    match_floor: float,
) -> tuple[list[dict], dict]:
    quality_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    baseline_groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    for item in quality:
        quality_groups[(str(item["image_id"]), str(item["category_name"]))].append(item)
    for index, item in enumerate(baseline):
        baseline_groups[(str(item["image_id"]), str(item["category_name"]))].append((index, item))

    evidence = [(0.0, 0.0, 1e-12) for _ in baseline]
    for key, base_items in baseline_groups.items():
        candidates = quality_groups.get(key, [])
        if not candidates:
            continue
        left = np.asarray([item[1]["bbox"] for item in base_items], dtype=np.float32)
        right = np.asarray([item["bbox"] for item in candidates], dtype=np.float32)
        overlaps = iou_matrix(left, right)
        best = overlaps.argmax(axis=1)
        values = overlaps[np.arange(len(left)), best]
        for local, overlap in enumerate(values):
            if float(overlap) < match_floor:
                continue
            target = base_items[local][0]
            evidence[target] = (1.0, float(overlap), float(candidates[int(best[local])]["score"]))

    output, matched, changed = [], 0, 0
    ratios = []
    for item, (present, overlap, vf_score) in zip(baseline, evidence, strict=True):
        base_score = max(float(item["score"]), 1e-12)
        vf_score = max(float(vf_score), 1e-12)
        features = {
            "log_base_score": math.log(base_score),
            "vf_present": present,
            "vf_iou": overlap,
            "log_vf_score": math.log(vf_score),
            "vf_log_advantage": math.log(vf_score) - math.log(base_score),
        }
        calibrated = max(probability(features, profile), 1e-12)
        new_score = base_score ** (1.0 - blend) * calibrated ** blend
        row = dict(item)
        row["score"] = new_score
        output.append(row)
        matched += int(present)
        changed += int(new_score != float(item["score"]))
        ratios.append(new_score / base_score)

    before_identity = identity_digest(baseline)
    after_identity = identity_digest(output)
    if before_identity != after_identity or len(output) != len(baseline):
        raise RuntimeError("rescoring changed membership or geometry")
    return output, {
        "predictions": len(output),
        "matched_predictions": matched,
        "unmatched_predictions": len(output) - matched,
        "scores_changed": changed,
        "identity_sha256": before_identity,
        "score_ratio": {
            "minimum": float(np.min(ratios)) if ratios else None,
            "median": float(np.median(ratios)) if ratios else None,
            "maximum": float(np.max(ratios)) if ratios else None,
        },
    }


def build(
    baseline_path: Path, quality_path: Path, profile_path: Path,
    output_path: Path, report_path: Path,
) -> dict:
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite output or report")
    baseline = load_predictions(baseline_path)
    quality = load_predictions(quality_path)
    profile, blend, match_floor = deployment_config(profile_path)
    output, audit = rescore(baseline, quality, profile, blend, match_floor)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    report = {
        **audit,
        "baseline": str(baseline_path.resolve()),
        "baseline_sha256": sha256_file(baseline_path),
        "quality": str(quality_path.resolve()),
        "quality_sha256": sha256_file(quality_path),
        "profile": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
        "blend": blend,
        "match_floor": match_floor,
        "output": str(output_path.resolve()),
        "output_sha256": sha256_file(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--quality", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.baseline, args.quality, args.profile, args.output, args.report
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
