"""Evaluate membership-preserving Varifocal score transfer on exact V7 dev predictions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import load_cell
from scripts.analyze_quality_rescue import summarize
from scripts.analyze_safe_r2_crossfit import fuse_v7


MATCH_IOU = 0.70
VIEW_WEIGHTS = {"original": 488 / 788, "grid_crops": 300 / 788}
EXPECTED_V7 = {"original": 68.70285282028146, "grid_crops": 69.11447644900377}
VARIANTS = (
    "baseline",
    "match_boost_110",
    "geometric_raise",
    "max_raise",
    "probability_union",
)


def iou_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float32)
    upper_left = np.maximum(left[:, None, :2], right[None, :, :2])
    lower_right = np.minimum(left[:, None, 2:], right[None, :, 2:])
    side = np.maximum(lower_right - upper_left, 0.0)
    intersection = side[..., 0] * side[..., 1]
    left_area = np.maximum(left[:, 2] - left[:, 0], 0.0) * np.maximum(
        left[:, 3] - left[:, 1], 0.0
    )
    right_area = np.maximum(right[:, 2] - right[:, 0], 0.0) * np.maximum(
        right[:, 3] - right[:, 1], 0.0
    )
    return intersection / np.maximum(
        left_area[:, None] + right_area[None, :] - intersection, 1e-12
    )


def match_quality_scores(baseline: list[dict], quality: list[dict]) -> list[dict]:
    base_groups: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    quality_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for index, item in enumerate(baseline):
        base_groups[(item["image_id"], item["category_name"])].append((index, item))
    for item in quality:
        quality_groups[(item["image_id"], item["category_name"])].append(item)
    matches = [{"matched": False, "iou": 0.0, "quality_score": 0.0} for _ in baseline]
    for key, base_items in base_groups.items():
        quality_items = quality_groups.get(key, [])
        if not quality_items:
            continue
        left = np.asarray([item[1]["bbox"] for item in base_items], dtype=np.float32)
        right = np.asarray([item["bbox"] for item in quality_items], dtype=np.float32)
        matrix = iou_matrix(left, right)
        best = matrix.argmax(axis=1)
        overlap = matrix[np.arange(len(left)), best]
        for local_index, value in enumerate(overlap):
            if float(value) < MATCH_IOU:
                continue
            baseline_index = base_items[local_index][0]
            matches[baseline_index] = {
                "matched": True,
                "iou": float(value),
                "quality_score": float(quality_items[int(best[local_index])]["score"]),
            }
    return matches


def transfer_score(base_score: float, match: dict, variant: str) -> float:
    if variant == "baseline" or not match["matched"]:
        return base_score
    quality_score = match["quality_score"]
    if variant == "match_boost_110":
        return min(1.0, base_score * 1.10)
    if variant == "geometric_raise":
        return max(base_score, math.sqrt(max(base_score * quality_score, 0.0)))
    if variant == "max_raise":
        return max(base_score, quality_score)
    if variant == "probability_union":
        return 1.0 - (1.0 - base_score) * (1.0 - quality_score)
    raise ValueError(variant)


def apply_variant(baseline: list[dict], matches: list[dict], variant: str) -> list[dict]:
    output = []
    for item, match in zip(baseline, matches):
        updated = dict(item)
        updated["score"] = transfer_score(float(item["score"]), match, variant)
        output.append(updated)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = {variant: {"views": {}} for variant in VARIANTS}
    diagnostics = {}
    for view in ("original", "grid_crops"):
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        baseline_metrics = summarize(baseline, cell_a["ground_truth"], cell_a["sizes"])
        if not math.isclose(
            float(baseline_metrics["score_proxy_20_60_20"]), EXPECTED_V7[view],
            rel_tol=0.0, abs_tol=1e-10,
        ):
            raise ValueError(f"exact V7 baseline mismatch in {view}: {baseline_metrics}")
        quality = json.loads((
            args.quality_evaluation / f"varifocal/varifocal_{view}_predictions.json"
        ).read_text())
        matches = match_quality_scores(baseline, quality)
        diagnostics[view] = {
            "baseline_predictions": len(baseline),
            "quality_predictions": len(quality),
            "matched_predictions": sum(item["matched"] for item in matches),
            "membership_and_coordinates_unchanged": True,
        }
        for variant in VARIANTS:
            predictions = apply_variant(baseline, matches, variant)
            summary = summarize(predictions, cell_a["ground_truth"], cell_a["sizes"])
            records[variant]["views"][view] = summary
            print(
                f"{view} {variant}: score={summary['score_proxy_20_60_20']:.9f} "
                f"map50={summary['map50_macro']:.9f}", flush=True,
            )
    baseline_weighted = sum(
        VIEW_WEIGHTS[view]
        * records["baseline"]["views"][view]["score_proxy_20_60_20"]
        for view in VIEW_WEIGHTS
    )
    table = []
    for variant in VARIANTS:
        weighted = sum(
            VIEW_WEIGHTS[view]
            * records[variant]["views"][view]["score_proxy_20_60_20"]
            for view in VIEW_WEIGHTS
        )
        row = {
            "variant": variant,
            "weighted_score": weighted,
            "weighted_delta": weighted - baseline_weighted,
            "views": records[variant]["views"],
        }
        table.append(row)
    table.sort(key=lambda row: row["weighted_score"], reverse=True)
    result = {
        "match_iou": MATCH_IOU,
        "diagnostics": diagnostics,
        "baseline_weighted": baseline_weighted,
        "ranking": table,
        "warning": "Frozen-development ablation only; no Final/Test authorization.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "baseline_weighted": baseline_weighted,
        "ranking": [{
            "variant": row["variant"],
            "weighted_score": row["weighted_score"],
            "weighted_delta": row["weighted_delta"],
            "original": row["views"]["original"]["score_proxy_20_60_20"],
            "grid_crops": row["views"]["grid_crops"]["score_proxy_20_60_20"],
        } for row in table],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
