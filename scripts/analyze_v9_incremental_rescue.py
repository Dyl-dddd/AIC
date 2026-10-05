"""Evaluate conservative V9 consensus rescues on top of the frozen V7 detector.

This is a development-only analysis.  It never reads Test labels or writes a
submission.  V9 predictions may add low-ranked boxes, but the V7 membership,
coordinates, and ordering are left unchanged.
"""

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

from scripts.analyze_final_submission_postprocess import load_cell, metrics50
from scripts.analyze_safe_r2_crossfit import fuse_v7
from steel_defect.geometry import box_iou


VIEWS = ("original", "grid_crops")
VIEW_WEIGHTS = {"original": 488 / 788, "grid_crops": 300 / 788}
SCORE_FLOORS = (0.001, 0.003, 0.01, 0.03)
SUPPORT_IOUS = (0.30, 0.50, 0.70)
BASE_EXCLUSIONS = (0.30, 0.50, 0.70)
TOPKS = (1, 2)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def group_predictions(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["image_id"], row["category_name"])].append(row)
    return grouped


def max_iou(box: list[float], rows: list[dict]) -> float:
    if not rows:
        return 0.0
    overlaps = box_iou(
        np.asarray(box, dtype=np.float32),
        np.asarray([row["bbox"] for row in rows], dtype=np.float32),
    )
    return float(np.max(overlaps))


def best_match(box: list[float], rows: list[dict]) -> tuple[float, dict | None]:
    if not rows:
        return 0.0, None
    overlaps = box_iou(
        np.asarray(box, dtype=np.float32),
        np.asarray([row["bbox"] for row in rows], dtype=np.float32),
    )
    index = int(np.argmax(overlaps))
    return float(overlaps[index]), rows[index]


def build_features(
    baseline: list[dict], p1: list[dict], p2: list[dict], ground_truth: dict,
) -> list[dict]:
    base_groups = group_predictions(baseline)
    p2_groups = group_predictions(p2)
    output = []
    for row in p1:
        key = row["image_id"], row["category_name"]
        support_iou, support = best_match(row["bbox"], p2_groups.get(key, []))
        base_iou = max_iou(row["bbox"], base_groups.get(key, []))
        gt_rows = [
            {"bbox": box}
            for box in ground_truth[row["image_id"]].get(row["category_name"], [])
        ]
        gt_iou = max_iou(row["bbox"], gt_rows)
        p1_score = max(float(row["score"]), 1e-12)
        p2_score = max(float(support["score"]), 1e-12) if support else 1e-12
        output.append({
            "prediction": row,
            "support_prediction": support,
            "p1_score": p1_score,
            "p2_score": p2_score,
            "joint_score": math.sqrt(p1_score * p2_score),
            "support_iou": support_iou,
            "base_iou": base_iou,
            "gt_iou": gt_iou,
            "label": int(gt_iou >= 0.50),
        })
    return output


def nms_features(rows: list[dict], limit: int, iou: float = 0.50) -> list[dict]:
    kept: list[dict] = []
    for row in sorted(rows, key=lambda item: item["joint_score"], reverse=True):
        if any(
            other["prediction"]["category_name"] == row["prediction"]["category_name"]
            and max_iou(row["prediction"]["bbox"], [other["prediction"]]) > iou
            for other in kept
        ):
            continue
        kept.append(row)
        if len(kept) >= limit:
            break
    return kept


def select_rescues(
    features: list[dict], score_floor: float, support_iou: float,
    base_exclusion: float, topk: int,
) -> list[dict]:
    eligible = [
        row for row in features
        if row["joint_score"] >= score_floor
        and row["support_iou"] >= support_iou
        and row["base_iou"] < base_exclusion
    ]
    by_image: dict[str, list[dict]] = defaultdict(list)
    for row in eligible:
        by_image[row["prediction"]["image_id"]].append(row)
    selected = []
    for rows in by_image.values():
        selected.extend(nms_features(rows, topk))
    return selected


def append_low_ranked(baseline: list[dict], selected: list[dict]) -> list[dict]:
    minimum = min(float(row["score"]) for row in baseline)
    ordered = sorted(selected, key=lambda row: row["joint_score"], reverse=True)
    denominator = max(len(ordered), 1)
    rescues = []
    for rank, row in enumerate(ordered):
        prediction = dict(row["prediction"])
        # Strictly below every protected baseline score while retaining rescue order.
        prediction["score"] = minimum * (0.49 - 0.48 * rank / denominator)
        rescues.append(prediction)
    return [*baseline, *rescues]


def config_key(config: tuple[float, float, float, int]) -> str:
    floor, support, exclusion, topk = config
    return f"f{floor:g}_s{support:.2f}_b{exclusion:.2f}_k{topk}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--v9-cells", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configs = [
        (floor, support, exclusion, topk)
        for floor in SCORE_FLOORS
        for support in SUPPORT_IOUS
        for exclusion in BASE_EXCLUSIONS
        for topk in TOPKS
    ]
    view_records = {}
    for view in VIEWS:
        print(f"LOAD {view}", flush=True)
        cell_a = load_cell(args.v7_cells / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(args.v7_cells / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        p1 = read_json(args.v9_cells / f"v9_model_b_quality_p1_last_9a4ad2b8fe_{view}" / "predictions.json")
        p2 = read_json(args.v9_cells / f"v9_model_b_quality_p2_last_224153cc09_{view}" / "predictions.json")
        features = build_features(baseline, p1, p2, cell_a["ground_truth"])
        baseline_metrics = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        print(
            f"FEATURES {view}: base={len(baseline)} p1={len(p1)} p2={len(p2)} "
            f"v9_tp_candidates={sum(row['label'] for row in features)}",
            flush=True,
        )
        results = {}
        for number, config in enumerate(configs, start=1):
            selected = select_rescues(features, *config)
            predictions = append_low_ranked(baseline, selected)
            metric = metrics50(predictions, cell_a["ground_truth"], cell_a["sizes"])
            metric["gain"] = metric["score"] - baseline_metrics["score"]
            metric["rescue_count"] = len(selected)
            metric["rescue_labeled_positive"] = sum(row["label"] for row in selected)
            results[config_key(config)] = metric
            if number % 12 == 0:
                print(f"EVAL {view}: {number}/{len(configs)}", flush=True)
        view_records[view] = {
            "baseline": baseline_metrics,
            "results": results,
            "feature_summary": {
                "p1_predictions": len(p1),
                "p2_predictions": len(p2),
                "p1_gt_overlaps": sum(row["label"] for row in features),
                "consensus_iou_050": sum(row["support_iou"] >= 0.50 for row in features),
            },
        }

    ranking = []
    for config in configs:
        key = config_key(config)
        views = {view: view_records[view]["results"][key] for view in VIEWS}
        ranking.append({
            "config": key,
            "weighted_gain": sum(VIEW_WEIGHTS[v] * views[v]["gain"] for v in VIEWS),
            "robust_gain": min(views[v]["gain"] for v in VIEWS),
            "views": views,
        })
    ranking.sort(key=lambda row: (row["robust_gain"], row["weighted_gain"]), reverse=True)
    payload = {
        "status": "analysis_only",
        "method": "V9 P1/P2 consensus rescue below all protected V7 scores",
        "view_weights": VIEW_WEIGHTS,
        "views": view_records,
        "ranking": ranking,
        "best_robust": ranking[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(ranking[0], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
