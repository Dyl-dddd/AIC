"""Test a localization-aware IoU regressor on V12 group-OOF candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import metrics50
from scripts.analyze_v12_perclass_ranker_oof import FOLDS, FEATURES, PERMITTED
from scripts.analyze_v16_candidate_filter import make_records, read_json, score_oof
from scripts.analyze_v9_nonlinear_oof import matrix
from scripts.analyze_v9_support_oof import VIEW_WEIGHTS
from steel_defect.geometry import box_iou


def set_iou_targets(records: dict, cells: dict) -> None:
    for view, rows in records.items():
        by_image_class = defaultdict(list)
        for row in rows:
            item = row["prediction"]
            by_image_class[(item["image_id"], item["category_name"])].append(row)
        for (image_id, class_name), group in by_image_class.items():
            boxes = np.asarray([row["prediction"]["bbox"] for row in group], dtype=np.float32)
            truth = cells[view]["ground_truth"][image_id].get(class_name, [])
            if truth:
                maxima = np.zeros(len(group), dtype=np.float32)
                for target in truth:
                    maxima = np.maximum(maxima, box_iou(np.asarray(target, dtype=np.float32), boxes))
            else:
                maxima = np.zeros(len(group), dtype=np.float32)
            for row, value in zip(group, maxima, strict=True):
                row["iou_target"] = max(0.0, (float(value) - 0.25) / 0.75)


def fit(train: list[dict], seed: int) -> XGBRegressor:
    labels = np.asarray([row["iou_target"] for row in train], dtype=np.float32)
    weights = 1.0 + 12.0 * (labels > 0.0).astype(np.float32)
    model = XGBRegressor(
        n_estimators=180,
        max_depth=3,
        learning_rate=0.035,
        min_child_weight=12.0,
        subsample=0.82,
        colsample_bytree=0.88,
        reg_alpha=0.30,
        reg_lambda=10.0,
        objective="reg:squarederror",
        tree_method="hist",
        max_bin=128,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(matrix(train, FEATURES), labels, sample_weight=weights)
    return model


def score_regressor(records: dict) -> None:
    combined = [*records["original"], *records["grid_crops"]]
    for fold in range(FOLDS):
        for class_index, class_name in enumerate(PERMITTED):
            training = [row for row in combined if row["fold"] != fold and row["class_name"] == class_name]
            heldout = [row for row in combined if row["fold"] == fold and row["class_name"] == class_name]
            if {row["group"] for row in training} & {row["group"] for row in heldout}:
                raise ValueError("group leakage")
            model = fit(training, 6201 + 101 * fold + class_index)
            scores = model.predict(matrix(heldout, FEATURES))
            for row, score in zip(heldout, scores, strict=True):
                row["iou_estimate"] = max(min(float(score), 1.0), 1e-6)
            print(f"REGRESS fold={fold} class={class_name}", flush=True)


def blend(v12: list[dict], records: list[dict], alpha: float) -> list[dict]:
    output = []
    for item, row in zip(v12, records, strict=True):
        changed = dict(item)
        old = max(float(item["score"]), 1e-12)
        estimated = max(float(row["iou_estimate"]), 1e-12)
        changed["score"] = old ** (1.0 - alpha) * estimated**alpha
        output.append(changed)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v7-cells", type=Path, required=True)
    parser.add_argument("--quality-evaluation", type=Path, required=True)
    parser.add_argument("--v9-cells", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--v12-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells, _, records = make_records(args)
    v12 = score_oof(records, read_json(args.v12_results)["decision"]["class_betas"])
    set_iou_targets(records, cells)
    score_regressor(records)
    candidates = {"v12": v12}
    for alpha in (0.15, 0.25, 0.40, 0.60):
        candidates[f"iou_blend_{alpha:.2f}"] = {
            view: blend(v12[view], records[view], alpha) for view in VIEW_WEIGHTS
        }
    report = {"protocol": {"folds": FOLDS, "group_oof": True, "target": "clipped (IoU-0.25)/0.75", "features": list(FEATURES)}, "experiments": {}}
    for name, views in candidates.items():
        report["experiments"][name] = {}
        for view, rows in views.items():
            metric = metrics50(rows, cells[view]["ground_truth"], cells[view]["sizes"])
            report["experiments"][name][view] = metric
            print(f"{name} {view}: score={metric['score']:.6f} map50={metric['map50']:.6f} tp={metric['tp']}", flush=True)
    baseline = report["experiments"]["v12"]
    for views in report["experiments"].values():
        for view, metric in views.items():
            metric["delta_vs_v12"] = {key: metric[key] - baseline[view][key] for key in ("score", "map50", "tp", "fp")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
