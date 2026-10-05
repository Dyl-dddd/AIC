"""Strict group-OOF comparison and conservative blend of V10 and V12 ranks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import metrics50
from scripts.analyze_v12_perclass_ranker_oof import FOLDS, PERMITTED
from scripts.analyze_v16_candidate_filter import make_records, read_json, score_oof
from scripts.analyze_v9_nonlinear_oof import FEATURE_SETS, fit, matrix
from scripts.analyze_v9_support_oof import VIEW_WEIGHTS


def score_v10(records: dict, beta: float) -> dict[str, list[dict]]:
    combined = [*records["original"], *records["grid_crops"]]
    names = FEATURE_SETS["nonlinear_vf_v9"]
    probabilities = {}
    for fold in range(FOLDS):
        train = [row for row in combined if row["fold"] != fold]
        heldout = [row for row in combined if row["fold"] == fold]
        if {row["group"] for row in train} & {row["group"] for row in heldout}:
            raise ValueError("group leakage")
        model = fit(train, names, 1701 + fold)
        predictions = model.predict_proba(matrix(heldout, names))[:, 1]
        for row, value in zip(heldout, predictions, strict=True):
            probabilities[id(row)] = float(value)
        print(f"V10 fold={fold}", flush=True)
    output = {}
    for view, rows in records.items():
        values = []
        for row in rows:
            item = dict(row["prediction"])
            base = max(float(item["score"]), 1e-12)
            prob = max(probabilities[id(row)], 1e-12)
            item["score"] = base ** (1.0 - beta) * prob**beta
            values.append(item)
        output[view] = values
    return output


def blend(left: list[dict], right: list[dict], alpha: float, classes: set[str] | None = None) -> list[dict]:
    result = []
    for old, new in zip(left, right, strict=True):
        if old["image_id"] != new["image_id"] or old["category_name"] != new["category_name"] or old["bbox"] != new["bbox"]:
            raise ValueError("V10/V12 identity mismatch")
        item = dict(old)
        if classes is None or item["category_name"] in classes:
            item["score"] = max(float(old["score"]), 1e-12)**(1.0-alpha) * max(float(new["score"]), 1e-12)**alpha
        result.append(item)
    return result


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
    v10 = score_v10(records, 0.65)
    candidates = {"v12": v12, "v10_b065": v10}
    for alpha in (0.15, 0.30, 0.50):
        candidates[f"v10_blend_{alpha:.2f}"] = {view: blend(v12[view], v10[view], alpha) for view in VIEW_WEIGHTS}
    report = {"protocol": {"folds": FOLDS, "group_oof": True, "v10_beta": 0.65}, "experiments": {}}
    for name, views in candidates.items():
        report["experiments"][name] = {}
        for view, rows in views.items():
            metric = metrics50(rows, cells[view]["ground_truth"], cells[view]["sizes"])
            report["experiments"][name][view] = metric
            print(f"{name} {view}: score={metric['score']:.6f} map50={metric['map50']:.6f} tp={metric['tp']}", flush=True)
    class_delta = {}
    for class_name in PERMITTED:
        class_delta[class_name] = {
            view: report["experiments"]["v10_b065"][view]["per_class"][class_name]["ap50"]
            - report["experiments"]["v12"][view]["per_class"][class_name]["ap50"]
            for view in VIEW_WEIGHTS
        }
    report["class_ap50_v10_minus_v12"] = class_delta
    selected = {name for name, values in class_delta.items() if min(values.values()) >= 0.005}
    report["eligible_classes"] = sorted(selected)
    if selected:
        report["experiments"]["class_hybrid"] = {}
        for view in VIEW_WEIGHTS:
            rows = blend(v12[view], v10[view], 1.0, selected)
            metric = metrics50(rows, cells[view]["ground_truth"], cells[view]["sizes"])
            report["experiments"]["class_hybrid"][view] = metric
            print(f"class_hybrid {view}: score={metric['score']:.6f} map50={metric['map50']:.6f} tp={metric['tp']}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
