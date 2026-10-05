"""Evaluate simple test-time score calibration on V12 group-OOF results."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import metrics50
from scripts.analyze_v16_candidate_filter import make_records, read_json, score_oof
from scripts.analyze_v9_support_oof import VIEW_WEIGHTS


def transform(rows: list[dict], records: list[dict], method: str, alpha: float) -> list[dict]:
    by_image = defaultdict(list)
    by_image_class = defaultdict(list)
    for index, item in enumerate(rows):
        by_image[item["image_id"]].append(index)
        by_image_class[(item["image_id"], item["category_name"])].append(index)
    image_max = {name: max(float(rows[index]["score"]) for index in indices) for name, indices in by_image.items()}
    class_max = {name: max(float(rows[index]["score"]) for index in indices) for name, indices in by_image_class.items()}
    output = []
    for item, record in zip(rows, records, strict=True):
        image_id = item["image_id"]
        class_name = item["category_name"]
        score = float(item["score"])
        if method == "image_max":
            factor = max(image_max[image_id], 1e-12) ** (-alpha)
        elif method == "class_max":
            factor = max(class_max[(image_id, class_name)], 1e-12) ** (-alpha)
        elif method == "image_count":
            factor = (len(by_image[image_id]) / 200.0) ** (-alpha)
        elif method == "class_count":
            factor = (len(by_image_class[(image_id, class_name)]) / 30.0) ** (-alpha)
        elif method == "unsupported":
            support = sum(record["features"][f"{prefix}_present"] for prefix in ("vf", "p1", "p2"))
            factor = alpha if support == 0.0 else 1.0
        elif method == "all_supported":
            support = sum(record["features"][f"{prefix}_present"] for prefix in ("vf", "p1", "p2"))
            factor = alpha if support == 3.0 else 1.0
        else:
            raise ValueError(method)
        changed = dict(item)
        changed["score"] = min(max(score * factor, 1e-12), 1.0)
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
    specs = [("baseline", "baseline", 0.0)]
    for method in ("image_max", "class_max", "image_count", "class_count"):
        for alpha in (-0.30, -0.15, 0.15, 0.30):
            specs.append((f"{method}_{alpha:+.2f}", method, alpha))
    for method in ("unsupported", "all_supported"):
        for alpha in (0.70, 0.85, 1.15, 1.30):
            specs.append((f"{method}_{alpha:.2f}", method, alpha))
    report = {"experiments": {}}
    for name, method, alpha in specs:
        report["experiments"][name] = {}
        for view in VIEW_WEIGHTS:
            rows = v12[view] if method == "baseline" else transform(v12[view], records[view], method, alpha)
            metric = metrics50(rows, cells[view]["ground_truth"], cells[view]["sizes"])
            report["experiments"][name][view] = {key: metric[key] for key in ("score", "map50", "tp", "fp")}
        a = report["experiments"][name]
        print(f"{name}: original={a['original']['score']:.6f}, grid={a['grid_crops']['score']:.6f}", flush=True)
    base = report["experiments"]["baseline"]
    for metric in report["experiments"].values():
        for view in VIEW_WEIGHTS:
            metric[view]["delta_score"] = metric[view]["score"] - base[view]["score"]
            metric[view]["delta_map50"] = metric[view]["map50"] - base[view]["map50"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
