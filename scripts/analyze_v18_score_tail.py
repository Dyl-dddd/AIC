"""Fast group-OOF audit of score-tail removal on frozen V12 candidates."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import metrics50
from scripts.analyze_v16_candidate_filter import make_records, read_json, score_oof
from scripts.analyze_v9_support_oof import VIEW_WEIGHTS


def filter_rows(rows: list[dict], q: float, per_class: bool) -> list[dict]:
    if per_class:
        scores = defaultdict(list)
        for row in rows:
            scores[row["category_name"]].append(float(row["score"]))
        floors = {name: float(np.quantile(values, q)) for name, values in scores.items()}
        return [row for row in rows if float(row["score"]) >= floors[row["category_name"]]]
    floor = float(np.quantile([float(row["score"]) for row in rows], q))
    return [row for row in rows if float(row["score"]) >= floor]


def shared_floor(scored: dict[str, list[dict]], q: float, weighted: bool) -> float:
    score_parts = []
    weight_parts = []
    for view, rows in scored.items():
        values = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        score_parts.append(values)
        weight_parts.append(np.full(len(values), VIEW_WEIGHTS[view] / len(values), dtype=np.float64))
    scores = np.concatenate(score_parts)
    if not weighted:
        return float(np.quantile(scores, q))
    weights = np.concatenate(weight_parts)
    order = np.argsort(scores, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(scores[order[np.searchsorted(cumulative, q)]])


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
    scored = score_oof(records, read_json(args.v12_results)["decision"]["class_betas"])
    configs = [("baseline", 0.0, "baseline")]
    for q in (0.01, 0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.30, 0.50):
        configs.append((f"per_view_tail_{q:.2f}", q, "per_view"))
    for q in (0.05, 0.10, 0.12, 0.14, 0.16, 0.18):
        configs.append((f"pooled_tail_{q:.2f}", q, "pooled"))
        configs.append((f"weighted_tail_{q:.2f}", q, "weighted"))
    for q in (0.05, 0.10, 0.20):
        configs.append((f"class_tail_{q:.2f}", q, "class"))
    report = {"experiments": {}}
    for name, q, mode in configs:
        report["experiments"][name] = {}
        floor = shared_floor(scored, q, weighted=(mode == "weighted")) if mode in {"pooled", "weighted"} else None
        for view in VIEW_WEIGHTS:
            if mode == "baseline":
                rows = scored[view]
            elif floor is not None:
                rows = [row for row in scored[view] if float(row["score"]) >= floor]
            else:
                rows = filter_rows(scored[view], q, mode == "class")
            metric = metrics50(rows, cells[view]["ground_truth"], cells[view]["sizes"])
            report["experiments"][name][view] = {key: metric[key] for key in ("score", "map50", "tp", "fp", "predictions")}
        a = report["experiments"][name]
        print(name, {view: (round(a[view]["score"], 6), a[view]["tp"], a[view]["predictions"]) for view in VIEW_WEIGHTS}, flush=True)
    baseline = report["experiments"]["baseline"]
    for metrics in report["experiments"].values():
        for view, values in metrics.items():
            values["delta_score"] = values["score"] - baseline[view]["score"]
            values["delta_tp"] = values["tp"] - baseline[view]["tp"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
