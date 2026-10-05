"""Audit stage-2 results and run bounded CPU-only score diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stage1_cache import metrics50
from scripts.run_semifinal_stage1 import BASE_SHA, NEW_SHA, SPLIT_SHA, write_json
from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import classwise_nms
from steel_defect.metrics import _match_class, average_precision


PERMITTED = [name for name in CLASS_NAMES if name != "qilie"]
THRESHOLDS = [
    0.0001, 0.00015, 0.0002, 0.0003, 0.0005, 0.0007, 0.001,
    0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1,
    0.2, 0.3, 0.5,
]


def load_cell(folder: Path) -> dict:
    report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    predictions = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
    with gzip.open(folder / "candidates.jsonl.gz", "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        rows = [json.loads(line) for line in handle]
    if header["signature"] != report["signature"]:
        raise ValueError(f"cache signature mismatch: {folder.name}")
    ground_truth = {row["image_id"]: row["ground_truth"] for row in rows}
    sizes = {row["image_id"]: tuple(row["image_size"]) for row in rows}
    return {
        "report": report,
        "predictions": predictions,
        "ground_truth": ground_truth,
        "sizes": sizes,
        "rows": rows,
    }


def filter_by_class_thresholds(predictions: list[dict], thresholds: dict[str, float]) -> list[dict]:
    return [p for p in predictions if float(p["score"]) >= thresholds[p["category_name"]]]


def class_threshold_search(predictions: list[dict], ground_truth: dict, sizes: dict) -> dict:
    per_class: dict[str, list[dict]] = {}
    for name in PERMITTED:
        values = []
        class_predictions = [p for p in predictions if p["category_name"] == name]
        for threshold in THRESHOLDS:
            selected = [p for p in class_predictions if float(p["score"]) >= threshold]
            tp, fp, count = _match_class(selected, ground_truth, name, 0.5)
            values.append({
                "threshold": threshold,
                "tp": int(tp.sum()),
                "fp": int(fp.sum()),
                "gt": count,
                "ap50": average_precision(tp, fp, count),
            })
        per_class[name] = values

    def score(indices: dict[str, int]) -> float:
        chosen = [per_class[name][indices[name]] for name in PERMITTED]
        tp = sum(item["tp"] for item in chosen)
        fp = sum(item["fp"] for item in chosen)
        gt = sum(item["gt"] for item in chosen)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(gt, 1)
        map50 = float(np.mean([item["ap50"] for item in chosen]))
        return 20 * precision + 60 * recall + 20 * map50

    starts = [0, 3, 6, 10, 13]
    best_indices = None
    best_score = -1.0
    for start in starts:
        indices = {name: start for name in PERMITTED}
        for _ in range(12):
            changed = False
            for name in PERMITTED:
                current = indices[name]
                candidates = []
                for index in range(len(THRESHOLDS)):
                    indices[name] = index
                    candidates.append((score(indices), -THRESHOLDS[index], index))
                selected = max(candidates)[2]
                indices[name] = selected
                changed |= selected != current
            if not changed:
                break
        candidate_score = score(indices)
        if candidate_score > best_score:
            best_score = candidate_score
            best_indices = indices.copy()

    assert best_indices is not None
    thresholds = {name: THRESHOLDS[best_indices[name]] for name in PERMITTED}
    selected = filter_by_class_thresholds(predictions, thresholds)
    result = metrics50(selected, ground_truth, sizes)
    return {
        "thresholds": thresholds,
        "result": result,
        "grid_score": best_score,
        "search_grid": THRESHOLDS,
    }


def class_routing_search(variants: dict[str, list[dict]], ground_truth: dict, sizes: dict) -> dict:
    """Jointly select a bounded prediction source and threshold per class."""
    options: dict[str, list[dict]] = {}
    for name in PERMITTED:
        class_options = []
        for variant_name, predictions in variants.items():
            class_predictions = [p for p in predictions if p["category_name"] == name]
            for threshold in THRESHOLDS:
                selected = [p for p in class_predictions if float(p["score"]) >= threshold]
                tp, fp, count = _match_class(selected, ground_truth, name, 0.5)
                class_options.append({
                    "variant": variant_name,
                    "threshold": threshold,
                    "tp": int(tp.sum()),
                    "fp": int(fp.sum()),
                    "gt": count,
                    "ap50": average_precision(tp, fp, count),
                })
        options[name] = class_options

    def score(indices: dict[str, int]) -> float:
        chosen = [options[name][indices[name]] for name in PERMITTED]
        tp = sum(item["tp"] for item in chosen)
        fp = sum(item["fp"] for item in chosen)
        gt = sum(item["gt"] for item in chosen)
        return (
            20 * tp / max(tp + fp, 1)
            + 60 * tp / max(gt, 1)
            + 20 * float(np.mean([item["ap50"] for item in chosen]))
        )

    best_indices = None
    best_score = -1.0
    variant_names = list(variants)
    for start_variant in variant_names:
        start_index = variant_names.index(start_variant) * len(THRESHOLDS)
        indices = {name: start_index for name in PERMITTED}
        for _ in range(12):
            changed = False
            for name in PERMITTED:
                current = indices[name]
                candidates = []
                for index, option in enumerate(options[name]):
                    indices[name] = index
                    candidates.append((score(indices), -option["threshold"], index))
                selected = max(candidates)[2]
                indices[name] = selected
                changed |= selected != current
            if not changed:
                break
        candidate_score = score(indices)
        if candidate_score > best_score:
            best_score = candidate_score
            best_indices = indices.copy()

    assert best_indices is not None
    selection = {name: options[name][best_indices[name]] for name in PERMITTED}
    predictions = []
    for name, choice in selection.items():
        predictions.extend(
            prediction for prediction in variants[choice["variant"]]
            if prediction["category_name"] == name
            and float(prediction["score"]) >= choice["threshold"]
        )
    return {
        "selection": selection,
        "result": metrics50(predictions, ground_truth, sizes),
        "grid_score": best_score,
        "variant_count": len(variants),
    }


def ensemble_predictions(
    first: list[dict], second: list[dict], second_scale: float, nms_iou: float,
) -> list[dict]:
    grouped: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for prediction in first:
        grouped[prediction["image_id"]].append((prediction, 1.0))
    for prediction in second:
        grouped[prediction["image_id"]].append((prediction, second_scale))
    output = []
    class_ids = {name: index for index, name in enumerate(PERMITTED)}
    for image_id, rows in grouped.items():
        boxes = np.asarray([row[0]["bbox"] for row in rows], dtype=np.float32)
        scores = np.asarray([min(1.0, float(row[0]["score"]) * row[1]) for row in rows], dtype=np.float32)
        classes = np.asarray([class_ids[row[0]["category_name"]] for row in rows], dtype=np.int64)
        for index in classwise_nms(boxes, scores, classes, nms_iou):
            output.append({
                "image_id": image_id,
                "category_name": PERMITTED[int(classes[index])],
                "bbox": [int(value) for value in boxes[index]],
                "score": round(float(scores[index]), 7),
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    preflight = json.loads((args.results / "preflight.json").read_text(encoding="utf-8"))
    if not (
        preflight["scope"] == "complete_dev"
        and preflight["stage"] == 2
        and preflight["source_bytes_verified"]
        and not preflight["training_started"]
        and preflight["split_sha256"] == SPLIT_SHA
        and preflight["baseline_sha256"] == BASE_SHA
        and preflight["candidate_sha256"] == NEW_SHA
    ):
        raise ValueError("preflight evidence mismatch")

    comparison = json.loads((args.results / "comparison.json").read_text(encoding="utf-8"))
    cells = {}
    audit = {}
    threshold_results = {}
    for folder in sorted(path for path in args.results.iterdir() if path.is_dir()):
        cell = load_cell(folder)
        report = cell["report"]
        if comparison[folder.name] != report or not report["complete"]:
            raise ValueError(f"incomplete or mismatched report: {folder.name}")
        if report["sources"] != 459 or report["signature"]["limit"] != 0:
            raise ValueError(f"not complete dev: {folder.name}")
        reproduced = metrics50(cell["predictions"], cell["ground_truth"], cell["sizes"])
        uploaded = report["all_exported"]["summary"]
        if abs(reproduced["score"] - uploaded["score_proxy_20_60_20"]) > 1e-9:
            raise ValueError(f"score mismatch: {folder.name}")
        audit[folder.name] = {
            "verified": True,
            "weights_sha256": report["signature"]["weights_sha256"],
            "sources": report["sources"],
            "views": report["views"],
            "predictions": report["predictions"],
            "seconds": report["seconds"],
            "all_exported": reproduced,
        }
        threshold_results[folder.name] = class_threshold_search(
            cell["predictions"], cell["ground_truth"], cell["sizes"]
        )
        cells[folder.name] = cell
        print(f"AUDIT {folder.name}: {reproduced['score']:.6f}", flush=True)

    ensemble_specs = {
        "original_old1024_plus_new1280": (
            "old_sliding_original_s1024_low", "new_semifinal_grid_original_s1280_low"
        ),
        "crops_old1024_plus_new1024": (
            "old_sliding_grid_crops_s1024_low", "new_semifinal_grid_grid_crops_s1024_low"
        ),
    }
    ensemble_results = {}
    for label, (first_name, second_name) in ensemble_specs.items():
        first, second = cells[first_name], cells[second_name]
        if first["ground_truth"] != second["ground_truth"] or first["sizes"] != second["sizes"]:
            raise ValueError(f"ensemble views differ: {label}")
        variants = []
        variant_predictions = {"first": first["predictions"], "second": second["predictions"]}
        best = None
        for scale in (0.5, 1.0, 2.0):
            for iou in (0.35, 0.55, 0.75):
                predictions = ensemble_predictions(
                    first["predictions"], second["predictions"], scale, iou
                )
                result = metrics50(predictions, first["ground_truth"], first["sizes"])
                record = {
                    "second_score_scale": scale,
                    "nms_iou": iou,
                    "result": result,
                }
                variants.append(record)
                variant_predictions[f"ensemble_scale{scale:g}_iou{iou:g}"] = predictions
                if best is None or result["score"] > best[0]:
                    best = (result["score"], predictions, record)
        assert best is not None
        ensemble_results[label] = {
            "first": first_name,
            "second": second_name,
            "variants": variants,
            "best_raw": best[2],
            "best_class_thresholds": class_threshold_search(
                best[1], first["ground_truth"], first["sizes"]
            ),
            "class_routing": class_routing_search(
                variant_predictions, first["ground_truth"], first["sizes"]
            ),
        }
        print(f"ENSEMBLE {label}: {best[0]:.6f}", flush=True)

    write_json(args.output / "audit.json", audit)
    write_json(args.output / "class_thresholds.json", threshold_results)
    write_json(args.output / "ensemble_diagnostics.json", ensemble_results)

    lines = [
        "# 第二阶段完整审计与决策分析", "",
        "计分：`20×P_micro + 60×R_micro + 20×mAP50_macro`。以下均为冻结开发集本地参考分，不是官方 Test 成绩。", "",
        "## 原始结果", "",
        "|单元|视图|全导出P|全导出R|AP50|参考分|相对旧原图|", "|---|---|---:|---:|---:|---:|---:|",
    ]
    baseline = audit["old_sliding_original_s1024_low"]["all_exported"]["score"]
    for name, item in audit.items():
        result = item["all_exported"]
        view = "裁块" if "grid_crops" in name else "原图"
        lines.append(
            f"|{name}|{view}|{result['p_micro']:.5f}|{result['r_micro']:.5f}|"
            f"{result['map50']:.5f}|{result['score']:.3f}|{result['score']-baseline:+.3f}|"
        )
    lines.extend(["", "## 同一开发集逐类阈值上限诊断", "",
        "|单元|统一最低阈值分|逐类阈值分|表观增益|", "|---|---:|---:|---:|"])
    for name, item in threshold_results.items():
        raw = audit[name]["all_exported"]["score"]
        tuned = item["result"]["score"]
        lines.append(f"|{name}|{raw:.3f}|{tuned:.3f}|{tuned-raw:+.3f}|")
    lines.extend(["", "逐类阈值直接在同一 dev 上优化，属于乐观上限；未经独立数据确认不得直接冻结为提交策略。", "",
        "## 有界双权重融合诊断", "",
        "|组合|最佳原始分|逐类阈值表观分|逐类来源路由表观分|", "|---|---:|---:|---:|"])
    for label, item in ensemble_results.items():
        lines.append(
            f"|{label}|{item['best_raw']['result']['score']:.3f}|"
            f"{item['best_class_thresholds']['result']['score']:.3f}|"
            f"{item['class_routing']['result']['score']:.3f}|"
        )
    lines.extend(["", "融合结果同样使用 dev 选参，只用于判断模型互补性；是否符合最终作品算力和规则约束需另行冻结确认。"])
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
