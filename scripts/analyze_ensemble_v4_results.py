"""Audit Ensemble V4 results and test bounded Model A/B complementarity.

This script never runs a neural network. It replays the frozen-development
candidate caches, applies a common post-processing policy, and evaluates a
small, explicit fusion grid. The final class router is a diagnostic upper
bound because it is calibrated on the same development split.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_stage1_cache import metrics50, replay
from scripts.run_semifinal_stage1 import write_json
from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import classwise_nms
from steel_defect.inference import InferenceOptions
from steel_defect.metrics import _match_class, average_precision
from steel_defect.runtime import sha256_file


PERMITTED = [name for name in CLASS_NAMES if name != "qilie"]
VIEW_WEIGHTS = {"original": 0.6192893401015228, "grid_crops": 0.38071065989847713}
MODEL_B_SCALES = (0.35, 0.50, 0.65, 0.80, 1.00, 1.25)
CROSS_MODEL_IOUS = (0.55, 0.60, 0.65, 0.68, 0.72, 0.75)
THRESHOLDS = (
    0.0001,
    0.00015,
    0.0002,
    0.0003,
    0.0005,
    0.0007,
    0.001,
    0.002,
    0.003,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.1,
    0.2,
    0.3,
    0.4,
)


def load_cell(folder: Path) -> dict:
    report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    exported = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
    with gzip.open(folder / "candidates.jsonl.gz", "rt", encoding="utf-8") as handle:
        header = json.loads(next(handle))
        rows = [json.loads(line) for line in handle]
    if header.get("signature") != report.get("signature"):
        raise ValueError(f"cache signature mismatch: {folder}")
    ground_truth = {row["image_id"]: row["ground_truth"] for row in rows}
    sizes = {row["image_id"]: tuple(row["image_size"]) for row in rows}
    return {
        "folder": folder,
        "report": report,
        "rows": rows,
        "ground_truth": ground_truth,
        "sizes": sizes,
        "exported": exported,
        "cache_sha256": sha256_file(folder / "candidates.jsonl.gz"),
    }


def validate_pair(model_a: dict, model_b: dict, view: str) -> None:
    for cell in (model_a, model_b):
        report = cell["report"]
        if not report.get("complete") or report["signature"].get("view") != view:
            raise ValueError(f"incomplete or wrong-view cell: {cell['folder']}")
        if report["signature"].get("limit") != 0:
            raise ValueError(f"partial development result: {cell['folder']}")
        reproduced = metrics50(cell["exported"], cell["ground_truth"], cell["sizes"])
        uploaded = report["all_exported"]["summary"]
        if abs(reproduced["score"] - uploaded["score_proxy_20_60_20"]) > 1e-9:
            raise ValueError(f"uploaded metric mismatch: {cell['folder']}")
    if model_a["ground_truth"] != model_b["ground_truth"]:
        raise ValueError(f"ground-truth mismatch for {view}")
    if model_a["sizes"] != model_b["sizes"]:
        raise ValueError(f"image-size mismatch for {view}")
    a_ids = model_a["report"]["signature"].get("source_ids")
    b_ids = model_b["report"]["signature"].get("source_ids")
    if a_ids != b_ids:
        raise ValueError(f"source split mismatch for {view}")


def base_options(cell: dict) -> InferenceOptions:
    options = InferenceOptions(**cell["report"]["signature"]["options"])
    return replace(
        options,
        conf=0.0001,
        iou=0.68,
        edge_penalty=0.50,
        merge="nms",
        class_thresholds={},
    )


def fuse_predictions(
    model_a: list[dict], model_b: list[dict], model_b_scale: float, nms_iou: float
) -> list[dict]:
    grouped: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for prediction in model_a:
        grouped[prediction["image_id"]].append((prediction, 1.0))
    for prediction in model_b:
        grouped[prediction["image_id"]].append((prediction, model_b_scale))

    class_ids = {name: index for index, name in enumerate(PERMITTED)}
    output: list[dict] = []
    for image_id, rows in grouped.items():
        rows = [row for row in rows if row[0]["category_name"] in class_ids]
        if not rows:
            continue
        boxes = np.asarray([row[0]["bbox"] for row in rows], dtype=np.float32)
        scores = np.asarray(
            [min(1.0, float(row[0]["score"]) * row[1]) for row in rows], dtype=np.float32
        )
        classes = np.asarray(
            [class_ids[row[0]["category_name"]] for row in rows], dtype=np.int64
        )
        for index in classwise_nms(boxes, scores, classes, nms_iou):
            output.append(
                {
                    "image_id": image_id,
                    "category_name": PERMITTED[int(classes[index])],
                    "bbox": [int(round(value)) for value in boxes[index]],
                    "score": round(float(scores[index]), 7),
                }
            )
    return output


def weighted_summary(results: dict[str, dict]) -> dict:
    weighted = sum(VIEW_WEIGHTS[view] * results[view]["score"] for view in VIEW_WEIGHTS)
    return {
        "weighted_score": weighted,
        "robust_min_score": min(results[view]["score"] for view in VIEW_WEIGHTS),
        "views": results,
    }


def option_statistics(predictions: list[dict], ground_truth: dict, class_name: str, threshold: float) -> dict:
    selected = [
        prediction
        for prediction in predictions
        if prediction["category_name"] == class_name
        and float(prediction["score"]) >= threshold
    ]
    tp, fp, count = _match_class(selected, ground_truth, class_name, 0.5)
    return {
        "gt": count,
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "ap50": average_precision(tp, fp, count),
    }


def score_from_class_stats(stats: dict[str, dict[str, dict]]) -> dict:
    results = {}
    for view in VIEW_WEIGHTS:
        rows = [stats[name][view] for name in PERMITTED]
        gt = sum(row["gt"] for row in rows)
        tp = sum(row["tp"] for row in rows)
        fp = sum(row["fp"] for row in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(gt, 1)
        map50 = float(np.mean([row["ap50"] for row in rows if row["gt"]]))
        results[view] = {
            "p_micro": precision,
            "r_micro": recall,
            "map50": map50,
            "score": 20 * precision + 60 * recall + 20 * map50,
            "gt": gt,
            "tp": tp,
            "fp": fp,
            "fn": gt - tp,
        }
    return weighted_summary(results)


def shared_class_router(variants: dict[str, dict[str, list[dict]]], cells: dict[str, dict]) -> dict:
    options: dict[str, list[dict]] = {}
    for class_name in PERMITTED:
        class_options = []
        for variant_name, per_view in variants.items():
            for threshold in THRESHOLDS:
                stats = {
                    view: option_statistics(
                        per_view[view], cells[view]["ground_truth"], class_name, threshold
                    )
                    for view in VIEW_WEIGHTS
                }
                class_options.append(
                    {
                        "variant": variant_name,
                        "threshold": threshold,
                        "stats": stats,
                    }
                )
        options[class_name] = class_options

    starts = []
    for variant_name in variants:
        start = next(
            index
            for index, option in enumerate(options[PERMITTED[0]])
            if option["variant"] == variant_name and option["threshold"] == 0.0001
        )
        starts.append(start)

    best = None
    for start in starts:
        indices = {name: start for name in PERMITTED}
        for _ in range(12):
            changed = False
            for class_name in PERMITTED:
                previous = indices[class_name]
                candidates = []
                for index in range(len(options[class_name])):
                    indices[class_name] = index
                    selected_stats = {
                        name: options[name][indices[name]]["stats"] for name in PERMITTED
                    }
                    summary = score_from_class_stats(selected_stats)
                    candidates.append(
                        (
                            summary["weighted_score"],
                            summary["robust_min_score"],
                            -options[class_name][index]["threshold"],
                            index,
                        )
                    )
                indices[class_name] = max(candidates)[3]
                changed |= indices[class_name] != previous
            if not changed:
                break

        selected = {name: options[name][indices[name]] for name in PERMITTED}
        summary = score_from_class_stats(
            {name: selected[name]["stats"] for name in PERMITTED}
        )
        candidate = (summary["weighted_score"], summary["robust_min_score"])
        if best is None or candidate > best[0]:
            best = (candidate, selected, summary)

    assert best is not None
    selection = {
        name: {
            "variant": best[1][name]["variant"],
            "threshold": best[1][name]["threshold"],
            "stats": best[1][name]["stats"],
        }
        for name in PERMITTED
    }
    return {
        "selection": selection,
        "summary": best[2],
        "warning": "Same-development-set class routing is an optimistic diagnostic, not an official score.",
    }


def class_comparison(cells: dict[str, dict], base_predictions: dict[str, dict[str, list[dict]]]) -> dict:
    comparison = {}
    for view in VIEW_WEIGHTS:
        comparison[view] = {}
        for class_name in PERMITTED:
            comparison[view][class_name] = {
                model: option_statistics(
                    base_predictions[model][view],
                    cells[view]["ground_truth"],
                    class_name,
                    0.0001,
                )
                for model in ("model_a", "model_b")
            }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a-original", type=Path, required=True)
    parser.add_argument("--model-a-grid", type=Path, required=True)
    parser.add_argument("--model-b-original", type=Path, required=True)
    parser.add_argument("--model-b-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    cells = {
        "original": {
            "model_a": load_cell(args.model_a_original),
            "model_b": load_cell(args.model_b_original),
        },
        "grid_crops": {
            "model_a": load_cell(args.model_a_grid),
            "model_b": load_cell(args.model_b_grid),
        },
    }
    for view, pair in cells.items():
        validate_pair(pair["model_a"], pair["model_b"], view)

    base_predictions = {"model_a": {}, "model_b": {}}
    base_results = {"model_a": {}, "model_b": {}}
    for view, pair in cells.items():
        for model in ("model_a", "model_b"):
            cell = pair[model]
            predictions = replay(cell["rows"], base_options(cell))
            base_predictions[model][view] = predictions
            base_results[model][view] = metrics50(
                predictions, cell["ground_truth"], cell["sizes"]
            )

    audit = {
        "model_a": {
            "weights_sha256": cells["original"]["model_a"]["report"]["signature"]["weights_sha256"],
            "summary": weighted_summary(base_results["model_a"]),
        },
        "model_b": {
            "weights_sha256": cells["original"]["model_b"]["report"]["signature"]["weights_sha256"],
            "summary": weighted_summary(base_results["model_b"]),
        },
        "split_sha256": cells["original"]["model_a"]["report"]["signature"]["split_sha256"],
        "candidate_cache_sha256": {
            view: {
                model: cells[view][model]["cache_sha256"]
                for model in ("model_a", "model_b")
            }
            for view in VIEW_WEIGHTS
        },
    }

    fusion_grid = []
    best_fusion_predictions = None
    best_fusion_key = None
    for scale in MODEL_B_SCALES:
        for iou in CROSS_MODEL_IOUS:
            per_view_predictions = {}
            per_view_results = {}
            for view in VIEW_WEIGHTS:
                pair = cells[view]
                predictions = fuse_predictions(
                    base_predictions["model_a"][view],
                    base_predictions["model_b"][view],
                    scale,
                    iou,
                )
                per_view_predictions[view] = predictions
                per_view_results[view] = metrics50(
                    predictions,
                    pair["model_a"]["ground_truth"],
                    pair["model_a"]["sizes"],
                )
            summary = weighted_summary(per_view_results)
            tag = f"fusion_b{scale:g}_iou{iou:g}"
            record = {
                "tag": tag,
                "model_b_scale": scale,
                "cross_model_nms_iou": iou,
                **summary,
            }
            fusion_grid.append(record)
            key = (summary["weighted_score"], summary["robust_min_score"])
            if best_fusion_key is None or key > best_fusion_key:
                best_fusion_key = key
                best_fusion_predictions = per_view_predictions
    fusion_grid.sort(
        key=lambda item: (item["weighted_score"], item["robust_min_score"]), reverse=True
    )
    best_fusion = fusion_grid[0]
    assert best_fusion_predictions is not None

    router_variants = {
        "model_a": base_predictions["model_a"],
        "model_b": base_predictions["model_b"],
        "fusion": best_fusion_predictions,
    }
    router_cells = {
        view: cells[view]["model_a"] for view in VIEW_WEIGHTS
    }
    router = shared_class_router(router_variants, router_cells)

    comparison = class_comparison(router_cells, base_predictions)
    decision = {
        "protected_official_baseline": 66.16,
        "model_a_dev_baseline": audit["model_a"]["summary"]["weighted_score"],
        "model_b_dev_baseline": audit["model_b"]["summary"]["weighted_score"],
        "best_global_fusion": best_fusion,
        "global_fusion_beats_model_a_both_views": all(
            best_fusion["views"][view]["score"]
            > audit["model_a"]["summary"]["views"][view]["score"]
            for view in VIEW_WEIGHTS
        ),
        "class_router_diagnostic": router["summary"],
        "class_router_selection": {
            name: {
                "variant": item["variant"],
                "threshold": item["threshold"],
            }
            for name, item in router["selection"].items()
        },
        "next_gate": (
            "BUILD_INFERENCE_PACKAGE"
            if router["summary"]["weighted_score"]
            > audit["model_a"]["summary"]["weighted_score"] + 0.30
            else "STOP_ENSEMBLE_AND_REDESIGN"
        ),
        "warning": "All scores are frozen-development proxies, not official Test scores.",
    }

    write_json(args.output / "audit.json", audit)
    write_json(args.output / "class_comparison.json", comparison)
    write_json(args.output / "fusion_grid.json", fusion_grid)
    write_json(args.output / "class_router.json", router)
    write_json(args.output / "decision.json", decision)

    lines = [
        "# Ensemble V4 result analysis",
        "",
        "All values are frozen-development proxies under `20P + 60R + 20mAP50`; they are not official Test scores.",
        "",
        "## Raw comparison",
        "",
        "|System|Original|Grid crops|Weighted|Worst view|",
        "|---|---:|---:|---:|---:|",
    ]
    for model in ("model_a", "model_b"):
        item = audit[model]["summary"]
        lines.append(
            f"|{model}|{item['views']['original']['score']:.3f}|"
            f"{item['views']['grid_crops']['score']:.3f}|{item['weighted_score']:.3f}|"
            f"{item['robust_min_score']:.3f}|"
        )
    lines.extend(
        [
            "",
            "## Bounded global fusion",
            "",
            "|B scale|Cross-model NMS|Original|Grid crops|Weighted|Worst view|",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in fusion_grid[:10]:
        lines.append(
            f"|{item['model_b_scale']:.2f}|{item['cross_model_nms_iou']:.2f}|"
            f"{item['views']['original']['score']:.3f}|"
            f"{item['views']['grid_crops']['score']:.3f}|"
            f"{item['weighted_score']:.3f}|{item['robust_min_score']:.3f}|"
        )
    lines.extend(
        [
            "",
            "## Shared class router diagnostic",
            "",
            "|Class|Source|Threshold|",
            "|---|---|---:|",
        ]
    )
    for name, item in router["selection"].items():
        lines.append(f"|{name}|{item['variant']}|{item['threshold']:.6g}|")
    summary = router["summary"]
    lines.extend(
        [
            "",
            f"Router original: `{summary['views']['original']['score']:.3f}`",
            f"Router grid crops: `{summary['views']['grid_crops']['score']:.3f}`",
            f"Router weighted: `{summary['weighted_score']:.3f}`",
            f"Router worst view: `{summary['robust_min_score']:.3f}`",
            "",
            "The router is calibrated on this same development set and is therefore an optimistic upper bound. It must be frozen before any official submission and should be validated against rule constraints on multi-model inference.",
            "",
            f"Decision gate: `{decision['next_gate']}`.",
        ]
    )
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
