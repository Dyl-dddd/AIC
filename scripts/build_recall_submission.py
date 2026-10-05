"""Build a memory-bounded recall-oriented submission from frozen caches."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.classes import CLASS_NAMES
from steel_defect.recall_booster import (
    build_action_arrays,
    iter_candidate_records,
    matched_ground_truth,
)
from steel_defect.voc import discover_pairs, parse_voc


def load_actions(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = payload.get("actions", payload.get("action_milp_selected", payload))
    if not isinstance(actions, list) or not actions:
        raise ValueError("Action file must contain a non-empty action list")
    required = {"source_class", "target_class", "scale"}
    for action in actions:
        if not required <= set(action):
            raise ValueError(f"Invalid action: {action}")
        if action["source_class"] not in CLASS_NAMES or action["target_class"] not in CLASS_NAMES:
            raise ValueError(f"Unknown class in action: {action}")
        if len(action["scale"]) != 2 or any(float(value) <= 0 for value in action["scale"]):
            raise ValueError(f"Invalid scale in action: {action}")
        if len(action.get("shift", [0, 0])) != 2:
            raise ValueError(f"Invalid shift in action: {action}")
    return actions


def load_ground_truth(source: Path, split_manifest: Path, split: str) -> dict[str, dict[str, list[list[float]]]]:
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    selected = set(manifest["splits"][split])
    pairs, missing_xml, _ = discover_pairs(source)
    if missing_xml:
        raise ValueError(f"Missing XML for {len(missing_xml)} images")
    output = {}
    for image_path, xml_path in pairs:
        relative = image_path.relative_to(source).as_posix()
        if relative not in selected:
            continue
        record = parse_voc(image_path, xml_path)
        per_class: dict[str, list[list[float]]] = {}
        for annotation in record.annotations:
            per_class.setdefault(annotation.class_name, []).append(list(annotation.box))
        output[relative] = per_class
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument(
        "--extra-action", nargs=6, action="append", metavar=("SOURCE", "TARGET", "SX", "SY", "DX", "DY"),
        help="Append a frozen source/target/scale/center-shift action without rerunning the optimizer",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pre-nms-iou", type=float, default=0.5)
    parser.add_argument("--final-nms-iou", type=float, default=0.0)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument(
        "--candidate-score-threshold", type=float, default=0.0,
        help="Discard raw candidates below this score before NMS and action expansion",
    )
    parser.add_argument("--sweep-score-floor", type=float, action="append", default=[])
    parser.add_argument("--rounding", choices=("nearest", "outward"), default="outward")
    parser.add_argument("--ground-truth-source", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.score_floor <= 1:
        parser.error("--score-floor must be within [0,1]")
    if not 0 <= args.candidate_score_threshold <= 1:
        parser.error("--candidate-score-threshold must be within [0,1]")
    if bool(args.ground_truth_source) != bool(args.split_manifest):
        parser.error("--ground-truth-source and --split-manifest must be provided together")
    if args.output is None and args.ground_truth_source is None:
        parser.error("Provide --output and/or ground truth options")

    actions = load_actions(args.actions)
    for values in args.extra_action or []:
        source, target, scale_x, scale_y, shift_x, shift_y = values
        if source not in CLASS_NAMES or target not in CLASS_NAMES:
            parser.error(f"Unknown class in --extra-action: {values}")
        actions.append({
            "source_class": source,
            "target_class": target,
            "scale": [float(scale_x), float(scale_y)],
            "shift": [float(shift_x), float(shift_y)],
        })
    ground_truth = (
        load_ground_truth(args.ground_truth_source, args.split_manifest, args.split)
        if args.ground_truth_source else None
    )
    iterators = [iter_candidate_records(path) for path in args.cache]
    output_handle = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")
        output_handle.write("[")
    first_prediction = True
    image_count = prediction_count = 0
    score_floors = sorted(set([args.score_floor, *args.sweep_score_floor]))
    sweep_stats = {
        value: {"hit": 0, "total": 0, "per_class": Counter(), "per_class_gt": Counter()}
        for value in score_floors
    }
    missed_images: list[dict] = []
    try:
        for records in zip_longest(*iterators):
            if any(record is None for record in records):
                raise ValueError("Candidate caches contain different image counts")
            image_ids = {record["image_id"] for record in records}
            if len(image_ids) != 1:
                raise ValueError(f"Candidate caches are out of order: {sorted(image_ids)}")
            image_id = next(iter(image_ids))
            boxes, scores, classes, image_size = build_action_arrays(
                records, CLASS_NAMES, actions, args.pre_nms_iou,
                0.0 if args.sweep_score_floor else args.score_floor, args.final_nms_iou,
                args.candidate_score_threshold,
            )
            if args.rounding == "outward":
                rounded = np.concatenate(
                    (np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:])), axis=1
                ).astype(np.int64)
            else:
                rounded = np.rint(boxes).astype(np.int64)
            width, height = image_size
            rounded[:, [0, 2]] = rounded[:, [0, 2]].clip(0, width)
            rounded[:, [1, 3]] = rounded[:, [1, 3]].clip(0, height)
            valid = (rounded[:, 2] > rounded[:, 0]) & (rounded[:, 3] > rounded[:, 1])
            rounded, scores, classes = rounded[valid], scores[valid], classes[valid]
            prediction_count += len(rounded)
            if ground_truth is not None:
                gt_key = image_id if image_id in ground_truth else Path(image_id).name
                for score_floor in score_floors:
                    image_hit, image_total, class_counts = matched_ground_truth(
                        rounded.astype(np.float32), np.maximum(scores, score_floor), classes,
                        ground_truth.get(gt_key, {}), CLASS_NAMES,
                    )
                    stats = sweep_stats[score_floor]
                    stats["hit"] += image_hit
                    stats["total"] += image_total
                    for name, (class_hit, class_total) in class_counts.items():
                        stats["per_class"][name] += class_hit
                        stats["per_class_gt"][name] += class_total
                        if score_floor == args.score_floor and class_hit < class_total:
                            missed_images.append({
                                "image_id": image_id,
                                "class_name": name,
                                "true_positives": class_hit,
                                "ground_truth": class_total,
                            })
            if output_handle:
                submission_id = Path(image_id).name
                submission_scores = np.maximum(scores, args.score_floor)
                for box, score, class_id in zip(rounded, submission_scores, classes, strict=True):
                    item = {
                        "image_id": submission_id,
                        "category_name": CLASS_NAMES[int(class_id)],
                        "bbox": box.tolist(),
                        "score": round(float(score), 7),
                    }
                    if not first_prediction:
                        output_handle.write(",")
                    output_handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
                    first_prediction = False
            image_count += 1
            primary = sweep_stats[args.score_floor]
            if not args.quiet:
                print(f"[{image_count}] {image_id}: predictions={len(rounded)}, TP={primary['hit']}/{primary['total']}")
            if args.limit and image_count >= args.limit:
                break
    finally:
        if output_handle:
            output_handle.write("]")
            output_handle.close()

    primary = sweep_stats[args.score_floor]
    hit = primary["hit"]
    total = primary["total"]
    per_class = primary["per_class"]
    per_class_gt = primary["per_class_gt"]
    report = {
        "images": image_count,
        "predictions": prediction_count,
        "actions": len(actions),
        "extra_actions": args.extra_action or [],
        "pre_nms_iou": args.pre_nms_iou,
        "final_nms_iou": args.final_nms_iou,
        "score_floor": args.score_floor,
        "candidate_score_threshold": args.candidate_score_threshold,
        "rounding": args.rounding,
        "true_positives": hit,
        "false_negatives": total - hit,
        "ground_truth": total,
        "micro_recall": hit / total if total else None,
        "proxy_score": 100.0 * hit / total if total else None,
        "per_class": {
            name: {"true_positives": per_class[name], "ground_truth": per_class_gt[name]}
            for name in CLASS_NAMES if per_class_gt[name]
        },
        "missed_images": missed_images,
        "score_floor_sweep": {
            str(value): {
                "true_positives": stats["hit"],
                "false_negatives": stats["total"] - stats["hit"],
                "micro_recall": stats["hit"] / stats["total"] if stats["total"] else None,
                "proxy_score": 100.0 * stats["hit"] / stats["total"] if stats["total"] else None,
            }
            for value, stats in sweep_stats.items()
        },
        "candidate_caches": [str(path.resolve()) for path in args.cache],
        "action_file": str(args.actions.resolve()),
        "submission": str(args.output.resolve()) if args.output else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
