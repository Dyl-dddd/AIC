"""Build a submission from independently filtered baseline and incremental layers."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_recall_submission import load_actions, load_ground_truth
from steel_defect.classes import CLASS_NAMES
from steel_defect.recall_booster import build_action_arrays, iter_candidate_records, matched_ground_truth


def parse_extra(values: list[list[str]] | None) -> list[dict]:
    actions = []
    for source, target, sx, sy, dx, dy in values or []:
        if source not in CLASS_NAMES or target not in CLASS_NAMES:
            raise ValueError(f"Unknown class in extra action: {source}->{target}")
        actions.append({
            "source_class": source,
            "target_class": target,
            "scale": [float(sx), float(sy)],
            "shift": [float(dx), float(dy)],
        })
    return actions


def cache_weight_sha256(path: Path) -> str | None:
    """Read cache provenance without loading candidate arrays."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        prefix = handle.read(8192)
    match = re.search(r'"weights_sha256"\s*:\s*"([0-9a-fA-F]{64})"', prefix)
    return match.group(1).lower() if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--base-actions", type=Path, required=True)
    parser.add_argument("--base-extra-action", nargs=6, action="append")
    parser.add_argument("--incremental-cache", type=Path, action="append", required=True)
    parser.add_argument("--incremental-actions", type=Path, required=True)
    parser.add_argument(
        "--additional-cache", type=Path, action="append",
        help="Optional third independent candidate layer (repeat for multiple views)",
    )
    parser.add_argument(
        "--additional-actions", type=Path,
        help="Frozen action policy for --additional-cache",
    )
    parser.add_argument(
        "--rescue-cache", type=Path, action="append",
        help="Optional fourth independent rescue layer (repeat for multiple views)",
    )
    parser.add_argument(
        "--rescue-actions", type=Path,
        help="Frozen action policy for --rescue-cache",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pre-nms-iou", type=float, default=0.5)
    parser.add_argument("--final-nms-iou", type=float, default=0.0)
    parser.add_argument("--score-floor", type=float, default=0.001)
    parser.add_argument(
        "--exclude-category", action="append", default=[],
        help="Omit categories absent from this round's official test set (repeatable)",
    )
    parser.add_argument("--candidate-score-threshold", type=float, default=1e-5)
    parser.add_argument("--rounding", choices=("nearest", "outward"), default="nearest")
    parser.add_argument("--ground-truth-source", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    excluded_categories = set(args.exclude_category)
    unknown_exclusions = excluded_categories - set(CLASS_NAMES)
    if unknown_exclusions:
        parser.error(f"Unknown categories to exclude: {sorted(unknown_exclusions)}")
    if not 0 <= args.score_floor <= 1:
        parser.error("--score-floor must be within [0,1]")
    allowed_classes = [name for name in CLASS_NAMES if name not in excluded_categories]
    allowed_class_ids = np.asarray(
        [CLASS_NAMES.index(name) for name in allowed_classes], dtype=np.int64
    )
    if args.output is None and args.ground_truth_source is None:
        parser.error("Provide --output and/or ground truth options")
    if bool(args.ground_truth_source) != bool(args.split_manifest):
        parser.error("--ground-truth-source and --split-manifest must be provided together")
    if bool(args.additional_cache) != bool(args.additional_actions):
        parser.error("--additional-cache and --additional-actions must be provided together")
    if bool(args.rescue_cache) != bool(args.rescue_actions):
        parser.error("--rescue-cache and --rescue-actions must be provided together")

    cache_paths = [
        *args.base_cache, *args.incremental_cache,
        *(args.additional_cache or []), *(args.rescue_cache or []),
    ]
    weight_hashes = {str(path): cache_weight_sha256(path) for path in cache_paths}
    known_hashes = {value for value in weight_hashes.values() if value is not None}
    if args.output and len(known_hashes) > 1:
        parser.error(
            "Competition submission mixes checkpoints from different training stages; "
            f"cache weight hashes: {weight_hashes}"
        )

    base_actions = load_actions(args.base_actions) + parse_extra(args.base_extra_action)
    incremental_actions = load_actions(args.incremental_actions)
    additional_actions = load_actions(args.additional_actions) if args.additional_actions else []
    rescue_actions = load_actions(args.rescue_actions) if args.rescue_actions else []
    ground_truth = (
        load_ground_truth(args.ground_truth_source, args.split_manifest, args.split)
        if args.ground_truth_source else None
    )
    base_iterators = [iter_candidate_records(path) for path in args.base_cache]
    incremental_iterators = [iter_candidate_records(path) for path in args.incremental_cache]
    additional_iterators = [iter_candidate_records(path) for path in args.additional_cache or []]
    rescue_iterators = [iter_candidate_records(path) for path in args.rescue_cache or []]
    all_iterators = [*base_iterators, *incremental_iterators, *additional_iterators, *rescue_iterators]
    output_handle = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.output.open("w", encoding="utf-8")
        output_handle.write("[")
    first_prediction = True
    image_count = prediction_count = hit = total = 0
    layer_predictions = Counter()
    per_class_hit = Counter()
    per_class_total = Counter()
    missed_images = []
    try:
        for records in zip_longest(*all_iterators):
            if any(record is None for record in records):
                raise ValueError("Candidate caches contain different image counts")
            image_ids = {record["image_id"] for record in records}
            if len(image_ids) != 1:
                raise ValueError(f"Candidate caches are out of order: {sorted(image_ids)}")
            image_id = next(iter(image_ids))
            base_records = records[:len(base_iterators)]
            incremental_end = len(base_iterators) + len(incremental_iterators)
            incremental_records = records[len(base_iterators):incremental_end]
            additional_end = incremental_end + len(additional_iterators)
            additional_records = records[incremental_end:additional_end]
            rescue_records = records[additional_end:]
            base_boxes, base_scores, base_classes, base_size = build_action_arrays(
                base_records, CLASS_NAMES, base_actions, args.pre_nms_iou,
                args.score_floor, args.final_nms_iou, args.candidate_score_threshold,
            )
            inc_boxes, inc_scores, inc_classes, inc_size = build_action_arrays(
                incremental_records, CLASS_NAMES, incremental_actions, args.pre_nms_iou,
                args.score_floor, args.final_nms_iou, args.candidate_score_threshold,
            )
            if tuple(base_size) != tuple(inc_size):
                raise ValueError(f"Layer image-size mismatch for {image_id}")
            layer_boxes = [base_boxes, inc_boxes]
            layer_scores = [base_scores, inc_scores]
            layer_classes = [base_classes, inc_classes]
            if additional_records:
                add_boxes, add_scores, add_classes, add_size = build_action_arrays(
                    additional_records, CLASS_NAMES, additional_actions, args.pre_nms_iou,
                    args.score_floor, args.final_nms_iou, args.candidate_score_threshold,
                )
                if tuple(base_size) != tuple(add_size):
                    raise ValueError(f"Additional-layer image-size mismatch for {image_id}")
                layer_boxes.append(add_boxes)
                layer_scores.append(add_scores)
                layer_classes.append(add_classes)
                layer_predictions["additional"] += len(add_boxes)
            if rescue_records:
                rescue_boxes, rescue_scores, rescue_classes, rescue_size = build_action_arrays(
                    rescue_records, CLASS_NAMES, rescue_actions, args.pre_nms_iou,
                    args.score_floor, args.final_nms_iou, args.candidate_score_threshold,
                )
                if tuple(base_size) != tuple(rescue_size):
                    raise ValueError(f"Rescue-layer image-size mismatch for {image_id}")
                layer_boxes.append(rescue_boxes)
                layer_scores.append(rescue_scores)
                layer_classes.append(rescue_classes)
                layer_predictions["rescue"] += len(rescue_boxes)
            boxes = np.concatenate(layer_boxes, axis=0)
            scores = np.concatenate(layer_scores, axis=0)
            classes = np.concatenate(layer_classes, axis=0)
            layer_predictions["base"] += len(base_boxes)
            layer_predictions["incremental"] += len(inc_boxes)
            if args.rounding == "outward":
                rounded = np.concatenate((np.floor(boxes[:, :2]), np.ceil(boxes[:, 2:])), axis=1).astype(np.int64)
            else:
                rounded = np.rint(boxes).astype(np.int64)
            width, height = base_size
            rounded[:, [0, 2]] = rounded[:, [0, 2]].clip(0, width)
            rounded[:, [1, 3]] = rounded[:, [1, 3]].clip(0, height)
            valid = (rounded[:, 2] > rounded[:, 0]) & (rounded[:, 3] > rounded[:, 1])
            if excluded_categories:
                valid &= np.isin(classes, allowed_class_ids)
            rounded, scores, classes = rounded[valid], scores[valid], classes[valid]
            prediction_count += len(rounded)
            if ground_truth is not None:
                gt_key = image_id if image_id in ground_truth else Path(image_id).name
                if gt_key not in ground_truth:
                    raise ValueError(
                        f"No ground truth for {image_id}; check --split-manifest and --split"
                    )
                image_hit, image_total, counts = matched_ground_truth(
                    rounded.astype(np.float32), scores, classes,
                    {
                        name: boxes for name, boxes in ground_truth.get(gt_key, {}).items()
                        if name in allowed_classes
                    }, CLASS_NAMES,
                )
                hit += image_hit
                total += image_total
                for name, (class_hit, class_total) in counts.items():
                    per_class_hit[name] += class_hit
                    per_class_total[name] += class_total
                    if class_hit < class_total:
                        missed_images.append({
                            "image_id": image_id,
                            "class_name": name,
                            "true_positives": class_hit,
                            "ground_truth": class_total,
                        })
            if output_handle:
                submission_id = Path(image_id).name
                for box, score, class_id in zip(rounded, scores, classes, strict=True):
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
            if not args.quiet:
                print(f"[{image_count}] {image_id}: base={len(base_boxes)} incremental={len(inc_boxes)}")
    finally:
        if output_handle:
            output_handle.write("]")
            output_handle.close()

    report = {
        "images": image_count,
        "predictions": prediction_count,
        "layer_predictions": dict(layer_predictions),
        "base_actions": len(base_actions),
        "incremental_actions": len(incremental_actions),
        "additional_actions": len(additional_actions),
        "rescue_actions": len(rescue_actions),
        "score_floor": args.score_floor,
        "excluded_categories": sorted(excluded_categories),
        "true_positives": hit,
        "false_negatives": total - hit,
        "ground_truth": total,
        "micro_recall": hit / total if total else None,
        "proxy_score": 100.0 * hit / total if total else None,
        "per_class": {
            name: {"true_positives": per_class_hit[name], "ground_truth": per_class_total[name]}
            for name in allowed_classes if per_class_total[name]
        },
        "missed_images": missed_images,
        "base_caches": [str(path.resolve()) for path in args.base_cache],
        "incremental_caches": [str(path.resolve()) for path in args.incremental_cache],
        "additional_caches": [str(path.resolve()) for path in args.additional_cache or []],
        "rescue_caches": [str(path.resolve()) for path in args.rescue_cache or []],
        "cache_weight_sha256": weight_hashes,
        "output": str(args.output.resolve()) if args.output else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
