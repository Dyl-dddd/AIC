"""Search global class/geometry actions that rescue misses from an exact layered anchor."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_layered_submission import parse_extra
from scripts.build_recall_submission import load_actions, load_ground_truth
from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import box_iou
from steel_defect.recall_booster import (
    agnostic_nms,
    build_action_arrays,
    iter_candidate_records,
    scale_boxes,
)


def unmatched_ground_truth(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    ground_truth: dict[str, list[list[float]]],
) -> list[tuple[str, int, np.ndarray]]:
    misses: list[tuple[str, int, np.ndarray]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        gt = np.asarray(ground_truth.get(class_name, []), dtype=np.float32).reshape(-1, 4)
        if not len(gt):
            continue
        selected = np.flatnonzero(classes == class_id)
        selected = selected[np.argsort(scores[selected])[::-1]]
        matched: set[int] = set()
        for prediction_index in selected:
            ious = box_iou(boxes[prediction_index], gt)
            order = np.argsort(ious)[::-1]
            match = next((int(index) for index in order if int(index) not in matched), None)
            if match is not None and float(ious[match]) >= 0.5:
                matched.add(match)
                if len(matched) == len(gt):
                    break
        misses.extend(
            (class_name, index, gt[index]) for index in range(len(gt)) if index not in matched
        )
    return misses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--base-actions", type=Path, required=True)
    parser.add_argument("--base-extra-action", nargs=6, action="append")
    parser.add_argument("--incremental-cache", type=Path, action="append", required=True)
    parser.add_argument("--incremental-actions", type=Path, required=True)
    parser.add_argument("--additional-cache", type=Path, action="append", required=True)
    parser.add_argument("--additional-actions", type=Path, required=True)
    parser.add_argument("--rescue-cache", type=Path, action="append", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--scale", nargs=2, type=float, action="append", required=True)
    parser.add_argument("--shift", nargs=2, type=float, action="append")
    parser.add_argument("--pre-nms-iou", type=float, default=0.5)
    parser.add_argument("--candidate-score-threshold", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_actions = load_actions(args.base_actions) + parse_extra(args.base_extra_action)
    incremental_actions = load_actions(args.incremental_actions)
    additional_actions = load_actions(args.additional_actions)
    ground_truth = load_ground_truth(args.source, args.split_manifest, args.split)
    scales = [tuple(map(float, scale)) for scale in args.scale]
    shifts = [tuple(map(float, shift)) for shift in (args.shift or [[0.0, 0.0]])]

    base_iters = [iter_candidate_records(path) for path in args.base_cache]
    inc_iters = [iter_candidate_records(path) for path in args.incremental_cache]
    add_iters = [iter_candidate_records(path) for path in args.additional_cache]
    rescue_iters = [iter_candidate_records(path) for path in args.rescue_cache]
    iterators = [*base_iters, *inc_iters, *add_iters, *rescue_iters]

    action_hits: dict[tuple[int, str, float, float, float, float], set[str]] = {}
    action_costs: Counter[tuple[int, float, float, float, float]] = Counter()
    anchor_hits = total = 0
    anchor_misses: list[str] = []

    for records in zip_longest(*iterators):
        if any(record is None for record in records):
            raise ValueError("Candidate caches contain different image counts")
        image_ids = {record["image_id"] for record in records}
        if len(image_ids) != 1:
            raise ValueError(f"Candidate caches are out of order: {sorted(image_ids)}")
        image_id = next(iter(image_ids))
        i0 = len(base_iters)
        i1 = i0 + len(inc_iters)
        i2 = i1 + len(add_iters)
        base_records = records[:i0]
        inc_records = records[i0:i1]
        add_records = records[i1:i2]
        rescue_records = records[i2:]

        layer_arrays = []
        for layer_records, actions in (
            (base_records, base_actions),
            (inc_records, incremental_actions),
            (add_records, additional_actions),
        ):
            layer_arrays.append(
                build_action_arrays(
                    layer_records, CLASS_NAMES, actions, args.pre_nms_iou,
                    0.001, 0.0, args.candidate_score_threshold,
                )
            )
        sizes = {tuple(item[3]) for item in layer_arrays}
        if len(sizes) != 1:
            raise ValueError(f"Anchor layer image-size mismatch for {image_id}")
        image_size = next(iter(sizes))
        anchor_boxes = np.concatenate([item[0] for item in layer_arrays])
        anchor_scores = np.concatenate([item[1] for item in layer_arrays])
        anchor_classes = np.concatenate([item[2] for item in layer_arrays])
        gt_key = image_id if image_id in ground_truth else Path(image_id).name
        image_gt = ground_truth.get(gt_key, {})
        image_total = sum(len(value) for value in image_gt.values())
        misses = unmatched_ground_truth(anchor_boxes, anchor_scores, anchor_classes, image_gt)
        anchor_hits += image_total - len(misses)
        total += image_total
        anchor_misses.extend(f"{image_id}#{index}:{name}" for name, index, _ in misses)

        candidates = [c for record in rescue_records for c in record["candidates"]]
        raw_boxes = np.asarray([c["bbox"] for c in candidates], dtype=np.float32).reshape(-1, 4)
        raw_scores = np.asarray([c["score"] for c in candidates], dtype=np.float32)
        raw_classes = np.asarray([c["class_id"] for c in candidates], dtype=np.int64)
        selected = raw_scores >= args.candidate_score_threshold
        raw_boxes, raw_scores, raw_classes = agnostic_nms(
            raw_boxes[selected], raw_scores[selected], raw_classes[selected], args.pre_nms_iou
        )

        transformed: dict[tuple[int, float, float, float, float], np.ndarray] = {}
        for source_id in range(len(CLASS_NAMES)):
            source_boxes = raw_boxes[raw_classes == source_id]
            for sx, sy in scales:
                for dx, dy in shifts:
                    boxes, _ = scale_boxes(source_boxes, image_size, sx, sy, dx, dy)
                    key = (source_id, sx, sy, dx, dy)
                    transformed[key] = boxes
                    action_costs[key] += len(boxes)
        for target_name, gt_index, gt_box in misses:
            miss_key = f"{image_id}#{gt_index}:{target_name}"
            for transform_key, boxes in transformed.items():
                if len(boxes) and float(box_iou(gt_box, boxes).max()) >= 0.5:
                    key = (*transform_key[:1], target_name, *transform_key[1:])
                    action_hits.setdefault(key, set()).add(miss_key)

    covered: set[str] = set()
    remaining = set(action_hits)
    actions = []
    while remaining:
        def rank(key):
            cost_key = (key[0], key[2], key[3], key[4], key[5])
            gain = len(action_hits[key] - covered)
            return gain / max(action_costs[cost_key], 1), gain, -action_costs[cost_key]
        key = max(remaining, key=rank)
        new_hits = action_hits[key] - covered
        if not new_hits:
            break
        source_id, target_name, sx, sy, dx, dy = key
        cost_key = (source_id, sx, sy, dx, dy)
        actions.append({
            "source_class": CLASS_NAMES[source_id], "target_class": target_name,
            "scale": [sx, sy], "shift": [dx, dy],
            "new_hits": sorted(new_hits), "emitted_boxes": action_costs[cost_key],
        })
        covered.update(new_hits)
        remaining.remove(key)

    payload = {
        "actions": actions,
        "anchor_hits": anchor_hits,
        "ground_truth": total,
        "anchor_misses": anchor_misses,
        "rescuable_misses": sorted(covered),
        "combined_upper_hits": anchor_hits + len(covered),
        "candidate_actions": len(action_hits),
        "rescue_caches": [str(path.resolve()) for path in args.rescue_cache],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
