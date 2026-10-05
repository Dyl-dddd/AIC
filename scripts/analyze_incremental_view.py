"""Find the smallest useful actions contributed by one new inference view.

The baseline and new-view candidates are NMS-filtered independently.  Only
ground-truth boxes missed by the baseline are considered possible rescues, so
the report measures genuinely incremental coverage instead of total coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_recall_coverage import read_cache, transform_boxes
from steel_defect.geometry import box_iou
from steel_defect.voc import discover_pairs, parse_voc


def nms_items(items: list[dict], iou: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray([item["bbox"] for item in items], dtype=np.float32).reshape(-1, 4)
    classes = np.asarray([item["class_id"] for item in items], dtype=np.int64)
    scores = np.asarray([item["score"] for item in items], dtype=np.float32)
    if not len(boxes) or iou <= 0:
        return boxes, classes, scores
    xywh = boxes.copy()
    xywh[:, 2:] -= xywh[:, :2]
    keep = np.asarray(
        cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou),
        dtype=np.int64,
    ).reshape(-1)
    return boxes[keep], classes[keep], scores[keep]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-cache", type=Path, action="append", required=True)
    parser.add_argument("--new-cache", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--scale", nargs=2, type=float, action="append", required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    selected = set(manifest["splits"][args.split])
    classes = manifest["classes"]
    pairs, missing_xml, _ = discover_pairs(args.source)
    if missing_xml:
        raise ValueError(f"Missing XML for {len(missing_xml)} images")
    records = [
        parse_voc(image_path, xml_path)
        for image_path, xml_path in pairs
        if image_path.relative_to(args.source).as_posix() in selected
    ]
    base_caches = [read_cache(path) for path in args.base_cache]
    new_cache = read_cache(args.new_cache)
    scales = tuple((float(x), float(y)) for x, y in args.scale)
    action_hits: dict[tuple[int, str, int], set[str]] = {}
    action_costs: Counter[tuple[int, int]] = Counter()
    rescues: dict[str, dict] = {}
    baseline_hits = 0
    total = 0

    for record in records:
        image_id = record.image_path.relative_to(args.source).as_posix()
        base_items = [
            candidate
            for cache in base_caches
            for candidate in cache["images"][image_id]["candidates"]
        ]
        new_item = new_cache["images"][image_id]
        image_size = tuple(new_item["image_size"])
        base_boxes, _, _ = nms_items(base_items, args.nms_iou)
        new_boxes, new_classes, _ = nms_items(new_item["candidates"], args.nms_iou)
        base_transformed = transform_boxes(base_boxes, image_size, scales, 0.0)
        new_action_boxes: dict[tuple[int, int], np.ndarray] = {}
        for source_id in range(len(classes)):
            source_boxes = new_boxes[new_classes == source_id]
            for scale_index, scale in enumerate(scales):
                transformed = transform_boxes(source_boxes, image_size, (scale,), 0.0)
                new_action_boxes[(source_id, scale_index)] = transformed
                action_costs[(source_id, scale_index)] += len(transformed)

        for annotation_index, annotation in enumerate(record.annotations):
            gt = np.asarray(annotation.box, dtype=np.float32)
            baseline_max = float(box_iou(gt, base_transformed).max()) if len(base_transformed) else 0.0
            total += 1
            if baseline_max >= args.iou:
                baseline_hits += 1
                continue
            gt_key = f"{image_id}#{annotation_index}:{annotation.class_name}"
            matching_actions = []
            best_iou = 0.0
            for (source_id, scale_index), boxes in new_action_boxes.items():
                if not len(boxes):
                    continue
                maximum = float(box_iou(gt, boxes).max())
                best_iou = max(best_iou, maximum)
                if maximum >= args.iou:
                    key = (source_id, annotation.class_name, scale_index)
                    action_hits.setdefault(key, set()).add(gt_key)
                    matching_actions.append({
                        "source_class": classes[source_id],
                        "target_class": annotation.class_name,
                        "scale": list(scales[scale_index]),
                        "max_iou": maximum,
                    })
            if matching_actions:
                rescues[gt_key] = {
                    "image_id": image_id,
                    "annotation_index": annotation_index,
                    "target_class": annotation.class_name,
                    "ground_truth_box": list(annotation.box),
                    "baseline_max_iou": baseline_max,
                    "new_view_best_iou": best_iou,
                    "matching_actions": sorted(
                        matching_actions, key=lambda item: item["max_iou"], reverse=True
                    ),
                }

    covered: set[str] = set()
    remaining = set(action_hits)
    greedy = []
    while remaining:
        key = max(
            remaining,
            key=lambda item: (
                len(action_hits[item] - covered) / max(action_costs[(item[0], item[2])], 1),
                len(action_hits[item] - covered),
                -action_costs[(item[0], item[2])],
            ),
        )
        new_hits = action_hits[key] - covered
        if not new_hits:
            break
        source_id, target_class, scale_index = key
        covered.update(new_hits)
        remaining.remove(key)
        greedy.append({
            "source_class": classes[source_id],
            "target_class": target_class,
            "scale": list(scales[scale_index]),
            "new_hits": sorted(new_hits),
            "emitted_boxes": action_costs[(source_id, scale_index)],
        })

    report = {
        "base_caches": [str(path.resolve()) for path in args.base_cache],
        "new_cache": str(args.new_cache.resolve()),
        "nms_iou": args.nms_iou,
        "iou": args.iou,
        "scales": [list(scale) for scale in scales],
        "ground_truth": total,
        "baseline_upper_hits": baseline_hits,
        "incremental_rescues": len(rescues),
        "combined_upper_hits": baseline_hits + len(rescues),
        "rescues": rescues,
        "greedy_incremental_actions": greedy,
        "greedy_emitted_boxes": sum(item["emitted_boxes"] for item in greedy),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
