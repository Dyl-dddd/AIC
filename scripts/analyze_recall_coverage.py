"""Measure class-agnostic proposal coverage for recall-oriented inference.

This diagnostic never reads test labels and never creates a submission.  It
uses the frozen development split to estimate how many ground-truth boxes have
at least one geometric proposal at a requested IoU.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import cv2
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.geometry import box_iou
from steel_defect.voc import discover_pairs, parse_voc


POLICIES: dict[str, tuple[tuple[float, float], ...]] = {
    "identity": ((1.0, 1.0),),
    "compact": (
        (1.0, 1.0), (0.75, 0.75), (1.25, 1.25), (1.5, 1.5),
        (0.75, 1.0), (1.25, 1.0), (1.0, 0.75), (1.0, 1.25),
        (1.0, 1.5), (1.0, 2.0), (1.0, 3.0), (1.5, 1.0),
        (2.0, 1.0), (3.0, 1.0),
    ),
    "broad": tuple(
        (scale_x, scale_y)
        for scale_x in (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0)
        for scale_y in (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0)
    ),
}


def read_cache(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        images = {}
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("type") == "candidate_cache_header":
                    continue
                image_id = item.pop("image_id")
                images[image_id] = item
        return {"images": images}
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def transform_boxes(
    boxes: np.ndarray,
    image_size: tuple[int, int],
    scales: tuple[tuple[float, float], ...],
    shift_fraction: float,
) -> np.ndarray:
    if not len(boxes):
        return np.empty((0, 4), dtype=np.float32)
    width, height = image_size
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    sizes = boxes[:, 2:] - boxes[:, :2]
    shifts = ((0.0, 0.0),)
    if shift_fraction > 0:
        f = shift_fraction
        shifts = ((0.0, 0.0), (-f, 0.0), (f, 0.0), (0.0, -f), (0.0, f))
    variants = []
    for scale_x, scale_y in scales:
        scaled = sizes * np.asarray([scale_x, scale_y], dtype=np.float32)
        for shift_x, shift_y in shifts:
            shifted_center = centers + scaled * np.asarray([shift_x, shift_y], dtype=np.float32)
            variants.append(np.concatenate((shifted_center - scaled * 0.5, shifted_center + scaled * 0.5), axis=1))
    output = np.concatenate(variants, axis=0)
    output[:, [0, 2]] = output[:, [0, 2]].clip(0, width)
    output[:, [1, 3]] = output[:, [1, 3]].clip(0, height)
    valid = (output[:, 2] > output[:, 0] + 1) & (output[:, 3] > output[:, 1] + 1)
    return output[valid]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, action="append", required=True, help="Repeat to evaluate the union of view caches")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, help="Optional path for the JSON report")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--image", action="append", help="Limit diagnostics to a split image basename; repeat as needed")
    parser.add_argument("--policy", choices=tuple(POLICIES), default="identity")
    parser.add_argument("--scale", nargs=2, type=float, action="append", metavar=("SX", "SY"), help="Override policy with one or more explicit scale pairs")
    parser.add_argument("--shift-fraction", type=float, default=0.0)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=0, help="Keep only the highest-scoring K raw candidates per image before transforms")
    parser.add_argument("--agnostic-nms-iou", type=float, default=0.0, help="Apply class-agnostic OpenCV NMS before transforms; 0 disables it")
    parser.add_argument("--class-aware", action="store_true", help="Only allow candidates predicted as the GT class")
    parser.add_argument("--report-confusions", action="store_true", help="Report wrong predicted classes that can rescue class misses")
    parser.add_argument("--report-ground-truth", action="store_true", help="Include per-annotation maximum IoU and hit status")
    parser.add_argument("--report-scale-greedy", action="store_true", help="Report a greedy minimum scale subset by GT coverage")
    parser.add_argument(
        "--report-action-greedy",
        action="store_true",
        help="Greedily select predicted-class -> output-class + scale actions by coverage per emitted box",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.98,
        help="Stop action selection after this micro recall is reached",
    )
    parser.add_argument(
        "--report-action-milp",
        action="store_true",
        help="Solve the minimum-emitted-box action cover at target recall",
    )
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    selected = set(manifest["splits"][args.split])
    pairs, missing_xml, _ = discover_pairs(args.source)
    if missing_xml:
        raise ValueError(f"Missing XML for {len(missing_xml)} images")
    records = [
        parse_voc(image_path, xml_path)
        for image_path, xml_path in pairs
        if image_path.relative_to(args.source).as_posix() in selected
        and (not args.image or image_path.name in args.image)
    ]
    if not records:
        raise ValueError("No split images matched the requested image filter")
    caches = [read_cache(path) for path in args.cache]
    scales = tuple((float(x), float(y)) for x, y in args.scale) if args.scale else POLICIES[args.policy]
    hit = 0
    total = 0
    proposals = 0
    per_class: dict[str, list[int]] = {}
    max_ious: list[float] = []
    ground_truth_details: list[dict] = []
    confusion_rescues: Counter[tuple[str, str]] = Counter()
    scale_hits: list[set[int]] = [set() for _ in scales]
    action_hits: dict[tuple[int, str, int], set[int]] = {}
    action_costs: Counter[tuple[int, int]] = Counter()
    for record in records:
        image_id = record.image_path.relative_to(args.source).as_posix()
        raw_items = [cache["images"][image_id] for cache in caches]
        image_sizes = {tuple(item["image_size"]) for item in raw_items}
        if len(image_sizes) != 1:
            raise ValueError(f"Candidate caches disagree on image size for {image_id}: {sorted(image_sizes)}")
        raw_boxes = np.asarray(
            [candidate["bbox"] for item in raw_items for candidate in item["candidates"]],
            dtype=np.float32,
        ).reshape(-1, 4)
        raw_classes = np.asarray(
            [candidate["class_id"] for item in raw_items for candidate in item["candidates"]],
            dtype=np.int64,
        )
        raw_scores = np.asarray(
            [candidate["score"] for item in raw_items for candidate in item["candidates"]],
            dtype=np.float32,
        )
        if args.top_k > 0 and len(raw_boxes) > args.top_k:
            keep = np.argpartition(raw_scores, -args.top_k)[-args.top_k:]
            raw_boxes = raw_boxes[keep]
            raw_classes = raw_classes[keep]
            raw_scores = raw_scores[keep]
        if args.agnostic_nms_iou > 0 and len(raw_boxes):
            xywh = raw_boxes.copy()
            xywh[:, 2:] -= xywh[:, :2]
            keep = cv2.dnn.NMSBoxes(
                xywh.tolist(), raw_scores.tolist(), 0.0, args.agnostic_nms_iou
            )
            keep = np.asarray(keep, dtype=np.int64).reshape(-1)
            raw_boxes = raw_boxes[keep]
            raw_classes = raw_classes[keep]
            raw_scores = raw_scores[keep]
        image_size = next(iter(image_sizes))
        boxes = transform_boxes(raw_boxes, image_size, scales, args.shift_fraction)
        proposals += len(boxes)
        by_class = {}
        if args.class_aware or args.report_confusions:
            for class_id, class_name in enumerate(manifest["classes"]):
                by_class[class_name] = transform_boxes(
                    raw_boxes[raw_classes == class_id], image_size, scales, args.shift_fraction
                )
        action_boxes: dict[tuple[int, int], np.ndarray] = {}
        if args.report_action_greedy or args.report_action_milp:
            for source_id in range(len(manifest["classes"])):
                source_boxes = raw_boxes[raw_classes == source_id]
                for scale_index, scale in enumerate(scales):
                    transformed = transform_boxes(
                        source_boxes, image_size, (scale,), args.shift_fraction
                    )
                    action_boxes[(source_id, scale_index)] = transformed
                    action_costs[(source_id, scale_index)] += len(transformed)
        for annotation in record.annotations:
            ground_truth_index = total
            candidate_boxes = by_class.get(annotation.class_name, boxes)
            maximum = (
                float(box_iou(np.asarray(annotation.box, dtype=np.float32), candidate_boxes).max())
                if len(candidate_boxes) else 0.0
            )
            if args.report_confusions:
                class_maxima = {
                    class_name: (
                        float(box_iou(np.asarray(annotation.box, dtype=np.float32), class_boxes).max())
                        if len(class_boxes) else 0.0
                    )
                    for class_name, class_boxes in by_class.items()
                }
                correct_maximum = class_maxima.get(annotation.class_name, 0.0)
                source_class, rescue_maximum = max(class_maxima.items(), key=lambda item: item[1])
                if correct_maximum < args.iou <= rescue_maximum:
                    confusion_rescues[(source_class, annotation.class_name)] += 1
            matched = int(maximum >= args.iou)
            counts = per_class.setdefault(annotation.class_name, [0, 0])
            counts[0] += matched
            counts[1] += 1
            hit += matched
            total += 1
            max_ious.append(maximum)
            if args.report_ground_truth:
                ground_truth_details.append({
                    "image_id": image_id,
                    "class_name": annotation.class_name,
                    "box": list(annotation.box),
                    "max_iou": maximum,
                    "hit": bool(matched),
                })
            if args.report_scale_greedy:
                source_boxes = raw_boxes
                if args.class_aware:
                    class_id = manifest["classes"].index(annotation.class_name)
                    source_boxes = raw_boxes[raw_classes == class_id]
                for scale_index, scale in enumerate(scales):
                    scaled_boxes = transform_boxes(source_boxes, image_size, (scale,), args.shift_fraction)
                    scale_maximum = (
                        float(box_iou(np.asarray(annotation.box, dtype=np.float32), scaled_boxes).max())
                        if len(scaled_boxes) else 0.0
                    )
                    if scale_maximum >= args.iou:
                        scale_hits[scale_index].add(ground_truth_index)
            if args.report_action_greedy or args.report_action_milp:
                for (source_id, scale_index), transformed in action_boxes.items():
                    if not len(transformed):
                        continue
                    action_maximum = float(
                        box_iou(np.asarray(annotation.box, dtype=np.float32), transformed).max()
                    )
                    if action_maximum >= args.iou:
                        key = (source_id, annotation.class_name, scale_index)
                        action_hits.setdefault(key, set()).add(ground_truth_index)
    greedy_scales = []
    covered: set[int] = set()
    remaining = set(range(len(scales)))
    while remaining:
        selected = max(remaining, key=lambda index: len(scale_hits[index] - covered))
        gain = len(scale_hits[selected] - covered)
        if gain == 0:
            break
        covered.update(scale_hits[selected])
        remaining.remove(selected)
        greedy_scales.append({
            "scale": list(scales[selected]),
            "new_hits": gain,
            "cumulative_hits": len(covered),
            "cumulative_recall": len(covered) / max(total, 1),
        })
    action_greedy = []
    action_covered: set[int] = set()
    action_remaining = set(action_hits)
    target_hits = int(np.ceil(args.target_recall * total))
    while action_remaining and len(action_covered) < target_hits:
        def action_rank(key: tuple[int, str, int]) -> tuple[float, int, int]:
            gain = len(action_hits[key] - action_covered)
            cost = action_costs[(key[0], key[2])]
            return (gain / max(cost, 1), gain, -cost)

        selected_action = max(action_remaining, key=action_rank)
        new_hits = action_hits[selected_action] - action_covered
        if not new_hits:
            break
        source_id, target_class, scale_index = selected_action
        cost = action_costs[(source_id, scale_index)]
        action_covered.update(new_hits)
        action_remaining.remove(selected_action)
        action_greedy.append({
            "source_class": manifest["classes"][source_id],
            "target_class": target_class,
            "scale": list(scales[scale_index]),
            "new_hits": len(new_hits),
            "emitted_boxes": cost,
            "cumulative_boxes": sum(item["emitted_boxes"] for item in action_greedy) + cost,
            "cumulative_hits": len(action_covered),
            "cumulative_recall": len(action_covered) / max(total, 1),
        })
    action_milp = []
    action_milp_covered: set[int] = set()
    action_milp_status = "not_requested"
    if args.report_action_milp:
        action_keys = sorted(action_hits)
        action_count = len(action_keys)
        # Binary action variables followed by one binary "uncovered" variable per GT.
        variable_count = action_count + total
        objective = np.zeros(variable_count, dtype=np.float64)
        for index, (source_id, _target_class, scale_index) in enumerate(action_keys):
            objective[index] = action_costs[(source_id, scale_index)]
        # A tiny miss penalty only breaks equal-cost ties; action count remains dominant.
        objective[action_count:] = 1e-3
        constraints = lil_matrix((total + 1, variable_count), dtype=np.float64)
        for action_index, key in enumerate(action_keys):
            for ground_truth_index in action_hits[key]:
                constraints[ground_truth_index, action_index] = -1.0
        for ground_truth_index in range(total):
            constraints[ground_truth_index, action_count + ground_truth_index] = -1.0
        max_misses = total - target_hits
        constraints[total, action_count:] = 1.0
        lower = np.full(total + 1, -np.inf, dtype=np.float64)
        upper = np.concatenate((np.full(total, -1.0), np.asarray([max_misses], dtype=np.float64)))
        result = milp(
            c=objective,
            integrality=np.ones(variable_count, dtype=np.int8),
            bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
            constraints=LinearConstraint(constraints.tocsr(), lower, upper),
            options={"time_limit": 120.0, "mip_rel_gap": 0.0},
        )
        action_milp_status = str(result.message)
        if result.x is not None:
            chosen = np.flatnonzero(result.x[:action_count] > 0.5)
            for action_index in chosen:
                source_id, target_class, scale_index = action_keys[int(action_index)]
                action_milp_covered.update(action_hits[action_keys[int(action_index)]])
                action_milp.append({
                    "source_class": manifest["classes"][source_id],
                    "target_class": target_class,
                    "scale": list(scales[scale_index]),
                    "hits": len(action_hits[action_keys[int(action_index)]]),
                    "emitted_boxes": action_costs[(source_id, scale_index)],
                })
    report = {
        "policy": args.policy,
        "caches": [str(path.resolve()) for path in args.cache],
        "class_aware": args.class_aware,
        "shift_fraction": args.shift_fraction,
        "top_k": args.top_k,
        "agnostic_nms_iou": args.agnostic_nms_iou,
        "scale_variants": len(scales),
        "proposal_boxes": proposals,
        "cross_class_submission_boxes": proposals * len(manifest["classes"]),
        "hits": hit,
        "ground_truth": total,
        "micro_recall_upper_bound": hit / total,
        "proxy_score_upper_bound": 100.0 * hit / total,
        "per_class": per_class,
        "ground_truth_details": ground_truth_details if args.report_ground_truth else [],
        "confusion_rescues": {
            f"{source}->{target}": count
            for (source, target), count in confusion_rescues.most_common()
        },
        "scale_individual_hits": [
            {"scale": list(scale), "hits": len(scale_hits[index])}
            for index, scale in enumerate(scales)
        ] if args.report_scale_greedy else [],
        "scale_greedy_order": greedy_scales,
        "action_greedy_order": action_greedy if args.report_action_greedy else [],
        "action_greedy_boxes": (
            sum(item["emitted_boxes"] for item in action_greedy)
            if args.report_action_greedy else 0
        ),
        "action_greedy_hits": len(action_covered) if args.report_action_greedy else 0,
        "action_greedy_recall": (
            len(action_covered) / max(total, 1) if args.report_action_greedy else 0.0
        ),
        "action_milp_status": action_milp_status,
        "action_milp_selected": action_milp,
        "action_milp_boxes": sum(item["emitted_boxes"] for item in action_milp),
        "action_milp_hits": len(action_milp_covered),
        "action_milp_recall": len(action_milp_covered) / max(total, 1),
        "max_iou_quantiles": dict(zip(
            ("q00", "q01", "q02", "q05", "q10", "q25", "q50", "q100"),
            np.quantile(max_ious, (0, .01, .02, .05, .10, .25, .50, 1)).tolist(),
        )),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
