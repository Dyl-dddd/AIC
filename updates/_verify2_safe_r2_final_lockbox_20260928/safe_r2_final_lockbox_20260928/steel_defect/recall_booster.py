"""Memory-bounded helpers for recall-oriented single-checkpoint inference."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from .geometry import box_iou


class _JsonStreamReader:
    """Decode one JSON value at a time without loading a cache into memory."""

    def __init__(self, handle, chunk_size: int = 1024 * 1024):
        self.handle = handle
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.position = 0
        self.eof = False

    def _fill(self) -> None:
        self.buffer = self.buffer[self.position:]
        self.position = 0
        chunk = self.handle.read(self.chunk_size)
        self.eof = not chunk
        self.buffer += chunk

    def peek(self) -> str:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                raise ValueError("Unexpected end of candidate cache")
            self._fill()

    def expect(self, character: str) -> None:
        if self.peek() != character:
            raise ValueError(f"Expected {character!r} in candidate cache")
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                item, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if self.eof:
                    raise ValueError("Invalid or truncated candidate cache") from None
                self._fill()
            else:
                if end == len(self.buffer) and not self.eof:
                    self._fill()
                    continue
                self.position = end
                return item


def iter_candidate_records(path: str | Path) -> Iterator[dict]:
    """Yield candidate-cache image records from legacy JSON or JSONL gzip."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    if path.name.endswith((".jsonl", ".jsonl.gz")):
        with opener(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("type") == "candidate_cache_header":
                    continue
                yield item
        return
    with opener(path, "rt", encoding="utf-8") as handle:
        reader = _JsonStreamReader(handle)
        reader.expect("{")
        found_images = False
        while reader.peek() != "}":
            key = reader.value()
            reader.expect(":")
            if key == "images":
                found_images = True
                reader.expect("{")
                previous_id = None
                while reader.peek() != "}":
                    image_id = reader.value()
                    reader.expect(":")
                    record = reader.value()
                    if previous_id is not None and image_id < previous_id:
                        raise ValueError("Candidate cache images are not sorted")
                    previous_id = image_id
                    yield {"image_id": image_id, **record}
                    if reader.peek() != "}":
                        reader.expect(",")
                reader.expect("}")
            else:
                reader.value()
            if reader.peek() != "}":
                reader.expect(",")
        reader.expect("}")
        if not found_images:
            raise ValueError("Candidate cache has no images mapping")


def agnostic_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(boxes) or iou_threshold <= 0:
        return boxes, scores, classes
    xywh = boxes.astype(np.float32, copy=True)
    xywh[:, 2:] -= xywh[:, :2]
    keep = cv2.dnn.NMSBoxes(
        xywh.tolist(), scores.astype(float).tolist(), 0.0, float(iou_threshold)
    )
    keep = np.asarray(keep, dtype=np.int64).reshape(-1)
    return boxes[keep], scores[keep], classes[keep]


def scale_boxes(
    boxes: np.ndarray,
    image_size: tuple[int, int],
    scale_x: float,
    scale_y: float,
    shift_x: float = 0.0,
    shift_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale boxes about their centers and return boxes plus input-valid mask."""
    if not len(boxes):
        return np.empty((0, 4), dtype=np.float32), np.zeros(0, dtype=bool)
    width, height = image_size
    centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    sizes = (boxes[:, 2:] - boxes[:, :2]) * np.asarray(
        [scale_x, scale_y], dtype=np.float32
    )
    centers = centers + sizes * np.asarray([shift_x, shift_y], dtype=np.float32)
    output = np.concatenate((centers - sizes * 0.5, centers + sizes * 0.5), axis=1)
    output[:, [0, 2]] = output[:, [0, 2]].clip(0, width)
    output[:, [1, 3]] = output[:, [1, 3]].clip(0, height)
    valid = (output[:, 2] > output[:, 0] + 1) & (output[:, 3] > output[:, 1] + 1)
    return output[valid], valid


def build_action_arrays(
    cache_items: Iterable[dict],
    class_names: list[str],
    actions: list[dict],
    pre_nms_iou: float,
    score_floor: float,
    final_nms_iou: float = 0.0,
    candidate_score_threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    """Apply a frozen action policy to the union of candidate-cache views."""
    items = list(cache_items)
    if not items:
        raise ValueError("At least one cache item is required")
    image_sizes = {tuple(item["image_size"]) for item in items}
    if len(image_sizes) != 1:
        raise ValueError(f"Candidate caches disagree on image size: {sorted(image_sizes)}")
    image_size = next(iter(image_sizes))
    candidates = [candidate for item in items for candidate in item["candidates"]]
    boxes = np.asarray([item["bbox"] for item in candidates], dtype=np.float32).reshape(-1, 4)
    scores = np.asarray([item["score"] for item in candidates], dtype=np.float32)
    classes = np.asarray([item["class_id"] for item in candidates], dtype=np.int64)
    if candidate_score_threshold > 0:
        selected = scores >= float(candidate_score_threshold)
        boxes, scores, classes = boxes[selected], scores[selected], classes[selected]
    boxes, scores, classes = agnostic_nms(boxes, scores, classes, pre_nms_iou)

    name_to_id = {name: index for index, name in enumerate(class_names)}
    output_boxes: list[np.ndarray] = []
    output_scores: list[np.ndarray] = []
    output_classes: list[np.ndarray] = []
    for action in actions:
        source_id = name_to_id[action["source_class"]]
        target_id = name_to_id[action["target_class"]]
        selected = classes == source_id
        source_boxes = boxes[selected]
        source_scores = scores[selected]
        scale_x, scale_y = map(float, action["scale"])
        shift_x, shift_y = map(float, action.get("shift", (0.0, 0.0)))
        transformed, valid = scale_boxes(
            source_boxes, image_size, scale_x, scale_y, shift_x, shift_y
        )
        if not len(transformed):
            continue
        output_boxes.append(transformed)
        output_scores.append(np.maximum(source_scores[valid], score_floor))
        output_classes.append(np.full(len(transformed), target_id, dtype=np.int64))
    if not output_boxes:
        empty = np.empty((0, 4), dtype=np.float32)
        return empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64), image_size
    boxes = np.concatenate(output_boxes)
    scores = np.concatenate(output_scores)
    classes = np.concatenate(output_classes)
    if final_nms_iou > 0:
        kept: list[np.ndarray] = []
        for target_id in np.unique(classes):
            indices = np.flatnonzero(classes == target_id)
            xywh = boxes[indices].astype(np.float32, copy=True)
            xywh[:, 2:] -= xywh[:, :2]
            local_keep = cv2.dnn.NMSBoxes(
                xywh.tolist(), scores[indices].astype(float).tolist(), 0.0, float(final_nms_iou)
            )
            kept.append(indices[np.asarray(local_keep, dtype=np.int64).reshape(-1)])
        keep = np.concatenate(kept) if kept else np.empty(0, dtype=np.int64)
        order = np.argsort(scores[keep])[::-1]
        keep = keep[order]
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    return boxes, scores, classes, image_size


def matched_ground_truth(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    ground_truth: dict[str, list[list[float]]],
    class_names: list[str],
    iou_threshold: float = 0.5,
) -> tuple[int, int, dict[str, tuple[int, int]]]:
    """Apply the repository's score-ordered one-to-one matcher for one image."""
    hit = 0
    total = 0
    per_class: dict[str, tuple[int, int]] = {}
    for class_id, class_name in enumerate(class_names):
        gt = np.asarray(ground_truth.get(class_name, []), dtype=np.float32).reshape(-1, 4)
        total += len(gt)
        if not len(gt):
            continue
        selected = np.flatnonzero(classes == class_id)
        selected = selected[np.argsort(scores[selected])[::-1]]
        matched: set[int] = set()
        for prediction_index in selected:
            ious = box_iou(boxes[prediction_index], gt)
            order = np.argsort(ious)[::-1]
            match = next((int(index) for index in order if int(index) not in matched), None)
            if match is not None and float(ious[match]) >= iou_threshold:
                matched.add(match)
                if len(matched) == len(gt):
                    break
        hit += len(matched)
        per_class[class_name] = (len(matched), len(gt))
    return hit, total, per_class
