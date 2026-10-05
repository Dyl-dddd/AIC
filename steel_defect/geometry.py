"""Tiling and box post-processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Tile:
    x: int
    y: int
    width: int
    height: int
    image_width: int
    image_height: int

    @property
    def is_left_edge(self) -> bool:
        return self.x == 0

    @property
    def is_top_edge(self) -> bool:
        return self.y == 0

    @property
    def is_right_edge(self) -> bool:
        return self.x + self.width >= self.image_width

    @property
    def is_bottom_edge(self) -> bool:
        return self.y + self.height >= self.image_height


def _positions(length: int, tile_size: int, overlap: float) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, int(round(tile_size * (1.0 - overlap))))
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[Tile]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    xs = _positions(width, tile_size, overlap)
    ys = _positions(height, tile_size, overlap)
    return [
        Tile(x, y, min(tile_size, width - x), min(tile_size, height - y), width, height)
        for y in ys
        for x in xs
    ]


def generate_grid_tiles(
    width: int,
    height: int,
    tile_width: int,
    tile_height: int,
    rows: int,
    cols: int,
) -> list[Tile]:
    """Generate an edge-aligned fixed grid with evenly distributed overlap.

    The semifinal release uses 1387x1516 crops from 4096x3000 images on a
    2x3 grid. Evenly spacing the first and last crop gives x=[0, 1354, 2709]
    and y=[0, 1484], matching that construction exactly.
    """
    if min(width, height, tile_width, tile_height, rows, cols) <= 0:
        raise ValueError("image size, tile size, rows, and cols must be positive")
    tile_width = min(tile_width, width)
    tile_height = min(tile_height, height)

    def positions(length: int, tile: int, count: int) -> list[int]:
        last = length - tile
        if count == 1 or last == 0:
            return [0]
        return sorted({int(round(index * last / (count - 1))) for index in range(count)})

    xs = positions(width, tile_width, cols)
    ys = positions(height, tile_height, rows)
    return [
        Tile(x, y, tile_width, tile_height, width, height)
        for y in ys
        for x in xs
    ]


def generate_strip_tiles(
    width: int,
    height: int,
    strip_width: int,
    overlap: float,
) -> list[Tile]:
    """Full-height vertical strips sliding along x.

    Long defects such as zonglie lose their complete vertical extent when a
    1024-px square tile can only show a fragment; the fragment localization
    then lands below IoU 0.5. A strip keeps the full image height at native
    resolution and trades horizontal field of view for it. Horizontal
    positions follow the same overlap policy as square tiles, so strips and
    square tiles can be mixed inside one dataset or one inference pass.
    """
    if strip_width <= 0:
        raise ValueError("strip_width must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    xs = _positions(width, strip_width, overlap)
    return [
        Tile(x, 0, min(strip_width, width - x), height, width, height)
        for x in xs
    ]


def clip_box_to_tile(
    box: Iterable[float],
    tile: Tile,
    min_visibility: float = 0.35,
    min_size: float = 2.0,
) -> tuple[float, float, float, float] | None:
    """Clip an xyxy box to a tile and return tile-local coordinates."""
    x1, y1, x2, y2 = map(float, box)
    ix1 = max(x1, tile.x)
    iy1 = max(y1, tile.y)
    ix2 = min(x2, tile.x + tile.width)
    iy2 = min(y2, tile.y + tile.height)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw < min_size or ih < min_size:
        return None
    area = max(1e-9, (x2 - x1) * (y2 - y1))
    visibility = (iw * ih) / area
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    center_inside = tile.x <= cx < tile.x + tile.width and tile.y <= cy < tile.y + tile.height
    # A very long defect may cross several tiles while its full-box visibility
    # stays below the threshold in every tile. Keep those meaningful fragments;
    # otherwise they become false background examples. The global training view
    # remains responsible for learning the complete long-object extent.
    spans_tile_x = x1 <= tile.x and x2 >= tile.x + tile.width
    spans_tile_y = y1 <= tile.y and y2 >= tile.y + tile.height
    if visibility < min_visibility and not center_inside and not (spans_tile_x or spans_tile_y):
        return None
    return ix1 - tile.x, iy1 - tile.y, ix2 - tile.x, iy2 - tile.y


def xyxy_to_yolo(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = map(float, box)
    return (
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area1 + area2 - inter, 1e-9)


def classwise_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Greedy class-aware NMS over original-image xyxy boxes."""
    if len(boxes) == 0:
        return []
    keep: list[int] = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        while len(order):
            current = int(order[0])
            keep.append(current)
            if len(order) == 1:
                break
            rest = order[1:]
            order = rest[box_iou(boxes[current], boxes[rest]) <= iou_threshold]
    return sorted(keep, key=lambda idx: float(scores[idx]), reverse=True)


def classwise_weighted_box_fusion(
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse overlapping predictions from multiple views of the same model."""
    if len(boxes) == 0:
        return boxes.copy(), scores.copy(), classes.copy()
    fused_boxes: list[np.ndarray] = []
    fused_scores: list[float] = []
    fused_classes: list[int] = []
    for class_id in np.unique(classes):
        indices = np.flatnonzero(classes == class_id)
        order = indices[np.argsort(scores[indices])[::-1]]
        clusters: list[dict[str, object]] = []
        for index in order:
            box = boxes[index].astype(np.float64)
            score = float(scores[index])
            best_cluster = None
            best_iou = iou_threshold
            for cluster in clusters:
                iou = float(box_iou(box, np.asarray([cluster["box"]], dtype=np.float64))[0])
                if iou > best_iou:
                    best_iou = iou
                    best_cluster = cluster
            if best_cluster is None:
                clusters.append({"box": box, "weighted_sum": box * score, "weight": score, "score": score})
            else:
                best_cluster["weighted_sum"] = best_cluster["weighted_sum"] + box * score
                best_cluster["weight"] = float(best_cluster["weight"]) + score
                best_cluster["box"] = best_cluster["weighted_sum"] / max(float(best_cluster["weight"]), 1e-9)
                best_cluster["score"] = max(float(best_cluster["score"]), score)
        for cluster in clusters:
            fused_boxes.append(np.asarray(cluster["box"], dtype=np.float32))
            fused_scores.append(float(cluster["score"]))
            fused_classes.append(int(class_id))
    order = np.argsort(np.asarray(fused_scores))[::-1]
    return (
        np.asarray(fused_boxes, dtype=np.float32)[order],
        np.asarray(fused_scores, dtype=np.float32)[order],
        np.asarray(fused_classes, dtype=np.int64)[order],
    )


def edge_score_factor(box: Iterable[float], tile: Tile, margin: int = 12, penalty: float = 0.85) -> float:
    """Down-weight partial boxes touching an internal tile edge."""
    x1, y1, x2, y2 = map(float, box)
    touches_internal = (
        (x1 <= margin and not tile.is_left_edge)
        or (y1 <= margin and not tile.is_top_edge)
        or (x2 >= tile.width - margin and not tile.is_right_edge)
        or (y2 >= tile.height - margin and not tile.is_bottom_edge)
    )
    return penalty if touches_internal else 1.0
