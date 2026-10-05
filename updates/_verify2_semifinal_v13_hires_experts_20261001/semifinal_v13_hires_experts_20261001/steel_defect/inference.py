"""Single-model multi-view inference for high-resolution steel images."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .deblur import adaptive_deblur, apply_clahe
from .geometry import (
    Tile,
    classwise_nms,
    classwise_weighted_box_fusion,
    edge_score_factor,
    generate_tiles,
    generate_grid_tiles,
)
from .image_io import imread


@dataclass(frozen=True)
class InferenceOptions:
    tile_size: int = 1024
    overlap: float = 0.25
    imgsz: int = 1024
    batch: int = 4
    conf: float = 0.01
    iou: float = 0.55
    local_iou: float = 0.70
    device: str = "0"
    half: bool = True
    tta: bool = False
    deblur: bool = False
    deblur_threshold: float = 85.0
    clahe: bool = False
    edge_margin: int = 12
    edge_penalty: float = 1.0
    global_pass: bool = False
    merge: str = "nms"
    max_det: int = 1000
    class_thresholds: dict[str, float] = field(default_factory=dict)
    tile_layout: str = "sliding"

    def __post_init__(self) -> None:
        if self.tile_layout not in {"sliding", "semifinal_grid"}:
            raise ValueError("tile_layout must be sliding or semifinal_grid")
        if self.tile_size <= 0 or self.imgsz <= 0 or self.batch <= 0:
            raise ValueError("tile_size, imgsz, and batch must be positive")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError("conf must be in [0, 1]")
        if not 0.0 <= self.edge_penalty <= 1.0:
            raise ValueError("edge_penalty must be in [0, 1]")
        if self.deblur_threshold <= 0:
            raise ValueError("deblur_threshold must be positive")


def _enhance(image: np.ndarray, options: InferenceOptions) -> np.ndarray:
    output = (
        adaptive_deblur(image, threshold=options.deblur_threshold).image
        if options.deblur
        else image
    )
    if options.clahe:
        output = apply_clahe(output)
    return output


def _append_results(results, views, detections, options: InferenceOptions) -> None:
    for result, (tile, scale_x, scale_y, is_global) in zip(results, views, strict=True):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
        for box, score, class_id in zip(boxes, scores, classes, strict=True):
            touches_edge = False if is_global else edge_score_factor(
                box, tile, margin=options.edge_margin, penalty=0.0
            ) == 0.0
            x1, y1, x2, y2 = box
            detections.append(
                (
                    [x1 * scale_x + tile.x, y1 * scale_y + tile.y, x2 * scale_x + tile.x, y2 * scale_y + tile.y],
                    float(score),
                    int(class_id),
                    touches_edge,
                    is_global,
                )
            )


def _flush(model, images, views, detections, options: InferenceOptions) -> None:
    if not images:
        return
    results = model.predict(
        source=images,
        imgsz=options.imgsz,
        conf=options.conf,
        iou=options.local_iou,
        device=options.device,
        half=options.half,
        verbose=False,
        augment=options.tta,
        max_det=options.max_det,
    )
    _append_results(results, views, detections, options)
    images.clear()
    views.clear()


def predict_candidates(
    model, image_path: str | Path, options: InferenceOptions
) -> tuple[list[dict], tuple[int, int]]:
    """Run all views once and return unmerged original-coordinate candidates."""
    image_path = Path(image_path)
    image = imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise OSError(f"Cannot read {image_path}")
    return predict_candidates_array(model, image, options)


def inference_tiles(width: int, height: int, options: InferenceOptions) -> list[Tile]:
    """Use the same rectangular grid as preparation; never leave coverage gaps."""
    if options.tile_layout == "sliding":
        return generate_tiles(width, height, options.tile_size, options.overlap)
    if width > 4096 or height > 3000:
        raise ValueError("semifinal_grid supports images up to 4096x3000; use sliding otherwise")
    return generate_grid_tiles(width, height, 1387, 1516, rows=2, cols=3)


def predict_candidates_array(
    model, image: np.ndarray, options: InferenceOptions,
) -> tuple[list[dict], tuple[int, int]]:
    """Evaluate an in-memory grayscale view without lossy temporary JPEGs."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected nonempty grayscale image")
    height, width = image.shape
    detections: list[tuple[list[float], float, int, bool, bool]] = []
    batch_images: list[np.ndarray] = []
    batch_views: list[tuple[Tile, float, float, bool]] = []

    for tile in inference_tiles(width, height, options):
        crop = image[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
        crop = cv2.cvtColor(_enhance(crop, options), cv2.COLOR_GRAY2BGR)
        batch_images.append(crop)
        batch_views.append((tile, 1.0, 1.0, False))
        if len(batch_images) >= options.batch:
            _flush(model, batch_images, batch_views, detections, options)

    if options.global_pass:
        scale = min(1.0, options.imgsz / max(width, height))
        global_width = max(1, int(round(width * scale)))
        global_height = max(1, int(round(height * scale)))
        global_view = cv2.resize(image, (global_width, global_height), interpolation=cv2.INTER_AREA)
        global_view = cv2.cvtColor(_enhance(global_view, options), cv2.COLOR_GRAY2BGR)
        batch_images.append(global_view)
        batch_views.append((Tile(0, 0, global_width, global_height, global_width, global_height), 1.0 / scale, 1.0 / scale, True))
    _flush(model, batch_images, batch_views, detections, options)

    if not detections:
        return [], (width, height)
    boxes = np.asarray([item[0] for item in detections], dtype=np.float32)
    scores = np.asarray([item[1] for item in detections], dtype=np.float32)
    classes = np.asarray([item[2] for item in detections], dtype=np.int64)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    valid = (
        (boxes[:, 2] > boxes[:, 0] + 1)
        & (boxes[:, 3] > boxes[:, 1] + 1)
        & (classes >= 0)
    )
    output = []
    for index in np.flatnonzero(valid):
        output.append(
            {
                "bbox": [round(float(value), 4) for value in boxes[index]],
                "score": round(float(scores[index]), 7),
                "class_id": int(classes[index]),
                "touches_internal_edge": bool(detections[index][3]),
                "is_global": bool(detections[index][4]),
            }
        )
    return output, (width, height)


def merge_candidates(
    candidates: list[dict],
    image_size: tuple[int, int],
    class_names: list[str],
    options: InferenceOptions,
) -> list[dict]:
    """Apply edge weighting, fusion/NMS, and per-class thresholds deterministically."""
    if not candidates:
        return []
    width, height = image_size
    boxes = np.asarray([item["bbox"] for item in candidates], dtype=np.float32)
    scores = np.asarray(
        [
            float(item["score"]) * (options.edge_penalty if item.get("touches_internal_edge") else 1.0)
            for item in candidates
        ],
        dtype=np.float32,
    )
    classes = np.asarray([item["class_id"] for item in candidates], dtype=np.int64)
    valid = (classes >= 0) & (classes < len(class_names))
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    if options.merge == "wbf":
        boxes, scores, classes = classwise_weighted_box_fusion(boxes, scores, classes, options.iou)
    elif options.merge == "nms":
        keep = classwise_nms(boxes, scores, classes, options.iou)
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
    else:
        raise ValueError(f"Unknown merge method: {options.merge}")

    output = []
    for box, score, class_id in zip(boxes, scores, classes, strict=True):
        class_name = class_names[int(class_id)]
        threshold = options.class_thresholds.get(class_name, options.conf)
        if float(score) < threshold:
            continue
        x1, y1, x2, y2 = box
        output.append(
            {
                "category_name": class_name,
                "bbox": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
                "score": round(float(score), 6),
            }
        )
    return output


def predict_image(model, image_path: str | Path, class_names: list[str], options: InferenceOptions) -> list[dict]:
    candidates, image_size = predict_candidates(model, image_path, options)
    return merge_candidates(candidates, image_size, class_names, options)
