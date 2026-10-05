"""Stream-validate a large JSON-array submission with bounded memory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.classes import CLASS_NAMES


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        finished = False
        while True:
            chunk = handle.read(chunk_size)
            eof = not chunk
            buffer = buffer[position:] + chunk
            position = 0
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if not started:
                    if position >= len(buffer):
                        break
                    if buffer[position] != "[":
                        raise ValueError("Submission root must be a JSON array")
                    started = True
                    position += 1
                    continue
                while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                    position += 1
                if position >= len(buffer):
                    break
                if buffer[position] == "]":
                    finished = True
                    position += 1
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position != len(buffer) or not eof:
                        tail = buffer[position:] + handle.read()
                        if tail.strip():
                            raise ValueError("Unexpected data after JSON array")
                    return
                try:
                    item, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError("Truncated or invalid JSON object") from None
                    break
                yield item
                position = end
            if eof:
                break
        if not started or not finished:
            raise ValueError("Truncated JSON array")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--exclude-category", action="append", default=[],
        help="Disallow a training category in this submission (repeatable).",
    )
    args = parser.parse_args()

    excluded_categories = set(args.exclude_category)
    unknown_exclusions = excluded_categories - set(CLASS_NAMES)
    if unknown_exclusions:
        raise ValueError(f"Unknown categories to exclude: {sorted(unknown_exclusions)}")
    allowed_categories = set(CLASS_NAMES) - excluded_categories

    expected_images = {
        path.name for path in args.source.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    }
    if not expected_images:
        raise ValueError(f"No images found under {args.source}")
    required = {"image_id", "category_name", "bbox", "score"}
    seen_images: set[str] = set()
    class_counts: Counter[str] = Counter()
    prediction_count = 0
    for prediction_count, item in enumerate(iter_json_array(args.submission), start=1):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"Prediction {prediction_count} must contain exactly {sorted(required)}")
        image_id = item["image_id"]
        category = item["category_name"]
        bbox = item["bbox"]
        score = item["score"]
        if image_id not in expected_images:
            raise ValueError(f"Prediction {prediction_count} has unknown image_id {image_id!r}")
        if category not in allowed_categories:
            raise ValueError(f"Prediction {prediction_count} has unknown category {category!r}")
        if (
            not isinstance(bbox, list) or len(bbox) != 4
            or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
            or min(bbox) < 0 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]
        ):
            raise ValueError(f"Prediction {prediction_count} has invalid bbox {bbox!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"Prediction {prediction_count} has invalid score {score!r}")
        seen_images.add(image_id)
        class_counts[category] += 1

    missing = sorted(expected_images - seen_images)
    report = {
        "valid": True,
        "predictions": prediction_count,
        "expected_images": len(expected_images),
        "images_with_predictions": len(seen_images),
        "missing_images": missing,
        "allowed_categories": sorted(allowed_categories),
        "class_counts": dict(class_counts),
        "submission_bytes": args.submission.stat().st_size,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
