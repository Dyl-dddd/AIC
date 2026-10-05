from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.classes import CLASS_NAMES
from steel_defect.deblur import laplacian_variance
from steel_defect.image_io import imread
from steel_defect.voc import discover_pairs, parse_voc


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": round(float(np.percentile(arr, p)), 3) for p in (0, 10, 25, 50, 75, 90, 100)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit VOC annotations and image quality.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON report path")
    parser.add_argument("--quality-samples", type=int, default=300)
    parser.add_argument("--allow-unknown", action="store_true")
    args = parser.parse_args()

    pairs, missing_xml, orphan_xml = discover_pairs(args.source)
    counts: Counter[str] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    area_ratios: list[float] = []
    errors: list[str] = []
    records = []
    for image_path, xml_path in pairs:
        try:
            record = parse_voc(image_path, xml_path, strict=not args.allow_unknown)
        except Exception as exc:
            errors.append(str(exc))
            continue
        records.append(record)
        for ann in record.annotations:
            x1, y1, x2, y2 = ann.box
            w, h = x2 - x1, y2 - y1
            counts[ann.class_name] += 1
            widths.append(w)
            heights.append(h)
            area_ratios.append((w * h) / (record.width * record.height))

    blur_scores: list[float] = []
    if records and args.quality_samples > 0:
        indices = np.linspace(0, len(records) - 1, min(args.quality_samples, len(records)), dtype=int)
        for idx in indices:
            image = imread(records[int(idx)].image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                preview = cv2.resize(image, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                blur_scores.append(laplacian_variance(preview))

    report = {
        "source": str(args.source.resolve()),
        "paired_images": len(pairs),
        "valid_records": len(records),
        "missing_xml": [str(p) for p in missing_xml],
        "orphan_xml": [str(p) for p in orphan_xml],
        "errors": errors,
        "class_counts": {name: counts[name] for name in CLASS_NAMES},
        "box_width_px": percentile_summary(widths),
        "box_height_px": percentile_summary(heights),
        "box_area_ratio": percentile_summary(area_ratios),
        "quarter_scale_laplacian_variance": percentile_summary(blur_scores),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
