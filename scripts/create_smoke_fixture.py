"""Create a tiny deterministic VOC dataset for end-to-end pipeline smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.classes import CLASS_NAMES
from steel_defect.image_io import imwrite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--images", type=int, default=10)
    args = parser.parse_args()
    if args.images < 5:
        raise SystemExit("Use at least five images so a grouped train/val split is meaningful")
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    width, height = 256, 192
    boxes = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        column, row = class_id % 3, class_id // 3
        x1, y1 = 12 + column * 80, 10 + row * 58
        boxes.append((class_name, (x1, y1, x1 + 25, y1 + 18)))

    for index in range(args.images):
        stem = f"coil{index:03d}-Raw01-f_00001"
        image = rng.normal(105 + index, 8, size=(height, width)).clip(0, 255).astype(np.uint8)
        for class_id, (_, (x1, y1, x2, y2)) in enumerate(boxes):
            value = 155 + (class_id % 3) * 25
            cv2.rectangle(image, (x1, y1), (x2, y2), int(value), thickness=-1)
        if not imwrite(args.output / f"{stem}.jpg", image):
            raise OSError(f"Could not write smoke image {stem}")

        root = ET.Element("annotation")
        ET.SubElement(root, "filename").text = f"{stem}.jpg"
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "1"
        for class_name, (x1, y1, x2, y2) in boxes:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = class_name
            box = ET.SubElement(obj, "bndbox")
            for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (x1, y1, x2, y2), strict=True):
                ET.SubElement(box, key).text = str(value)
        ET.ElementTree(root).write(args.output / f"{stem}.xml", encoding="utf-8", xml_declaration=True)
    print(f"Created {args.images} VOC smoke images in {args.output.resolve()}")


if __name__ == "__main__":
    main()
