from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal_grounding.data import load_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a normalized bbox on the visible image.")
    parser.add_argument("--data", required=True, help="Path to official JSON.")
    parser.add_argument("--root", default=None, help="Dataset root. Defaults to the JSON parent.")
    parser.add_argument("--sample-id", default=None, help="Query ID. Defaults to the first sample.")
    parser.add_argument("--output", default="runs/multimodal_grounding/sample_bbox.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_annotations(args.data, root=args.root, require_bbox=True)
    sample = next((item for item in samples if item.sample_id == args.sample_id), samples[0])
    if args.sample_id is not None and sample.sample_id != args.sample_id:
        raise KeyError(f"sample-id {args.sample_id!r} was not found.")

    with Image.open(sample.visible) as image:
        image = image.convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = sample.bbox or (0.0, 0.0, 1.0, 1.0)
        box = [x1 * width, y1 * height, x2 * width, y2 * height]
        draw = ImageDraw.Draw(image)
        line_width = max(3, round(min(width, height) * 0.004))
        draw.rectangle(box, outline=(255, 40, 40), width=line_width)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, quality=95)
        print(f"saved {sample.sample_id} visualization to {output}")


if __name__ == "__main__":
    main()
