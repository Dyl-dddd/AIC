from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from multimodal_grounding.data import MODALITY_KEYS, check_files, load_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit AIC multimodal grounding JSON data.")
    parser.add_argument("--data", required=True, help="Path to official JSON annotation file.")
    parser.add_argument("--root", default=None, help="Dataset root. Defaults to the JSON parent.")
    parser.add_argument("--require-bbox", action="store_true", help="Require every sample to include bbox.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_annotations(args.data, root=args.root, require_bbox=args.require_bbox)
    print(f"samples: {len(samples)}")
    image_groups = Counter(str(sample.visible) for sample in samples)
    print(f"image groups: {len(image_groups)}")
    print(f"queries per image: min={min(image_groups.values())}, max={max(image_groups.values())}")

    missing = check_files(samples)
    print(f"missing modality files: {len(missing)}")
    for sample_id, key, path in missing[:20]:
        print(f"  {sample_id} {key}: {path}")

    if missing:
        return

    for key in MODALITY_KEYS:
        sizes: Counter[tuple[int, int]] = Counter()
        modes: Counter[str] = Counter()
        for sample in samples:
            with Image.open(getattr(sample, key)) as image:
                sizes[image.size] += 1
                modes[image.mode] += 1
        print(f"{key} modes: {dict(modes)}")
        print(f"{key} sizes: {dict(sizes.most_common(5))}")

    labeled = [sample for sample in samples if sample.bbox is not None]
    print(f"labeled samples: {len(labeled)}")
    if labeled:
        boxes = np.asarray([sample.bbox for sample in labeled], dtype=np.float32)
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        print(f"bbox width mean={widths.mean():.4f}, p50={np.percentile(widths, 50):.4f}")
        print(f"bbox height mean={heights.mean():.4f}, p50={np.percentile(heights, 50):.4f}")
        print(f"bbox area mean={areas.mean():.5f}, p50={np.percentile(areas, 50):.5f}")
        print(f"first sample: {labeled[0].sample_id} | {labeled[0].query}")


if __name__ == "__main__":
    main()
