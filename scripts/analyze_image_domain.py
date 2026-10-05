"""Compare aggregate image-domain statistics across one or more directories.

The analysis is deterministic and bounded. It reports only aggregate metadata
and pixel summaries; image names and raw pixels are never written to output.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.image_io import imread


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in (0, 10, 25, 50, 75, 90, 100)}


def _sample_indices(count: int, limit: int) -> np.ndarray:
    if count <= 0 or limit <= 0:
        return np.asarray([], dtype=np.int64)
    return np.linspace(0, count - 1, min(count, limit), dtype=np.int64)


def analyze_directory(source: Path, max_samples: int, preview_size: int) -> dict:
    source = source.resolve()
    images = sorted(
        path for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    metadata_errors = 0
    valid_paths: list[Path] = []
    for path in images:
        try:
            with Image.open(path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                valid_paths.append(path)
        except Exception:
            metadata_errors += 1

    means: list[float] = []
    stds: list[float] = []
    low_clip: list[float] = []
    high_clip: list[float] = []
    sharpness: list[float] = []
    edge_density: list[float] = []
    histogram = np.zeros(256, dtype=np.float64)
    pixel_errors = 0
    for index in _sample_indices(len(valid_paths), max_samples):
        image = imread(valid_paths[int(index)], cv2.IMREAD_GRAYSCALE)
        if image is None:
            pixel_errors += 1
            continue
        height, width = image.shape
        scale = min(1.0, preview_size / max(height, width))
        if scale < 1.0:
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        means.append(float(image.mean()))
        stds.append(float(image.std()))
        low_clip.append(float(np.mean(image <= 5)))
        high_clip.append(float(np.mean(image >= 250)))
        sharpness.append(float(cv2.Laplacian(image, cv2.CV_64F).var()))
        edges = cv2.Canny(image, 50, 150)
        edge_density.append(float(np.mean(edges > 0)))
        histogram += np.bincount(image.ravel(), minlength=256)

    histogram /= max(histogram.sum(), 1.0)
    return {
        "image_count": len(images),
        "valid_metadata": len(valid_paths),
        "metadata_errors": metadata_errors,
        "dimensions": dict(dimensions),
        "modes": dict(modes),
        "pixel_sample_count": len(means),
        "pixel_errors": pixel_errors,
        "mean_intensity": _percentiles(means),
        "contrast_std": _percentiles(stds),
        "low_clip_fraction": _percentiles(low_clip),
        "high_clip_fraction": _percentiles(high_clip),
        "preview_laplacian_variance": _percentiles(sharpness),
        "preview_edge_density": _percentiles(edge_density),
        "histogram_256": [round(float(value), 10) for value in histogram],
    }


def _js_divergence(left: list[float], right: list[float]) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    midpoint = 0.5 * (p + q)

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0
        return float(np.sum(first[mask] * np.log2(first[mask] / np.maximum(second[mask], 1e-300))))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded aggregate image-domain comparison")
    parser.add_argument(
        "--source",
        action="append",
        nargs=2,
        metavar=("NAME", "DIRECTORY"),
        required=True,
        help="Repeat for each named image directory",
    )
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--preview-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.max_samples <= 0 or args.preview_size <= 0:
        raise SystemExit("--max-samples and --preview-size must be positive")
    names = [name for name, _ in args.source]
    if len(names) != len(set(names)):
        raise SystemExit("Source names must be unique")

    datasets = {
        name: analyze_directory(Path(directory), args.max_samples, args.preview_size)
        for name, directory in args.source
    }
    comparisons: dict[str, dict[str, float]] = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            comparisons[f"{left}__vs__{right}"] = {
                "histogram_js_divergence_bits": round(
                    _js_divergence(datasets[left]["histogram_256"], datasets[right]["histogram_256"]),
                    8,
                )
            }
    report = {
        "scope": {
            "max_pixel_samples_per_source": args.max_samples,
            "preview_max_side": args.preview_size,
            "metadata_scan": "all discovered images",
        },
        "datasets": datasets,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = {
        "scope": report["scope"],
        "datasets": {
            name: {key: value for key, value in stats.items() if key != "histogram_256"}
            for name, stats in datasets.items()
        },
        "comparisons": comparisons,
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
