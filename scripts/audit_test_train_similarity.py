"""Read-only screen for near-duplicate official train/Test images.

No labels are loaded and no Test annotations are inferred. This only measures
whether filenames sharing a source prefix are visually related.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/official/train"
TEST = ROOT / "data/official/test"
SPLIT = ROOT / "data/official_v2_20260831b/splits.json"


def feature(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    thumb = cv2.resize(image, (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    thumb = cv2.GaussianBlur(thumb, (5, 5), 0)
    thumb -= thumb.mean()
    thumb /= max(float(np.linalg.norm(thumb)), 1e-8)
    return thumb.ravel()


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))["splits"]
    grouped = defaultdict(list)
    for name in split["train"]:
        grouped[name.split("-")[0]].append(name)
    scores = []
    for test_path in sorted(TEST.glob("*.jpg")):
        matches = grouped.get(test_path.name.split("-")[0], [])
        if not matches:
            continue
        target = feature(test_path)
        comparisons = []
        for name in matches:
            path = TRAIN / name
            if path.exists():
                comparisons.append((float(target @ feature(path)), name))
        if comparisons:
            best, nearest = max(comparisons)
            scores.append({"test": test_path.name, "nearest_train": nearest,
                           "correlation": round(best, 4), "same_prefix_train_count": len(comparisons)})
    scores.sort(key=lambda row: row["correlation"], reverse=True)
    print(json.dumps({"overlap_test_images": len(scores),
                      "correlation_ge_0_9": sum(row["correlation"] >= .9 for row in scores),
                      "correlation_ge_0_8": sum(row["correlation"] >= .8 for row in scores),
                      "top": scores[:30]}, indent=2))


if __name__ == "__main__":
    main()
