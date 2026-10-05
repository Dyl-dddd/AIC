"""OpenCV image I/O that also works with non-ASCII Windows paths."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(
    path: str | Path,
    image: np.ndarray,
    extension: str | None = None,
    params: list[int] | None = None,
) -> bool:
    path = Path(path)
    suffix = extension or path.suffix
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    ok, encoded = cv2.imencode(suffix, image, params or [])
    if not ok:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded.tofile(str(path))
    except OSError:
        return False
    return True

