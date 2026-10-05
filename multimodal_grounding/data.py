from __future__ import annotations

import hashlib
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


MODALITY_KEYS = ("visible", "infrared", "depth")
TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class GroundingSample:
    sample_id: str
    visible: Path
    infrared: Path
    depth: Path
    query: str
    bbox: tuple[float, float, float, float] | None
    raw: dict[str, Any]

    def prediction_record(self, bbox: Iterable[float], decimals: int = 6) -> dict[str, Any]:
        record = dict(self.raw)
        record["bbox"] = [round(float(v), decimals) for v in bbox]
        return record


def load_annotations(
    json_path: str | Path,
    root: str | Path | None = None,
    require_bbox: bool = False,
) -> list[GroundingSample]:
    """Load the official query-indexed JSON format.

    The official format is:
    {
      "QueryID": {
        "visible": "Images/visible/xxxxx.png",
        "infrared": "Images/infrared/xxxxx.png",
        "depth": "Images/depth/xxxxx.png",
        "query": "...",
        "bbox": [x1, y1, x2, y2]
      }
    }
    """
    json_path = Path(json_path)
    data_root = Path(root) if root is not None else json_path.parent
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    if isinstance(payload, list):
        items = []
        for index, record in enumerate(payload):
            if not isinstance(record, dict):
                raise ValueError(f"Entry {index} is not an object.")
            sample_id = str(record.get("id") or record.get("query_id") or index)
            items.append((sample_id, record))
    elif isinstance(payload, dict):
        items = [(str(sample_id), record) for sample_id, record in payload.items()]
    else:
        raise ValueError("Annotation JSON must be a dict or a list.")

    samples: list[GroundingSample] = []
    for sample_id, record in items:
        if not isinstance(record, dict):
            raise ValueError(f"Entry {sample_id!r} is not an object.")
        missing = [key for key in (*MODALITY_KEYS, "query") if key not in record]
        if missing:
            raise ValueError(f"Entry {sample_id!r} is missing fields: {missing}")

        bbox = _parse_bbox(record.get("bbox"), sample_id) if "bbox" in record else None
        if require_bbox and bbox is None:
            raise ValueError(f"Entry {sample_id!r} has no bbox label.")

        samples.append(
            GroundingSample(
                sample_id=sample_id,
                visible=_resolve_path(data_root, record["visible"]),
                infrared=_resolve_path(data_root, record["infrared"]),
                depth=_resolve_path(data_root, record["depth"]),
                query=str(record["query"]),
                bbox=bbox,
                raw=dict(record),
            )
        )
    return samples


def _resolve_path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _parse_bbox(value: Any, sample_id: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"Entry {sample_id!r} bbox must be a 4-element list.")
    bbox = tuple(float(v) for v in value)
    if any(math.isnan(v) or math.isinf(v) for v in bbox):
        raise ValueError(f"Entry {sample_id!r} bbox contains NaN or Inf.")
    if not (0.0 <= bbox[0] < bbox[2] <= 1.0 and 0.0 <= bbox[1] < bbox[3] <= 1.0):
        raise ValueError(f"Entry {sample_id!r} bbox must be normalized [x1,y1,x2,y2].")
    return bbox


def tokenize_query(query: str, max_tokens: int, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if vocab_size < 3:
        raise ValueError("vocab_size must be >= 3.")
    words = TOKEN_RE.findall(query.lower())
    token_ids = [_hash_token(word, vocab_size) for word in words[:max_tokens]]
    if not token_ids:
        token_ids = [1]
    mask = [1] * len(token_ids)
    pad = max_tokens - len(token_ids)
    if pad > 0:
        token_ids.extend([0] * pad)
        mask.extend([0] * pad)
    return torch.tensor(token_ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


def _hash_token(token: str, vocab_size: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest()
    value = int.from_bytes(digest, "little")
    return 2 + value % (vocab_size - 2)


def read_multimodal_tensor(sample: GroundingSample, image_size: int) -> torch.Tensor:
    visible = _read_rgb(sample.visible, image_size)
    infrared = _read_rgb(sample.infrared, image_size)
    depth = _read_depth(sample.depth, image_size)
    stacked = np.concatenate([visible, infrared, depth], axis=0)
    return torch.from_numpy(stacked.astype(np.float32, copy=False))


def _read_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1)


def _read_depth(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    depth = array.astype(np.float32)
    valid = depth > 0
    if np.any(valid):
        valid_values = depth[valid]
        lo = float(np.percentile(valid_values, 1))
        hi = float(np.percentile(valid_values, 99))
        if hi <= lo:
            hi = float(valid_values.max())
            lo = float(valid_values.min())
        if hi > lo:
            depth = (np.clip(depth, lo, hi) - lo) / (hi - lo)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)
        depth[~valid] = 0.0
    else:
        depth = np.zeros_like(depth, dtype=np.float32)
    depth = np.asarray(
        Image.fromarray(depth).resize((image_size, image_size), Image.Resampling.BILINEAR),
        dtype=np.float32,
    )
    return depth[None, :, :]


class GroundingDataset(Dataset):
    def __init__(
        self,
        samples: list[GroundingSample],
        image_size: int = 384,
        max_tokens: int = 32,
        vocab_size: int = 8192,
        augment: bool = False,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.max_tokens = max_tokens
        self.vocab_size = vocab_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = read_multimodal_tensor(sample, self.image_size)
        bbox = torch.tensor(sample.bbox, dtype=torch.float32) if sample.bbox is not None else None
        if self.augment and bbox is not None and random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            x1, y1, x2, y2 = bbox.tolist()
            bbox = torch.tensor([1.0 - x2, y1, 1.0 - x1, y2], dtype=torch.float32)
        tokens, mask = tokenize_query(sample.query, self.max_tokens, self.vocab_size)
        return {
            "sample": sample,
            "image": image,
            "tokens": tokens,
            "mask": mask,
            "bbox": bbox,
        }


def collate_grounding(batch: list[dict[str, Any]]) -> dict[str, Any]:
    bboxes = [item["bbox"] for item in batch]
    return {
        "samples": [item["sample"] for item in batch],
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "tokens": torch.stack([item["tokens"] for item in batch], dim=0),
        "mask": torch.stack([item["mask"] for item in batch], dim=0),
        "bbox": torch.stack(bboxes, dim=0) if all(b is not None for b in bboxes) else None,
    }


def check_files(samples: Iterable[GroundingSample]) -> list[tuple[str, str, Path]]:
    missing: list[tuple[str, str, Path]] = []
    for sample in samples:
        for key in MODALITY_KEYS:
            path = getattr(sample, key)
            if not path.exists():
                missing.append((sample.sample_id, key, path))
    return missing
