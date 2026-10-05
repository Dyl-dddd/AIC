import gzip
import json
from pathlib import Path

import pytest

from scripts.run_ensemble_v7_tta_submission_pipeline import validate_tta_cache
from steel_defect.classes import CLASS_NAMES


def write_cache(path: Path, *, tta: bool = True) -> None:
    header = {
        "type": "candidate_cache_header",
        "schema": 3,
        "weights_sha256": "abc",
        "classes": CLASS_NAMES,
        "options": {
            "tile_layout": "semifinal_grid",
            "tile_size": 1024,
            "overlap": 0.25,
            "imgsz": 1024,
            "batch": 1,
            "conf": 0.00001,
            "local_iou": 0.70,
            "half": True,
            "tta": tta,
            "deblur": False,
            "clahe": False,
            "global_pass": True,
            "max_det": 2000,
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        handle.write(json.dumps({"image_id": "sample.jpg", "candidates": []}) + "\n")


def test_validate_tta_cache_accepts_exact_policy(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl.gz"
    write_cache(path)
    assert validate_tta_cache(path, "abc", 1024) == 1


def test_validate_tta_cache_rejects_non_tta_cache(tmp_path: Path) -> None:
    path = tmp_path / "cache.jsonl.gz"
    write_cache(path, tta=False)
    with pytest.raises(ValueError, match="policy mismatch"):
        validate_tta_cache(path, "abc", 1024)
