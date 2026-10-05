import gzip
import json
from pathlib import Path

import pytest

from scripts.build_ensemble_submission_from_caches import build_ensemble_submissions
from scripts.run_ensemble_v5_submission_pipeline import validate_cache
from steel_defect.classes import CLASS_NAMES


def write_cache(path: Path, weight_sha: str, candidates: list[dict]) -> None:
    header = {
        "type": "candidate_cache_header",
        "schema": 3,
        "weights_sha256": weight_sha,
        "classes": CLASS_NAMES,
        "options": {"imgsz": 1024, "tta": False},
    }
    record = {"image_id": "sample.jpg", "image_size": [100, 100], "candidates": candidates}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        handle.write(json.dumps(record) + "\n")


def candidate(category: str, box: list[int], score: float) -> dict:
    return {
        "bbox": box,
        "score": score,
        "class_id": CLASS_NAMES.index(category),
        "touches_internal_edge": False,
        "is_global": True,
    }


def test_builder_generates_global_and_routed_variants(tmp_path: Path):
    cache_a = tmp_path / "a.jsonl.gz"
    cache_b = tmp_path / "b.jsonl.gz"
    write_cache(cache_a, "aaa", [candidate("zonglie", [1, 1, 20, 20], 0.8)])
    write_cache(
        cache_b,
        "bbb",
        [
            candidate("zonglie", [1, 1, 20, 20], 0.9),
            candidate("jiaza", [30, 30, 50, 50], 0.9),
            candidate("qilie", [60, 60, 80, 80], 0.9),
        ],
    )
    router = {
        name: {"source": "fusion", "threshold": 0.0001}
        for name in CLASS_NAMES
        if name != "qilie"
    }
    router["zonglie"] = {"source": "model_a", "threshold": 0.0001}
    router["jiaza"] = {"source": "model_b", "threshold": 0.5}
    profile = tmp_path / "profile.json"
    profile.write_text(
        json.dumps(
            {
                "model_a_sha256": "aaa",
                "model_b_sha256": "bbb",
                "per_model_nms_iou": 0.68,
                "edge_penalty": 0.5,
                "model_b_score_scale": 0.35,
                "cross_model_nms_iou": 0.72,
                "merge_conf": 0.0001,
                "class_router": router,
            }
        ),
        encoding="utf-8",
    )
    global_output = tmp_path / "global.json"
    routed_output = tmp_path / "routed.json"
    report = tmp_path / "report.json"
    result = build_ensemble_submissions(
        cache_a, cache_b, profile, global_output, routed_output, report
    )
    global_items = json.loads(global_output.read_text(encoding="utf-8"))
    routed_items = json.loads(routed_output.read_text(encoding="utf-8"))
    assert {item["category_name"] for item in global_items} == {"zonglie", "jiaza"}
    assert {item["category_name"] for item in routed_items} == {"zonglie", "jiaza"}
    assert next(item for item in routed_items if item["category_name"] == "zonglie")["score"] == 0.8
    assert next(item for item in routed_items if item["category_name"] == "jiaza")["score"] == 0.9
    assert result["images"] == 1
    assert result["variants"]["global"]["predictions"] == 2


def test_validate_cache_requires_exact_inference_policy(tmp_path: Path):
    cache = tmp_path / "policy.jsonl.gz"
    options = {
        "tile_layout": "semifinal_grid",
        "tile_size": 1024,
        "overlap": 0.25,
        "imgsz": 1280,
        "batch": 1,
        "conf": 0.00001,
        "local_iou": 0.70,
        "half": True,
        "tta": False,
        "deblur": False,
        "clahe": False,
        "global_pass": True,
        "max_det": 2000,
    }

    def write_policy_cache(payload: dict) -> None:
        header = {
            "type": "candidate_cache_header",
            "schema": 3,
            "weights_sha256": "bbb",
            "classes": CLASS_NAMES,
            "options": payload,
        }
        with gzip.open(cache, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(header) + "\n")
            handle.write(json.dumps({"image_id": "sample.jpg"}) + "\n")

    write_policy_cache(options)
    assert validate_cache(cache, "bbb", 1280) == 1

    write_policy_cache({**options, "global_pass": False})
    with pytest.raises(ValueError, match="inference policy mismatch"):
        validate_cache(cache, "bbb", 1280)
