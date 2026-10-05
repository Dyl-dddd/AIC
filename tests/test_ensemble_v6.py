import gzip
import json
from pathlib import Path

from scripts.build_source_aware_submission_from_caches import build_submission
from steel_defect.classes import CLASS_NAMES


def write_cache(path: Path, weight_sha: str, rows: list[dict], imgsz: int) -> None:
    header = {
        "type": "candidate_cache_header", "schema": 3,
        "weights_sha256": weight_sha, "classes": CLASS_NAMES,
        "options": {"imgsz": imgsz, "tta": False},
    }
    record = {"image_id": "sample.jpg", "image_size": [100, 100], "candidates": rows}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        handle.write(json.dumps(record) + "\n")


def candidate(box, score):
    return {
        "bbox": box, "score": score,
        "class_id": CLASS_NAMES.index("jieba"),
        "touches_internal_edge": False, "is_global": True,
    }


def test_source_aware_builder_uses_consensus_profile(tmp_path: Path):
    cache_a, cache_b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    write_cache(cache_a, "aaa", [candidate([0, 0, 20, 20], 0.4)], 1024)
    write_cache(cache_b, "bbb", [candidate([1, 1, 21, 21], 0.8)], 1280)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({
        "model_a_sha256": "aaa", "model_b_sha256": "bbb",
        "per_model_nms_iou": 0.68, "edge_penalty": 0.5, "merge_conf": 0.0001,
        "model_b_score_scale": 0.35, "match_iou": 0.70,
        "consensus_boost": 1.10, "coordinate_mode": "anchor",
        "unmatched_b_factor": 0.50, "final_nms_iou": 0.72,
    }), encoding="utf-8")
    output, report = tmp_path / "submission.json", tmp_path / "report.json"
    result = build_submission(cache_a, cache_b, profile, output, report)
    items = json.loads(output.read_text(encoding="utf-8"))
    assert result["images"] == 1
    assert len(items) == 1
    assert items[0]["bbox"] == [0, 0, 20, 20]
    assert items[0]["score"] == 0.44

