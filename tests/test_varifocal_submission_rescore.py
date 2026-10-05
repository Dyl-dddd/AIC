from __future__ import annotations

import json

from scripts.rescore_submission_with_varifocal import build, identity_digest, rescore


PROFILE = {
    "names": [
        "log_base_score", "vf_present", "vf_iou", "log_vf_score",
        "vf_log_advantage",
    ],
    "mean": [0.0] * 5,
    "scale": [1.0] * 5,
    "intercept": 0.0,
    "coefficients": [0.0, 1.0, 0.0, 0.0, 0.0],
}


def test_rescore_preserves_membership_and_geometry() -> None:
    baseline = [
        {"image_id": "a.jpg", "category_name": "jieba", "bbox": [0, 0, 10, 10], "score": 0.1},
        {"image_id": "a.jpg", "category_name": "zonglie", "bbox": [20, 20, 30, 30], "score": 0.2},
    ]
    quality = [
        {"image_id": "a.jpg", "category_name": "jieba", "bbox": [1, 1, 11, 11], "score": 0.4},
        {"image_id": "a.jpg", "category_name": "zonglie", "bbox": [100, 100, 110, 110], "score": 0.9},
    ]
    output, audit = rescore(baseline, quality, PROFILE, blend=0.75, match_floor=0.5)
    assert identity_digest(output) == identity_digest(baseline)
    assert audit["predictions"] == 2
    assert audit["matched_predictions"] == 1
    assert output[0]["score"] != baseline[0]["score"]
    assert output[1]["score"] != baseline[1]["score"]


def test_build_refuses_overwrite_and_writes_audit(tmp_path) -> None:
    baseline = tmp_path / "baseline.json"
    quality = tmp_path / "quality.json"
    profile = tmp_path / "profile.json"
    output = tmp_path / "output.json"
    report = tmp_path / "report.json"
    row = [{"image_id": "a.jpg", "category_name": "jieba", "bbox": [0, 0, 10, 10], "score": 0.1}]
    baseline.write_text(json.dumps(row), encoding="utf-8")
    quality.write_text(json.dumps(row), encoding="utf-8")
    profile.write_text(json.dumps({"blend": 0.75, "profile": PROFILE}), encoding="utf-8")
    result = build(baseline, quality, profile, output, report)
    assert result["predictions"] == 1
    assert result["identity_sha256"] == identity_digest(row)
    assert output.is_file() and report.is_file()
