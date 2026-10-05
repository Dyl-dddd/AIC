import gzip
import json
from pathlib import Path

from scripts.build_final_submission_from_cache import build_submission
from steel_defect.classes import CLASS_NAMES


def test_cache_builder_applies_thresholds_and_excludes_qilie(tmp_path: Path):
    cache = tmp_path / "cache.jsonl.gz"
    header = {"type": "candidate_cache_header", "schema": 3,
              "weights_sha256": "abc", "classes": CLASS_NAMES}
    record = {"image_id": "nested/example.jpg", "image_size": [100, 100], "candidates": [
        {"bbox": [1, 1, 20, 20], "score": 0.9, "class_id": CLASS_NAMES.index("qilie"),
         "touches_internal_edge": False, "is_global": True},
        {"bbox": [30, 30, 50, 50], "score": 0.2, "class_id": CLASS_NAMES.index("jieba"),
         "touches_internal_edge": False, "is_global": True},
        {"bbox": [60, 60, 80, 80], "score": 0.001, "class_id": CLASS_NAMES.index("gunyin"),
         "touches_internal_edge": False, "is_global": True},
    ]}
    with gzip.open(cache, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        handle.write(json.dumps(record) + "\n")
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(json.dumps({"thresholds": {
        name: (0.4 if name == "gunyin" else 0.0001)
        for name in CLASS_NAMES if name != "qilie"
    }}), encoding="utf-8")
    output, report = tmp_path / "submission.json", tmp_path / "report.json"
    result = build_submission(cache, output, report, thresholds, 0.00001, 0.68, 0.5, "abc")
    items = json.loads(output.read_text(encoding="utf-8"))
    assert result["images"] == 1 and result["predictions"] == 1
    assert items == [{"image_id": "example.jpg", "category_name": "jieba",
                      "bbox": [30, 30, 50, 50], "score": 0.2}]
