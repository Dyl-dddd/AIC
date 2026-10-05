"""Regression tests for large-cache streaming and local score analysis."""

from __future__ import annotations

import gzip
from io import StringIO
import json

import numpy as np

from scripts.analyze_submission_tradeoff import ap50_from_ranked_labels, label_image_class
from scripts.build_layered_submission import cache_weight_sha256
from steel_defect.metrics import average_precision
from steel_defect.recall_booster import _JsonStreamReader, iter_candidate_records


def test_chunked_json_reader_handles_split_values():
    reader = _JsonStreamReader(StringIO(' {"name":"abcdef","count":12345} '), chunk_size=3)
    reader.expect("{")
    assert reader.value() == "name"
    reader.expect(":")
    assert reader.value() == "abcdef"
    reader.expect(",")
    assert reader.value() == "count"
    reader.expect(":")
    assert reader.value() == 12345
    reader.expect("}")


def test_legacy_cache_streams_images_and_provenance(tmp_path):
    path = tmp_path / "cache.json.gz"
    digest = "a" * 64
    payload = {
        "signature": {"weights_sha256": digest},
        "images": {
            "a.jpg": {"image_size": [8, 8], "candidates": []},
            "b.jpg": {"image_size": [8, 8], "candidates": []},
        },
    }
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    assert cache_weight_sha256(path) == digest
    assert [item["image_id"] for item in iter_candidate_records(path)] == ["a.jpg", "b.jpg"]


def test_compact_ap_matches_reference():
    scores = np.asarray([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    assert ap50_from_ranked_labels(scores, labels, 2) == average_precision(
        labels.astype(float), 1.0 - labels, 2
    )
    predictions = [
        {"bbox": [0, 0, 10, 10], "score": 0.9},
        {"bbox": [0, 0, 10, 10], "score": 0.8},
    ]
    assert label_image_class(predictions, [[0, 0, 10, 10]]).tolist() == [1, 0]
