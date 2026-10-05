from copy import deepcopy

import numpy as np

from scripts.analyze_safe_r2_nonlinear import (
    duplicate_robust_weights,
    membership_digest,
    rerank,
)


def _record(view: str, label: int, group: str, score: float = 0.2) -> dict:
    return {
        "view": view,
        "label": label,
        "group": group,
        "prediction": {
            "image_id": f"{group}.jpg",
            "category_name": "jieba",
            "bbox": [1.0, 2.0, 3.0, 4.0],
            "score": score,
        },
    }


def test_duplicate_robust_weights_neutralize_rows_within_group():
    records = []
    for view in ("original", "grid_crops"):
        for label in (0, 1):
            records.append(_record(view, label, "g1"))
            records.extend(_record(view, label, "g2") for _ in range(3))
    weights = duplicate_robust_weights(records)
    for view in ("original", "grid_crops"):
        for label in (0, 1):
            one = [
                index for index, record in enumerate(records)
                if record["view"] == view and record["label"] == label
                and record["group"] == "g1"
            ]
            three = [
                index for index, record in enumerate(records)
                if record["view"] == view and record["label"] == label
                and record["group"] == "g2"
            ]
            assert np.sum(weights[one]) == np.sum(weights[three])


def test_rerank_changes_only_score_and_membership_hash_ignores_score():
    records = [_record("original", 0, "g1", 0.2), _record("original", 1, "g2", 0.8)]
    before = [deepcopy(record["prediction"]) for record in records]
    after = rerank(records, np.asarray([0.9, 0.1]), blend=0.5)
    assert [item["score"] for item in before] != [item["score"] for item in after]
    assert membership_digest(before) == membership_digest(after)
    for left, right in zip(before, after, strict=True):
        assert left["image_id"] == right["image_id"]
        assert left["category_name"] == right["category_name"]
        assert left["bbox"] == right["bbox"]
