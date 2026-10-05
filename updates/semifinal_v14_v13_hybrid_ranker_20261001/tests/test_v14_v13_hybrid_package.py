from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
CLASSES = {
    "jieba", "zonglie", "jiaza", "yiwuyaru", "huashang",
    "mamianmakeng", "yanghuatiepi", "gunyin",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hybrid_spec_and_models() -> None:
    spec = json.loads((ROOT / "v14_hybrid_spec.json").read_text(encoding="utf-8"))
    assert set(spec["classes"]) == CLASSES
    assert {
        name for name, row in spec["classes"].items() if row["feature_set"] == "v14"
    } == {"jieba", "jiaza", "mamianmakeng"}
    assert spec["strict_group_oof"]["hybrid_weighted_delta_vs_v7"] > spec["strict_group_oof"]["v12_weighted_delta_vs_v7"]
    assert spec["strict_group_oof"]["hybrid_original_delta_vs_v12"] > 0
    assert spec["strict_group_oof"]["hybrid_grid_crops_delta_vs_v12"] > 0
    for class_name, row in spec["classes"].items():
        path = ROOT / "models" / row["model"]
        assert digest(path) == row["sha256"]
        model = XGBClassifier()
        model.load_model(path)
        assert model.n_features_in_ == len(spec["feature_sets"][row["feature_set"]]), class_name


def test_generator_is_membership_preserving() -> None:
    path = ROOT / "generate_v14_v13_hybrid_ranker_submission.py"
    source = path.read_text(encoding="utf-8")
    ast.parse(source)
    assert "identity_digest(output) != identity" in source
    assert 'archive.write(source, "submission.json")' in source
    assert "LOC_SHA256" in source
