from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RECALL = ROOT / "configs/train/semifinal_v13_hires_recall_bce_4090.yaml"
LOCALIZE = ROOT / "configs/train/semifinal_v13_hires_localize_vfl_4090.yaml"
PIPELINE = ROOT / "scripts/run_v13_hires_experts_pipeline.py"


def load_train(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["train"]


def test_complementary_profiles_are_frozen() -> None:
    recall = load_train(RECALL)
    localize = load_train(LOCALIZE)
    assert recall["imgsz"] == localize["imgsz"] == 1536
    assert recall["epochs"] == localize["epochs"] == 12
    assert recall["loss"] == "bce"
    assert localize["loss"] == "varifocal"
    assert recall["mosaic"] > 0 and localize["mosaic"] == 0
    assert localize["box"] > recall["box"]
    assert localize["dfl"] > recall["dfl"]
    assert recall["val"] is False and localize["val"] is False
    assert recall["defer_validation"] is True and localize["defer_validation"] is True
    assert recall["workers"] == localize["workers"] == 0


def test_pipeline_has_no_test_or_submission_entrypoint() -> None:
    source = PIPELINE.read_text(encoding="utf-8")
    ast.parse(source)
    assert '"test_access": False' in source
    assert '"submission_creation": False' in source
    assert "semifinal_v13_hires_experts_results_20261001.tar.gz" in source
    assert "--fixed-export-threshold\", \"0.0001" in source
