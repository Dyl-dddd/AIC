import hashlib
import json
from pathlib import Path

import yaml
import pytest

from scripts.run_quality_aware_experiment import (
    GATE,
    arm_candidate,
    baseline_candidate,
    decide,
    fixed_selection_is_current,
    optimizer_update_total,
    verify_package_manifest,
)
from steel_defect.training_audit import training_code_signature
from scripts.train import validate_resume_contract


ROOT = Path(__file__).resolve().parents[1]


def row(weighted, original_score, grid_score, map50=0.5, recall=0.97, sha="x"):
    return {
        "sha256": sha,
        "weighted_score": weighted,
        "robust_min_score": min(original_score, grid_score),
        "checkpoint": f"{sha}.pt",
        "views": {
            "original": {
                "score_proxy_20_60_20": original_score,
                "map50_macro": map50,
                "map75_macro": map50 - 0.1,
                "map50_95_macro": map50 - 0.15,
                "recall_micro": recall,
                "false_positives_per_image": 10.0,
            },
            "grid_crops": {
                "score_proxy_20_60_20": grid_score,
                "map50_macro": map50,
                "map75_macro": map50 - 0.1,
                "map50_95_macro": map50 - 0.15,
                "recall_micro": recall,
                "false_positives_per_image": 10.0,
            },
        },
        "_per_class_ap50": {
            "original": {"a": map50},
            "grid_crops": {"a": map50},
        },
    }


def test_checkpoint_rows_separate_protected_baseline_from_trained_candidates():
    selection = {"candidates": [
        row(68.0, 68.0, 68.0, sha="base"),
        row(68.1, 68.2, 68.0, sha="early"),
        row(68.3, 68.4, 68.2, sha="late"),
    ]}
    assert baseline_candidate(selection, "base")["sha256"] == "base"
    assert arm_candidate(selection, "base")["sha256"] == "late"


def test_release_gate_requires_both_views_and_recall_protection():
    baseline = row(68.0, 68.0, 68.0, map50=0.50, recall=0.970, sha="base")
    control = row(68.1, 68.1, 68.1, map50=0.51, recall=0.970, sha="control")
    quality = row(68.4, 68.4, 68.4, map50=0.52, recall=0.970, sha="quality")
    passed = decide(baseline, control, quality)
    assert passed["gate"] == "PASS_STAGE1"
    degraded = row(68.4, 68.5, 68.05, map50=0.52, recall=0.970, sha="bad")
    stopped = decide(baseline, control, degraded)
    assert stopped["gate"] == "STOP"
    assert any(item.startswith("grid_crops:") for item in stopped["failures"])


def test_matched_compute_configs_differ_only_in_intended_loss_fields():
    control = yaml.safe_load((
        ROOT / "configs/train/semifinal_quality_bce_control_4090.yaml"
    ).read_text(encoding="utf-8"))["train"]
    quality = yaml.safe_load((
        ROOT / "configs/train/semifinal_quality_varifocal_4090.yaml"
    ).read_text(encoding="utf-8"))["train"]
    ignored = {
        "loss", "name", "varifocal_gamma", "varifocal_alpha", "quality_power",
    }
    assert {key: value for key, value in control.items() if key not in ignored} == {
        key: value for key, value in quality.items() if key not in ignored
    }
    assert control["loss"] == "bce"
    assert quality["loss"] == "varifocal"
    assert GATE["minimum_weighted_gain_over_control"] == 0.20


def test_optimizer_update_total_handles_resume_counter_reset():
    records = [
        {"optimizer_updates": 10, "microbatches": 80},
        {"optimizer_updates": 20, "microbatches": 160},
        {"optimizer_updates": 7, "microbatches": 80},
        {"optimizer_updates": 15, "microbatches": 160},
    ]
    assert optimizer_update_total(records) == 35


def test_optimizer_update_total_detects_equal_counter_at_resume():
    records = [
        {"optimizer_updates": 10, "microbatches": 80},
        {"optimizer_updates": 10, "microbatches": 80},
        {"optimizer_updates": 20, "microbatches": 160},
    ]
    assert optimizer_update_total(records) == 30


def test_training_code_signature_is_independent_of_cwd(tmp_path, monkeypatch):
    expected = training_code_signature(ROOT)
    monkeypatch.chdir(tmp_path)
    assert training_code_signature(ROOT) == expected
    assert "scripts/train.py" in expected


def test_fixed_selection_contract_rejects_scans_or_stale_checkpoints():
    selection = {
        "checkpoint_policy": "explicit",
        "export_thresholds": [0.0001],
        "candidates": [{"sha256": "base"}, {"sha256": "terminal"}],
    }
    assert fixed_selection_is_current(selection, {"base", "terminal"})
    selection["export_thresholds"] = [0.0001, 0.001]
    assert not fixed_selection_is_current(selection, {"base", "terminal"})


def test_package_manifest_is_required_and_verified(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_package_manifest(tmp_path)
    payload = tmp_path / "payload.txt"
    payload.write_text("governed", encoding="utf-8")
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    (tmp_path / "QUALITY_AWARE_PACKAGE_MANIFEST.json").write_text(
        json.dumps({"payload.txt": digest}), encoding="utf-8"
    )
    assert verify_package_manifest(tmp_path) == {"payload.txt": digest}
    payload.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError):
        verify_package_manifest(tmp_path)


def test_exact_resume_rejects_code_mixing():
    previous = {
        "dataset_signature": {"sha256": "data", "files": 2},
        "code_sha256": {"scripts/train.py": "old"},
        "config": {"epochs": 8, "exist_ok": True},
    }
    current = {
        "dataset_signature": {"sha256": "data", "files": 2},
        "code_sha256": {"scripts/train.py": "new"},
        "config": {"epochs": 8, "exist_ok": True, "resume": "last.pt"},
    }
    with pytest.raises(ValueError, match="Resume code differs"):
        validate_resume_contract(previous, current)

