from pathlib import Path

import yaml

from scripts.run_quality_aware_experiment import (
    GATE,
    arm_candidate,
    baseline_candidate,
    decide,
)


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
                "recall_micro": recall,
            },
            "grid_crops": {
                "score_proxy_20_60_20": grid_score,
                "map50_macro": map50,
                "recall_micro": recall,
            },
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

