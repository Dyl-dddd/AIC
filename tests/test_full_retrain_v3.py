from pathlib import Path

import yaml

from scripts.prepare_data import repeat_global_train_views
from scripts.select_full_retrain_checkpoint import best_export


def test_repeat_global_train_views_only_repeats_train_globals(tmp_path: Path):
    for kind in ("images", "labels"):
        (tmp_path / kind / "train").mkdir(parents=True)
    for stem in ("global", "crop"):
        (tmp_path / "images" / "train" / f"{stem}.jpg").write_bytes(b"image")
        (tmp_path / "labels" / "train" / f"{stem}.txt").write_text("", encoding="utf-8")
    metadata = [
        {"tile": "global", "split": "train", "is_global": True, "is_repeat": False},
        {"tile": "crop", "split": "train", "is_global": False, "is_repeat": False},
        {"tile": "dev", "split": "dev", "is_global": True, "is_repeat": False},
    ]
    result = repeat_global_train_views(metadata, tmp_path, 3)
    repeats = [item for item in result if item.get("repeat_kind") == "global_balance"]
    assert len(repeats) == 2
    assert all(item["split"] == "train" and item["is_global"] for item in repeats)
    assert (tmp_path / "images" / "train" / "global__globalrep2.jpg").is_file()


def test_training_profiles_are_full_network_and_bounded():
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((root / "configs/data/semifinal_full_retrain_v3.yaml").read_text(encoding="utf-8"))["prepare"]
    phase1 = yaml.safe_load((root / "configs/train/semifinal_full_retrain_phase1_4090.yaml").read_text(encoding="utf-8"))["train"]
    phase2 = yaml.safe_load((root / "configs/train/semifinal_full_retrain_phase2_4090.yaml").read_text(encoding="utf-8"))["train"]
    assert phase1["model"] == "yolo11s.pt" and "freeze" not in phase1
    assert phase1["epochs"] == 50 and phase1["imgsz"] == 1024
    assert phase2["epochs"] == 10 and phase2["lr0"] < phase1["lr0"]
    assert phase2["mosaic"] == 0.0
    assert data["workers"] == phase1["workers"] == phase2["workers"] == 0
    assert phase1["name"] == "full_retrain_v3_fix1_phase1"
    assert phase2["name"] == "full_retrain_v3_fix1_phase2"


def test_best_export_uses_actual_score_then_fewer_boxes():
    report = {"threshold_scan": [
        {"score_proxy_20_60_20": 65.0, "precision_micro": 0.1, "export_count": 100},
        {"score_proxy_20_60_20": 65.0, "precision_micro": 0.1, "export_count": 80},
        {"score_proxy_20_60_20": 64.9, "precision_micro": 0.9, "export_count": 10},
    ]}
    assert best_export(report)["export_count"] == 80

