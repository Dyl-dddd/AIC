from pathlib import Path
import tarfile

import yaml

from scripts.run_ensemble_v4_training_pipeline import package_results
from steel_defect.runtime import sha256_file


YOLO11M_SHA = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"


def test_ensemble_v4_profiles_are_diverse_and_memory_bounded():
    root = Path(__file__).resolve().parents[1]
    phase1 = yaml.safe_load(
        (root / "configs/train/semifinal_ensemble_v4_yolo11m_phase1_4090.yaml").read_text(
            encoding="utf-8"
        )
    )["train"]
    phase2 = yaml.safe_load(
        (root / "configs/train/semifinal_ensemble_v4_yolo11m_phase2_4090.yaml").read_text(
            encoding="utf-8"
        )
    )["train"]

    assert phase1["model"] == "yolo11m.pt"
    assert phase1["imgsz"] == phase2["imgsz"] == 1280
    assert phase1["batch"] == phase2["batch"] == 4
    assert phase1["workers"] == phase2["workers"] == 0
    assert phase1["epochs"] == 45
    assert phase2["epochs"] == 8
    assert phase2["lr0"] < phase1["lr0"]
    assert phase2["mosaic"] == 0.0
    assert phase1["seed"] != phase2["seed"]
    assert "freeze" not in phase1 and "freeze" not in phase2


def test_bundled_yolo11m_has_expected_digest():
    root = Path(__file__).resolve().parents[1]
    assert sha256_file(root / "yolo11m.pt") == YOLO11M_SHA


def test_compact_result_archive_contains_final_weight_and_reports(tmp_path: Path):
    project = tmp_path / "iron"
    pipeline = project / "runs/semifinal/pipeline"
    phase1 = project / "runs/semifinal/phase1"
    phase2 = project / "runs/semifinal/phase2"

    paths = [
        pipeline / "state.json",
        pipeline / "preflight.json",
        pipeline / "phase1_eval/selection.json",
        pipeline / "phase1_eval/SUMMARY.md",
        pipeline / "phase2_eval/selection.json",
        pipeline / "phase2_eval/SUMMARY.md",
        pipeline / "final/FINAL.json",
        pipeline / "final/model_b_yolo11m_best.pt",
    ]
    for run_dir in (phase1, phase2):
        paths.extend(
            run_dir / name
            for name in (
                "args.yaml",
                "results.csv",
                "training_contract.json",
                "completion.json",
            )
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    result = package_results(project, pipeline, phase1, phase2)
    archive_path = Path(result["archive"])
    assert archive_path.is_file()
    with tarfile.open(archive_path, "r:gz") as archive:
        names = archive.getnames()
    assert "runs/semifinal/pipeline/final/model_b_yolo11m_best.pt" in names
    assert len(names) == result["members"] == 16
