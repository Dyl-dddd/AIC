"""Run matched YOLO11m quality/localization refinements; never touches Test."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_retrain_pipeline import completed, run_logged, train_command
from scripts.run_semifinal_stage1 import SPLIT_SHA, verified_file, write_json
from steel_defect.runtime import sha256_file


MODEL_B_SHA256 = "02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090"
ARMS = {
    "quality_p1": ROOT / "configs/train/semifinal_v9_model_b_quality_p1_4090.yaml",
    "quality_p2": ROOT / "configs/train/semifinal_v9_model_b_quality_p2_4090.yaml",
}
RUN_NAMES = {
    "quality_p1": "v9_model_b_quality_p1",
    "quality_p2": "v9_model_b_quality_p2",
}


def verify_transfer(run_dir: Path) -> dict:
    path = run_dir / "initial_transfer.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_sha256") != MODEL_B_SHA256:
        raise ValueError(f"wrong initialization source: {run_dir}")
    if payload.get("tensor_matches") != payload.get("tensor_total"):
        raise ValueError(f"incomplete tensor transfer: {run_dir}")
    if payload.get("parameter_elements_matched") != payload.get("parameter_elements_total"):
        raise ValueError(f"incomplete parameter transfer: {run_dir}")
    return payload


def decision(selection: dict) -> dict:
    rows = selection["candidates"]
    baseline = next(row for row in rows if row["sha256"] == MODEL_B_SHA256)
    new_rows = [row for row in rows if row["sha256"] != MODEL_B_SHA256]
    candidate = max(new_rows, key=lambda row: row["weighted_score"])
    views = {}
    failures = []
    for view in ("original", "grid_crops"):
        old = baseline["views"][view]
        new = candidate["views"][view]
        row = {
            "score_gain": new["score_proxy_20_60_20"] - old["score_proxy_20_60_20"],
            "map50_gain": new["map50_macro"] - old["map50_macro"],
            "recall_change": new["recall_micro"] - old["recall_micro"],
        }
        views[view] = row
        if row["score_gain"] < 0.10:
            failures.append(f"{view}:score_gain<0.10")
        if row["map50_gain"] < 0.01:
            failures.append(f"{view}:map50_gain<0.01")
        if row["recall_change"] < -0.003:
            failures.append(f"{view}:recall_drop>0.003")
    weighted_gain = candidate["weighted_score"] - baseline["weighted_score"]
    if weighted_gain < 0.35:
        failures.append("weighted_gain<0.35")
    return {
        "status": "PROMOTE_TO_FUSION_STUDY" if not failures else "STOP",
        "baseline": baseline,
        "candidate": candidate,
        "weighted_gain": weighted_gain,
        "views": views,
        "failures": failures,
        "test_or_submission_authorized": False,
    }


def package_results(project: Path, output: Path, runs: dict[str, Path]) -> dict:
    destination = project.parent / "semifinal_v9_quality_localizer_results_20260930.tar.gz"
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite result archive: {destination}")
    files = [
        output / "preflight.json",
        output / "selection/selection.json",
        output / "selection/SUMMARY.md",
        output / "DECISION.json",
    ]
    for run_dir in runs.values():
        files.extend(
            run_dir / name for name in (
                "args.yaml", "results.csv", "training_contract.json", "completion.json",
                "initial_transfer.json", "optimizer_steps.jsonl", "batch_health.jsonl",
                "weights/last.pt",
            )
        )
    files.extend(
        path for path in (output / "selection/cells").rglob("*") if path.is_file()
    )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing result artifacts: {missing}")
    unique = list(dict.fromkeys(path.resolve() for path in files))
    manifest = {
        path.relative_to(project).as_posix(): sha256_file(path) for path in unique
    }
    manifest_path = output / "RESULT_MANIFEST.json"
    write_json(manifest_path, manifest)
    unique.append(manifest_path.resolve())
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"partial archive exists: {temporary}")
    with tarfile.open(temporary, "w:gz") as archive:
        for path in unique:
            archive.add(path, arcname=path.relative_to(project).as_posix(), recursive=False)
    temporary.replace(destination)
    return {
        "archive": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "members": len(unique),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = project / "runs/semifinal/v9_quality_localizer_pipeline"
    model_b = project / "runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt"
    split = project / "data/official_v2_20260831b/splits.json"
    dataset = project / "data/semifinal_yolo_balanced_v3/steel_defect.yaml"
    verified_file(model_b, MODEL_B_SHA256)
    verified_file(split, SPLIT_SHA)
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("V9 experiment requires RTX 4090")
    free, total = torch.cuda.mem_get_info()
    if free < 18 * 1024**3:
        raise RuntimeError(f"less than 18 GiB free VRAM: {free / 1024**3:.2f}")
    witness = torch.randn(64, 64, device="cuda", requires_grad=True)
    witness_loss = (witness @ witness.T).square().mean()
    witness_loss.backward()
    if not torch.isfinite(witness_loss) or not torch.isfinite(witness.grad).all():
        raise RuntimeError("CUDA witness failed")
    preflight = {
        "created": datetime.now(timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "free_vram_gib": free / 1024**3,
        "total_vram_gib": total / 1024**3,
        "model_b": str(model_b),
        "model_b_sha256": MODEL_B_SHA256,
        "split_sha256": SPLIT_SHA,
        "arms": {name: sha256_file(path) for name, path in ARMS.items()},
        "test_access": False,
        "submission_creation": False,
    }
    del witness, witness_loss
    torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    python = sys.executable
    runs = {
        name: project / "runs/semifinal" / RUN_NAMES[name] for name in ARMS
    }
    for name, config in ARMS.items():
        run_dir = runs[name]
        if not completed(run_dir):
            run_logged(
                train_command(python, project, config, run_dir, model_b),
                project,
                output / f"{name}_train.log",
            )
        verify_transfer(run_dir)
        last = run_dir / "weights/last.pt"
        if not last.is_file():
            raise FileNotFoundError(last)

    selection_dir = output / "selection"
    selection_path = selection_dir / "selection.json"
    if not selection_path.is_file():
        command = [
            python, "-u", str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
            "--project-root", str(project), "--run-dir", str(runs["quality_p1"]),
            "--output", str(selection_dir), "--imgsz", "1280", "--batch", "1",
            "--fixed-export-threshold", "0.0001",
        ]
        for run_dir in runs.values():
            command.extend(["--checkpoint", str(run_dir / "weights/last.pt")])
        command.extend(["--external-checkpoint", str(model_b)])
        run_logged(command, project, output / "selection.log")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    result = decision(selection)
    write_json(output / "DECISION.json", result)
    result_archive = package_results(project, output, runs)
    result["result_archive"] = result_archive
    write_json(output / "DECISION.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
