"""Train a diverse YOLO11m model for the semifinal two-model ensemble."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_retrain_pipeline import (
    completed,
    load_selected,
    run_logged,
    train_command,
)
from scripts.run_semifinal_stage1 import SPLIT_SHA, verified_file, write_json
from steel_defect.runtime import sha256_file


YOLO11M_SHA = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"


def package_results(
    project: Path,
    pipeline: Path,
    phase1: Path,
    phase2: Path,
) -> dict:
    destination = project.parent / "semifinal_ensemble_v4_yolo11m_results_20260926.tar.gz"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    files = [
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
        files.extend(
            run_dir / name
            for name in (
                "args.yaml",
                "results.csv",
                "training_contract.json",
                "completion.json",
            )
        )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot package missing result files: {missing}")

    with tarfile.open(temporary, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=path.relative_to(project).as_posix(), recursive=False)
    temporary.replace(destination)
    return {
        "archive": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "members": len(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    python = sys.executable
    output = project / "runs/semifinal/ensemble_v4_yolo11m_pipeline"
    state_path = output / "state.json"
    model_b_pretrained = ROOT / "yolo11m.pt"
    split = project / "data/official_v2_20260831b/splits.json"
    source = project / "data/official/train"

    verified_file(model_b_pretrained, YOLO11M_SHA)
    verified_file(split, SPLIT_SHA)
    if not source.is_dir():
        raise FileNotFoundError(f"training source missing: {source}")

    package_manifest = ROOT / "ENSEMBLE_TRAIN_PACKAGE_MANIFEST.json"
    if package_manifest.is_file():
        for relative, digest in json.loads(
            package_manifest.read_text(encoding="utf-8")
        ).items():
            verified_file(ROOT / relative, digest)

    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("ensemble training requires the requested RTX 4090")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 18 * 1024**3:
        raise RuntimeError(f"less than 18 GiB free VRAM: {free_bytes / 1024**3:.2f} GiB")

    torch.manual_seed(1337)
    witness = torch.randn(64, 64, device="cuda", requires_grad=True)
    witness_loss = (witness @ witness.T).square().mean()
    witness_loss.backward()
    if not torch.isfinite(witness_loss) or not torch.isfinite(witness.grad).all():
        raise RuntimeError("CUDA kernel witness failed")

    shm = os.statvfs("/dev/shm") if Path("/dev/shm").is_dir() else None
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "free_vram_gib": free_bytes / 1024**3,
        "total_vram_gib": total_bytes / 1024**3,
        "torch": torch.__version__,
        "python": sys.version,
        "yolo11m_sha256": YOLO11M_SHA,
        "split_sha256": SPLIT_SHA,
        "kernel_loss": witness_loss.item(),
        "kernel_gradient_norm": witness.grad.norm().item(),
        "dataloader_workers": 0,
        "train_imgsz": 1280,
        "train_batch": 4,
        "shared_memory_gib": (
            None if shm is None else shm.f_frsize * shm.f_bavail / 1024**3
        ),
    }
    del witness, witness_loss
    torch.cuda.empty_cache()

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    dataset = project / "data/semifinal_yolo_balanced_v3"
    dataset_summary = dataset / "metadata/summary.json"
    if dataset_summary.is_file():
        prepared = json.loads(dataset_summary.read_text(encoding="utf-8"))
        if prepared.get("global_repeat") != 3:
            raise ValueError(f"existing dataset has incompatible preparation: {dataset_summary}")
        print(f"Reuse prepared dataset: {dataset}", flush=True)
    else:
        if dataset.exists() and any(dataset.iterdir()):
            raise ValueError(f"partial dataset exists; move it aside before retrying: {dataset}")
        run_logged(
            [
                python,
                "-u",
                str(ROOT / "scripts/prepare_data.py"),
                "--config",
                str(ROOT / "configs/data/semifinal_full_retrain_v3.yaml"),
            ],
            project,
            output / "prepare.log",
        )

    prepared = json.loads(dataset_summary.read_text(encoding="utf-8"))
    train_global = int(prepared["train_global_views"])
    train_crop = int(prepared["train_crop_views"])
    global_fraction = train_global / max(train_global + train_crop, 1)
    if not 0.55 <= global_fraction <= 0.65:
        raise ValueError(f"unexpected train global-view fraction: {global_fraction:.4f}")

    state = {
        "status": "prepared",
        "updated": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "dataset_summary": prepared,
        "train_global_fraction": global_fraction,
        "model_a": "existing YOLO11s final model",
        "model_b": "YOLO11m 1280",
    }
    write_json(state_path, state)
    if args.prepare_only:
        return

    phase1 = project / "runs/semifinal/ensemble_v4_yolo11m_phase1"
    if not completed(phase1):
        run_logged(
            train_command(
                python,
                project,
                ROOT / "configs/train/semifinal_ensemble_v4_yolo11m_phase1_4090.yaml",
                phase1,
                model_b_pretrained,
            ),
            project,
            output / "phase1_train.log",
        )

    phase1_selection = output / "phase1_eval/selection.json"
    if not phase1_selection.is_file():
        run_logged(
            [
                python,
                "-u",
                str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
                "--project-root",
                str(project),
                "--run-dir",
                str(phase1),
                "--output",
                str(output / "phase1_eval"),
                "--imgsz",
                "1280",
                "--batch",
                "1",
            ],
            project,
            output / "phase1_select.log",
        )
    phase1_best = load_selected(phase1_selection)
    state.update(
        status="phase1_selected",
        phase1_checkpoint=str(phase1_best),
        phase1_sha256=sha256_file(phase1_best),
        updated=datetime.now(timezone.utc).isoformat(),
    )
    write_json(state_path, state)

    phase2 = project / "runs/semifinal/ensemble_v4_yolo11m_phase2"
    if not completed(phase2):
        run_logged(
            train_command(
                python,
                project,
                ROOT / "configs/train/semifinal_ensemble_v4_yolo11m_phase2_4090.yaml",
                phase2,
                phase1_best,
            ),
            project,
            output / "phase2_train.log",
        )

    phase2_selection = output / "phase2_eval/selection.json"
    if not phase2_selection.is_file():
        run_logged(
            [
                python,
                "-u",
                str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
                "--project-root",
                str(project),
                "--run-dir",
                str(phase2),
                "--external-checkpoint",
                str(phase1_best),
                "--output",
                str(output / "phase2_eval"),
                "--imgsz",
                "1280",
                "--batch",
                "1",
            ],
            project,
            output / "phase2_select.log",
        )

    final_best = load_selected(phase2_selection)
    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_weight = final_dir / "model_b_yolo11m_best.pt"
    if not final_weight.is_file() or sha256_file(final_weight) != sha256_file(final_best):
        shutil.copy2(final_best, final_weight)

    final_selection = json.loads(
        phase2_selection.read_text(encoding="utf-8")
    )["selected"]
    final = {
        "status": "complete",
        "selected_source": str(final_best),
        "final_weight": str(final_weight),
        "final_sha256": sha256_file(final_weight),
        "selection": final_selection,
        "architecture": "YOLO11m",
        "train_imgsz": 1280,
        "training_from_public_pretrained": True,
        "all_layers_trainable": True,
        "official_test_score": None,
        "warning": "Development selection is not an official Test score.",
    }
    write_json(final_dir / "FINAL.json", final)
    state.update(
        status="complete",
        final=final,
        updated=datetime.now(timezone.utc).isoformat(),
    )
    write_json(state_path, state)

    result_archive = package_results(project, output, phase1, phase2)
    final["result_archive"] = result_archive
    write_json(final_dir / "FINAL.json", final)
    state.update(final=final, result_archive=result_archive)
    write_json(state_path, state)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
