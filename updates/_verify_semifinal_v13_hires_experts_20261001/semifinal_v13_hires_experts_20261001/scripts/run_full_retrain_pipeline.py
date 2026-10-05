"""One-command full retraining, checkpoint selection, refinement, and finalization."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_semifinal_stage1 import SPLIT_SHA, verified_file, write_json
from steel_defect.runtime import sha256_file


YOLO11S_SHA = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"


def run_logged(command: list[str], cwd: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    print("RUN:", " ".join(command), flush=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\nRUN: " + " ".join(command) + "\n")
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f"command failed with exit code {code}: {' '.join(command)}")


def completed(run_dir: Path) -> bool:
    return (run_dir / "completion.json").is_file()


def train_command(python: str, project: Path, config: Path, run_dir: Path,
                  model: Path | None = None) -> list[str]:
    command = [python, "-u", str(ROOT / "scripts/train.py"), "--config", str(config)]
    if model is not None:
        command.extend(["--model", str(model)])
    last = run_dir / "weights" / "last.pt"
    if last.is_file() and not completed(run_dir):
        command.extend(["--resume", str(last)])
    return command


def load_selected(path: Path) -> Path:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(payload["selected"]["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    python = sys.executable
    # Use fresh run directories for the shared-memory fix.  In particular, do not
    # resume a checkpoint created by the old workers=2 profile: Ultralytics stores
    # trainer arguments inside last.pt and may restore the unsafe worker count.
    output = project / "runs/semifinal/full_retrain_v3_fix1_pipeline"
    state_path = output / "state.json"
    yolo = project / "yolo11s.pt"
    split = project / "data/official_v2_20260831b/splits.json"
    source = project / "data/official/train"
    verified_file(yolo, YOLO11S_SHA)
    verified_file(split, SPLIT_SHA)
    if not source.is_dir():
        raise FileNotFoundError(f"training source missing: {source}")
    package_manifest = ROOT / "FULL_RETRAIN_PACKAGE_MANIFEST.json"
    if package_manifest.is_file():
        for relative, digest in json.loads(package_manifest.read_text(encoding="utf-8")).items():
            verified_file(ROOT / relative, digest)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("full retraining requires the requested RTX 4090")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 10 * 1024**3:
        raise RuntimeError(f"less than 10 GiB free VRAM: {free_bytes / 1024**3:.2f} GiB")
    torch.manual_seed(42)
    witness = torch.randn(64, 64, device="cuda", requires_grad=True)
    loss = (witness @ witness.T).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(witness.grad).all():
        raise RuntimeError("CUDA kernel witness failed")
    shm = os.statvfs("/dev/shm") if Path("/dev/shm").is_dir() else None
    environment = {"gpu": torch.cuda.get_device_name(0), "free_vram_gib": free_bytes / 1024**3,
        "total_vram_gib": total_bytes / 1024**3, "torch": torch.__version__,
        "python": sys.version, "yolo11s_sha256": YOLO11S_SHA, "split_sha256": SPLIT_SHA,
        "kernel_loss": loss.item(), "kernel_gradient_norm": witness.grad.norm().item(),
        "dataloader_workers": 0,
        "shared_memory_gib": None if shm is None else shm.f_frsize * shm.f_bavail / 1024**3}
    del witness, loss
    torch.cuda.empty_cache()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    dataset = project / "data/semifinal_yolo_balanced_v3"
    summary = dataset / "metadata/summary.json"
    if summary.is_file():
        prepared = json.loads(summary.read_text(encoding="utf-8"))
        if prepared.get("global_repeat") != 3:
            raise ValueError(f"existing dataset has incompatible preparation: {summary}")
        print(f"Reuse prepared dataset: {dataset}", flush=True)
    else:
        if dataset.exists() and any(dataset.iterdir()):
            raise ValueError(f"partial dataset exists; move it aside before retrying: {dataset}")
        run_logged([python, "-u", str(ROOT / "scripts/prepare_data.py"), "--config",
                    str(ROOT / "configs/data/semifinal_full_retrain_v3.yaml")],
                   project, output / "prepare.log")
    prepared = json.loads(summary.read_text(encoding="utf-8"))
    train_global = int(prepared["train_global_views"])
    train_crop = int(prepared["train_crop_views"])
    mixture = train_global / max(train_global + train_crop, 1)
    if not 0.55 <= mixture <= 0.65:
        raise ValueError(f"unexpected train global-view fraction: {mixture:.4f}")
    state = {"status": "prepared", "updated": datetime.now(timezone.utc).isoformat(),
        "environment": environment, "dataset_summary": prepared, "train_global_fraction": mixture}
    write_json(state_path, state)
    if args.prepare_only:
        return

    phase1 = project / "runs/semifinal/full_retrain_v3_fix1_phase1"
    if not completed(phase1):
        run_logged(train_command(python, project,
            ROOT / "configs/train/semifinal_full_retrain_phase1_4090.yaml", phase1),
            project, output / "phase1_train.log")
    phase1_selection = output / "phase1_eval/selection.json"
    if not phase1_selection.is_file():
        run_logged([python, "-u", str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
            "--project-root", str(project), "--run-dir", str(phase1),
            "--output", str(output / "phase1_eval")], project, output / "phase1_select.log")
    phase1_best = load_selected(phase1_selection)
    state.update(status="phase1_selected", phase1_checkpoint=str(phase1_best),
                 phase1_sha256=sha256_file(phase1_best), updated=datetime.now(timezone.utc).isoformat())
    write_json(state_path, state)

    phase2 = project / "runs/semifinal/full_retrain_v3_fix1_phase2"
    if not completed(phase2):
        run_logged(train_command(python, project,
            ROOT / "configs/train/semifinal_full_retrain_phase2_4090.yaml", phase2, phase1_best),
            project, output / "phase2_train.log")
    phase2_selection = output / "phase2_eval/selection.json"
    if not phase2_selection.is_file():
        run_logged([python, "-u", str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
            "--project-root", str(project), "--run-dir", str(phase2),
            "--external-checkpoint", str(phase1_best), "--output", str(output / "phase2_eval")],
            project, output / "phase2_select.log")
    final_best = load_selected(phase2_selection)
    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_weight = final_dir / "best_score_model.pt"
    if not final_weight.is_file() or sha256_file(final_weight) != sha256_file(final_best):
        shutil.copy2(final_best, final_weight)
    final_selection = json.loads(phase2_selection.read_text(encoding="utf-8"))["selected"]
    final = {"status": "complete", "selected_source": str(final_best),
        "final_weight": str(final_weight), "final_sha256": sha256_file(final_weight),
        "selection": final_selection, "training_from_public_pretrained": True,
        "all_layers_trainable": True, "official_test_score": None,
        "warning": "Development selection is not an official Test score."}
    write_json(final_dir / "FINAL.json", final)
    state.update(status="complete", final=final, updated=datetime.now(timezone.utc).isoformat())
    write_json(state_path, state)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

