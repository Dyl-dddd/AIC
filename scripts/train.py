from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import hashlib
import json

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from steel_defect.focal_trainer import FocalDetectionTrainer, configure_focal
from steel_defect.quality_trainer import QualityAwareDetectionTrainer, configure_quality_loss
from steel_defect.runtime import load_yaml, platform_manifest, write_manifest
from steel_defect.training_audit import OptimizerAudit, cuda_witness, dataset_signature, audit_initial_transfer, save_raw_diagnostic_checkpoint, freeze_bn_statistics, training_code_signature
from steel_defect.governance import digest_file


DEFAULTS = {
    "model": "configs/yolo11s-p2.yaml", "pretrained": "yolo11s.pt", "loss": "bce",
    "epochs": 150, "imgsz": 1024, "batch": -1, "device": "0", "workers": 8,
    "project": "runs/steel", "name": "p2_bce", "patience": 35,
    "focal_gamma": 2.0, "focal_alpha": 0.25, "varifocal_gamma": 2.0,
    "varifocal_alpha": 0.75, "quality_power": 1.0,
    "cache": False, "optimizer": "AdamW",
    "lr0": 0.003, "lrf": 0.01, "weight_decay": 0.0005, "warmup_epochs": 3.0,
    "cos_lr": True, "close_mosaic": 15, "mosaic": 0.8, "mixup": 0.10,
    "degrees": 2.0, "translate": 0.08, "scale": 0.35, "fliplr": 0.5,
    "flipud": 0.5, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.25,
    "erasing": 0.15, "amp": True, "deterministic": True, "plots": True,
    "val": True, "save_period": 10, "exist_ok": False, "seed": 42,
    "multi_scale": False, "rect": False, "fraction": 1.0,
    "nbs": 64,
    "freeze_bn": False,
    "defer_validation": False,
}


class DeferredValidationDetectionTrainer(DetectionTrainer):
    """Train and checkpoint without Ultralytics' accumulating validation loop.

    The governed pipeline evaluates saved checkpoints separately on original
    images, so the built-in tiled validation would be redundant and can exceed
    host RAM on memory-constrained Windows machines.
    """

    def validate(self):
        fitness = 0.0 if self.fitness is None else float(self.fitness)
        # Keep results.csv rectangular even when the final-epoch validation
        # hook is invoked. These are explicit placeholders, not evaluation
        # results; governed original-image evaluation happens afterwards.
        metrics = {
            "metrics/precision(B)": 0.0,
            "metrics/recall(B)": 0.0,
            "metrics/mAP50(B)": 0.0,
            "metrics/mAP50-95(B)": 0.0,
            "val/box_loss": 0.0,
            "val/cls_loss": 0.0,
            "val/dfl_loss": 0.0,
        }
        return metrics, fitness

    def final_eval(self):
        return None


class DeferredQualityAwareDetectionTrainer(QualityAwareDetectionTrainer):
    """Quality-aware trainer with the governed external-validation contract."""

    validate = DeferredValidationDetectionTrainer.validate
    final_eval = DeferredValidationDetectionTrainer.final_eval


def parse_cache(value: str):
    lowered = value.lower()
    if lowered in {"false", "none", "0", "off"}:
        return False
    if lowered in {"true", "ram", "1", "on"}:
        return "ram"
    if lowered == "disk":
        return "disk"
    raise argparse.ArgumentTypeError("cache must be false, ram, or disk")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a reproducible single-model steel-defect detector.")
    parser.add_argument("--config", type=Path, help="YAML training profile")
    parser.add_argument("--data", default=argparse.SUPPRESS)
    parser.add_argument("--model", default=argparse.SUPPRESS)
    parser.add_argument("--pretrained", default=argparse.SUPPRESS)
    parser.add_argument("--resume", nargs="?", const="auto", default=argparse.SUPPRESS)
    parser.add_argument("--loss", choices=("bce", "focal", "varifocal"), default=argparse.SUPPRESS)
    numeric = {
        "epochs": int, "imgsz": int, "batch": int, "workers": int, "patience": int,
        "save-period": int, "seed": int, "freeze": int, "focal-gamma": float,
        "focal-alpha": float, "varifocal-gamma": float, "varifocal-alpha": float,
        "quality-power": float, "lr0": float, "lrf": float, "weight-decay": float,
        "warmup-epochs": float, "close-mosaic": int, "mosaic": float, "mixup": float,
        "degrees": float, "translate": float, "scale": float, "fliplr": float,
        "flipud": float, "hsv-h": float, "hsv-s": float, "hsv-v": float,
        "erasing": float, "fraction": float, "nbs": int,
    }
    for name, kind in numeric.items():
        parser.add_argument(f"--{name}", type=kind, default=argparse.SUPPRESS)
    for name in ("device", "project", "name", "optimizer"):
        parser.add_argument(f"--{name}", default=argparse.SUPPRESS)
    parser.add_argument("--cache", type=parse_cache, default=argparse.SUPPRESS)
    for name in ("amp", "deterministic", "plots", "val", "cos-lr", "multi-scale", "rect", "exist-ok", "freeze-bn", "defer-validation"):
        parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
    return parser


def resolve_config() -> dict:
    cli = vars(build_parser().parse_args())
    config_path = cli.pop("config", None)
    config = dict(DEFAULTS)
    config.update(load_yaml(config_path))
    config.update(cli)
    if "data" not in config and "resume" not in config:
        raise SystemExit("--data is required unless --resume is used")
    if config["loss"] == "focal":
        gamma, alpha = float(config["focal_gamma"]), float(config["focal_alpha"])
        if gamma < 0 or not 0.0 <= alpha <= 1.0:
            raise SystemExit("Focal loss requires gamma >= 0 and alpha in [0, 1]")
    if config["loss"] == "varifocal":
        gamma = float(config["varifocal_gamma"])
        alpha = float(config["varifocal_alpha"])
        power = float(config["quality_power"])
        if gamma < 0 or not 0.0 <= alpha <= 1.0 or power <= 0.0:
            raise SystemExit(
                "Varifocal loss requires gamma >= 0, alpha in [0, 1], and quality_power > 0"
            )
    return config


def find_resume_checkpoint(config: dict) -> Path:
    requested = str(config["resume"])
    if requested != "auto":
        path = Path(requested)
        if not path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        return path
    root = Path(config["project"])
    candidates = sorted(root.glob(f"{config['name']}*/weights/last.pt"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No last.pt found for {root / config['name']}")
    return candidates[-1]


def validate_resume_contract(previous: dict, current: dict) -> None:
    """Refuse a resume that would mix data, code, or governed settings."""
    if previous.get("dataset_signature") != current.get("dataset_signature"):
        raise ValueError("Resume data/labels differ from original run")
    if previous.get("code_sha256") != current.get("code_sha256"):
        raise ValueError("Resume code differs from original run")
    ignored = {"resume", "exist_ok"}
    previous_config = {
        key: value for key, value in previous.get("config", {}).items() if key not in ignored
    }
    current_config = {
        key: value for key, value in current.get("config", {}).items() if key not in ignored
    }
    if previous_config != current_config:
        raise ValueError("Exact resume config mismatch")


def main() -> None:
    config = resolve_config()
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not bool(config["deterministic"])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest_path = Path(config["project"]) / "_manifests" / f"{config['name']}_{timestamp}.json"
    manifest = platform_manifest(config)
    manifest["status"] = "starting"
    write_manifest(manifest_path, manifest)

    try:
        manifest["dataset_signature"] = dataset_signature(config["data"])
        manifest["code_sha256"] = training_code_signature(PACKAGE_ROOT)
        if not str(config["device"]).startswith("cpu"):
            manifest["cuda_witness"] = cuda_witness(config["seed"])
        resume_checkpoint = find_resume_checkpoint(config) if "resume" in config else None
        if resume_checkpoint is not None:
            checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
            if checkpoint.get("optimizer") is None or checkpoint.get("epoch", -1) < 0:
                raise ValueError("Checkpoint lacks optimizer/epoch; use new fine-tuning, not exact resume")
            contract = resume_checkpoint.parent.parent / "training_contract.json"
            if not contract.is_file():
                raise ValueError("Exact resume requires the original training contract")
            previous = json.loads(contract.read_text(encoding="utf-8"))
            validate_resume_contract(previous, manifest)
            del checkpoint
        model = YOLO(str(resume_checkpoint or config["model"]))
        if resume_checkpoint is None and str(config["pretrained"]).lower() != "none":
            model.load(str(config["pretrained"]))
        trainer = DeferredValidationDetectionTrainer if config["defer_validation"] else None
        if config["loss"] == "focal":
            if config["defer_validation"]:
                raise ValueError("Deferred validation currently supports BCE training only")
            configure_focal(config["focal_gamma"], config["focal_alpha"])
            trainer = FocalDetectionTrainer
        elif config["loss"] == "varifocal":
            configure_quality_loss(
                config["varifocal_gamma"], config["varifocal_alpha"], config["quality_power"]
            )
            trainer = (
                DeferredQualityAwareDetectionTrainer
                if config["defer_validation"] else QualityAwareDetectionTrainer
            )
        audit = OptimizerAudit()
        model.add_callback("on_train_start", audit.start)
        model.add_callback("on_train_batch_end", audit.batch)
        model.add_callback("on_fit_epoch_end", audit.epoch)
        if config["freeze_bn"]:
            # _model_train resets modes each epoch; apply immediately before every forward.
            model.add_callback("on_train_batch_start", lambda t: freeze_bn_statistics(t.model))
        data_root = Path(config["data"]).parent
        diagnostic_path = data_root / "diagnostic_manifest.json"
        if diagnostic_path.is_file() and json.loads(diagnostic_path.read_text(encoding="utf-8")).get("diagnostic_only"):
            model.add_callback("on_train_end", save_raw_diagnostic_checkpoint)
        def save_contract(t):
            write_manifest(Path(t.save_dir) / "training_contract.json", manifest)
        model.add_callback("on_train_start", save_contract)
        public_source = config["pretrained"] if str(config["pretrained"]).lower() != "none" else config["model"]
        if resume_checkpoint is None and Path(str(public_source)).is_file() and str(public_source).endswith(".pt"):
            model.add_callback("on_train_start", lambda t: audit_initial_transfer(t, public_source))
        manifest["status"] = "training"
        write_manifest(manifest_path, manifest)
        internal = {
            "pretrained", "loss", "focal_gamma", "focal_alpha",
            "varifocal_gamma", "varifocal_alpha", "quality_power",
            "resume", "freeze_bn", "defer_validation",
        }
        train_args = {key: value for key, value in config.items() if key not in internal}
        # Defensive filtering for project-only controls.  These options choose
        # our custom trainer behavior and must never reach Ultralytics, whose
        # argument validator rejects unknown keys before training starts.
        for key in internal:
            train_args.pop(key, None)
        if resume_checkpoint is not None:
            train_args["resume"] = str(resume_checkpoint)
        result = model.train(trainer=trainer, **train_args)
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        write_manifest(manifest_path, manifest)
        raise
    manifest["status"] = "complete"
    manifest["optimizer_updates_this_invocation"] = audit.updates
    manifest["epochs_completed_this_invocation"] = audit.epochs
    manifest["exit_code"] = 0
    write_manifest(Path(model.trainer.save_dir) / "completion.json", manifest)
    manifest["result_save_dir"] = str(getattr(result, "save_dir", ""))
    manifest["result_dict"] = getattr(result, "results_dict", {})
    write_manifest(manifest_path, manifest)
    print(f"Experiment manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
