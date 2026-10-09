"""Training witnesses: count successful optimizer steps (not micro-batches)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from copy import deepcopy

import yaml

from .governance import digest_file


def dataset_signature(data_path):
    data_path = Path(data_path)
    spec = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = Path(spec.get("path", data_path.parent))
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    files = {data_path.resolve()}
    for split in ("train", "val"):
        for value in ([spec[split]] if isinstance(spec[split], str) else spec[split]):
            path = Path(value)
            path = path if path.is_absolute() else root / path
            if path.is_file() and path.suffix == ".txt":
                files.add(path)
                images = []
                for line in path.read_text(encoding="utf-8").splitlines():
                    image = Path(line)
                    images.append(image if image.is_absolute() else path.parent / image)
            else:
                images = [p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if not images:
                raise ValueError(f"Empty training data split: {path}")
            for image in images:
                files.add(image.resolve())
                parts = list(image.parts)
                if "images" not in parts:
                    raise ValueError(f"Cannot infer label path: {image}")
                parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
                label = Path(*parts).with_suffix(".txt")
                if not label.is_file():
                    raise ValueError(f"Missing label (negative images require empty file): {label}")
                files.add(label.resolve())
    for extra in (root / "metadata" / "splits.json", root / "diagnostic_manifest.json"):
        if extra.is_file():
            files.add(extra)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update((str(path) + ":" + digest_file(path) + "\n").encode())
    return {"sha256": digest.hexdigest(), "files": len(files)}


def training_code_signature(root):
    """Hash governed Python sources relative to an explicit package root."""
    root = Path(root).resolve()
    return {
        path.relative_to(root).as_posix(): digest_file(path)
        for folder in ("steel_defect", "scripts")
        for path in sorted((root / folder).glob("*.py"))
    }


def cuda_witness(seed=42):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU training")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = torch.randn(32, 32, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss) or not torch.isfinite(x.grad).all() or not x.grad.abs().sum() > 0:
        raise RuntimeError("Seeded forward/backward witness failed")
    return {"seed": seed, "loss": float(loss.detach()), "gradient_norm": float(x.grad.norm())}


class OptimizerAudit:
    def __init__(self):
        self.updates = 0
        self.microbatches = 0
        self.start_time = time.monotonic()
        self.epochs = 0
        self.last_epoch = -1
        self.assignment = {}
        self.zero_positive_batches = 0

    def start(self, trainer):
        self.path = Path(trainer.save_dir) / "optimizer_steps.jsonl"
        self.batch_path = Path(trainer.save_dir) / "batch_health.jsonl"
        # Post-step hooks are NOT called when GradScaler skips an invalid step.
        trainer.optimizer.register_step_post_hook(self.step)
        if not hasattr(trainer.model, "criterion"):
            trainer.model.criterion = trainer.model.init_criterion()
        def capture_assignment(module, inputs, output):
            if not trainer.model.training:
                return
            scores, foreground = output[2].detach(), output[3].detach()
            positives = scores[scores > 0]
            self.assignment = {"gt_objects": int(inputs[5].sum()),
                               "foreground_anchors": int(foreground.sum()),
                               "target_quality_sum": float(scores.sum()),
                               "positive_quality_mean": float(positives.mean()) if positives.numel() else 0.0,
                               "positive_quality_max": float(positives.max()) if positives.numel() else 0.0,
                               "matched_confidence_mean": float(inputs[0].detach()[scores > 0].mean()) if positives.numel() else 0.0}
        trainer.model.criterion.assigner.register_forward_hook(capture_assignment)

    def step(self, optimizer, args, kwargs):
        self.updates += 1

    def batch(self, trainer):
        import torch

        self.microbatches += 1
        if not torch.isfinite(trainer.loss).all():
            raise FloatingPointError("Non-finite training loss; learning gate failed")
        record = {"epoch": trainer.epoch + 1, "microbatch": self.microbatches,
                  "optimizer_updates": self.updates, "loss": float(trainer.loss.detach()), **self.assignment}
        with self.batch_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if record.get("gt_objects", 0) > 0:
            self.zero_positive_batches = self.zero_positive_batches + 1 if abs(record["loss"]) < 1e-12 else 0
        if self.zero_positive_batches >= 25:
            raise FloatingPointError("Degenerate learning: zero loss on 25 positive-GT batches")

    def epoch(self, trainer):
        # Ultralytics also calls on_fit_epoch_end during final best.pt validation.
        if trainer.epoch == self.last_epoch:
            return
        self.last_epoch = trainer.epoch
        self.epochs += 1
        record = {"epoch": trainer.epoch + 1, "optimizer_updates": self.updates,
                  "microbatches": self.microbatches, "accumulate": trainer.accumulate,
                  "batch_size": trainer.batch_size, "nbs": trainer.args.nbs,
                  "lr": [g["lr"] for g in trainer.optimizer.param_groups],
                  "seconds": time.monotonic() - self.start_time}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"AUDIT epoch={record['epoch']} actual_optimizer_updates={self.updates} microbatches={self.microbatches}", flush=True)


def audit_initial_transfer(trainer, public_weights):
    """Compare the final nc-adjusted training model against public weights once."""
    import torch
    from ultralytics import YOLO
    reference = YOLO(str(public_weights)).model.float().cpu().state_dict()
    target = trainer.model.state_dict()
    named_parameters = dict(trainer.model.named_parameters())
    report = {"source": str(public_weights), "source_sha256": digest_file(Path(public_weights)),
              "meaning": "Exact same-key/shape/value match before training; not a semantic remapping claim",
              "tensor_matches": 0, "tensor_total": len(target), "parameter_elements_matched": 0,
              "parameter_elements_total": sum(p.numel() for p in named_parameters.values()), "modules": {}}
    for key, tensor in target.items():
        matched = key in reference and tensor.shape == reference[key].shape and torch.equal(tensor.detach().cpu(), reference[key])
        report["tensor_matches"] += int(matched)
        if matched and key in named_parameters:
            report["parameter_elements_matched"] += tensor.numel()
        prefix = ".".join(key.split(".")[:2])
        module = report["modules"].setdefault(prefix, {"matched": 0, "total": 0})
        module["matched"] += int(matched)
        module["total"] += 1
    (Path(trainer.save_dir) / "initial_transfer.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TRANSFER public tensor match={report['tensor_matches']}/{report['tensor_total']} "
          f"parameter elements={report['parameter_elements_matched']}/{report['parameter_elements_total']}", flush=True)


def save_raw_diagnostic_checkpoint(trainer):
    """Keep the last non-EMA model for normalization/EMA diagnosis, not submission."""
    import torch

    model = deepcopy(trainer.model).float().cpu()
    if hasattr(model, "criterion"):
        delattr(model, "criterion")
    model.eval()
    torch.save({"model": model, "train_args": vars(trainer.args), "epoch": trainer.epoch,
                "optimizer": None, "diagnostic_only": True}, Path(trainer.save_dir) / "weights" / "raw_last_diagnostic.pt")


def freeze_bn_statistics(model):
    """Freeze running moments only; affine weights remain trainable."""
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
