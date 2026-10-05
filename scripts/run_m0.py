"""Bounded M0 queue: stock BCE -> (only if learned) P2 BCE; never starts long training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from steel_defect.classes import CLASS_NAMES
from steel_defect.governance import digest_file
from steel_defect.metrics import evaluate_predictions
from steel_defect.runtime import write_manifest
from steel_defect.training_audit import dataset_signature


def learning_gate(metrics, updates, completed, numerical_healthy=True):
    summary = metrics["summary"]
    return (completed and numerical_healthy and 500 <= updates <= 1000
            and all(math.isfinite(summary[key]) and summary[key] >= .90
                    for key in ("map50_macro", "recall_macro")))


def evaluate_diagnostic(run_dir: Path, data_dir: Path):
    from ultralytics import YOLO
    completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    if completion.get("dataset_signature") != dataset_signature(data_dir / "diagnostic.yaml"):
        raise ValueError("Diagnostic training data changed after training")
    diagnostic = json.loads((data_dir / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    if not diagnostic.get("diagnostic_only") or not diagnostic.get("train_equals_validation"):
        raise ValueError("Not an intentional train=self-evaluation diagnostic")
    items = diagnostic["items"]
    truth = {}
    for item in items:
        truth[item["id"] + ".jpg"] = {}
        for class_id, box in item["annotations"]:
            truth[item["id"] + ".jpg"].setdefault(CLASS_NAMES[class_id], []).append(box)
    weights = run_dir / "weights" / "best.pt"
    model = YOLO(str(weights))
    if list(model.names.values()) != CLASS_NAMES:
        raise ValueError("Checkpoint class order differs from diagnostic")
    predictions, seen = [], set()
    for result in model.predict(source=[str((data_dir / "images" / "diagnostic" / name).resolve()) for name in truth],
                                imgsz=1024, batch=2, conf=.001, iou=.70, max_det=300,
                                device="0", half=False, stream=True, verbose=False):
        name = Path(result.path).name
        if name in seen or name not in truth:
            raise ValueError("Duplicate/unknown diagnostic prediction image")
        seen.add(name)
        for box, score, cls in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            predictions.append({"image_id": name, "category_name": CLASS_NAMES[int(cls)], "bbox": box, "score": score})
    if seen != set(truth):
        raise ValueError("Diagnostic evaluation scope incomplete")
    metrics = evaluate_predictions(predictions, truth, CLASS_NAMES, operating_threshold=.05,
                                   image_sizes={name: (1024, 1024) for name in truth})
    updates = completion["optimizer_updates_this_invocation"]
    with (run_dir / "results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    nonfinite = [{"epoch": row["epoch"], "field": key, "value": value} for row in rows
                 for key, value in row.items() if (key.startswith("train/") or key.startswith("val/"))
                 and not math.isfinite(float(value))]
    numerical_healthy = bool(rows) and not nonfinite
    passed = learning_gate(metrics, updates, completion["status"] == "complete", numerical_healthy)
    report = {"diagnostic_only": True, "not_generalization": True, "scope_complete": True,
              "images": len(seen), "actual_optimizer_updates": updates,
              "numerical_healthy": numerical_healthy, "nonfinite_epoch_metrics": nonfinite,
              "weights_sha256": digest_file(weights), "diagnostic_sha256": digest_file(data_dir / "diagnostic_manifest.json"),
              "gate": {"passed": passed, "ap50_min": .9, "recall_min": .9, "fixed_confidence": .05,
                       "update_budget": [500, 1000]}, **metrics}
    write_manifest(run_dir / "learning_gate.json", report)
    write_manifest(run_dir / "diagnostic_predictions.json", {"predictions": predictions})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--fp32", action="store_true", help="Single-factor AMP-off diagnostic")
    parser.add_argument("--lr0", type=float, help="Explicit diagnostic LR ablation")
    parser.add_argument("--freeze-bn", action="store_true", help="Single-factor running-stat freeze diagnostic")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Queue output exists; do not rerun a live/completed queue")
    args.output.mkdir(parents=True)
    files = [p for folder in ("steel_defect", "scripts") for p in Path(folder).glob("*.py")]
    files.append(Path("configs/train/diagnostic_stock_v2.yaml"))
    code = {str(p): digest_file(p) for p in files}
    state = {"started_at": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
             "scope": "R002 and conditionally R003 only; full training needs next decision",
             "status": "running", "code_sha256": code, "runs": {}}
    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(args.output / "state.json", state)
    save()
    try:
        variants = [("R002_stock_bce" + args.suffix, []), ("R003_p2_bce" + args.suffix, ["--model", "configs/yolo11s-p2.yaml", "--pretrained", "yolo11s.pt"])]
        common = (["--no-amp"] if args.fp32 else []) + (["--lr0", str(args.lr0)] if args.lr0 is not None else [])
        if args.freeze_bn:
            common.append("--freeze-bn")
        for name, extras in variants:
            if any(digest_file(Path(path)) != sha for path, sha in code.items()):
                raise ValueError("Code changed during the queue; review before continuing")
            run_dir = Path("runs/diagnostic_v2") / name
            if run_dir.exists():
                raise ValueError(f"Run already exists: {run_dir}; refusing silent resume")
            command = [sys.executable, "-u", "scripts/train.py", "--config", "configs/train/diagnostic_stock_v2.yaml", "--name", name, *common, *extras]
            state["runs"][name] = {"status": "training", "command": command}
            save()
            print(f"Starting {name}", flush=True)
            with (args.output / f"{name}.stdout.log").open("w", encoding="utf-8") as out, (args.output / f"{name}.stderr.log").open("w", encoding="utf-8") as err:
                child = subprocess.Popen(command, stdout=out, stderr=err, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
                state["runs"][name]["pid"] = child.pid
                save()
                code_return = child.wait()
            state["runs"][name]["returncode"] = code_return
            if code_return:
                state["runs"][name]["status"] = "failed"
                raise RuntimeError(f"{name} exited {code_return}; see preserved logs")
            report = evaluate_diagnostic(run_dir, Path("data/diagnostic_v2"))
            state["runs"][name].update(status="gate_passed" if report["gate"]["passed"] else "gate_failed",
                                       summary=report["summary"], actual_optimizer_updates=report["actual_optimizer_updates"])
            save()
            if not report["gate"]["passed"]:
                state["status"] = "stopped_learning_gate"
                save()
                print(f"{name} failed learning gate; no further experiment launched", flush=True)
                return
        state["status"] = "learning_checks_passed_next_decision_required"
        save()
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
