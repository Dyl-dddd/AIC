"""Bounded M1 queue: five-epoch baselines and original dev eval at epochs 1/3/5."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import os
import ctypes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_m0 import learning_gate
from steel_defect.governance import digest_file
from steel_defect.runtime import write_manifest


def check_gate(path):
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("scope_complete") or not report["gate"]["passed"] or not learning_gate(
            report, report["actual_optimizer_updates"], True, report.get("numerical_healthy", False)):
        raise ValueError(f"Learning gate not passed: {path}")
    if report["weights_sha256"] != digest_file(path.parent / "weights" / "best.pt"):
        raise ValueError("Gate checkpoint changed")
    if report["diagnostic_sha256"] != digest_file(Path("data/diagnostic_v2/diagnostic_manifest.json")):
        raise ValueError("Diagnostic provenance changed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock-gate", type=Path, required=True)
    parser.add_argument("--p2-gate", type=Path, help="Only provide when P2 has separately passed M0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_gate(args.stock_gate)
    if args.p2_gate:
        check_gate(args.p2_gate)
    summary_path = Path("data/official_yolo_hybrid_v2/metadata/summary.json")
    prepared = json.loads(summary_path.read_text(encoding="utf-8"))
    if prepared["records"] != {"train": 2395, "dev": 459} or prepared["repeated_train_tiles"] != 0:
        raise ValueError("Unexpected v2 data preparation")
    if args.output.exists():
        raise ValueError("Queue output exists; refusing implicit rerun")
    args.output.mkdir(parents=True)
    state = {"status": "running", "pid": os.getpid(), "runs": {},
             "scope": "M1 only. No M2/long training/test submission.",
             "started_at": datetime.now(timezone.utc).isoformat(),
             "source_split_sha256": digest_file(Path("data/official_v2_20260831b/splits.json")),
             "profile_sha256": digest_file(Path("configs/train/official_v2_baseline.yaml"))}
    code = {str(p): digest_file(p) for folder in ("scripts", "steel_defect") for p in Path(folder).glob("*.py")}
    state["code_sha256"] = code
    def save():
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_manifest(args.output / "state.json", state)
    def run(label, command):
        if any(digest_file(Path(p)) != sha for p, sha in code.items()):
            raise ValueError("Code changed while M1 queued; review before continuing")
        state["current_step"] = label
        print(label, flush=True)
        with (args.output / f"{label}.stdout.log").open("w", encoding="utf-8") as out, (args.output / f"{label}.stderr.log").open("w", encoding="utf-8") as err:
            child = subprocess.Popen([sys.executable, "-u", *command], stdout=out, stderr=err,
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            state["child_pid"] = child.pid
            save()
            code_return = child.wait()
        state["child_pid"] = None
        if code_return:
            raise RuntimeError(f"{label} exited {code_return}; see saved logs")
    save()
    keep_awake = False
    if os.name == "nt":
        # Process-scoped idle-sleep prevention; no persistent power-plan edits.
        keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        state["idle_sleep_prevention"] = keep_awake
        save()
    try:
        variants = [("R020_stock_bce", [])]
        if args.p2_gate:
            variants.append(("R021_p2_bce", ["--model", "configs/yolo11s-p2.yaml", "--pretrained", "yolo11s.pt"]))
        for name, extras in variants:
            root = Path("runs/v2_baselines") / name
            if root.exists():
                raise ValueError(f"Training directory exists: {root}")
            state["runs"][name] = {"status": "training", "evaluations": []}
            run(name + "_train", ["scripts/train.py", "--config", "configs/train/official_v2_baseline.yaml", "--name", name, *extras])
            completion = json.loads((root / "completion.json").read_text(encoding="utf-8"))
            if completion["status"] != "complete" or completion["epochs_completed_this_invocation"] != 5:
                raise ValueError("Incomplete baseline training")
            with (root / "results.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if len(rows) != 5 or any(not math.isfinite(float(v)) for r in rows for k, v in r.items() if k.startswith(("train/", "val/"))):
                raise ValueError("Invalid baseline numerical/epoch history")
            for epoch in (1, 3, 5):
                checkpoint = root / "weights" / f"epoch{epoch - 1}.pt"
                output = root / f"original_dev_epoch{epoch}"
                run(name + f"_dev_e{epoch}", ["scripts/eval.py", "--weights", str(checkpoint),
                    "--source", "data/official/train", "--split-manifest", "data/official_v2_20260831b/splits.json",
                    "--split", "dev", "--output", str(output), "--tile-size", "1024", "--imgsz", "1024",
                    "--overlap", "0.25", "--batch", "4", "--conf", "0.001", "--global-pass", "--no-half", "--no-tune-thresholds"])
                result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
                if not result["scope_complete"] or result["evaluated_images"] != 459:
                    raise ValueError("Original-image eval incomplete")
                if not all(math.isfinite(float(v)) for k, v in result["summary"].items() if isinstance(v, (int, float))):
                    raise ValueError("Non-finite original-image metric")
                state["runs"][name]["evaluations"].append({"epoch": epoch, "checkpoint": str(checkpoint), "summary": result["summary"]})
                save()
            evaluations = state["runs"][name]["evaluations"]
            best = max(evaluations, key=lambda item: (item["summary"]["map50_macro"], item["summary"]["recall_macro"]))
            shutil.copy2(best["checkpoint"], root / "weights" / "best_original.pt")
            write_manifest(root / "original_selection.json", {"selection_metric": "dev map50_macro, then recall_macro; not an official total score", **best})
            state["runs"][name]["status"] = "complete"
            state["runs"][name]["selected_epoch"] = best["epoch"]
            save()
        state["status"] = "complete_analysis_required_before_M2"
        save()
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    finally:
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
