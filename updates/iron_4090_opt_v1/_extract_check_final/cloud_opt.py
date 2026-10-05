"""Safe optimized-training entry point layered on the original cloud package.

This file adds new experiments without overwriting cloud.py, existing data, or
historical runs.  The original package verifier and CUDA preflight still run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cloud as base


ROOT = Path(__file__).resolve().parent
VARIANTS = {
    "repeat3": Path("configs/cloud/optimized_repeat3_4090.yaml"),
    "repeat3_aug": Path("configs/cloud/optimized_repeat3_aug_4090.yaml"),
}
GENERATED_YAML = Path("data/official_yolo_hybrid_v2/steel_defect_repeat3.yaml")
ORIGINAL_EXECUTE = base.execute
ORIGINAL_VERIFY = base.verify
ORIGINAL_FINISH_EVALUATIONS = base.finish_evaluations


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_update(full: bool = False) -> None:
    manifest_path = ROOT / "OPTIMIZED_UPDATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            errors.append(f"Unsafe update path: {relative}")
            continue
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing update file: {relative}")
        elif path.stat().st_size != item["size"]:
            errors.append(f"Update size mismatch: {relative}")
        elif full and file_sha256(path) != item["sha256"]:
            errors.append(f"Update hash mismatch: {relative}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"OPT UPDATE PASS: {len(manifest['files'])} files")


def combined_verify(full: bool = False) -> None:
    ORIGINAL_VERIFY(full)
    verify_update(full)


def optimized_execute(command, logfile) -> None:
    command = list(command)
    if command and command[0] == "scripts/train.py":
        command[0] = "optimized/train.py"
    ORIGINAL_EXECUTE(command, logfile)


def recall_micro(metrics_path: Path) -> float:
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    classes = report.get("per_class", {}).values()
    true_positives = sum(int(item.get("true_positives", 0)) for item in classes)
    ground_truth = sum(int(item.get("ground_truth", 0)) for item in classes)
    return true_positives / ground_truth if ground_truth else 0.0


def finish_evaluations_with_score_proxy(root, control, config, state) -> None:
    """Keep the AP selection and also save a recall-aligned competition checkpoint."""
    ORIGINAL_FINISH_EVALUATIONS(root, control, config, state)
    enriched = []
    for item in state["evaluations"]:
        candidate = dict(item)
        candidate["recall_micro"] = recall_micro(Path(item["metrics"]))
        enriched.append(candidate)
    best = max(enriched, key=lambda item: (
        item["recall_micro"],
        item["summary"]["recall_macro"],
        item["summary"]["map50_macro"],
    ))
    selected = Path(root) / "weights" / "best_score_proxy.pt"
    shutil.copy2(best["checkpoint"], selected)
    record = {
        **best,
        "checkpoint": str(selected),
        "weights_sha256": base.sha(selected),
        "split_sha256": base.sha(base.SPLIT),
        "recipe_sha256": base.recipe_hash(config),
        "stage": state["stage"],
        "criterion": "dev micro recall, then macro recall and AP50; aligned to current recall-dominant public score",
    }
    output = Path(root) / "score_proxy_selection.json"
    base.write(output, record)
    state["score_proxy_selection"] = str(output)
    base.write(Path(control) / "state.json", state)


def configure(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown optimized variant: {variant}")
    base.PROFILE = VARIANTS[variant]
    base.DATA_YAMLS = {
        str(GENERATED_YAML): "data/official_yolo_hybrid_v2",
        "data/diagnostic_v2/diagnostic.yaml": "data/diagnostic_v2",
    }
    base.verify = combined_verify
    base.execute = optimized_execute
    base.finish_evaluations = finish_evaluations_with_score_proxy


def prepare(cap: float) -> None:
    command = [
        sys.executable,
        "optimized/build_rare_repeat_manifest.py",
        "--dataset-root", "data/official_yolo_hybrid_v2",
        "--repeat-cap", str(cap),
        "--output-list", "data/official_yolo_hybrid_v2/rare_repeat_train_cap3.txt",
        "--output-yaml", str(GENERATED_YAML),
        "--report", "runs/cloud4090_opt/_preparation/rare_repeat_report.json",
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def show_status() -> None:
    roots = [Path("runs/cloud4090_opt/_queues"), Path("runs/cloud4090/_queues")]
    found = False
    for queue_root in roots:
        for path in sorted(queue_root.glob("*/state.json")):
            found = True
            state = json.loads(path.read_text(encoding="utf-8"))
            print(json.dumps({
                "queue": str(path.parent),
                "status": state.get("status"),
                "stage": state.get("stage"),
                "epochs": state.get("epochs"),
                "current_step": state.get("current_step"),
                "error": state.get("error"),
            }, ensure_ascii=False))
    if not found:
        print("No queue state found")


def compare() -> None:
    rows = []
    for root in (Path("runs/cloud4090"), Path("runs/cloud4090_opt")):
        for run_root in root.glob("*"):
            path = run_root / "score_proxy_selection.json"
            if not path.is_file():
                path = run_root / "original_selection.json"
            if not path.is_file():
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            summary = result.get("summary", {})
            micro = result.get("recall_micro")
            if micro is None and result.get("metrics") and Path(result["metrics"]).is_file():
                micro = recall_micro(Path(result["metrics"]))
            rows.append({
                "run": path.parent.name,
                "epoch": result.get("epoch"),
                "recall_micro": micro,
                "map50_macro": summary.get("map50_macro"),
                "recall_macro": summary.get("recall_macro"),
                "tail_map50_macro": summary.get("tail_map50_macro"),
                "tail_recall_macro": summary.get("tail_recall_macro"),
                "checkpoint": result.get("checkpoint"),
            })
    rows.sort(key=lambda item: (
        item["recall_micro"] if item["recall_micro"] is not None else -1,
        item["recall_macro"] if item["recall_macro"] is not None else -1,
        item["map50_macro"] if item["map50_macro"] is not None else -1,
    ), reverse=True)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def add_variant(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)


def main() -> None:
    base.os.chdir(ROOT)
    base.os.environ.setdefault("MPLBACKEND", "Agg")
    base.os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    base.os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)

    preparation = sub.add_parser("prepare")
    preparation.add_argument("--repeat-cap", type=float, default=3.0)
    validation = sub.add_parser("verify")
    validation.add_argument("--full", action="store_true")
    preflight = sub.add_parser("preflight")
    add_variant(preflight)
    run_parser = sub.add_parser("run")
    add_variant(run_parser)
    run_parser.add_argument("--stage", choices=["smoke", "baseline", "long"], required=True)
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--epochs", type=int, default=60)
    run_parser.add_argument("--batch", type=int, choices=[2, 4, 8])
    run_parser.add_argument("--workers", type=int)
    run_parser.add_argument("--after", type=Path)
    run_parser.add_argument("--dry-run", action="store_true")
    recovery = sub.add_parser("evaluate-run")
    add_variant(recovery)
    recovery.add_argument("--name", required=True)
    sub.add_parser("status")
    sub.add_parser("compare")
    args = parser.parse_args()

    if args.action == "prepare":
        if not 1.0 < args.repeat_cap <= 3.0:
            parser.error("--repeat-cap must be within (1, 3]")
        prepare(args.repeat_cap)
    elif args.action == "verify":
        combined_verify(args.full)
    elif args.action == "status":
        show_status()
    elif args.action == "compare":
        compare()
    else:
        configure(args.variant)
        if args.action == "preflight":
            combined_verify(False)
            base.preflight()
        elif args.action == "run":
            if args.dry_run:
                print("OPT NOTE: execution replaces scripts/train.py with optimized/train.py")
            base.run(args)
        else:
            base.evaluate_run(args)


if __name__ == "__main__":
    main()
