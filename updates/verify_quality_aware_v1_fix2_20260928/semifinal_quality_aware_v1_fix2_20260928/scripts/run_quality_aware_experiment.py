"""Run a matched-compute BCE vs IoU-quality Varifocal experiment on RTX 4090.

This is a development experiment, never a Test or final-lockbox pipeline.  It
starts both arms from the same protected Model A checkpoint, evaluates every
saved checkpoint on the frozen dual-view dev split, and emits a pass/stop
decision.  No leaderboard submission is created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import torch
import ultralytics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_full_retrain_pipeline import run_logged
from scripts.run_semifinal_stage1 import SPLIT_SHA, verified_file, write_json
from steel_defect.quality_trainer import SUPPORTED_ULTRALYTICS, ensure_ultralytics_compatibility
from steel_defect.runtime import sha256_file


MODEL_A_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
EXPECTED_GLOBAL_REPEAT = 3
ARMS = {
    "bce_control": ROOT / "configs/train/semifinal_quality_bce_control_4090.yaml",
    "varifocal": ROOT / "configs/train/semifinal_quality_varifocal_4090.yaml",
}
GATE = {
    "minimum_weighted_gain_over_control": 0.20,
    "minimum_each_view_score_gain": 0.10,
    "minimum_each_view_map50_gain": 0.005,
    "maximum_each_view_recall_drop": 0.002,
}


def completed(run_dir: Path) -> bool:
    return (run_dir / "completion.json").is_file()


def train_command(python: str, config: Path, run_dir: Path, base: Path) -> list[str]:
    command = [
        python, "-u", str(ROOT / "scripts/train.py"),
        "--config", str(config), "--model", str(base),
    ]
    last = run_dir / "weights/last.pt"
    if last.is_file() and not completed(run_dir):
        command.extend(["--resume", str(last)])
    return command


def arm_candidate(selection: dict, base_sha: str) -> dict:
    trained = [item for item in selection["candidates"] if item["sha256"] != base_sha]
    if not trained:
        raise RuntimeError("checkpoint selector returned no trained arm checkpoint")
    return max(
        trained,
        key=lambda item: (float(item["weighted_score"]), float(item["robust_min_score"])),
    )


def baseline_candidate(selection: dict, base_sha: str) -> dict:
    matches = [item for item in selection["candidates"] if item["sha256"] == base_sha]
    if len(matches) != 1:
        raise RuntimeError(f"expected one protected baseline row, found {len(matches)}")
    return matches[0]


def metric(row: dict, view: str, key: str) -> float:
    aliases = {
        "score": ("score_proxy_20_60_20",),
        "map50": ("map50_macro", "map50"),
        "recall": ("recall_micro", "recall"),
    }
    payload = row["views"][view]
    for name in aliases[key]:
        if name in payload:
            return float(payload[name])
    raise KeyError(f"missing {key} in {view}: {sorted(payload)}")


def decide(baseline: dict, control: dict, quality: dict) -> dict:
    failures = []
    weighted_gain = float(quality["weighted_score"]) - float(control["weighted_score"])
    if weighted_gain < GATE["minimum_weighted_gain_over_control"]:
        failures.append("weighted_gain_over_control<0.20")
    per_view = {}
    for view in ("original", "grid_crops"):
        score_gain = metric(quality, view, "score") - metric(control, view, "score")
        map_gain = metric(quality, view, "map50") - metric(control, view, "map50")
        recall_drop = metric(quality, view, "recall") - metric(control, view, "recall")
        per_view[view] = {
            "quality": quality["views"][view],
            "control": control["views"][view],
            "baseline": baseline["views"][view],
            "score_gain_over_control": score_gain,
            "map50_gain_over_control": map_gain,
            "recall_delta_vs_control": recall_drop,
        }
        if score_gain < GATE["minimum_each_view_score_gain"]:
            failures.append(f"{view}:score_gain<0.10")
        if map_gain < GATE["minimum_each_view_map50_gain"]:
            failures.append(f"{view}:map50_gain<0.005")
        if recall_drop < -GATE["maximum_each_view_recall_drop"]:
            failures.append(f"{view}:recall_drop>0.002")
    return {
        "gate": "PASS_STAGE1" if not failures else "STOP",
        "stage2_exact_v7_tta_fusion_allowed": not failures,
        "leaderboard_submission_allowed": False,
        "failures": failures,
        "thresholds": GATE,
        "weighted_gain_over_control": weighted_gain,
        "weighted_gain_over_protected_baseline": (
            float(quality["weighted_score"]) - float(baseline["weighted_score"])
        ),
        "views": per_view,
    }


def render_report(decision: dict, rows: dict) -> str:
    lines = [
        "# Quality-aware detector matched-compute experiment", "",
        "Both arms started from the same protected Model A checkpoint and used "
        "the same data, seed, optimizer, augmentation, resolution, batch, and epoch budget. "
        "The only intended change is BCE versus IoU-quality Varifocal classification loss.", "",
        "|Arm|Original|Grid crops|Weighted|", "|---|---:|---:|---:|",
    ]
    for name in ("protected_baseline", "bce_control", "varifocal"):
        row = rows[name]
        lines.append(
            f"|{name}|{metric(row, 'original', 'score'):.6f}|"
            f"{metric(row, 'grid_crops', 'score'):.6f}|{float(row['weighted_score']):.6f}|"
        )
    lines.extend([
        "", "## Decision", "",
        f"Gate: `{decision['gate']}`.", "",
        f"Weighted gain over control: `{decision['weighted_gain_over_control']:+.6f}`.", "",
        "A pass authorizes only exact V7 TTA fusion evaluation with frozen Model B. "
        "It never authorizes a Test or leaderboard submission.", "",
        "Failures: " + (", ".join(decision["failures"]) if decision["failures"] else "none"),
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    python = sys.executable
    output = project / "runs/semifinal/quality_aware_v1_pipeline"
    base = project / "runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt"
    split = project / "data/official_v2_20260831b/splits.json"
    source = project / "data/official/train"
    verified_file(base, MODEL_A_SHA256)
    verified_file(split, SPLIT_SHA)
    ensure_ultralytics_compatibility()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("quality-aware experiment requires RTX 4090")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    if free_bytes < 10 * 1024**3:
        raise RuntimeError(f"less than 10 GiB free VRAM: {free_bytes / 1024**3:.2f} GiB")

    package_manifest = ROOT / "QUALITY_AWARE_PACKAGE_MANIFEST.json"
    if package_manifest.is_file():
        for relative, digest in json.loads(package_manifest.read_text(encoding="utf-8")).items():
            verified_file(ROOT / relative, digest)
    environment = {
        "gpu": torch.cuda.get_device_name(0),
        "free_vram_gib": free_bytes / 1024**3,
        "total_vram_gib": total_bytes / 1024**3,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "required_ultralytics": SUPPORTED_ULTRALYTICS,
        "python": sys.version,
        "model_a_sha256": MODEL_A_SHA256,
        "split_sha256": SPLIT_SHA,
        "test_or_final_used": False,
        "time": datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", environment)
    print(json.dumps(environment, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    dataset = project / "data/semifinal_yolo_balanced_v3"
    summary = dataset / "metadata/summary.json"
    if not summary.is_file():
        run_logged([
            python, "-u", str(ROOT / "scripts/prepare_data.py"), "--config",
            str(ROOT / "configs/data/semifinal_full_retrain_v3.yaml"),
        ], project, output / "prepare.log")
    prepared = json.loads(summary.read_text(encoding="utf-8"))
    if int(prepared.get("global_repeat", -1)) != EXPECTED_GLOBAL_REPEAT:
        raise ValueError("prepared dataset does not match the frozen balanced-v3 contract")

    if args.smoke_only:
        smoke = output / "smoke"
        command = [
            python, "-u", str(ROOT / "scripts/train.py"), "--config", str(ARMS["varifocal"]),
            "--model", str(base), "--project", str(smoke.parent), "--name", smoke.name,
            "--epochs", "1", "--fraction", "0.02", "--save-period", "1",
        ]
        run_logged(command, project, output / "smoke.log")
        write_json(output / "SMOKE_COMPLETE.json", {"status": "complete", "loss": "varifocal"})
        return

    selections = {}
    rows = {}
    run_dirs = {}
    for arm, config in ARMS.items():
        run_dir = project / "runs/semifinal" / f"quality_v1_{arm}"
        run_dirs[arm] = run_dir
        if not completed(run_dir):
            run_logged(train_command(python, config, run_dir, base), project, output / f"{arm}_train.log")
        selection_dir = output / f"{arm}_eval"
        selection_path = selection_dir / "selection.json"
        if not selection_path.is_file():
            run_logged([
                python, "-u", str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
                "--project-root", str(project), "--run-dir", str(run_dir),
                "--external-checkpoint", str(base), "--output", str(selection_dir),
            ], project, output / f"{arm}_select.log")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        selections[arm] = selection
        rows[arm] = arm_candidate(selection, MODEL_A_SHA256)

    baseline_a = baseline_candidate(selections["bce_control"], MODEL_A_SHA256)
    baseline_b = baseline_candidate(selections["varifocal"], MODEL_A_SHA256)
    if baseline_a["views"] != baseline_b["views"] or baseline_a["weighted_score"] != baseline_b["weighted_score"]:
        raise RuntimeError("protected baseline replay differs between arms")
    rows["protected_baseline"] = baseline_a
    decision = decide(baseline_a, rows["bce_control"], rows["varifocal"])
    result = {"environment": environment, "rows": rows, "decision": decision}
    write_json(output / "QUALITY_AWARE_DECISION.json", result)
    (output / "REPORT.md").write_text(render_report(decision, rows), encoding="utf-8")

    final_dir = output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for name in ("bce_control", "varifocal"):
        source_weight = Path(rows[name]["checkpoint"])
        target = final_dir / f"{name}.pt"
        if not target.is_file():
            shutil.copy2(source_weight, target)
        if sha256_file(target) != rows[name]["sha256"]:
            raise RuntimeError(f"copied {name} checkpoint digest mismatch")

    audit_dir = output / "audit"
    for arm, run_dir in run_dirs.items():
        destination = audit_dir / arm
        destination.mkdir(parents=True, exist_ok=True)
        for relative in (
            "completion.json", "training_contract.json", "optimizer_steps.jsonl",
            "batch_health.jsonl", "results.csv", "args.yaml",
        ):
            source_path = run_dir / relative
            if source_path.is_file():
                shutil.copy2(source_path, destination / relative)
    archive = project.parent / "quality_aware_v1_results_20260928.tar.gz"
    if archive.exists():
        raise FileExistsError(f"refusing to overwrite result archive: {archive}")
    archive_members = [
        output / "preflight.json",
        output / "QUALITY_AWARE_DECISION.json",
        output / "REPORT.md",
        output / "final",
        output / "audit",
        output / "bce_control_eval/selection.json",
        output / "bce_control_eval/SUMMARY.md",
        output / "varifocal_eval/selection.json",
        output / "varifocal_eval/SUMMARY.md",
        *sorted(output.glob("*.log")),
    ]
    with tarfile.open(archive, "w:gz") as handle:
        for member in archive_members:
            if member.exists():
                handle.add(member, arcname=(Path(output.name) / member.relative_to(output)).as_posix())
    print(json.dumps({
        "decision": decision,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

