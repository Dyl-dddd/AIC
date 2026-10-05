"""Run matched-compute BCE vs task-aligned quality-Varifocal on RTX 4090.

This is a development experiment, never a Test or final-lockbox pipeline.  It
starts both arms from the same protected Model A checkpoint, evaluates every
saved checkpoint on the frozen dual-view dev split, and emits a pass/stop
decision.  No leaderboard submission is created.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import json
import math
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
from steel_defect.training_audit import dataset_signature, training_code_signature


MODEL_A_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
EXPECTED_GLOBAL_REPEAT = 3
EXPECTED_EPOCHS = 8
RUN_TAG = "quality_aware_v1_fix3_20260928"
ARMS = {
    "bce_control": ROOT / "configs/train/semifinal_quality_bce_control_4090.yaml",
    "varifocal": ROOT / "configs/train/semifinal_quality_varifocal_4090.yaml",
}
GATE = {
    "minimum_weighted_gain_over_control": 0.20,
    "minimum_each_view_score_gain": 0.10,
    "minimum_each_view_map50_gain": 0.005,
    "maximum_each_view_recall_drop": 0.002,
    "minimum_each_view_map75_gain": 0.005,
    "minimum_each_view_map50_95_gain": 0.0,
    "maximum_per_class_ap50_drop": 0.02,
    "maximum_false_positives_per_image_ratio": 1.05,
}


def completed(run_dir: Path) -> bool:
    return (run_dir / "completion.json").is_file()


def verify_package_manifest(root: Path) -> dict[str, str]:
    manifest_path = root / "QUALITY_AWARE_PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"required package manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("quality-aware package manifest is empty or invalid")
    for relative, digest in manifest.items():
        path = root / relative
        if path.resolve() == manifest_path.resolve():
            raise ValueError("package manifest must not self-reference")
        verified_file(path, digest)
    return manifest


def train_command(python: str, config: Path, run_dir: Path, base: Path) -> list[str]:
    command = [
        python, "-u", str(ROOT / "scripts/train.py"),
        "--config", str(config), "--model", str(base),
        "--project", str(run_dir.parent), "--name", run_dir.name,
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
        "map75": ("map75_macro", "map75"),
        "map50_95": ("map50_95_macro", "map50_95"),
        "recall": ("recall_micro", "recall"),
        "false_positives_per_image": ("false_positives_per_image",),
    }
    payload = row["views"][view]
    for name in aliases[key]:
        if name in payload:
            return float(payload[name])
    raise KeyError(f"missing {key} in {view}: {sorted(payload)}")


def parse_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def optimizer_update_total(records: list[dict]) -> int:
    """Sum cumulative optimizer counters across any exact-resume segments."""
    if not records:
        raise ValueError("optimizer audit is empty")
    total = 0
    segment_max = 0
    previous = -1
    previous_microbatches = -1
    for record in records:
        current = int(record["optimizer_updates"])
        microbatches = int(record.get("microbatches", -1))
        if current < previous or (
            previous_microbatches >= 0 and microbatches >= 0
            and microbatches <= previous_microbatches
        ):
            total += segment_max
            segment_max = 0
        segment_max = max(segment_max, current)
        previous = current
        previous_microbatches = microbatches
    return total + segment_max


def current_training_code_signature() -> dict[str, str]:
    return training_code_signature(ROOT)


def current_evaluation_code_signature() -> dict[str, str]:
    paths = [
        ROOT / "scripts/select_full_retrain_checkpoint.py",
        ROOT / "scripts/run_semifinal_stage1.py",
        *sorted((ROOT / "steel_defect").glob("*.py")),
    ]
    return {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in paths}


def run_audit(run_dir: Path, expected_dataset_signature: dict) -> dict:
    contract_path = run_dir / "training_contract.json"
    completion_path = run_dir / "completion.json"
    if not contract_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError(f"incomplete governed training run: {run_dir}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if contract.get("dataset_signature") != expected_dataset_signature:
        raise ValueError(f"dataset signature mismatch in {run_dir}")
    if contract.get("code_sha256") != current_training_code_signature():
        raise ValueError(f"training code signature mismatch in {run_dir}")
    if contract.get("status") not in {"starting", "training", "complete"}:
        raise ValueError(f"invalid training contract status in {run_dir}")
    if completion.get("status") != "complete" or int(completion.get("exit_code", -1)) != 0:
        raise ValueError(f"training did not complete cleanly: {run_dir}")
    optimizer_rows = parse_jsonl(run_dir / "optimizer_steps.jsonl")
    batch_rows = parse_jsonl(run_dir / "batch_health.jsonl")
    if len(optimizer_rows) != EXPECTED_EPOCHS:
        raise ValueError(
            f"expected exactly {EXPECTED_EPOCHS} completed epochs in {run_dir}, "
            f"found {len(optimizer_rows)}"
        )
    transfer_path = run_dir / "initial_transfer.json"
    if not transfer_path.is_file():
        raise FileNotFoundError(f"missing initial transfer witness: {transfer_path}")
    transfer = json.loads(transfer_path.read_text(encoding="utf-8"))
    if transfer.get("source_sha256") != MODEL_A_SHA256:
        raise ValueError(f"initial checkpoint differs from protected Model A in {run_dir}")
    if (
        int(transfer.get("parameter_elements_matched", -1))
        != int(transfer.get("parameter_elements_total", -2))
    ):
        raise ValueError(f"protected Model A was not transferred exactly in {run_dir}")
    if int(transfer.get("tensor_matches", -1)) != int(transfer.get("tensor_total", -2)):
        raise ValueError(f"protected Model A buffers were not transferred exactly in {run_dir}")
    return {
        "dataset_signature": contract["dataset_signature"],
        "config": contract["config"],
        "optimizer_updates": optimizer_update_total(optimizer_rows),
        "microbatches": len(batch_rows),
        "epochs": len(optimizer_rows),
        "all_losses_finite": all(
            isinstance(row.get("loss"), (int, float)) and math.isfinite(float(row["loss"]))
            for row in batch_rows
        ),
        "initial_transfer": transfer,
    }


def validate_matched_compute(left: dict, right: dict) -> None:
    if left["dataset_signature"] != right["dataset_signature"]:
        raise ValueError("A/B dataset signatures differ")
    for key in ("optimizer_updates", "microbatches", "epochs"):
        if left[key] != right[key]:
            raise ValueError(f"A/B {key} differ: {left[key]} != {right[key]}")
    if not left["all_losses_finite"] or not right["all_losses_finite"]:
        raise ValueError("non-finite training loss recorded in an experiment arm")
    allowed = {
        "loss", "name", "project", "varifocal_gamma", "varifocal_alpha", "quality_power",
    }
    left_config = {key: value for key, value in left["config"].items() if key not in allowed}
    right_config = {key: value for key, value in right["config"].items() if key not in allowed}
    if left_config != right_config:
        raise ValueError("A/B realized training contracts differ outside intended loss fields")


def validate_prepared_dataset(dataset: Path) -> dict:
    summary_path = dataset / "metadata/summary.json"
    manifest_path = dataset / "metadata/splits.json"
    yaml_path = dataset / "steel_defect.yaml"
    for path in (summary_path, manifest_path, yaml_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(summary.get("global_repeat", -1)) != EXPECTED_GLOBAL_REPEAT:
        raise ValueError("prepared dataset global_repeat mismatch")
    if summary.get("records") != {"train": 2395, "dev": 459}:
        raise ValueError(f"prepared source counts mismatch: {summary.get('records')}")
    if manifest.get("frozen_source_sha256") != SPLIT_SHA:
        raise ValueError("prepared dataset was not derived from the protected split manifest")
    expected_sources = {"train": 2395, "dev": 459, "final": 291}
    expected_groups = {"train": 1399, "dev": 258, "final": 185}
    if {key: len(manifest["splits"][key]) for key in expected_sources} != expected_sources:
        raise ValueError("prepared frozen source counts mismatch")
    if {key: len(manifest["groups"][key]) for key in expected_groups} != expected_groups:
        raise ValueError("prepared frozen group counts mismatch")
    group_sets = {key: set(manifest["groups"][key]) for key in expected_groups}
    if any(group_sets[left] & group_sets[right] for left, right in (
        ("train", "dev"), ("train", "final"), ("dev", "final")
    )):
        raise ValueError("official groups overlap across frozen splits")
    final_images = dataset / "images/final"
    if final_images.exists() and any(final_images.rglob("*")):
        raise ValueError("final-holdout images entered the prepared training dataset")
    signature = dataset_signature(yaml_path)
    return {"summary": summary, "manifest": str(manifest_path), "signature": signature}


def fixed_selection_is_current(selection: dict, expected_shas: set[str]) -> bool:
    return (
        selection.get("checkpoint_policy") == "explicit"
        and selection.get("export_thresholds") == [0.0001]
        and {item.get("sha256") for item in selection.get("candidates", [])} == expected_shas
    )


def validate_selection_reports(selection: dict, selection_dir: Path) -> None:
    expected_code = current_evaluation_code_signature()
    root = selection_dir.resolve()
    for candidate in selection.get("candidates", []):
        digest = candidate.get("sha256")
        if set(candidate.get("reports", {})) != {"original", "grid_crops"}:
            raise ValueError(f"selection candidate has incomplete views: {digest}")
        for view, report_name in candidate.get("reports", {}).items():
            report_path = Path(report_name).resolve()
            if root not in report_path.parents or not report_path.is_file():
                raise ValueError(f"selection report escaped or is missing: {report_path}")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            signature = report.get("signature", {})
            if not report.get("complete"):
                raise ValueError(f"incomplete selection report: {report_path}")
            if signature.get("weights_sha256") != digest:
                raise ValueError(f"selection/report checkpoint mismatch: {report_path}")
            if signature.get("split_sha256") != SPLIT_SHA:
                raise ValueError(f"selection/report split mismatch: {report_path}")
            if signature.get("view") != view or signature.get("limit") != 0:
                raise ValueError(f"selection/report view or limit mismatch: {report_path}")
            if signature.get("threshold_scan") != [0.0001]:
                raise ValueError(f"selection/report threshold mismatch: {report_path}")
            if signature.get("code") != expected_code:
                raise ValueError(f"selection/report code signature mismatch: {report_path}")
            if len(signature.get("source_ids", [])) != 459:
                raise ValueError(f"selection/report source count mismatch: {report_path}")
            if len(set(signature.get("source_ids", []))) != 459:
                raise ValueError(f"selection/report source IDs are duplicated: {report_path}")
            options = signature.get("options", {})
            expected_options = {
                "conf": 0.0001,
                "imgsz": 1024,
                "tile_layout": "semifinal_grid",
                "global_pass": True,
                "max_det": 2000,
                "edge_penalty": 0.85,
            }
            if any(options.get(key) != value for key, value in expected_options.items()):
                raise ValueError(f"selection/report inference policy mismatch: {report_path}")


def copy_recomputable_evaluation_artifacts(
    source_report: Path, destination: Path, stem: str
) -> dict[str, str]:
    """Copy predictions and compact GT so Stage-1 metrics can be recomputed offline."""
    source_predictions = source_report.parent / "predictions.json"
    source_candidates = source_report.parent / "candidates.jsonl.gz"
    if not source_predictions.is_file() or not source_candidates.is_file():
        raise FileNotFoundError(f"evaluation artifacts missing beside {source_report}")
    prediction_target = destination / f"{stem}_predictions.json"
    ground_truth_target = destination / f"{stem}_ground_truth.jsonl.gz"
    shutil.copy2(source_predictions, prediction_target)
    with gzip.open(source_candidates, "rt", encoding="utf-8") as source_handle, gzip.open(
        ground_truth_target, "wt", encoding="utf-8"
    ) as target_handle:
        for line in source_handle:
            record = json.loads(line)
            if record.get("type") == "header":
                compact = record
            else:
                compact = {
                    key: record[key]
                    for key in ("image_id", "source_id", "image_size", "ground_truth")
                }
            target_handle.write(json.dumps(compact, separators=(",", ":")) + "\n")
    return {
        "predictions": prediction_target.as_posix(),
        "ground_truth": ground_truth_target.as_posix(),
    }


def per_class_ap50(row: dict, view: str) -> dict[str, float]:
    if "_per_class_ap50" in row:
        return {name: float(value) for name, value in row["_per_class_ap50"][view].items()}
    report = Path(row["reports"][view])
    payload = json.loads(report.read_text(encoding="utf-8"))
    return {
        name: float(values["ap50"])
        for name, values in payload["all_exported"]["per_class"].items()
        if int(values["ground_truth"]) > 0 and name != "qilie"
    }


def decide(baseline: dict, control: dict, quality: dict) -> dict:
    failures = []
    weighted_gain = float(quality["weighted_score"]) - float(control["weighted_score"])
    if weighted_gain < GATE["minimum_weighted_gain_over_control"]:
        failures.append("weighted_gain_over_control<0.20")
    per_view = {}
    for view in ("original", "grid_crops"):
        score_gain = metric(quality, view, "score") - metric(control, view, "score")
        map_gain = metric(quality, view, "map50") - metric(control, view, "map50")
        map75_gain = metric(quality, view, "map75") - metric(control, view, "map75")
        map50_95_gain = metric(quality, view, "map50_95") - metric(control, view, "map50_95")
        recall_drop = metric(quality, view, "recall") - metric(control, view, "recall")
        control_fp = metric(control, view, "false_positives_per_image")
        quality_fp = metric(quality, view, "false_positives_per_image")
        fp_ratio = quality_fp / control_fp if control_fp > 0.0 else (
            1.0 if quality_fp == 0.0 else float("inf")
        )
        quality_per_class = per_class_ap50(quality, view)
        control_per_class = per_class_ap50(control, view)
        class_deltas = {
            name: quality_per_class[name] - control_per_class[name]
            for name in sorted(control_per_class)
        }
        per_view[view] = {
            "quality": quality["views"][view],
            "control": control["views"][view],
            "baseline": baseline["views"][view],
            "score_gain_over_control": score_gain,
            "map50_gain_over_control": map_gain,
            "map75_gain_over_control": map75_gain,
            "map50_95_gain_over_control": map50_95_gain,
            "recall_delta_vs_control": recall_drop,
            "false_positives_per_image_ratio_vs_control": fp_ratio,
            "per_class_ap50_delta_vs_control": class_deltas,
        }
        if score_gain < GATE["minimum_each_view_score_gain"]:
            failures.append(f"{view}:score_gain<0.10")
        if map_gain < GATE["minimum_each_view_map50_gain"]:
            failures.append(f"{view}:map50_gain<0.005")
        if map75_gain < GATE["minimum_each_view_map75_gain"]:
            failures.append(f"{view}:map75_gain<0.005")
        if map50_95_gain < GATE["minimum_each_view_map50_95_gain"]:
            failures.append(f"{view}:map50_95_decreased")
        if recall_drop < -GATE["maximum_each_view_recall_drop"]:
            failures.append(f"{view}:recall_drop>0.002")
        if fp_ratio > GATE["maximum_false_positives_per_image_ratio"]:
            failures.append(f"{view}:false_positives_per_image_increase>5%")
        if min(class_deltas.values(), default=0.0) < -GATE["maximum_per_class_ap50_drop"]:
            failures.append(f"{view}:per_class_ap50_drop>0.02")
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
        "The only intended change is BCE versus TAL-soft-target quality-weighted "
        "Varifocal classification loss.", "",
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
    output = project / "runs/semifinal" / f"{RUN_TAG}_pipeline"
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

    package_manifest = verify_package_manifest(ROOT)
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
        "package_manifest_members": len(package_manifest),
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
    prepared_contract = validate_prepared_dataset(dataset)
    prepared = prepared_contract["summary"]
    environment["prepared_dataset_signature"] = prepared_contract["signature"]
    write_json(output / "preflight.json", environment)

    if args.smoke_only:
        smoke = output / "smoke"
        command = [
            python, "-u", str(ROOT / "scripts/train.py"), "--config", str(ARMS["varifocal"]),
            "--model", str(base), "--project", str(smoke.parent), "--name", smoke.name,
            "--epochs", "1", "--fraction", "0.02", "--save-period", "1",
        ]
        run_logged(command, project, output / "smoke.log")
        run_logged([
            python, "-u", str(ROOT / "scripts/verify_quality_checkpoint_compatibility.py"),
            "--checkpoint", str(smoke / "weights/last.pt"),
            "--report", str(output / "smoke_checkpoint_compatibility.json"),
            "--expected-epochs", "1",
        ], project, output / "smoke_reload.log")
        write_json(output / "SMOKE_COMPLETE.json", {"status": "complete", "loss": "varifocal"})
        return

    selections = {}
    rows = {}
    run_dirs = {}
    audits = {}
    for arm, config in ARMS.items():
        run_dir = project / "runs/semifinal" / f"{RUN_TAG}_{arm}"
        run_dirs[arm] = run_dir
        if not completed(run_dir):
            run_logged(train_command(python, config, run_dir, base), project, output / f"{arm}_train.log")
        run_logged([
            python, "-u", str(ROOT / "scripts/verify_quality_checkpoint_compatibility.py"),
            "--checkpoint", str(run_dir / "weights/last.pt"),
            "--report", str(output / f"{arm}_checkpoint_compatibility.json"),
            "--expected-epochs", str(EXPECTED_EPOCHS),
        ], project, output / f"{arm}_reload.log")
        audits[arm] = run_audit(run_dir, prepared_contract["signature"])
        selection_dir = output / f"{arm}_eval"
        selection_path = selection_dir / "selection.json"
        terminal = run_dir / "weights/last.pt"
        if not terminal.is_file():
            raise FileNotFoundError(terminal)
        expected_shas = {MODEL_A_SHA256, sha256_file(terminal)}
        if selection_path.is_file():
            previous_selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if not fixed_selection_is_current(previous_selection, expected_shas):
                raise ValueError(f"stale fixed-endpoint selection: {selection_path}")
        if not selection_path.is_file():
            run_logged([
                python, "-u", str(ROOT / "scripts/select_full_retrain_checkpoint.py"),
                "--project-root", str(project), "--run-dir", str(run_dir),
                "--checkpoint", str(terminal),
                "--external-checkpoint", str(base), "--output", str(selection_dir),
                "--fixed-export-threshold", "0.0001",
            ], project, output / f"{arm}_select.log")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if not fixed_selection_is_current(selection, expected_shas):
            raise ValueError(f"fixed-endpoint selector output is invalid: {selection_path}")
        validate_selection_reports(selection, selection_dir)
        selections[arm] = selection
        rows[arm] = arm_candidate(selection, MODEL_A_SHA256)

    validate_matched_compute(audits["bce_control"], audits["varifocal"])

    baseline_a = baseline_candidate(selections["bce_control"], MODEL_A_SHA256)
    baseline_b = baseline_candidate(selections["varifocal"], MODEL_A_SHA256)
    for view in ("original", "grid_crops"):
        for key in (
            "score", "map50", "map75", "map50_95", "recall",
            "false_positives_per_image",
        ):
            if not math.isclose(
                metric(baseline_a, view, key), metric(baseline_b, view, key),
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise RuntimeError(f"protected baseline replay differs between arms: {view}/{key}")
    rows["protected_baseline"] = baseline_a
    decision = decide(baseline_a, rows["bce_control"], rows["varifocal"])
    result = {
        "environment": environment,
        "matched_compute_audit": audits,
        "rows": rows,
        "decision": decision,
    }
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
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_index = {}
    shutil.copy2(split, audit_dir / "frozen_split_manifest.json")
    for arm, run_dir in run_dirs.items():
        destination = audit_dir / arm
        destination.mkdir(parents=True, exist_ok=True)
        for relative in (
            "completion.json", "training_contract.json", "optimizer_steps.jsonl",
            "batch_health.jsonl", "results.csv", "args.yaml", "initial_transfer.json",
        ):
            source_path = run_dir / relative
            if source_path.is_file():
                shutil.copy2(source_path, destination / relative)
        evaluation = audit_dir / "evaluation" / arm
        evaluation.mkdir(parents=True, exist_ok=True)
        arm_baseline = baseline_candidate(selections[arm], MODEL_A_SHA256)
        for row_name, row in (("protected_baseline", arm_baseline), (arm, rows[arm])):
            for view, report_path in row["reports"].items():
                source_report = Path(report_path)
                if not source_report.is_file():
                    raise FileNotFoundError(source_report)
                target_report = evaluation / f"{row_name}_{view}_metrics.json"
                shutil.copy2(source_report, target_report)
                artifacts = copy_recomputable_evaluation_artifacts(
                    source_report, evaluation, f"{row_name}_{view}"
                )
                audit_index[f"{arm}/{row_name}/{view}"] = {
                    "metrics": target_report.relative_to(output).as_posix(),
                    **{
                        key: Path(value).relative_to(output).as_posix()
                        for key, value in artifacts.items()
                    },
                }
    write_json(audit_dir / "AUDIT_INDEX.json", audit_index)
    archive = project.parent / "quality_aware_v1_fix3_results_20260928.tar.gz"
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
        *sorted(output.glob("*_checkpoint_compatibility.json")),
        *sorted(output.glob("*.log")),
    ]
    manifest_files = []
    for member in archive_members:
        if member.is_file():
            manifest_files.append(member)
        elif member.is_dir():
            manifest_files.extend(path for path in member.rglob("*") if path.is_file())
    result_manifest = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(set(manifest_files))
    }
    write_json(output / "RESULT_MANIFEST.json", result_manifest)
    archive_members.append(output / "RESULT_MANIFEST.json")
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

