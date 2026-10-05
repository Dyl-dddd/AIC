"""Run the one-time SAFE-R2 final-holdout evaluation; never tune or submit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import tarfile

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.analyze_final_submission_postprocess import VIEW_WEIGHTS, load_cell, metrics50, replay
from scripts.analyze_source_aware_fusion import source_aware_fuse
from scripts.run_semifinal_stage1 import SPLIT_SHA, evaluate_cell, write_json
from steel_defect.classes import CLASS_TO_ID
from steel_defect.governance import check_evaluation_scope, frozen_records, validate_split_manifest
from steel_defect.inference import InferenceOptions
from steel_defect.runtime import sha256_file
from steel_defect.safe_r2 import apply_safe_r2, load_profile
from steel_defect.voc import parse_voc


MODEL_A_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
MODEL_B_SHA256 = "02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090"
EXPECTED_SOURCES = 291
EXPECTED_ALLOWED_GT = 517
VIEWS = ("original", "grid_crops")
V7_FUSION = {
    "match_iou": 0.70,
    "boost": 1.10,
    "coord_mode": "anchor",
    "unmatched_b_factor": 0.50,
    "b_scale": 0.35,
    "final_nms_iou": 0.72,
}
LOCKBOX_GATE = {
    "minimum_weighted_score_delta": 0.15,
    "minimum_each_view_score_delta": 0.0,
    "minimum_each_view_map50_delta": 0.0,
    "maximum_each_view_recall_drop": 0.002,
    "require_exact_membership_count": True,
}


def verify_package() -> None:
    manifest_path = ROOT / "SAFE_R2_FINAL_LOCKBOX_PACKAGE_MANIFEST.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"package member failed SHA256 verification: {relative}")


def validate_frozen_profile(path: Path) -> dict:
    """Fail closed unless group-OOF explicitly authorized this exact profile."""
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen SAFE-R2 profile: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    profile = load_profile(raw)
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("SAFE-R2 profile is missing metadata")
    required = {
        "method": "SAFE-R2",
        "deployment_allowed": True,
        "gate": "PASS",
        "training_view": "original",
        "test_labels_used": False,
    }
    mismatched = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in required.items()
        if metadata.get(key) != value
    }
    selected = str(metadata.get("selected_from_group_oof", ""))
    if mismatched or not selected or selected == "A0_baseline":
        raise ValueError(
            "SAFE-R2 deployment is not authorized by the frozen group-OOF gate: "
            f"mismatched={mismatched}, selected={selected!r}"
        )
    return profile


def fuse_v7(cell_a: dict, cell_b: dict) -> list[dict]:
    if cell_a["ground_truth"] != cell_b["ground_truth"]:
        raise ValueError("Model A/B ground truth mismatch")
    if cell_a["sizes"] != cell_b["sizes"]:
        raise ValueError("Model A/B image-size mismatch")
    return source_aware_fuse(replay(cell_a), replay(cell_b), **V7_FUSION)


def _membership_counts(predictions: list[dict]) -> Counter:
    return Counter((item["image_id"], item["category_name"]) for item in predictions)


def apply_profile_to_view(
    cell_a: dict,
    cell_b: dict,
    baseline: list[dict],
    profile: dict,
) -> tuple[list[dict], dict]:
    """Apply one frozen profile while preserving every V7 anchor."""
    rows_a = {row["image_id"]: row for row in cell_a["rows"]}
    rows_b = {row["image_id"]: row for row in cell_b["rows"]}
    if set(rows_a) != set(rows_b):
        raise ValueError("Model A/B row sets differ")
    by_image: dict[str, list[dict]] = defaultdict(list)
    for item in baseline:
        by_image[item["image_id"]].append(item)

    output: list[dict] = []
    refined = 0
    reasons: Counter = Counter()
    support_buckets: Counter = Counter()
    for image_id in sorted(by_image):
        row_a, row_b = rows_a[image_id], rows_b[image_id]
        if tuple(row_a["image_size"]) != tuple(row_b["image_size"]):
            raise ValueError(f"Model A/B image-size mismatch: {image_id}")
        candidates_a: dict[int, list[dict]] = defaultdict(list)
        candidates_b: dict[int, list[dict]] = defaultdict(list)
        for item in row_a["candidates"]:
            candidates_a[int(item["class_id"])].append(item)
        for item in row_b["candidates"]:
            candidates_b[int(item["class_id"])].append(item)

        # Run class-by-class so irrelevant classes never enter provenance matching.
        per_class: dict[int, list[dict]] = defaultdict(list)
        for prediction in by_image[image_id]:
            per_class[CLASS_TO_ID[prediction["category_name"]]].append(prediction)
        for class_id in sorted(per_class):
            anchors = [
                {**item, "_lockbox_anchor_bbox": list(item["bbox"])}
                for item in per_class[class_id]
            ]
            changed = apply_safe_r2(
                anchors,
                candidates_a[class_id],
                candidates_b[class_id],
                tuple(row_a["image_size"]),
                profile,
                include_diagnostics=True,
            )
            for item in changed:
                diagnostic = item.pop("safe_r2")
                anchor_bbox = item.pop("_lockbox_anchor_bbox")
                # Match the integer/7-decimal serialization required by the
                # official submission validator.  A pathological sub-pixel
                # refinement falls back to the V7 anchor instead of deleting
                # a prediction.
                rounded_box = [int(round(float(value))) for value in item["bbox"]]
                if rounded_box[2] <= rounded_box[0] or rounded_box[3] <= rounded_box[1]:
                    rounded_box = [int(round(float(value))) for value in anchor_bbox]
                    diagnostic["refined"] = False
                    diagnostic["refinement_reason"] = "rounding_fallback"
                item["bbox"] = rounded_box
                item["score"] = round(float(item["score"]), 7)
                refined += int(diagnostic["refined"])
                reasons[diagnostic["refinement_reason"]] += 1
                for bucket in diagnostic["provenance_buckets"]:
                    support_buckets[bucket] += 1
                output.append(item)

    if len(output) != len(baseline) or _membership_counts(output) != _membership_counts(baseline):
        raise RuntimeError("SAFE-R2 violated its no-add/no-delete membership invariant")
    return output, {
        "predictions": len(output),
        "refined": refined,
        "refinement_reasons": dict(sorted(reasons.items())),
        "support_bucket_counts": dict(sorted(support_buckets.items())),
    }


def evaluate_gate(baseline: dict[str, dict], candidate: dict[str, dict]) -> dict:
    failures: list[str] = []
    baseline_weighted = sum(VIEW_WEIGHTS[v] * baseline[v]["score"] for v in VIEWS)
    candidate_weighted = sum(VIEW_WEIGHTS[v] * candidate[v]["score"] for v in VIEWS)
    delta = candidate_weighted - baseline_weighted
    if delta < LOCKBOX_GATE["minimum_weighted_score_delta"]:
        failures.append("weighted_score_delta<0.15")
    views = {}
    for view in VIEWS:
        score_delta = candidate[view]["score"] - baseline[view]["score"]
        map_delta = candidate[view]["map50"] - baseline[view]["map50"]
        recall_delta = candidate[view]["r_micro"] - baseline[view]["r_micro"]
        membership_equal = candidate[view]["predictions"] == baseline[view]["predictions"]
        if score_delta < LOCKBOX_GATE["minimum_each_view_score_delta"]:
            failures.append(f"{view}:score_decreased")
        if map_delta < LOCKBOX_GATE["minimum_each_view_map50_delta"]:
            failures.append(f"{view}:map50_decreased")
        if recall_delta < -LOCKBOX_GATE["maximum_each_view_recall_drop"]:
            failures.append(f"{view}:recall_drop>0.002")
        if LOCKBOX_GATE["require_exact_membership_count"] and not membership_equal:
            failures.append(f"{view}:membership_count_changed")
        views[view] = {
            "score_delta": score_delta,
            "map50_delta": map_delta,
            "recall_delta": recall_delta,
            "membership_equal": membership_equal,
        }
    return {
        "gate": "PASS" if not failures else "STOP",
        "deployment_allowed_after_lockbox": not failures,
        "failures": failures,
        "baseline_weighted_score": baseline_weighted,
        "candidate_weighted_score": candidate_weighted,
        "weighted_score_delta": delta,
        "views": views,
        "thresholds": LOCKBOX_GATE,
    }


def write_predictions(path: Path, predictions: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "safe_r2_predictions", "count": len(predictions)}) + "\n")
        for item in predictions:
            handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def render_report(decision: dict, baseline: dict, candidate: dict, profile_sha: str) -> str:
    lines = [
        "# SAFE-R2 one-time final-lockbox evaluation", "",
        "This report applies one frozen profile exactly once. No parameter search, "
        "threshold sweep, or final-label retuning was performed.", "",
        f"Profile SHA256: `{profile_sha}`", "",
        "|View|System|Score|P|R|mAP50|TP|FP|FN|Predictions|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for view in VIEWS:
        for name, records in (("V7 baseline", baseline), ("SAFE-R2", candidate)):
            row = records[view]
            lines.append(
                f"|{view}|{name}|{row['score']:.6f}|{row['p_micro']:.7f}|"
                f"{row['r_micro']:.7f}|{row['map50']:.7f}|{row['tp']}|{row['fp']}|"
                f"{row['fn']}|{row['predictions']}|"
            )
    lines.extend([
        "", "## Frozen decision", "",
        f"Gate: `{decision['gate']}`", "",
        f"Weighted delta: `{decision['weighted_score_delta']:+.6f}`", "",
        "Failures: " + (", ".join(f"`{x}`" for x in decision["failures"]) or "none"), "",
        "A STOP result is terminal for this profile. Do not inspect final-label errors, "
        "change the profile, and rerun this lockbox.",
    ])
    return "\n".join(lines) + "\n"


def package_results(output: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with tarfile.open(temporary, "w:gz") as archive:
        archive.add(output, arcname=output.name)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument(
        "--profile", type=Path,
        default=ROOT / "configs/inference/safe_r2_frozen_profile.json",
    )
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    project = args.project_root.resolve()
    source = project / "data/official/train"
    split = project / "data/official_v2_20260831b/splits.json"
    weight_a = project / "runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt"
    weight_b = project / "runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt"
    output = project / "runs/semifinal/safe_r2_final_lockbox"

    verify_package()
    profile = validate_frozen_profile(args.profile)
    profile_sha = sha256_file(args.profile)
    for path, expected in ((split, SPLIT_SHA), (weight_a, MODEL_A_SHA256), (weight_b, MODEL_B_SHA256)):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"missing or wrong protected artifact: {path}")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("final-lockbox evaluation requires RTX 4090")

    manifest = json.loads(split.read_text(encoding="utf-8"))
    validate_split_manifest(manifest)
    selected = manifest["splits"]["final"]
    if len(selected) != EXPECTED_SOURCES:
        raise ValueError(f"expected {EXPECTED_SOURCES} final-holdout sources")
    records = [parse_voc(source / name, (source / name).with_suffix(".xml")) for name in selected]
    subset = {**manifest, "splits": {"final": selected}, "groups": {"final": manifest["groups"]["final"]}}
    records = frozen_records(records, source, subset)["final"]
    records.sort(key=lambda record: str(record.image_path))
    counts = Counter(annotation.class_name for record in records for annotation in record.annotations)
    allowed_gt = sum(value for name, value in counts.items() if name != "qilie")
    if allowed_gt != EXPECTED_ALLOWED_GT:
        raise ValueError(f"unexpected final-holdout ground truth: {counts}")

    options_by_model = {
        "model_a": InferenceOptions(
            tile_layout="semifinal_grid", batch=1, conf=0.0001, iou=0.68,
            local_iou=0.70, half=True, tta=True, global_pass=True,
            max_det=2000, imgsz=1024, edge_penalty=0.50,
        ),
        "model_b": InferenceOptions(
            tile_layout="semifinal_grid", batch=1, conf=0.0001, iou=0.68,
            local_iou=0.70, half=True, tta=True, global_pass=True,
            max_det=2000, imgsz=1280, edge_penalty=0.50,
        ),
    }
    freeze_payload = {
        "schema": 1,
        "purpose": "one-time-safe-r2-final-lockbox",
        "profile_sha256": profile_sha,
        "split_sha256": SPLIT_SHA,
        "models": {"model_a": MODEL_A_SHA256, "model_b": MODEL_B_SHA256},
        "v7_fusion": V7_FUSION,
        "gate": LOCKBOX_GATE,
        "sources": EXPECTED_SOURCES,
        "allowed_ground_truth": EXPECTED_ALLOWED_GT,
        "tuning": False,
        "submission": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    freeze_path = output / "LOCKBOX_FREEZE.json"
    if freeze_path.exists():
        previous = json.loads(freeze_path.read_text(encoding="utf-8"))
        if previous != freeze_payload:
            raise ValueError("existing lockbox was frozen with a different profile or policy")
    else:
        write_json(freeze_path, freeze_payload)

    preflight = {
        **freeze_payload,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "profile_metadata": profile["metadata"],
    }
    write_json(output / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return
    if (output / "LOCKBOX_DECISION.json").exists():
        raise FileExistsError("this one-time lockbox already completed; reruns are forbidden")

    code = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in [
            Path(__file__).resolve(),
            ROOT / "scripts/analyze_final_submission_postprocess.py",
            ROOT / "scripts/analyze_source_aware_fusion.py",
            ROOT / "scripts/run_semifinal_stage1.py",
            *sorted((ROOT / "steel_defect").glob("*.py")),
        ]
    }
    reports = {}
    for model_name, weight, digest in (
        ("model_a", weight_a, MODEL_A_SHA256),
        ("model_b", weight_b, MODEL_B_SHA256),
    ):
        model = YOLO(str(weight))
        names = [model.names[index] for index in sorted(model.names)]
        if names != manifest["classes"]:
            raise ValueError(f"checkpoint class order mismatch: {weight}")
        options = options_by_model[model_name]
        policy = asdict(options)
        final_freeze = {
            "weights_sha256": digest,
            "split_sha256": SPLIT_SHA,
            "evaluation_policy": policy,
        }
        check_evaluation_scope(
            manifest, "final", selected, tune=False, limit=0,
            freeze=final_freeze, weights_sha=digest, manifest_sha=SPLIT_SHA,
            policy=policy,
        )
        for view in VIEWS:
            tag = f"{model_name}_tta_s{options.imgsz}_{view}"
            signature = {
                "schema": 1,
                "purpose": "safe_r2_one_time_final_lockbox",
                "weights_sha256": digest,
                "split_sha256": SPLIT_SHA,
                "profile_sha256": profile_sha,
                "code": code,
                "options": policy,
                "view": view,
                "limit": 0,
                "threshold_scan": [0.0001],
                "numpy": np.__version__,
                "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "source_ids": [record.image_path.relative_to(source).as_posix() for record in records],
            }
            report = evaluate_cell(
                model, records, source, manifest["classes"], options, view,
                output / "cells" / tag, signature, export_thresholds=(0.0001,),
            )
            reports[tag] = {
                "metrics": str((output / "cells" / tag / "metrics.json").resolve()),
                "seconds": report["seconds"],
            }
            write_json(output / "state.json", {"complete": False, "reports": reports})
        del model
        torch.cuda.empty_cache()

    baseline_metrics: dict[str, dict] = {}
    candidate_metrics: dict[str, dict] = {}
    diagnostics = {}
    for view in VIEWS:
        cell_a = load_cell(output / "cells" / f"model_a_tta_s1024_{view}")
        cell_b = load_cell(output / "cells" / f"model_b_tta_s1280_{view}")
        baseline = fuse_v7(cell_a, cell_b)
        candidate, diagnostic = apply_profile_to_view(cell_a, cell_b, baseline, profile)
        baseline_metrics[view] = metrics50(baseline, cell_a["ground_truth"], cell_a["sizes"])
        candidate_metrics[view] = metrics50(candidate, cell_a["ground_truth"], cell_a["sizes"])
        diagnostics[view] = diagnostic
        view_output = output / "evaluation" / view
        view_output.mkdir(parents=True, exist_ok=True)
        write_json(view_output / "baseline_metrics.json", baseline_metrics[view])
        write_json(view_output / "safe_r2_metrics.json", candidate_metrics[view])
        write_predictions(view_output / "baseline_predictions.jsonl.gz", baseline)
        write_predictions(view_output / "safe_r2_predictions.jsonl.gz", candidate)

    decision = evaluate_gate(baseline_metrics, candidate_metrics)
    decision.update({
        "profile_sha256": profile_sha,
        "profile_metadata": profile["metadata"],
        "test_labels_used_for_training_or_selection": False,
        "parameter_search_performed": False,
        "rerun_or_retuning_allowed": False,
    })
    write_json(output / "LOCKBOX_DECISION.json", decision)
    write_json(output / "metrics_full.json", {
        "baseline": baseline_metrics,
        "safe_r2": candidate_metrics,
        "diagnostics": diagnostics,
    })
    (output / "REPORT.md").write_text(
        render_report(decision, baseline_metrics, candidate_metrics, profile_sha),
        encoding="utf-8",
    )
    summary = {
        "complete": True,
        "decision": decision,
        "reports": reports,
        "next": (
            "PASS: an independently packaged Test builder may be prepared without reading "
            "final errors. STOP: retire this profile; do not retune on final labels."
        ),
    }
    write_json(output / "SUMMARY.json", summary)
    write_json(output / "state.json", {"complete": True, "reports": reports})
    archive = project.parent / "safe_r2_final_lockbox_results_20260928.tar.gz"
    package_results(output, archive)
    print(json.dumps({
        **summary,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
