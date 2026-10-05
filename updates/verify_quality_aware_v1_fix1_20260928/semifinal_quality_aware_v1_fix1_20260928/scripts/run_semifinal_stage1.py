"""Read-only, single-checkpoint geometry evaluation. Never trains a model."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import torch
import ultralytics
from ultralytics import YOLO

from steel_defect.geometry import clip_box_to_tile, generate_grid_tiles
from steel_defect.governance import check_evaluation_scope, frozen_records
from steel_defect.image_io import imread
from steel_defect.inference import InferenceOptions, merge_candidates, predict_candidates_array
from steel_defect.metrics import evaluate_predictions
from steel_defect.runtime import sha256_file
from steel_defect.voc import parse_voc

BASE_SHA = "5852470665ee87c29ac741e6dd6b786ec51c10e6f7ebf79d05d9e70ecf7562c7"
NEW_SHA = "3f2a01fa426b65c76da0272efca57e1aafb12a5b40618a6d425e8dae2b5ab773"
SPLIT_SHA = "c46359024fa2c06b4d03cef91e274588cf555a0d053108b840010df3a6573bf9"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def verified_file(path, expected):
    if not path.is_file():
        raise FileNotFoundError(f"Missing: {path}. Supply its actual path; do not rename another checkpoint.")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}\nexpected={expected}\nactual={actual}")
    return actual


def allowed_gt(annotations):
    result = {}
    for annotation in annotations:
        if annotation.class_name != "qilie":
            result.setdefault(annotation.class_name, []).append(list(annotation.box))
    return result


def crop_gt(record, tile):
    result = {}
    for annotation in record.annotations:
        if annotation.class_name == "qilie":
            continue
        # Same clipping convention as prepare_data. This is a derived-view
        # diagnostic, not new independent examples or official test labels.
        box = clip_box_to_tile(annotation.box, tile, min_visibility=0.35, min_size=2.0)
        if box is not None:
            result.setdefault(annotation.class_name, []).append(list(box))
    return result


def add_micro(metrics):
    classes = metrics["per_class"].values()
    gt = sum(item["ground_truth"] for item in classes)
    tp = sum(item["true_positives"] for item in classes)
    fp = sum(item["false_positives"] for item in classes)
    metrics["summary"].update(ground_truth=gt, true_positives=tp,
        false_positives=fp, false_negatives=gt-tp,
        recall_micro=tp/max(gt, 1), precision_micro=tp/max(tp+fp, 1))
    summary = metrics["summary"]
    summary["score_proxy_20_60_20"] = score_proxy(
        summary["precision_micro"], summary["recall_micro"], summary["map50_macro"])
    return metrics


def score_proxy(precision, recall, map50):
    """Known contest weights; local AP integration/aggregation remains a proxy."""
    return 20.0 * precision + 60.0 * recall + 20.0 * map50


def evaluate_cell(model, records, source, classes, options, view, destination, signature, export_thresholds=None):
    report_path = destination / "metrics.json"
    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        if previous.get("signature") != signature:
            raise ValueError(f"Existing results have a different signature: {destination}; use a new output directory")
        if previous.get("complete"):
            print(f"Reuse verified completed cell: {destination.name}", flush=True)
            return previous
    destination.mkdir(parents=True, exist_ok=True)
    predictions, gt, sizes = [], {}, {}
    started = time.monotonic()
    with gzip.open(destination / "candidates.jsonl.gz", "wt", encoding="utf-8") as cache:
        cache.write(json.dumps({"type": "header", "signature": signature}) + "\n")
        for index, record in enumerate(records, 1):
            image = imread(record.image_path, cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (record.height, record.width):
                raise ValueError(f"Image/XML size mismatch: {record.image_path}")
            source_id = record.image_path.relative_to(source).as_posix()
            if view == "original":
                views = [(source_id, image, allowed_gt(record.annotations))]
            else:
                tiles = generate_grid_tiles(record.width, record.height, 1387, 1516, 2, 3)
                views = [(f"{source_id}::x{t.x}_y{t.y}",
                          image[t.y:t.y+t.height, t.x:t.x+t.width], crop_gt(record, t)) for t in tiles]
            for image_id, pixels, truth in views:
                candidates, size = predict_candidates_array(model, pixels, options)
                result = merge_candidates(candidates, size, classes, options)
                predictions.extend({"image_id": image_id, **p} for p in result if p["category_name"] != "qilie")
                gt[image_id], sizes[image_id] = truth, size
                cache.write(json.dumps({"image_id": image_id, "source_id": source_id,
                    "image_size": size, "ground_truth": truth, "candidates": candidates}, separators=(",", ":")) + "\n")
            if index % 10 == 0 or index == len(records):
                print(f"{destination.name}: {index}/{len(records)} sources; {len(predictions)} predictions", flush=True)
                cache.flush()
    permitted = [name for name in classes if name != "qilie"]
    fixed = add_micro(evaluate_predictions(predictions, gt, permitted, operating_threshold=0.05, image_sizes=sizes))
    # Diagnostic P/R uses .05 while this AP uses all predictions: no score here.
    fixed["summary"].pop("score_proxy_20_60_20")
    exported = add_micro(evaluate_predictions(predictions, gt, permitted, operating_threshold=0.0, image_sizes=sizes))
    # Recompute AP on the actual filtered export. Do not combine thresholded
    # precision/recall with AP from an unfiltered prediction list.
    threshold_scan = []
    if export_thresholds is None:
        export_thresholds = (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2)
    if min(export_thresholds) < options.conf:
        raise ValueError("Cannot recover candidates below the generation threshold")
    for threshold in export_thresholds:
        filtered = [p for p in predictions if p["score"] >= threshold]
        result = add_micro(evaluate_predictions(filtered, gt, permitted,
            operating_threshold=0.0, image_sizes=sizes))
        threshold_scan.append({"export_threshold": threshold, "export_count": len(filtered), **result["summary"]})
    report = {"complete": True, "signature": signature, "sources": len(records), "views": len(gt),
        "predictions": len(predictions), "seconds": time.monotonic()-started,
        "fixed_005": fixed, "all_exported": exported, "threshold_scan": threshold_scan,
        "score_definition": "20*P_micro + 60*R_micro + 20*mAP50_macro, all on the same exported predictions; local AP protocol may differ from official scorer",
        "note": "grid_crops uses all derived crops, no sampled negatives; not directly comparable to earlier YOLO.val crop report"}
    write_json(destination / "predictions.json", predictions)
    write_json(report_path, report)
    print(json.dumps({"cell": destination.name, **fixed["summary"]}, ensure_ascii=False), flush=True)
    return report


def experiment_jobs(stage):
    if stage == 1:
        return [(label, layout, 1024, 0.001) for label in ("old", "new")
                for layout in ("sliding", "semifinal_grid")]
    if stage == 2:
        return [("old", "sliding", 1024, 0.0001),
                ("new", "semifinal_grid", 1024, 0.0001),
                ("new", "semifinal_grid", 1280, 0.0001)]
    raise ValueError("Unknown experiment stage")


def main(stage=1):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--limit", type=int, default=0, help="Smoke only. Zero runs all 459 dev sources.")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.limit and args.output == ROOT / "results":
        parser.error("Smoke tests require a separate --output directory")
    project = args.project_root.resolve()
    baseline = args.baseline or project / "best_score_proxy_epoch25.pt"
    candidate = args.candidate or project / "runs/semifinal/grid2x3_ft15_lr3e5_b2/weights/epoch5.pt"
    source = project / "data/official/train"
    split = project / "data/official_v2_20260831b/splits.json"
    verified_file(baseline, BASE_SHA)
    verified_file(candidate, NEW_SHA)
    verified_file(split, SPLIT_SHA)
    package_manifest = ROOT / f"STAGE{stage}_PACKAGE_MANIFEST.json"
    if package_manifest.exists():
        for relative, digest in json.loads(package_manifest.read_text(encoding="utf-8")).items():
            verified_file(ROOT / relative, digest)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU fallback")
    gpu = torch.cuda.get_device_name(0)
    if "4090" not in gpu:
        raise RuntimeError(f"This cloud job requires the requested 4090, got {gpu}")
    torch.manual_seed(42)
    witness = torch.randn(64, 64, device="cuda", requires_grad=True)
    loss = (witness @ witness.T).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(witness.grad).all():
        raise RuntimeError("CUDA kernel witness failed")
    environment = {"gpu": gpu, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "numpy": np.__version__, "python": sys.version, "kernel_loss": loss.item(), "kernel_gradient_norm": witness.grad.norm().item()}
    del witness, loss
    torch.cuda.empty_cache()
    manifest = json.loads(split.read_text(encoding="utf-8"))
    selected = manifest["splits"]["dev"]
    if len(selected) != 459:
        raise ValueError("Expected 459 frozen dev sources")
    records = [parse_voc(source / name, (source / name).with_suffix(".xml")) for name in selected]
    subset = {**manifest, "splits": {"dev": selected}, "groups": {"dev": manifest["groups"]["dev"]}}
    records = frozen_records(records, source, subset)["dev"]
    counts = Counter(a.class_name for r in records for a in r.annotations)
    if sum(v for k, v in counts.items() if k != "qilie") != 612:
        raise ValueError(f"Expected 612 allowed-class GT: {counts}")
    split_sets = {name: set(items) for name, items in manifest["splits"].items()}
    group_sets = {name: set(items) for name, items in manifest["groups"].items()}
    for name in split_sets:
        for other in split_sets:
            if name < other and (split_sets[name] & split_sets[other] or group_sets[name] & group_sets[other]):
                raise ValueError(f"Split leakage: {name}/{other}")
    records.sort(key=lambda r: str(r.image_path))
    if args.limit:
        records = records[:args.limit]
    code = {p.relative_to(ROOT).as_posix(): sha256_file(p)
        for p in [Path(__file__).resolve(), *sorted((ROOT / "steel_defect").glob("*.py"))]}
    write_json(args.output / "preflight.json", {"environment": environment, "gt_by_class": counts,
        "source_bytes_verified": True, "split_sha256": SPLIT_SHA, "baseline_sha256": BASE_SHA,
        "candidate_sha256": NEW_SHA, "training_started": False, "stage": stage,
        "scope": "smoke" if args.limit else "complete_dev"})
    print("Preflight passed; no training will be launched.", flush=True)
    if args.preflight_only:
        return
    reports = {}
    for label, layout, resolution, conf in experiment_jobs(stage):
        weights, weight_sha = (baseline, BASE_SHA) if label == "old" else (candidate, NEW_SHA)
        model = YOLO(str(weights))
        if [model.names[i] for i in sorted(model.names)] != manifest["classes"]:
            raise ValueError("Checkpoint class order mismatch")
        options = InferenceOptions(tile_layout=layout, batch=1, conf=conf, half=True,
            global_pass=True, max_det=2000, imgsz=resolution,
            edge_penalty=0.85 if stage == 2 else 1.0)
        thresholds = ((0.0001, 0.0003, 0.001, 0.003, 0.01) if stage == 2 else
                      (0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2))
        for view in ("original", "grid_crops"):
            cell = f"{label}_{layout}_{view}" + (f"_s{resolution}_low" if stage == 2 else "")
            signature = {"schema": 2, "stage": stage, "weights_sha256": weight_sha, "split_sha256": SPLIT_SHA,
                "code": code, "options": asdict(options), "view": view, "limit": args.limit,
                "threshold_scan": list(thresholds), "numpy": np.__version__,
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "source_ids": [r.image_path.relative_to(source).as_posix() for r in records]}
            check_evaluation_scope(manifest, "dev", selected, tune=False, limit=args.limit,
                freeze=None, weights_sha=weight_sha, manifest_sha=SPLIT_SHA, policy=asdict(options))
            reports[cell] = evaluate_cell(model, records, source, manifest["classes"], options,
                view, args.output / cell, signature, export_thresholds=thresholds)
            write_json(args.output / "comparison.json", reports)
        del model
        torch.cuda.empty_cache()
    lines = [f"# 第{stage}阶段真实评估结果", "", "仅完整 dev 可用于决策；裁块与原图不可当独立样本相加。", "",
        "|实验|源图/视图|AP50|宏召回@.05|FP/图@.05|全部导出微召回|全部导出参考分|秒|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell, report in reports.items():
        fixed = report["fixed_005"]["summary"]
        lines.append(f"|{cell}|{report['sources']}/{report['views']}|{fixed['map50_macro']:.5f}|{fixed['recall_macro']:.5f}|{fixed['false_positives_per_image']:.3f}|{report['all_exported']['summary']['recall_micro']:.5f}|{report['all_exported']['summary']['score_proxy_20_60_20']:.3f}|{report['seconds']:.1f}|")
    lines.extend(["", "## 输出阈值对照（本地参考分，不是官方实测分）", "",
        "|实验|阈值|P微平均|R微平均|mAP50|参考分|导出框数|", "|---|---:|---:|---:|---:|---:|---:|"])
    for cell, report in reports.items():
        for item in report["threshold_scan"]:
            lines.append(f"|{cell}|{item['export_threshold']}|{item['precision_micro']:.5f}|{item['recall_micro']:.5f}|{item['map50_macro']:.5f}|{item['score_proxy_20_60_20']:.3f}|{item['export_count']}|")
    lines.extend(["", "本报告不估算官方70分；不能把裁块诊断的GT数解释为新增独立样本。",
        "下一步：回传本报告、comparison.json、preflight.json，核对双视图门槛后才选择云端短训练。"])
    (args.output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Completed. Report: {args.output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
