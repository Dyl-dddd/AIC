"""Select retraining checkpoints with the actual score proxy on both dev views."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_semifinal_stage1 import SPLIT_SHA, evaluate_cell, verified_file, write_json
from steel_defect.governance import check_evaluation_scope, frozen_records
from steel_defect.inference import InferenceOptions
from steel_defect.runtime import sha256_file
from steel_defect.voc import parse_voc


ORIGINAL_WEIGHT = 488 / 788
CROP_WEIGHT = 300 / 788
OLD_ENDPOINTS = {"original": 64.178496, "grid_crops": 66.261375}


def epoch_key(path: Path) -> tuple[int, str]:
    match = re.fullmatch(r"epoch(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def discover_checkpoints(run_dir: Path, external: list[Path]) -> list[Path]:
    weights = run_dir / "weights"
    candidates = sorted(weights.glob("epoch*.pt"), key=epoch_key)
    for name in ("best.pt", "last.pt"):
        path = weights / name
        if path.is_file():
            candidates.append(path)
    candidates.extend(path for path in external if path.is_file())
    unique = []
    seen = set()
    for path in candidates:
        digest = sha256_file(path)
        if digest not in seen:
            seen.add(digest)
            unique.append(path.resolve())
    if not unique:
        raise FileNotFoundError(f"no checkpoints found in {weights}")
    return unique


def best_export(report: dict) -> dict:
    return max(
        report["threshold_scan"],
        key=lambda item: (
            item["score_proxy_20_60_20"], item["precision_micro"], -item["export_count"]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--external-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = project / "data/official/train"
    split = project / "data/official_v2_20260831b/splits.json"
    verified_file(split, SPLIT_SHA)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("checkpoint selection requires the requested RTX 4090")

    manifest = json.loads(split.read_text(encoding="utf-8"))
    selected = manifest["splits"]["dev"]
    if len(selected) != 459:
        raise ValueError("expected 459 frozen dev sources")
    records = [parse_voc(source / name, (source / name).with_suffix(".xml")) for name in selected]
    subset = {**manifest, "splits": {"dev": selected}, "groups": {"dev": manifest["groups"]["dev"]}}
    records = frozen_records(records, source, subset)["dev"]
    records.sort(key=lambda record: str(record.image_path))
    counts = Counter(annotation.class_name for record in records for annotation in record.annotations)
    if sum(value for name, value in counts.items() if name != "qilie") != 612:
        raise ValueError(f"unexpected ground truth: {counts}")

    if args.imgsz <= 0 or args.batch <= 0:
        raise ValueError("imgsz and batch must be positive")
    options = InferenceOptions(tile_layout="semifinal_grid", batch=args.batch, conf=0.0001,
        half=True, global_pass=True, max_det=2000, imgsz=args.imgsz, edge_penalty=0.85)
    thresholds = (0.0001, 0.0003, 0.001, 0.003, 0.01)
    code = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in
        [Path(__file__).resolve(), ROOT / "scripts/run_semifinal_stage1.py",
         *sorted((ROOT / "steel_defect").glob("*.py"))]}
    rows = []
    for checkpoint in discover_checkpoints(args.run_dir, args.external_checkpoint):
        digest = sha256_file(checkpoint)
        model = YOLO(str(checkpoint))
        if [model.names[index] for index in sorted(model.names)] != manifest["classes"]:
            raise ValueError(f"checkpoint class order mismatch: {checkpoint}")
        tag = f"{checkpoint.parent.parent.name}_{checkpoint.stem}_{digest[:10]}"
        view_reports = {}
        view_best = {}
        for view in ("original", "grid_crops"):
            cell = f"{tag}_{view}"
            signature = {"schema": 1, "purpose": "full_retrain_selection",
                "weights_sha256": digest, "split_sha256": SPLIT_SHA, "code": code,
                "options": asdict(options), "view": view, "limit": 0,
                "threshold_scan": list(thresholds), "numpy": np.__version__,
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "source_ids": [record.image_path.relative_to(source).as_posix() for record in records]}
            check_evaluation_scope(manifest, "dev", selected, tune=False, limit=0,
                freeze=None, weights_sha=digest, manifest_sha=SPLIT_SHA, policy=asdict(options))
            report = evaluate_cell(model, records, source, manifest["classes"], options,
                view, args.output / "cells" / cell, signature, export_thresholds=thresholds)
            view_reports[view] = str((args.output / "cells" / cell / "metrics.json").resolve())
            view_best[view] = best_export(report)
        weighted = (ORIGINAL_WEIGHT * view_best["original"]["score_proxy_20_60_20"]
                    + CROP_WEIGHT * view_best["grid_crops"]["score_proxy_20_60_20"])
        robust_min = min(view_best["original"]["score_proxy_20_60_20"],
                         view_best["grid_crops"]["score_proxy_20_60_20"])
        rows.append({"checkpoint": str(checkpoint), "sha256": digest, "views": view_best,
            "weighted_score": weighted, "robust_min_score": robust_min,
            "beats_old_both_views": all(view_best[name]["score_proxy_20_60_20"] >= OLD_ENDPOINTS[name]
                                         for name in OLD_ENDPOINTS),
            "reports": view_reports})
        del model
        torch.cuda.empty_cache()
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)

    rows.sort(key=lambda row: (row["weighted_score"], row["robust_min_score"]), reverse=True)
    result = {"selection_rule": "0.619289*original + 0.380711*grid_crops; tie by worst view",
        "score_definition": "20*P_micro + 60*R_micro + 20*mAP50_macro",
        "test_view_weights": {"original": ORIGINAL_WEIGHT, "grid_crops": CROP_WEIGHT},
        "selected": rows[0], "candidates": rows,
        "warning": "Frozen-dev checkpoint selection; not an official Test score."}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "selection.json", result)
    lines = ["# 完整重训练checkpoint筛选", "", f"选中：`{rows[0]['checkpoint']}`", "",
        "|checkpoint|原图分|裁块分|加权分|双视图超过旧端点|", "|---|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"|{Path(row['checkpoint']).name}|{row['views']['original']['score_proxy_20_60_20']:.3f}|"
                     f"{row['views']['grid_crops']['score_proxy_20_60_20']:.3f}|{row['weighted_score']:.3f}|"
                     f"{row['beats_old_both_views']}|")
    (args.output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Selected: {rows[0]['checkpoint']}", flush=True)


if __name__ == "__main__":
    main()

