"""Run full frozen-dev native TTA evaluation for protected Models A and B."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
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

from scripts.run_ensemble_v6_consensus_pipeline import sha256_file, verify_package
from scripts.run_semifinal_stage1 import SPLIT_SHA, evaluate_cell, write_json
from steel_defect.governance import check_evaluation_scope, frozen_records
from steel_defect.inference import InferenceOptions
from steel_defect.voc import parse_voc


MODEL_A_SHA256 = "d69b236a91599f850b495d7b2e60d87c0c4905713398baca21d18e564cc42a4d"
MODEL_B_SHA256 = "02f1f09b80addd462e8781d08c27b744b0175b4ab04dcc2df5191892c3a23090"


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
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    source = project / "data/official/train"
    split = project / "data/official_v2_20260831b/splits.json"
    weight_a = project / "runs/semifinal/full_retrain_v3_fix1_pipeline/final/best_score_model.pt"
    weight_b = project / "runs/semifinal/ensemble_v4_yolo11m_pipeline/final/model_b_yolo11m_best.pt"
    output = project / "runs/semifinal/ensemble_v6_tta_dev"

    verify_package()
    for path, expected in ((split, SPLIT_SHA), (weight_a, MODEL_A_SHA256), (weight_b, MODEL_B_SHA256)):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"missing or wrong protected artifact: {path}")
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("full TTA development evaluation requires RTX 4090")

    manifest = json.loads(split.read_text(encoding="utf-8"))
    selected = manifest["splits"]["dev"]
    if len(selected) != 459:
        raise ValueError("expected 459 frozen development sources")
    records = [parse_voc(source / name, (source / name).with_suffix(".xml")) for name in selected]
    subset = {**manifest, "splits": {"dev": selected}, "groups": {"dev": manifest["groups"]["dev"]}}
    records = frozen_records(records, source, subset)["dev"]
    records.sort(key=lambda record: str(record.image_path))
    counts = Counter(annotation.class_name for record in records for annotation in record.annotations)
    if sum(value for name, value in counts.items() if name != "qilie") != 612:
        raise ValueError(f"unexpected frozen development ground truth: {counts}")

    preflight = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "split_sha256": SPLIT_SHA, "dev_sources": len(records),
        "allowed_ground_truth": 612,
        "models": {
            "model_a": {"path": str(weight_a), "sha256": MODEL_A_SHA256, "imgsz": 1024},
            "model_b": {"path": str(weight_b), "sha256": MODEL_B_SHA256, "imgsz": 1280},
        },
        "tta": True, "views": ["original", "grid_crops"],
        "training": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "preflight.json", preflight)
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    if args.preflight_only:
        return

    code = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in [Path(__file__).resolve(), ROOT / "scripts/run_semifinal_stage1.py",
                     *sorted((ROOT / "steel_defect").glob("*.py"))]
    }
    reports = {}
    for model_name, weight, digest, imgsz in (
        ("model_a", weight_a, MODEL_A_SHA256, 1024),
        ("model_b", weight_b, MODEL_B_SHA256, 1280),
    ):
        model = YOLO(str(weight))
        names = [model.names[index] for index in sorted(model.names)]
        if names != manifest["classes"]:
            raise ValueError(f"checkpoint class order mismatch: {weight}")
        options = InferenceOptions(
            tile_layout="semifinal_grid", batch=1, conf=0.0001,
            iou=0.68, local_iou=0.70, half=True, tta=True,
            global_pass=True, max_det=2000, imgsz=imgsz, edge_penalty=0.50,
        )
        for view in ("original", "grid_crops"):
            tag = f"{model_name}_tta_s{imgsz}_{view}"
            signature = {
                "schema": 1, "purpose": "ensemble_v6_native_tta_dev",
                "weights_sha256": digest, "split_sha256": SPLIT_SHA,
                "code": code, "options": asdict(options), "view": view,
                "limit": 0, "threshold_scan": [0.0001],
                "numpy": np.__version__, "torch": torch.__version__,
                "ultralytics": ultralytics.__version__,
                "source_ids": [record.image_path.relative_to(source).as_posix() for record in records],
            }
            check_evaluation_scope(
                manifest, "dev", selected, tune=False, limit=0, freeze=None,
                weights_sha=digest, manifest_sha=SPLIT_SHA, policy=asdict(options),
            )
            report = evaluate_cell(
                model, records, source, manifest["classes"], options,
                view, output / "cells" / tag, signature,
                export_thresholds=(0.0001,),
            )
            reports[tag] = {
                "metrics": str((output / "cells" / tag / "metrics.json").resolve()),
                "score": report["all_exported"]["summary"]["score_proxy_20_60_20"],
                "seconds": report["seconds"],
            }
            write_json(output / "state.json", {"complete": False, "reports": reports})
        del model
        torch.cuda.empty_cache()

    state = {"complete": True, "preflight": preflight, "reports": reports}
    write_json(output / "state.json", state)
    archive = project.parent / "ensemble_v6_tta_dev_results_20260927.tar.gz"
    package_results(output, archive)
    summary = {
        "complete": True,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "reports": reports,
        "next": "Return this archive for bounded A/B/TTA fusion calibration before Test inference.",
    }
    write_json(output / "SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

