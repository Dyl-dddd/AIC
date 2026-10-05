"""Evaluate three single-model checkpoint interpolations on frozen dev."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_semifinal_soups import build_soup
from scripts.run_semifinal_stage1 import (
    BASE_SHA, NEW_SHA, SPLIT_SHA, evaluate_cell, verified_file, write_json,
)
from steel_defect.governance import check_evaluation_scope, frozen_records
from steel_defect.inference import InferenceOptions
from steel_defect.runtime import sha256_file
from steel_defect.voc import parse_voc


ALPHAS = (0.25, 0.50, 0.75)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/mnt/proj/iron"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--generated-weights", type=Path, default=ROOT / "generated_weights")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.limit and args.output == ROOT / "results":
        parser.error("smoke tests require a separate --output")

    project = args.project_root.resolve()
    baseline = args.baseline or project / "best_score_proxy_epoch25.pt"
    candidate = args.candidate or project / "runs/semifinal/grid2x3_ft15_lr3e5_b2/weights/epoch5.pt"
    source = project / "data/official/train"
    split = project / "data/official_v2_20260831b/splits.json"
    verified_file(baseline, BASE_SHA)
    verified_file(candidate, NEW_SHA)
    verified_file(split, SPLIT_SHA)
    package_manifest = ROOT / "STAGE3_PACKAGE_MANIFEST.json"
    if package_manifest.is_file():
        for relative, digest in json.loads(package_manifest.read_text(encoding="utf-8")).items():
            verified_file(ROOT / relative, digest)
    if not torch.cuda.is_available() or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"stage3 requires the requested RTX 4090; CUDA={torch.cuda.is_available()}")
    torch.manual_seed(42)
    witness = torch.randn(64, 64, device="cuda", requires_grad=True)
    loss = (witness @ witness.T).square().mean()
    loss.backward()
    if not torch.isfinite(loss) or not torch.isfinite(witness.grad).all():
        raise RuntimeError("CUDA kernel witness failed")
    environment = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "ultralytics": ultralytics.__version__, "numpy": np.__version__, "python": sys.version,
        "kernel_loss": loss.item(), "kernel_gradient_norm": witness.grad.norm().item()}
    del witness, loss
    torch.cuda.empty_cache()

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
        raise ValueError(f"unexpected permitted ground truth: {counts}")
    if args.limit:
        records = records[:args.limit]

    soups = {}
    for alpha in ALPHAS:
        tag = f"a{int(round(alpha * 100)):03d}"
        soups[tag] = build_soup(baseline, candidate, alpha, args.generated_weights / f"old_new_{tag}.pt")
        model = YOLO(soups[tag]["output"])
        if [model.names[index] for index in sorted(model.names)] != manifest["classes"]:
            raise ValueError(f"class order mismatch: {tag}")
        del model
    torch.cuda.empty_cache()
    write_json(args.output / "preflight.json", {"environment": environment, "gt_by_class": counts,
        "source_bytes_verified": True, "split_sha256": SPLIT_SHA, "baseline_sha256": BASE_SHA,
        "candidate_sha256": NEW_SHA, "soups": soups, "training_started": False, "stage": 3,
        "scope": "smoke" if args.limit else "complete_dev"})
    print("Preflight passed; checkpoint interpolation is single-model and no training was launched.", flush=True)
    if args.preflight_only:
        return

    code = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in
        [Path(__file__).resolve(), ROOT / "scripts/build_semifinal_soups.py",
         ROOT / "scripts/run_semifinal_stage1.py", *sorted((ROOT / "steel_defect").glob("*.py"))]}
    options = InferenceOptions(tile_layout="semifinal_grid", batch=1, conf=0.0001, half=True,
        global_pass=True, max_det=2000, imgsz=1024, edge_penalty=0.85)
    thresholds = (0.0001, 0.0003, 0.001, 0.003, 0.01)
    reports = {}
    for tag, metadata in soups.items():
        model = YOLO(metadata["output"])
        for view in ("original", "grid_crops"):
            cell = f"soup_{tag}_{view}_s1024_low"
            signature = {"schema": 1, "stage": 3, "alpha_new": metadata["alpha_new"],
                "weights_sha256": metadata["output_sha256"], "parent_old_sha256": BASE_SHA,
                "parent_new_sha256": NEW_SHA, "split_sha256": SPLIT_SHA, "code": code,
                "options": asdict(options), "view": view, "limit": args.limit,
                "threshold_scan": list(thresholds), "numpy": np.__version__,
                "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                "source_ids": [record.image_path.relative_to(source).as_posix() for record in records]}
            check_evaluation_scope(manifest, "dev", selected, tune=False, limit=args.limit,
                freeze=None, weights_sha=metadata["output_sha256"], manifest_sha=SPLIT_SHA,
                policy=asdict(options))
            reports[cell] = evaluate_cell(model, records, source, manifest["classes"], options,
                view, args.output / cell, signature, export_thresholds=thresholds)
            write_json(args.output / "comparison.json", reports)
        del model
        torch.cuda.empty_cache()

    lines = ["# 第三阶段单模型权重插值结果", "",
        "每个 checkpoint 仍是单模型；alpha 表示新 epoch5 的权重占比。以下为本地 dev 参考分。", "",
        "|单元|P|R|AP50|参考分|秒|", "|---|---:|---:|---:|---:|---:|"]
    for cell, report in reports.items():
        summary = report["all_exported"]["summary"]
        lines.append(f"|{cell}|{summary['precision_micro']:.5f}|{summary['recall_micro']:.5f}|"
                     f"{summary['map50_macro']:.5f}|{summary['score_proxy_20_60_20']:.3f}|{report['seconds']:.1f}|")
    lines.extend(["", "该结果仍需与第二阶段两个端点比较；不得把 dev 最优值当官方 Test 成绩。"]) 
    (args.output / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Completed. Report: {args.output / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()

