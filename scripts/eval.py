from __future__ import annotations

import argparse
import gzip
import json
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from steel_defect.classes import CLASS_NAMES
from steel_defect.inference import InferenceOptions, merge_candidates, predict_candidates
from steel_defect.metrics import evaluate_predictions, ground_truth_from_records, tune_class_thresholds
from steel_defect.runtime import apply_profile_defaults, sha256_file
from steel_defect.voc import discover_pairs, parse_voc
from steel_defect.governance import check_evaluation_scope, frozen_records


def read_json(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the complete original-image inference pipeline on real VOC GT.")
    parser.add_argument("--config", type=Path, help="YAML original-image evaluation profile")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--source", type=Path, help="Original image/XML dataset root")
    parser.add_argument("--split-manifest", type=Path, help="metadata/splits.json from prepare_data.py")
    parser.add_argument("--split", choices=("train", "calibration", "val", "dev", "final"), default="val")
    parser.add_argument("--freeze-manifest", type=Path, help="Required signed policy/model/split for final holdout")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-layout", choices=("sliding", "semifinal_grid"), default="sliding")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.005)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--local-iou", type=float, default=0.70)
    parser.add_argument("--operating-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--deblur", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--deblur-threshold", type=float, default=85.0)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--global-pass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--edge-margin", type=int, default=12)
    parser.add_argument("--edge-penalty", type=float, default=1.0)
    parser.add_argument("--merge", choices=("nms", "wbf"), default="nms")
    parser.add_argument("--max-det", type=int, default=2000)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--thresholds", type=Path, help="Fixed per-class thresholds from calibration")
    parser.add_argument("--tune-thresholds", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--candidate-cache", type=Path, help="Raw per-view candidate cache (.json or .json.gz)")
    parser.add_argument("--reuse-candidates", action="store_true", help="Skip network inference and replay the cache")
    parser.add_argument("--cache-only", action="store_true", help="Run model views and save raw candidates without merge/metrics")
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test only; 0 evaluates the complete split")
    args = apply_profile_defaults(parser, "evaluate")
    if args.weights is None or args.source is None or args.split_manifest is None:
        parser.error("--weights, --source and --split-manifest are required (directly or through --config)")
    for name in ("weights", "source", "split_manifest", "output"):
        setattr(args, name, Path(getattr(args, name)))
    args.candidate_cache = Path(args.candidate_cache) if args.candidate_cache else args.output / "raw_candidates.json.gz"
    if args.cache_only and args.reuse_candidates:
        parser.error("--cache-only cannot be combined with --reuse-candidates")

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    class_names = [str(name) for name in manifest["classes"]]
    class_thresholds = {}
    if args.thresholds:
        threshold_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
        class_thresholds = threshold_payload.get("thresholds", threshold_payload)
        unknown = sorted(set(class_thresholds) - set(class_names))
        if unknown:
            raise ValueError(f"Unknown classes in threshold file: {unknown}")
        class_thresholds = {str(name): float(value) for name, value in class_thresholds.items()}
    selected_paths = set(manifest["splits"][args.split])
    pairs, missing_xml, _ = discover_pairs(args.source)
    if missing_xml:
        raise ValueError(f"Missing XML for {len(missing_xml)} images")
    custom_mapping = None if class_names == CLASS_NAMES else {name: index for index, name in enumerate(class_names)}
    records = []
    for image_path, xml_path in pairs:
        relative = str(image_path.relative_to(args.source)).replace("\\", "/")
        if relative in selected_paths:
            records.append(parse_voc(image_path, xml_path, class_to_id=custom_mapping))
    records.sort(key=lambda record: str(record.image_path))
    policy_keys = ("tile_layout", "tile_size", "overlap", "imgsz", "conf", "iou", "local_iou", "operating_threshold",
                   "half", "tta", "deblur", "deblur_threshold", "clahe", "global_pass", "edge_margin",
                   "edge_penalty", "merge", "max_det", "beta")
    evaluation_policy = {key: getattr(args, key) for key in policy_keys}
    evaluation_policy["class_thresholds"] = class_thresholds
    freeze = read_json(args.freeze_manifest) if args.freeze_manifest else None
    check_evaluation_scope(manifest, args.split,
        [r.image_path.relative_to(args.source).as_posix() for r in records],
        tune=args.tune_thresholds, limit=args.limit, freeze=freeze,
        weights_sha=sha256_file(args.weights), manifest_sha=sha256_file(args.split_manifest), policy=evaluation_policy)
    if manifest.get("schema_version") == 2:
        # Verify the selected source bytes and labels against the frozen contract.
        subset = {**manifest, "splits": {args.split: manifest["splits"][args.split]},
                  "groups": {args.split: manifest["groups"][args.split]}}
        records = frozen_records(records, args.source, subset)[args.split]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No records selected for split={args.split}")

    half = args.half if args.half is not None else not str(args.device).lower().startswith("cpu")
    options = InferenceOptions(
        tile_layout=args.tile_layout,
        tile_size=args.tile_size, overlap=args.overlap, imgsz=args.imgsz, batch=args.batch,
        conf=args.conf, iou=args.iou, local_iou=args.local_iou, device=args.device, half=half,
        tta=args.tta, deblur=args.deblur, deblur_threshold=args.deblur_threshold,
        clahe=args.clahe, edge_margin=args.edge_margin, edge_penalty=args.edge_penalty,
        global_pass=args.global_pass,
        merge=args.merge, max_det=args.max_det,
    )

    generation_signature = {
        "cache_schema": 2,
        "tile_layout": args.tile_layout,
        "code_sha256": {p.name: sha256_file(p) for p in sorted((Path(__file__).resolve().parents[1] / "steel_defect").glob("*.py"))},
        "source_sha256": hashlib.sha256(json.dumps([
            [r.image_path.relative_to(args.source).as_posix(), sha256_file(r.image_path), sha256_file(r.xml_path)]
            for r in records], sort_keys=True).encode()).hexdigest(),
        "weights_sha256": sha256_file(args.weights),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split": args.split,
        "class_names": class_names,
        "tile_size": args.tile_size,
        "overlap": args.overlap,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "local_iou": args.local_iou,
        "half": half,
        "tta": args.tta,
        "deblur": args.deblur,
        "deblur_threshold": args.deblur_threshold,
        "clahe": args.clahe,
        "edge_margin": args.edge_margin,
        "global_pass": args.global_pass,
        "max_det": args.max_det,
    }
    cached_images: dict[str, dict] = {}
    model = None
    if args.reuse_candidates:
        if not args.candidate_cache.is_file():
            raise FileNotFoundError(f"Candidate cache not found: {args.candidate_cache}")
        cache_payload = read_json(args.candidate_cache)
        if cache_payload.get("signature") != generation_signature:
            raise ValueError("Candidate cache signature does not match the requested model/view pipeline")
        cached_images = cache_payload.get("images", {})
    else:
        model = YOLO(str(args.weights))
        model_names = [model.names[index] for index in sorted(model.names)] if isinstance(model.names, dict) else list(model.names)
        if model_names != class_names:
            raise ValueError(f"Checkpoint classes {model_names} do not match dataset classes {class_names}")

    predictions = []
    image_ids = {}
    cache_images: dict[str, dict] = {}
    for index, record in enumerate(records, start=1):
        image_id = str(record.image_path.relative_to(args.source)).replace("\\", "/")
        image_ids[str(record.image_path.resolve())] = image_id
        if args.reuse_candidates:
            if image_id not in cached_images:
                raise ValueError(f"Candidate cache is incomplete; missing {image_id}")
            raw = cached_images[image_id]
            candidates = raw["candidates"]
            image_size = tuple(raw["image_size"])
        else:
            candidates, image_size = predict_candidates(model, record.image_path, options)
            cache_images[image_id] = {"image_size": list(image_size), "candidates": candidates}
        if args.cache_only:
            print(f"[{index}/{len(records)}] {image_id}: raw_candidates={len(candidates)}")
            continue
        current = merge_candidates(candidates, image_size, class_names, options)
        predictions.extend({"image_id": image_id, **prediction} for prediction in current)
        print(f"[{index}/{len(records)}] {image_id}: predictions={len(current)}")

    if not args.reuse_candidates:
        write_json(args.candidate_cache, {"signature": generation_signature, "images": cache_images})
    if args.cache_only:
        candidate_count = sum(len(item["candidates"]) for item in cache_images.values())
        args.output.mkdir(parents=True, exist_ok=True)
        summary = {
            "evaluated_images": len(records),
            "raw_candidates": candidate_count,
            "candidate_cache": str(args.candidate_cache.resolve()),
            "generation_signature": generation_signature,
        }
        (args.output / "candidate_cache_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
        return

    ground_truth = ground_truth_from_records(records, image_ids)
    metrics = evaluate_predictions(
        predictions, ground_truth, class_names, operating_threshold=args.operating_threshold, beta=args.beta,
        class_thresholds=class_thresholds,
        image_sizes={image_ids[str(r.image_path.resolve())]: (r.width, r.height) for r in records},
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = {
        "evaluation_type": "real_gt",
        "ground_truth_source": str(args.source.resolve()),
        "split_manifest": str(args.split_manifest.resolve()),
        "split": args.split,
        "evaluated_images": len(records),
        "full_split_images": len(selected_paths),
        "scope_complete": not args.limit and len(records) == len(selected_paths),
        "weights": str(args.weights.resolve()),
        "candidate_cache": str(args.candidate_cache.resolve()),
        "reused_candidates": args.reuse_candidates,
        "hashes": {
            "weights": sha256_file(args.weights),
            "split_manifest": sha256_file(args.split_manifest),
        },
        "inference": vars(args),
        **metrics,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    if args.tune_thresholds:
        thresholds = tune_class_thresholds(predictions, ground_truth, class_names, beta=args.beta)
        (args.output / "thresholds.json").write_text(
            json.dumps({"beta": args.beta, "thresholds": thresholds}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(metrics["summary"], ensure_ascii=False, indent=2))
    print(f"Evaluation artifacts: {args.output.resolve()}")


if __name__ == "__main__":
    main()
