from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO

from steel_defect.classes import CLASS_NAMES
from steel_defect.inference import InferenceOptions, predict_candidates, predict_image
from steel_defect.runtime import apply_profile_defaults, sha256_file
from steel_defect.voc import IMAGE_EXTENSIONS


def load_thresholds(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds", payload)
    unknown = set(thresholds) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown classes in threshold file: {sorted(unknown)}")
    return {str(name): float(value) for name, value in thresholds.items()}


def validate_submission(items: list[dict]) -> None:
    for index, item in enumerate(items):
        if set(item) != {"image_id", "category_name", "bbox", "score"}:
            raise ValueError(f"Invalid fields at prediction {index}")
        if item["category_name"] not in CLASS_NAMES:
            raise ValueError(f"Invalid category at prediction {index}")
        x1, y1, x2, y2 = item["bbox"]
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"Invalid bbox at prediction {index}: {item['bbox']}")
        if not 0.0 <= item["score"] <= 1.0:
            raise ValueError(f"Invalid score at prediction {index}: {item['score']}")


def select_split_images(
    image_paths: list[Path], source: Path, split_manifest: Path | None, split: str,
) -> list[Path]:
    if split_manifest is None:
        return image_paths
    manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
    if split not in manifest.get("splits", {}):
        raise ValueError(f"Unknown split {split!r} in {split_manifest}")
    selected = set(manifest["splits"][split])
    by_relative = {path.relative_to(source).as_posix(): path for path in image_paths}
    missing = sorted(selected - set(by_relative))
    if missing:
        raise ValueError(f"Split {split!r} references {len(missing)} missing images: {missing[:5]}")
    return [by_relative[name] for name in sorted(selected)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window inference and JSON submission.")
    parser.add_argument("--config", type=Path, help="YAML inference profile")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--split-manifest", type=Path, help="Optional frozen split manifest")
    parser.add_argument("--split", default="dev", help="Manifest split to infer")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N sorted images for a smoke test")
    parser.add_argument("--output", type=Path, default=Path("submission.json"))
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-layout", choices=("sliding", "semifinal_grid"), default="sliding")
    parser.add_argument("--overlap", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--iou", type=float, default=0.55)
    parser.add_argument("--local-iou", type=float, default=0.70)
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tta", action="store_true", help="Single-model test-time augmentation")
    deblur_group = parser.add_mutually_exclusive_group()
    deblur_group.add_argument(
        "--deblur",
        dest="deblur",
        action="store_true",
        help="Enable clarity-gated deblurring (experimental; disabled by default)",
    )
    deblur_group.add_argument(
        "--no-deblur",
        dest="deblur",
        action="store_false",
        help="Disable clarity-gated deblurring for an A/B baseline",
    )
    parser.set_defaults(deblur=False)
    parser.add_argument("--deblur-threshold", type=float, default=85.0)
    parser.add_argument("--clahe", action="store_true")
    parser.add_argument("--edge-margin", type=int, default=12)
    parser.add_argument("--edge-penalty", type=float, default=1.0)
    parser.add_argument("--global-pass", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--strip-pass", action=argparse.BooleanOptionalAction, default=False,
        help="Add full-height strip views (long-defect context) to the candidate set",
    )
    parser.add_argument("--strip-width", type=int, default=810)
    parser.add_argument("--strip-overlap", type=float, default=0.2)
    parser.add_argument("--strip-imgsz", type=int, default=0, help="Strip-view inference size; 0 reuses --imgsz")
    parser.add_argument("--merge", choices=("nms", "wbf"), default="nms")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--thresholds", type=Path, help="Per-class threshold JSON from eval.py")
    parser.add_argument("--cache-only", action="store_true", help="Write raw candidates as streaming JSONL gzip instead of a submission")
    parser.add_argument("--candidate-cache", type=Path, default=Path("raw_candidates.jsonl.gz"))
    args = apply_profile_defaults(parser, "infer")
    if args.weights is None or args.source is None:
        parser.error("--weights and --source are required (directly or through --config)")
    args.weights = Path(args.weights)
    args.source = Path(args.source)
    args.output = Path(args.output)
    args.split_manifest = Path(args.split_manifest) if args.split_manifest else None
    args.thresholds = Path(args.thresholds) if args.thresholds else None

    model = YOLO(str(args.weights))
    model_names = (
        [model.names[index] for index in sorted(model.names)]
        if isinstance(model.names, dict)
        else list(model.names)
    )
    if model_names != CLASS_NAMES:
        raise ValueError(f"Checkpoint classes do not match official order: {model_names}")
    half = args.half if args.half is not None else not str(args.device).lower().startswith("cpu")
    options = InferenceOptions(
        tile_layout=args.tile_layout,
        tile_size=args.tile_size, overlap=args.overlap, imgsz=args.imgsz, batch=args.batch,
        conf=args.conf, iou=args.iou, local_iou=args.local_iou, device=args.device, half=half,
        tta=args.tta, deblur=args.deblur, deblur_threshold=args.deblur_threshold,
        clahe=args.clahe, edge_margin=args.edge_margin,
        edge_penalty=args.edge_penalty, global_pass=args.global_pass, merge=args.merge,
        strip_pass=args.strip_pass, strip_width=args.strip_width,
        strip_overlap=args.strip_overlap, strip_imgsz=args.strip_imgsz,
        max_det=args.max_det, class_thresholds=load_thresholds(args.thresholds),
    )
    image_paths = sorted(
        p for p in args.source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    image_paths = select_split_images(image_paths, args.source, args.split_manifest, args.split)
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.limit:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise SystemExit(f"No images found under {args.source}")
    duplicate_names = sorted(name for name, count in Counter(path.name for path in image_paths).items() if count > 1)
    if duplicate_names:
        raise SystemExit(f"Duplicate submission image_id values: {duplicate_names[:10]}")
    submission = []
    cache_handle = None
    if args.cache_only:
        args.candidate_cache = Path(args.candidate_cache)
        if not args.candidate_cache.name.endswith(".jsonl.gz"):
            parser.error("--candidate-cache must end in .jsonl.gz")
        args.candidate_cache.parent.mkdir(parents=True, exist_ok=True)
        cache_handle = gzip.open(args.candidate_cache, "wt", encoding="utf-8")
        header = {
            "type": "candidate_cache_header",
            "schema": 3,
            "weights_sha256": sha256_file(args.weights),
            "classes": CLASS_NAMES,
            "source": str(args.source.resolve()),
            "split_manifest": str(args.split_manifest.resolve()) if args.split_manifest else None,
            "split": args.split if args.split_manifest else None,
            "options": {
                "tile_layout": args.tile_layout,
                "tile_size": args.tile_size, "overlap": args.overlap, "imgsz": args.imgsz,
                "batch": args.batch, "conf": args.conf, "local_iou": args.local_iou,
                "device": str(args.device), "half": half, "tta": args.tta,
                "deblur": args.deblur, "clahe": args.clahe, "global_pass": args.global_pass,
                "strip_pass": args.strip_pass, "strip_width": args.strip_width,
                "strip_overlap": args.strip_overlap, "strip_imgsz": args.strip_imgsz,
                "max_det": args.max_det,
            },
        }
        cache_handle.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n")
    try:
        for index, image_path in enumerate(image_paths, start=1):
            if args.cache_only:
                candidates, image_size = predict_candidates(model, image_path, options)
                item = {
                    "image_id": image_path.relative_to(args.source).as_posix(),
                    "image_size": list(image_size),
                    "candidates": candidates,
                }
                cache_handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                cache_handle.flush()
                print(f"[{index}/{len(image_paths)}] {image_path.name}: raw_candidates={len(candidates)}")
            else:
                predictions = predict_image(model, image_path, CLASS_NAMES, options)
                submission.extend({"image_id": image_path.name, **prediction} for prediction in predictions)
                print(f"[{index}/{len(image_paths)}] {image_path.name}: total predictions={len(submission)}")
    finally:
        if cache_handle:
            cache_handle.close()

    if args.cache_only:
        print(f"Wrote streaming candidate cache to {args.candidate_cache.resolve()}")
        return

    validate_submission(submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(submission)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
