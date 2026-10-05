"""Build a source-aware consensus submission from verified Model A/B caches."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from itertools import zip_longest
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_source_aware_fusion import source_aware_fuse
from scripts.build_final_submission_from_cache import read_cache_header
from steel_defect.classes import CLASS_NAMES
from steel_defect.inference import InferenceOptions, merge_candidates
from steel_defect.recall_booster import iter_candidate_records
from steel_defect.runtime import sha256_file


EXCLUDED = {"qilie"}


def merge_model(record: dict, profile: dict) -> list[dict]:
    options = InferenceOptions(
        conf=float(profile["merge_conf"]),
        iou=float(profile["per_model_nms_iou"]),
        edge_penalty=float(profile["edge_penalty"]),
        merge="nms",
    )
    return [
        {"image_id": record["image_id"], **item}
        for item in merge_candidates(
            record["candidates"], tuple(record["image_size"]), CLASS_NAMES, options
        )
        if item["category_name"] not in EXCLUDED
    ]


def build_submission(cache_a: Path, cache_b: Path, profile_path: Path,
                     output: Path, report_path: Path) -> dict:
    if output.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite output or report")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    header_a = read_cache_header(cache_a)
    header_b = read_cache_header(cache_b)
    if header_a.get("weights_sha256") != profile["model_a_sha256"]:
        raise ValueError("Model A cache weight SHA256 mismatch")
    if header_b.get("weights_sha256") != profile["model_b_sha256"]:
        raise ValueError("Model B cache weight SHA256 mismatch")
    if header_a.get("classes") != CLASS_NAMES or header_b.get("classes") != CLASS_NAMES:
        raise ValueError("candidate cache class order mismatch")

    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    image_ids = set()
    first = True
    with ExitStack() as stack:
        handle = stack.enter_context(output.open("x", encoding="utf-8"))
        handle.write("[")
        pairs = zip_longest(iter_candidate_records(cache_a), iter_candidate_records(cache_b))
        for record_a, record_b in pairs:
            if record_a is None or record_b is None:
                raise ValueError("candidate caches contain different image counts")
            if record_a["image_id"] != record_b["image_id"]:
                raise ValueError("candidate cache image order mismatch")
            if tuple(record_a["image_size"]) != tuple(record_b["image_size"]):
                raise ValueError(f"candidate cache image-size mismatch: {record_a['image_id']}")
            image_id = Path(record_a["image_id"]).name
            if image_id in image_ids:
                raise ValueError(f"duplicate submission image ID: {image_id}")
            image_ids.add(image_id)
            model_a = merge_model(record_a, profile)
            model_b = merge_model(record_b, profile)
            fused = source_aware_fuse(
                model_a,
                model_b,
                match_iou=float(profile["match_iou"]),
                boost=float(profile["consensus_boost"]),
                coord_mode=str(profile["coordinate_mode"]),
                unmatched_b_factor=float(profile["unmatched_b_factor"]),
                b_scale=float(profile["model_b_score_scale"]),
                final_nms_iou=float(profile["final_nms_iou"]),
            )
            for item in fused:
                payload = {
                    "image_id": image_id,
                    "category_name": item["category_name"],
                    "bbox": item["bbox"],
                    "score": item["score"],
                }
                if not first:
                    handle.write(",")
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                first = False
                counts[item["category_name"]] += 1
        handle.write("]\n")

    report = {
        "images": len(image_ids),
        "predictions": sum(counts.values()),
        "class_counts": dict(counts),
        "submission": str(output.resolve()),
        "submission_sha256": sha256_file(output),
        "cache_a_sha256": sha256_file(cache_a),
        "cache_b_sha256": sha256_file(cache_b),
        "profile_sha256": sha256_file(profile_path),
        "model_a_sha256": profile["model_a_sha256"],
        "model_b_sha256": profile["model_b_sha256"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_submission(
        args.cache_a, args.cache_b, args.profile, args.output, args.report
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

