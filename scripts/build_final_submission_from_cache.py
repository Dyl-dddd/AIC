"""Build a semifinal submission deterministically from one raw candidate cache."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steel_defect.classes import CLASS_NAMES
from steel_defect.inference import InferenceOptions, merge_candidates
from steel_defect.recall_booster import iter_candidate_records
from steel_defect.runtime import sha256_file


EXCLUDED = {"qilie"}


def load_thresholds(path: Path | None, floor: float) -> dict[str, float]:
    if path is None:
        return {name: floor for name in CLASS_NAMES if name not in EXCLUDED}
    payload = json.loads(path.read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds", payload)
    unknown = set(thresholds) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"unknown threshold classes: {sorted(unknown)}")
    result = {name: float(value) for name, value in thresholds.items() if name not in EXCLUDED}
    if set(result) != set(CLASS_NAMES) - EXCLUDED:
        raise ValueError("threshold file must define every permitted semifinal category")
    if any(not floor <= value <= 1.0 for value in result.values()):
        raise ValueError("class thresholds must be between the generation floor and 1")
    return result


def read_cache_header(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        header = json.loads(handle.readline())
    if header.get("type") != "candidate_cache_header" or header.get("schema") != 3:
        raise ValueError("expected a schema-3 JSONL candidate cache")
    return header


def build_submission(cache: Path, output: Path, report_path: Path, thresholds_path: Path | None,
                     floor: float, nms_iou: float, edge_penalty: float,
                     expected_weight_sha256: str | None = None) -> dict:
    cache = cache.resolve(strict=True)
    if output.exists() or report_path.exists():
        raise FileExistsError("output or report already exists; refusing to overwrite")
    header = read_cache_header(cache)
    if expected_weight_sha256 and header.get("weights_sha256") != expected_weight_sha256:
        raise ValueError("candidate cache weight SHA256 mismatch")
    thresholds = load_thresholds(thresholds_path, floor)
    options = InferenceOptions(conf=floor, iou=nms_iou, edge_penalty=edge_penalty,
        merge="nms", class_thresholds=thresholds)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_ids: set[str] = set()
    class_counts: Counter[str] = Counter()
    predictions = 0
    first = True
    with output.open("x", encoding="utf-8") as handle:
        handle.write("[")
        for record in iter_candidate_records(cache):
            image_id = Path(record["image_id"]).name
            if image_id in image_ids:
                raise ValueError(f"duplicate image_id in candidate cache: {image_id}")
            image_ids.add(image_id)
            merged = merge_candidates(record["candidates"], tuple(record["image_size"]),
                                      CLASS_NAMES, options)
            for item in merged:
                if item["category_name"] in EXCLUDED:
                    continue
                payload = {"image_id": image_id, **item}
                if not first:
                    handle.write(",")
                handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
                first = False
                predictions += 1
                class_counts[item["category_name"]] += 1
        handle.write("]\n")
    report = {
        "cache": str(cache), "cache_sha256": sha256_file(cache),
        "weights_sha256": header["weights_sha256"], "images": len(image_ids),
        "predictions": predictions, "nms_iou": nms_iou,
        "edge_penalty": edge_penalty, "generation_floor": floor,
        "thresholds": thresholds, "excluded_categories": sorted(EXCLUDED),
        "class_counts": dict(class_counts), "submission": str(output.resolve()),
        "submission_sha256": sha256_file(output),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--floor", type=float, default=0.0001)
    parser.add_argument("--nms-iou", type=float, default=0.68)
    parser.add_argument("--edge-penalty", type=float, default=0.5)
    parser.add_argument("--expected-weight-sha256")
    args = parser.parse_args()
    report = build_submission(args.cache, args.output, args.report, args.thresholds,
        args.floor, args.nms_iou, args.edge_penalty, args.expected_weight_sha256)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
