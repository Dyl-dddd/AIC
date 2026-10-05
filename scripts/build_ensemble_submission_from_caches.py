"""Build audited global and class-routed submissions from two candidate caches."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import gzip
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_final_submission_from_cache import read_cache_header
from steel_defect.classes import CLASS_NAMES
from steel_defect.geometry import classwise_nms
from steel_defect.inference import InferenceOptions, merge_candidates
from steel_defect.recall_booster import iter_candidate_records
from steel_defect.runtime import sha256_file


EXCLUDED = {"qilie"}
PERMITTED = [name for name in CLASS_NAMES if name not in EXCLUDED]


def load_profile(path: Path) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "model_a_sha256",
        "model_b_sha256",
        "per_model_nms_iou",
        "edge_penalty",
        "model_b_score_scale",
        "cross_model_nms_iou",
        "merge_conf",
        "class_router",
    }
    missing = required - set(profile)
    if missing:
        raise ValueError(f"fusion profile missing fields: {sorted(missing)}")
    if set(profile["class_router"]) != set(PERMITTED):
        raise ValueError("class router must define every permitted category")
    for name, item in profile["class_router"].items():
        if item["source"] not in {"model_a", "model_b", "fusion"}:
            raise ValueError(f"invalid source for {name}: {item['source']}")
        if not 0.0 <= float(item["threshold"]) <= 1.0:
            raise ValueError(f"invalid threshold for {name}")
    return profile


def merge_model(record: dict, profile: dict) -> list[dict]:
    options = InferenceOptions(
        conf=float(profile["merge_conf"]),
        iou=float(profile["per_model_nms_iou"]),
        edge_penalty=float(profile["edge_penalty"]),
        merge="nms",
    )
    return [
        item
        for item in merge_candidates(
            record["candidates"], tuple(record["image_size"]), CLASS_NAMES, options
        )
        if item["category_name"] not in EXCLUDED
    ]


def fuse_image(model_a: list[dict], model_b: list[dict], profile: dict) -> list[dict]:
    rows = [(item, 1.0) for item in model_a]
    rows.extend((item, float(profile["model_b_score_scale"])) for item in model_b)
    if not rows:
        return []
    class_ids = {name: index for index, name in enumerate(PERMITTED)}
    boxes = np.asarray([item[0]["bbox"] for item in rows], dtype=np.float32)
    scores = np.asarray(
        [min(1.0, float(item[0]["score"]) * item[1]) for item in rows],
        dtype=np.float32,
    )
    classes = np.asarray(
        [class_ids[item[0]["category_name"]] for item in rows], dtype=np.int64
    )
    output = []
    for index in classwise_nms(
        boxes, scores, classes, float(profile["cross_model_nms_iou"])
    ):
        output.append(
            {
                "category_name": PERMITTED[int(classes[index])],
                "bbox": [int(round(value)) for value in boxes[index]],
                "score": round(float(scores[index]), 7),
            }
        )
    return output


def routed_image(
    model_a: list[dict], model_b: list[dict], fusion: list[dict], profile: dict
) -> list[dict]:
    sources = {"model_a": model_a, "model_b": model_b, "fusion": fusion}
    output = []
    for class_name, choice in profile["class_router"].items():
        output.extend(
            item
            for item in sources[choice["source"]]
            if item["category_name"] == class_name
            and float(item["score"]) >= float(choice["threshold"])
        )
    return output


def _write_item(handle, item: dict, first: bool) -> bool:
    if not first:
        handle.write(",")
    handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    return False


def build_ensemble_submissions(
    cache_a: Path,
    cache_b: Path,
    profile_path: Path,
    global_output: Path,
    routed_output: Path,
    report_path: Path,
) -> dict:
    for output in (global_output, routed_output, report_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite: {output}")
    profile = load_profile(profile_path)
    header_a = read_cache_header(cache_a)
    header_b = read_cache_header(cache_b)
    if header_a.get("weights_sha256") != profile["model_a_sha256"]:
        raise ValueError("Model A cache weight SHA256 mismatch")
    if header_b.get("weights_sha256") != profile["model_b_sha256"]:
        raise ValueError("Model B cache weight SHA256 mismatch")
    if header_a.get("classes") != CLASS_NAMES or header_b.get("classes") != CLASS_NAMES:
        raise ValueError("candidate cache class order mismatch")

    global_output.parent.mkdir(parents=True, exist_ok=True)
    routed_output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"global": Counter(), "routed": Counter()}
    images = 0
    first = {"global": True, "routed": True}

    with ExitStack() as stack:
        global_handle = stack.enter_context(global_output.open("x", encoding="utf-8"))
        routed_handle = stack.enter_context(routed_output.open("x", encoding="utf-8"))
        global_handle.write("[")
        routed_handle.write("[")
        pairs = zip_longest(iter_candidate_records(cache_a), iter_candidate_records(cache_b))
        seen_image_ids: set[str] = set()
        for record_a, record_b in pairs:
            if record_a is None or record_b is None:
                raise ValueError("candidate caches contain different image counts")
            if record_a["image_id"] != record_b["image_id"]:
                raise ValueError("candidate cache image order mismatch")
            if tuple(record_a["image_size"]) != tuple(record_b["image_size"]):
                raise ValueError(f"candidate cache image-size mismatch: {record_a['image_id']}")
            image_id = Path(record_a["image_id"]).name
            if image_id in seen_image_ids:
                raise ValueError(f"duplicate submission image ID: {image_id}")
            seen_image_ids.add(image_id)
            images += 1

            model_a = merge_model(record_a, profile)
            model_b = merge_model(record_b, profile)
            fusion = fuse_image(model_a, model_b, profile)
            routed = routed_image(model_a, model_b, fusion, profile)
            for variant, items, handle in (
                ("global", fusion, global_handle),
                ("routed", routed, routed_handle),
            ):
                for item in items:
                    payload = {"image_id": image_id, **item}
                    first[variant] = _write_item(handle, payload, first[variant])
                    counts[variant][item["category_name"]] += 1
        global_handle.write("]\n")
        routed_handle.write("]\n")

    report = {
        "images": images,
        "cache_a": str(cache_a.resolve()),
        "cache_a_sha256": sha256_file(cache_a),
        "cache_b": str(cache_b.resolve()),
        "cache_b_sha256": sha256_file(cache_b),
        "profile": str(profile_path.resolve()),
        "profile_sha256": sha256_file(profile_path),
        "model_a_sha256": profile["model_a_sha256"],
        "model_b_sha256": profile["model_b_sha256"],
        "variants": {
            "global": {
                "submission": str(global_output.resolve()),
                "submission_sha256": sha256_file(global_output),
                "predictions": sum(counts["global"].values()),
                "class_counts": dict(counts["global"]),
            },
            "routed": {
                "submission": str(routed_output.resolve()),
                "submission_sha256": sha256_file(routed_output),
                "predictions": sum(counts["routed"].values()),
                "class_counts": dict(counts["routed"]),
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--global-output", type=Path, required=True)
    parser.add_argument("--routed-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = build_ensemble_submissions(
        args.cache_a,
        args.cache_b,
        args.profile,
        args.global_output,
        args.routed_output,
        args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
