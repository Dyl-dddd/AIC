"""Compare the frozen numeric development tiles with the unlabeled Test tiles.

This is a domain/coverage audit, not an estimate of the competition score.
"""
from __future__ import annotations

from collections import Counter
import gzip
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from steel_defect.metrics import _match_class, average_precision

DEV = ROOT / "runs/semifinal/v39_numeric_adapt/dataset"
PIXELS = ROOT / "runs/semifinal/v25_test_audit_20261003/per_image_stats.json"
OUT = ROOT / "runs/semifinal/v43_test_matched_dev_20261004"
FEATURES = ("mean", "std", "black_fraction", "edge_density", "column_std", "row_std")
NAMES = ("jieba", "zonglie", "qilie", "jiaza", "yiwuyaru", "huashang", "mamianmakeng", "yanghuatiepi", "gunyin")
OOF = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof/v12_grid_crops_predictions.json"
CACHE = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells/model_a_tta_s1024_grid_crops/candidates.jsonl.gz"


def inspect(path: Path) -> dict:
    with Image.open(path) as im:
        im.draft("L", (512, 512))
        im = im.convert("L")
        im.thumbnail((512, 512))
        a = np.asarray(im)
    return {
        "file": path.name,
        "mean": float(a.mean()),
        "std": float(a.std()),
        "black_fraction": float((a <= 5).mean()),
        "edge_density": float((cv2.Canny(a, 50, 150) > 0).mean()),
        "column_std": float(a.mean(axis=0).std()),
        "row_std": float(a.mean(axis=1).std()),
    }


def label_counts(names: list[str]) -> dict:
    class_counts = Counter()
    positives = 0
    source_names = set()
    for name in names:
        source_names.add(name.split("__grid")[0])
        label = DEV / "labels/val" / Path(name).with_suffix(".txt").name
        if not label.exists():
            continue
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            positives += 1
        for line in lines:
            class_counts[NAMES[int(line.split()[0])]] += 1
    return {
        "tiles": len(names),
        "source_images": len(source_names),
        "tiles_with_any_label": positives,
        "ground_truth": dict(class_counts),
    }


def cell_id(name: str) -> str:
    stem = Path(name).stem
    base, index_text = stem.rsplit("__grid", 1)
    index = int(index_text)
    x = (0, 1354, 2709)[index % 3]
    y = (0, 1484)[index // 3]
    return f"{base}.jpg::x{x}_y{y}"


def oof_performance(names: list[str], predictions: list[dict], gt: dict) -> dict:
    allowed = {cell_id(name) for name in names}
    subset_gt = {image_id: gt[image_id] for image_id in allowed if image_id in gt}
    subset_predictions = [row for row in predictions if row["image_id"] in allowed]
    result = {"matched_cache_images": len(subset_gt), "predictions": len(subset_predictions), "classes": {}}
    for name in ("zonglie", "mamianmakeng", "yanghuatiepi", "jieba"):
        tp, fp, total_gt = _match_class(subset_predictions, subset_gt, name, .5)
        result["classes"][name] = {
            "ground_truth": int(total_gt),
            "predictions": int(len(tp)),
            "tp": int(tp.sum()),
            "ap50": average_precision(tp, fp, total_gt) if total_gt else None,
            "top100_tp": int(tp[:100].sum()),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cached = OUT / "dev_tiles.json"
    if cached.exists():
        dev = json.loads(cached.read_text(encoding="utf-8"))
    else:
        paths = sorted((DEV / "images/val").glob("*__grid*.jpg"))
        dev = []
        for i, path in enumerate(paths, 1):
            dev.append(inspect(path))
            if i % 250 == 0:
                print(f"DEV_STATS {i}/{len(paths)}", flush=True)
        cached.write_text(json.dumps(dev, ensure_ascii=False), encoding="utf-8")
    test = json.loads(PIXELS.read_text(encoding="utf-8"))["test_tile"]
    x = np.array([[row[f] for f in FEATURES] for row in dev], dtype=np.float64)
    y = np.array([[row[f] for f in FEATURES] for row in test], dtype=np.float64)
    med = np.median(np.vstack((x, y)), axis=0)
    scale = np.percentile(np.vstack((x, y)), 75, axis=0) - np.percentile(np.vstack((x, y)), 25, axis=0)
    scale = np.maximum(scale, 1e-6)
    dx, ty = (x - med) / scale, (y - med) / scale
    # A one-to-one nearest-neighbor subset: descriptive stress set, never tune on its score alone.
    costs = np.sum((ty[:, None, :] - dx[None, :, :]) ** 2, axis=2)
    match_test, match_dev = linear_sum_assignment(costs)
    matched = [dev[int(i)]["file"] for i in match_dev]
    median_std = float(np.median(y[:, FEATURES.index("std")]))
    low_names = [row["file"] for row in dev if row["std"] <= median_std]
    predictions = json.loads(OOF.read_text(encoding="utf-8"))
    with gzip.open(CACHE, "rt", encoding="utf-8") as handle:
        next(handle)
        gt = {row["image_id"]: row["ground_truth"] for line in handle if (row := json.loads(line))}
    shifts = {}
    for j, feature in enumerate(FEATURES):
        all_values = x[:, j]
        test_values = y[:, j]
        matched_values = x[match_dev, j]
        shifts[feature] = {
            "dev_median": float(np.median(all_values)),
            "test_median": float(np.median(test_values)),
            "matched_dev_median": float(np.median(matched_values)),
            "ks_dev_test": float(ks_2samp(all_values, test_values).statistic),
            "ks_p_dev_test": float(ks_2samp(all_values, test_values).pvalue),
            "ks_matched_test": float(ks_2samp(matched_values, test_values).statistic),
        }
    report = {
        "caveat": "Test has no labels; the 300 matched dev tiles are not independent by roll and cannot estimate official performance. Matching is only on six simple pixel features.",
        "features": list(FEATURES),
        "shifts": shifts,
        "all_dev": label_counts([row["file"] for row in dev]),
        "test_matched_dev": label_counts(matched),
        "low_contrast_dev": label_counts(low_names),
        "oof_performance": {
            "all_dev": oof_performance([row["file"] for row in dev], predictions, gt),
            "test_matched_dev": oof_performance(matched, predictions, gt),
            "low_contrast_dev": oof_performance(low_names, predictions, gt),
        },
        "low_contrast_threshold_test_std_median": median_std,
        "match_cost_median": float(np.median(costs[match_test, match_dev])),
        "match_cost_p90": float(np.percentile(costs[match_test, match_dev], 90)),
        "matched_dev_files": matched,
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    concise = {k: v for k, v in report.items() if k != "matched_dev_files"}
    print(json.dumps(concise, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
