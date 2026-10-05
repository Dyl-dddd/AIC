"""Audit frozen V12 failure modes and unlabeled Test covariate shift.

No model is trained; Test pixels and predictions are never treated as labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np
from scipy.stats import ks_2samp, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from steel_defect.classes import CLASS_NAMES
from steel_defect.metrics import _match_class, average_precision


CELLS = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
OOF = ROOT / "runs/semifinal/v20_crop_verifier_20261002/v12_oof"
PIXELS = ROOT / "runs/semifinal/v25_test_audit_20261003/per_image_stats.json"
SUBMISSION = Path("D:/新建文件夹 (8)/semifinal_v12_perclass_ranker_SUBMIT_ONLY.zip")
OUTPUT = ROOT / "runs/semifinal/v41_v12_bottleneck_audit_20261004/results.json"


def ground_truth(view: str) -> dict:
    cache = CELLS / f"model_a_tta_s1024_{view}" / "candidates.jsonl.gz"
    with gzip.open(cache, "rt", encoding="utf-8") as handle:
        next(handle)
        return {
            row["image_id"]: row["ground_truth"]
            for line in handle
            if (row := json.loads(line))
        }


def distribution(rows: list[dict], key: str) -> dict:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return {
        "count": len(values),
        "median": float(np.median(values)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "fraction_below_5": float(np.mean(values < 5)) if key == "std" else None,
        "fraction_below_10": float(np.mean(values < 10)) if key == "std" else None,
    }


def image_prediction_stats(predictions: list[dict], view: str) -> dict:
    by_image = defaultdict(list)
    by_class = defaultdict(list)
    for item in predictions:
        by_image[item["image_id"]].append(item)
        by_class[item["category_name"]].append(item)
    return {
        "view": view,
        "total": len(predictions),
        "images_with_prediction": len(by_image),
        "per_image_count_median": float(np.median([len(v) for v in by_image.values()])),
        "per_class": {
            name: {
                "count": len(by_class[name]),
                "score_median": float(np.median([float(row["score"]) for row in by_class[name]]))
                if by_class[name] else None,
                "score_p99": float(np.percentile([float(row["score"]) for row in by_class[name]], 99))
                if by_class[name] else None,
            }
            for name in CLASS_NAMES
        },
    }


def audit_view(view: str) -> dict:
    path = OOF / f"v12_{view}_predictions.json"
    predictions = json.loads(path.read_text(encoding="utf-8"))
    gt = ground_truth(view)
    by_class = defaultdict(list)
    for item in predictions:
        by_class[item["category_name"]].append(item)
    class_results = {}
    for name in CLASS_NAMES:
        if name == "qilie":
            continue
        rows = by_class[name]
        points = {}
        for threshold in (0.3, 0.5, 0.75):
            tp, fp, total_gt = _match_class(rows, gt, name, threshold)
            points[str(threshold)] = {
                "tp": int(tp.sum()), "fp": int(fp.sum()), "gt": total_gt,
                "ap": average_precision(tp, fp, total_gt),
                "top100_tp": int(tp[:100].sum()),
            }
        class_results[name] = points
    return {
        "n_images": len(gt),
        "n_predictions": len(predictions),
        "n_empty_gt_images": sum(not any(classes.values()) for classes in gt.values()),
        "classes": class_results,
        "prediction_distribution": image_prediction_stats(predictions, view),
    }


def test_distribution(pixel_stats: dict, predictions: list[dict]) -> dict:
    by_image = defaultdict(list)
    for row in predictions:
        by_image[row["image_id"]].append(row)
    result = {}
    for view, key in (("original", "test_full"), ("grid_crops", "test_tile")):
        pixels = pixel_stats[key]
        pairs = [(row, by_image[Path(row["file"]).name]) for row in pixels]
        scores = [float(max((p["score"] for p in preds), default=0.0)) for _, preds in pairs]
        counts = [len(preds) for _, preds in pairs]
        std = [float(row["std"]) for row, _ in pairs]
        corr = spearmanr(std, counts)
        result[view] = {
            "images": len(pairs),
            "prediction_count_median": float(np.median(counts)),
            "prediction_count_p90": float(np.percentile(counts, 90)),
            "highest_score_median": float(np.median(scores)),
            "spearman_contrast_vs_prediction_count": float(corr.statistic),
            "spearman_p": float(corr.pvalue),
            "zero_prediction_images": int(sum(c == 0 for c in counts)),
            "low_contrast_images_std_below_5": int(sum(s < 5 for s in std)),
        }
    return result


def main() -> None:
    pixel_stats = json.loads(PIXELS.read_text(encoding="utf-8"))
    pixel_shift = {}
    for train_key, test_key in (
        ("train_numeric_full", "test_full"),
        ("train_numeric_tile", "test_tile"),
    ):
        pair_key = f"{train_key}_vs_{test_key}"
        pixel_shift[pair_key] = {}
        for feature in ("mean", "std", "black_fraction", "edge_density", "column_std", "row_std"):
            train_values = [float(row[feature]) for row in pixel_stats[train_key]]
            test_values = [float(row[feature]) for row in pixel_stats[test_key]]
            ks = ks_2samp(train_values, test_values)
            pixel_shift[pair_key][feature] = {
                "train": distribution(pixel_stats[train_key], feature),
                "test": distribution(pixel_stats[test_key], feature),
                "ks_statistic": float(ks.statistic),
                "ks_p": float(ks.pvalue),
            }
    with ZipFile(SUBMISSION) as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError("Unexpected original V12 archive members")
        submission = json.loads(archive.read("submission.json"))
    test_image_ids = {
        "original": {Path(row["file"]).name for row in pixel_stats["test_full"]},
        "grid_crops": {Path(row["file"]).name for row in pixel_stats["test_tile"]},
    }
    if test_image_ids["original"] & test_image_ids["grid_crops"]:
        raise ValueError("Overlapping Test view image IDs")
    unknown = {row["image_id"] for row in submission} - set().union(*test_image_ids.values())
    if unknown:
        raise ValueError(f"Submission image IDs absent from Test audit: {len(unknown)}")
    results = {
        "warning": "Test is unlabeled; observed box counts are not official TP/FP; V12 OOF and Test are different samples.",
        "pixel_shift": pixel_shift,
        "frozen_dev": {view: audit_view(view) for view in ("original", "grid_crops")},
        "official_test_unlabeled": test_distribution(pixel_stats, submission),
        "official_test_prediction_distribution": {
            view: image_prediction_stats(
                [row for row in submission if row["image_id"] in test_image_ids[view]],
                view,
            )
            for view in ("original", "grid_crops")
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUTPUT}")
    for view, row in results["frozen_dev"].items():
        print(f"{view}: {row['n_images']} images, {row['n_predictions']} boxes")
        for name in ("zonglie", "jieba", "yanghuatiepi", "huashang"):
            points = row["classes"][name]
            print(f"  {name}: AP30={points['0.3']['ap']:.4f} AP50={points['0.5']['ap']:.4f} "
                  f"AP75={points['0.75']['ap']:.4f} top100TP={points['0.5']['top100_tp']}")
    for pair, items in pixel_shift.items():
        row = items["std"]
        print(f"{pair} std median: {row['train']['median']:.2f} -> {row['test']['median']:.2f}, "
              f"KS={row['ks_statistic']:.3f}, p={row['ks_p']:.3g}")


if __name__ == "__main__":
    main()
