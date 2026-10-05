"""Cross-fit image-texture evidence on frozen V12 candidates, without Test labels."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np
from xgboost import XGBClassifier

from scripts.analyze_final_submission_postprocess import load_cell
from scripts.analyze_safe_r2_crossfit import fuse_v7
from scripts.analyze_v12_perclass_ranker_oof import FEATURES, add_context, fit, matrix
from scripts.analyze_v9_support_oof import add_support, initial_records
from scripts.analyze_varifocal_oof_transfer import balanced_weights, load_groups
from steel_defect.metrics import _match_class, average_precision


ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "runs/semifinal/ensemble_v6_tta_dev_returned_20260927/ensemble_v6_tta_dev/cells"
QUALITY = ROOT / "runs/quality_aware_v1_fix3_results_received_20260928/quality_aware_v1_fix3_20260928_pipeline/audit/evaluation/varifocal"
V9 = ROOT / "runs/v9_quality_localizer_results_received_20260930/runs/semifinal/v9_quality_localizer_pipeline/selection/cells"
SPLITS = ROOT / "data/official_v2_20260831b/splits.json"
IMAGES = ROOT / "data/official/train"
OUT = ROOT / "runs/semifinal/v19_texture_reranker_20261002/results.json"
TARGETS = ("zonglie", "huashang", "yanghuatiepi")
BLENDS = (0.05, 0.10, 0.15, 0.20, 0.30)
TEXTURE_NAMES = (
    "inner_mean", "inner_std", "halo_mean", "halo_std", "mean_difference",
    "inner_gx", "inner_gy", "inner_lap", "halo_gx", "halo_gy", "halo_lap",
    "gradient_ratio", "lap_ratio", "log_width", "log_height", "log_aspect",
    "center_x", "center_y",
)


def _sums(integral: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]


def _stats(integral: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    area = np.maximum((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), 1)
    return _sums(integral, boxes) / area


def texture_features(rows: list[dict], image_root: Path) -> np.ndarray:
    by_source: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_source[row["prediction"]["image_id"].split("::", 1)[0]].append(index)
    output = np.zeros((len(rows), len(TEXTURE_NAMES)), dtype=np.float32)
    for number, (source, indices) in enumerate(sorted(by_source.items()), 1):
        image = cv2.imread(str(image_root / source), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_root / source)
        if image.ndim == 3:
            image = image[:, :, 0] if image.shape[2] == 1 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        scaled = cv2.resize(image, (max(1, width // 4), max(1, height // 4)), interpolation=cv2.INTER_AREA)
        gray = scaled.astype(np.float32) / 255.0
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        integrals = [cv2.integral(channel, sdepth=cv2.CV_64F) for channel in (gray, gray * gray, gx, gy, lap)]
        boxes = []
        bounds = []
        for index in indices:
            prediction = rows[index]["prediction"]
            suffix = prediction["image_id"].split("::", 1)
            ox, oy = 0, 0
            local_width, local_height = width, height
            if len(suffix) == 2:
                ox, oy = (int(value[1:]) for value in suffix[1].split("_"))
                local_width = min(1387, width - ox)
                local_height = min(1516, height - oy)
            x1, y1, x2, y2 = prediction["bbox"]
            boxes.append([x1 + ox, y1 + oy, x2 + ox, y2 + oy])
            bounds.append([ox, oy, ox + local_width, oy + local_height])
        original = np.asarray(boxes, dtype=np.float64)
        view_bounds = np.asarray(bounds, dtype=np.float64)
        scale = np.asarray([gray.shape[1] / width, gray.shape[0] / height] * 2)
        inner = np.rint(original * scale).astype(np.int64)
        inner[:, [0, 2]] = np.clip(inner[:, [0, 2]], 0, gray.shape[1])
        inner[:, [1, 3]] = np.clip(inner[:, [1, 3]], 0, gray.shape[0])
        inner[:, 2] = np.maximum(inner[:, 2], inner[:, 0] + 1)
        inner[:, 3] = np.maximum(inner[:, 3], inner[:, 1] + 1)
        inner[:, 2] = np.minimum(inner[:, 2], gray.shape[1])
        inner[:, 3] = np.minimum(inner[:, 3], gray.shape[0])
        expand_x = np.maximum(3, (inner[:, 2] - inner[:, 0]) // 5)
        expand_y = np.maximum(3, (inner[:, 3] - inner[:, 1]) // 5)
        halo = inner.copy()
        halo[:, 0] = np.maximum(0, halo[:, 0] - expand_x)
        halo[:, 1] = np.maximum(0, halo[:, 1] - expand_y)
        halo[:, 2] = np.minimum(gray.shape[1], halo[:, 2] + expand_x)
        halo[:, 3] = np.minimum(gray.shape[0], halo[:, 3] + expand_y)
        scaled_bounds = np.rint(view_bounds * scale).astype(np.int64)
        halo[:, 0] = np.maximum(halo[:, 0], scaled_bounds[:, 0])
        halo[:, 1] = np.maximum(halo[:, 1], scaled_bounds[:, 1])
        halo[:, 2] = np.minimum(halo[:, 2], scaled_bounds[:, 2])
        halo[:, 3] = np.minimum(halo[:, 3], scaled_bounds[:, 3])
        im, im2, igx, igy, ilap = integrals
        mean, halo_mean = _stats(im, inner), _stats(im, halo)
        std = np.sqrt(np.maximum(_stats(im2, inner) - mean * mean, 0.0))
        halo_std = np.sqrt(np.maximum(_stats(im2, halo) - halo_mean * halo_mean, 0.0))
        egx, egy, elap = (_stats(table, inner) for table in (igx, igy, ilap))
        hgx, hgy, hlap = (_stats(table, halo) for table in (igx, igy, ilap))
        bw = np.maximum(original[:, 2] - original[:, 0], 1.0)
        bh = np.maximum(original[:, 3] - original[:, 1], 1.0)
        vw = np.maximum(view_bounds[:, 2] - view_bounds[:, 0], 1.0)
        vh = np.maximum(view_bounds[:, 3] - view_bounds[:, 1], 1.0)
        features = np.column_stack((
            mean, std, halo_mean, halo_std, mean - halo_mean, egx, egy, elap,
            hgx, hgy, hlap, np.log((egx + 1e-4) / (egy + 1e-4)),
            np.log((elap + 1e-4) / (hlap + 1e-4)),
            np.log(bw / vw), np.log(bh / vh), np.log(bw / bh),
            (original[:, 0] + original[:, 2] - 2 * view_bounds[:, 0]) / (2 * vw),
            (original[:, 1] + original[:, 3] - 2 * view_bounds[:, 1]) / (2 * vh),
        )).astype(np.float32)
        if not np.isfinite(features).all():
            raise ValueError(f"nonfinite texture feature: {source}")
        output[indices] = features
        if number % 100 == 0:
            print(f"TEXTURE {number}/{len(by_source)}", flush=True)
    return output


def fit_texture(x: np.ndarray, y: np.ndarray, seed: int) -> XGBClassifier:
    weights = balanced_weights(np.zeros(len(y), dtype=np.int32), y).astype(np.float32)
    model = XGBClassifier(
        n_estimators=100, max_depth=2, learning_rate=0.035, min_child_weight=15,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=20.0,
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        max_bin=128, n_jobs=8, random_state=seed,
    )
    model.fit(x, y, sample_weight=weights, verbose=False)
    return model


def class_ap(rows: list[dict], scores: np.ndarray, ground_truth: dict, class_name: str) -> float:
    predictions = []
    for row, score in zip(rows, scores, strict=True):
        prediction = dict(row["prediction"])
        prediction["score"] = float(score)
        predictions.append(prediction)
    tp, fp, count = _match_class(predictions, ground_truth, class_name, 0.5)
    return float(average_precision(tp, fp, count))


def main() -> None:
    groups = load_groups(SPLITS)
    v12 = json.loads((ROOT / "runs/semifinal/v12_perclass_ranker_20261001/oof_results.json").read_text())
    records: dict[str, list[dict]] = {}
    truths: dict[str, dict] = {}
    textures: dict[str, np.ndarray] = {}
    for view in ("original", "grid_crops"):
        a = load_cell(CELLS / f"model_a_tta_s1024_{view}")
        b = load_cell(CELLS / f"model_b_tta_s1280_{view}")
        rows = initial_records(fuse_v7(a, b), a["ground_truth"], groups)
        add_support(rows, json.loads((QUALITY / f"varifocal_{view}_predictions.json").read_text()), "vf")
        for prefix, pattern in (("p1", "v9_model_b_quality_p1_last_*"), ("p2", "v9_model_b_quality_p2_last_*")):
            folder = next(V9.glob(f"{pattern}_{view}"))
            add_support(rows, json.loads((folder / "predictions.json").read_text()), prefix)
        add_context(rows, a["sizes"])
        selected = [row for row in rows if row["class_name"] in TARGETS]
        records[view], truths[view] = selected, a["ground_truth"]
        textures[view] = texture_features(selected, IMAGES)
        print(f"LOAD {view} rows={len(selected)}", flush=True)

    combined = [*records["original"], *records["grid_crops"]]
    texture = np.vstack((textures["original"], textures["grid_crops"]))
    result: dict = {"protocol": {"group_oof_folds": 5, "classes": TARGETS, "texture_features": TEXTURE_NAMES}, "classes": {}}
    for class_index, class_name in enumerate(TARGETS):
        positions = np.asarray([i for i, row in enumerate(combined) if row["class_name"] == class_name])
        rows = [combined[i] for i in positions]
        tex = texture[positions]
        labels = np.asarray([row["label"] for row in rows], dtype=np.int32)
        folds = np.asarray([row["fold"] for row in rows], dtype=np.int32)
        base_scores = np.asarray([row["prediction"]["score"] for row in rows], dtype=np.float64)
        baseline_probability = np.zeros(len(rows), dtype=np.float64)
        texture_probability = np.zeros(len(rows), dtype=np.float64)
        for fold in range(5):
            train, held = folds != fold, folds == fold
            if {rows[i]["group"] for i in np.flatnonzero(train)} & {rows[i]["group"] for i in np.flatnonzero(held)}:
                raise RuntimeError("group leakage")
            baseline_model = fit([row for row, keep in zip(rows, train, strict=True) if keep], 3101 + 101 * fold + list(v12["decision"]["class_betas"]).index(class_name))
            baseline_probability[held] = baseline_model.predict_proba(matrix([row for row, keep in zip(rows, held, strict=True) if keep], FEATURES))[:, 1]
            texture_model = fit_texture(tex[train], labels[train], 6401 + 101 * fold + class_index)
            texture_probability[held] = texture_model.predict_proba(tex[held])[:, 1]
            print(f"FIT {class_name} fold={fold} train={train.sum()} pos={labels[train].sum()}", flush=True)
        beta = float(v12["decision"]["class_betas"][class_name])
        v12_scores = np.maximum(base_scores, 1e-12) ** (1 - beta) * np.maximum(baseline_probability, 1e-12) ** beta
        view_len = len(records["original"])
        by_view: dict = {}
        for view in ("original", "grid_crops"):
            selected_positions = np.flatnonzero((positions < view_len) if view == "original" else (positions >= view_len))
            view_rows = [rows[i] for i in selected_positions]
            view_base = v12_scores[selected_positions]
            reported_ap = float(v12["decision"]["views"][view]["per_class"][class_name]["ap50"])
            reproduced_ap = class_ap(view_rows, view_base, truths[view], class_name)
            if abs(reported_ap - reproduced_ap) > 1e-6:
                raise RuntimeError(f"V12 reproduction mismatch {view}/{class_name}: {reproduced_ap} vs {reported_ap}")
            by_view[view] = {"v12_ap50": reproduced_ap, "v12_tp": int(v12["decision"]["views"][view]["per_class"][class_name]["tp"]), "blends": {}}
            for blend in BLENDS:
                score = np.maximum(view_base, 1e-12) ** (1 - blend) * np.maximum(texture_probability[selected_positions], 1e-12) ** blend
                ap = class_ap(view_rows, score, truths[view], class_name)
                by_view[view]["blends"][str(blend)] = {"ap50": ap, "delta": ap - reported_ap}
        result["classes"][class_name] = by_view
        print(f"RESULT {class_name} " + json.dumps(by_view), flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {OUT}", flush=True)


if __name__ == "__main__":
    main()
