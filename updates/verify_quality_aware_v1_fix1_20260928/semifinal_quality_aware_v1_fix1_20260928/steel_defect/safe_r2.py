"""Source-aware evidence reranking and guarded box refinement.

SAFE-R2 deliberately operates on an already selected prediction set.  It may
change scores (ranking) and, when strongly supported, coordinates, but it never
adds or removes a prediction.  Raw detector multiplicity is not treated as
independent evidence: at most one representative is retained from each of the
four A/B x local/global provenance buckets.

The module only depends on NumPy so fitted profiles can be replayed in the
submission builder without a GPU or a machine-learning runtime.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .classes import CLASS_NAMES, CLASS_TO_ID


FEATURE_NAMES = (
    "log_base_score",
    "ab_presence",
    "ab_iou",
    "source_count",
    "global_local_coherence",
    "dispersion",
    "edge_only",
    "log_area_fraction",
    "log_aspect_ratio",
)


DEFAULT_PROFILE: dict = {
    "support_iou": 0.50,
    # A fitted profile should explicitly opt into score changes.  Keeping this
    # at zero makes an absent profile a lossless pass-through.
    "score_blend": 0.0,
    "scaler": {
        "mean": {name: 0.0 for name in FEATURE_NAMES},
        "scale": {name: 1.0 for name in FEATURE_NAMES},
    },
    "linear": {
        "intercept": 0.0,
        "coefficients": {name: 0.0 for name in FEATURE_NAMES},
    },
    "refinement": {
        "enabled": False,
        "max_dispersion": 0.20,
        "minimum_anchor_iou": 0.75,
        "minimum_donors": 2,
        "require_non_edge_local": True,
        "score_power": 0.5,
        "anchor_iou_power": 1.0,
    },
}


def _deep_update(base: dict, update: Mapping) -> dict:
    output = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(output.get(key), dict):
            output[key] = _deep_update(output[key], value)
        else:
            output[key] = deepcopy(value)
    return output


def load_profile(profile: Mapping | str | Path | None = None) -> dict:
    """Load and validate a deterministic SAFE-R2 JSON profile."""
    if profile is None:
        supplied: Mapping = {}
    elif isinstance(profile, (str, Path)):
        supplied = json.loads(Path(profile).read_text(encoding="utf-8"))
    elif isinstance(profile, Mapping):
        supplied = profile
    else:
        raise TypeError("profile must be a mapping, JSON path, or None")
    result = _deep_update(DEFAULT_PROFILE, supplied)
    if not 0.0 <= float(result["support_iou"]) <= 1.0:
        raise ValueError("support_iou must be in [0, 1]")
    if not 0.0 <= float(result["score_blend"]) <= 1.0:
        raise ValueError("score_blend must be in [0, 1]")
    coefficients = result["linear"]["coefficients"]
    unknown = set(coefficients) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown SAFE-R2 feature coefficients: {sorted(unknown)}")
    for name in FEATURE_NAMES:
        scale = float(result["scaler"]["scale"].get(name, 1.0))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"scaler scale must be positive for {name}")
    refinement = result["refinement"]
    if not 0.0 <= float(refinement["max_dispersion"]) <= 1.0:
        raise ValueError("refinement.max_dispersion must be in [0, 1]")
    if not 0.0 <= float(refinement["minimum_anchor_iou"]) <= 1.0:
        raise ValueError("refinement.minimum_anchor_iou must be in [0, 1]")
    if int(refinement["minimum_donors"]) < 2:
        raise ValueError("refinement.minimum_donors must be at least two")
    return result


def _box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return float(intersection / max(area_a + area_b - intersection, 1e-12))


def _prediction_class_id(item: Mapping) -> int:
    if "class_id" in item:
        class_id = int(item["class_id"])
    elif "category_name" in item:
        try:
            class_id = CLASS_TO_ID[str(item["category_name"])]
        except KeyError as error:
            raise ValueError(f"unknown category_name: {item['category_name']}") from error
    else:
        raise ValueError("prediction must have class_id or category_name")
    if not 0 <= class_id < len(CLASS_NAMES):
        raise ValueError(f"class_id out of range: {class_id}")
    return class_id


def _valid_box(box: Sequence[float]) -> bool:
    return (
        len(box) == 4
        and all(math.isfinite(float(value)) for value in box)
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def select_provenance_representatives(
    prediction: Mapping,
    model_a_candidates: Sequence[Mapping],
    model_b_candidates: Sequence[Mapping],
    support_iou: float = 0.50,
) -> dict[str, dict]:
    """Select at most one class-consistent representative per provenance bucket.

    Selection is deterministic: anchor IoU, then raw score, then earlier input
    order.  Repeated detections from one view therefore cannot manufacture
    additional evidence.
    """
    anchor = prediction["bbox"]
    if not _valid_box(anchor):
        raise ValueError("prediction bbox must be valid xyxy")
    class_id = _prediction_class_id(prediction)
    selected: dict[str, tuple[tuple[float, float, int], dict]] = {}
    for model_name, candidates in (("a", model_a_candidates), ("b", model_b_candidates)):
        for index, candidate in enumerate(candidates):
            if _prediction_class_id(candidate) != class_id or not _valid_box(candidate["bbox"]):
                continue
            overlap = _box_iou(anchor, candidate["bbox"])
            if overlap < support_iou:
                continue
            view_name = "global" if bool(candidate.get("is_global", False)) else "local"
            bucket = f"{model_name}_{view_name}"
            key = (overlap, float(candidate["score"]), -index)
            if bucket not in selected or key > selected[bucket][0]:
                representative = dict(candidate)
                representative["anchor_iou"] = overlap
                representative["model"] = model_name
                representative["provenance_bucket"] = bucket
                selected[bucket] = (key, representative)
    return {bucket: selected[bucket][1] for bucket in sorted(selected)}


def extract_features(
    prediction: Mapping,
    representatives: Mapping[str, Mapping],
    image_size: tuple[int, int],
) -> dict[str, float]:
    """Extract the intentionally low-capacity SAFE-R2 feature vector."""
    width, height = map(float, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive (width, height)")
    box = np.asarray(prediction["bbox"], dtype=np.float64)
    box_width = max(float(box[2] - box[0]), 1e-12)
    box_height = max(float(box[3] - box[1]), 1e-12)
    base_score = float(prediction["score"])
    reps = list(representatives.values())
    a_reps = [item for item in reps if item["model"] == "a"]
    b_reps = [item for item in reps if item["model"] == "b"]

    ab_ious = [_box_iou(a["bbox"], b["bbox"]) for a in a_reps for b in b_reps]
    coherence: list[float] = []
    for model_name in ("a", "b"):
        local = representatives.get(f"{model_name}_local")
        global_ = representatives.get(f"{model_name}_global")
        if local is not None and global_ is not None:
            coherence.append(_box_iou(local["bbox"], global_["bbox"]))
    pairwise = [
        _box_iou(reps[left]["bbox"], reps[right]["bbox"])
        for left in range(len(reps))
        for right in range(left + 1, len(reps))
    ]
    dispersion = 1.0 if not pairwise else float(np.median(1.0 - np.asarray(pairwise)))
    edge_only = bool(reps) and all(
        not bool(item.get("is_global", False))
        and bool(item.get("touches_internal_edge", False))
        for item in reps
    )
    return {
        "log_base_score": math.log(max(base_score, 1e-12)),
        "ab_presence": float(bool(a_reps) and bool(b_reps)),
        "ab_iou": max(ab_ious, default=0.0),
        "source_count": float(len(reps)),
        "global_local_coherence": float(np.mean(coherence)) if coherence else 0.0,
        "dispersion": dispersion,
        "edge_only": float(edge_only),
        "log_area_fraction": math.log(max(box_width * box_height / (width * height), 1e-12)),
        "log_aspect_ratio": math.log(box_width / box_height),
    }


def quality_score(features: Mapping[str, float], profile: Mapping) -> float:
    """Apply the profile's scaler and linear sigmoid head."""
    total = float(profile["linear"].get("intercept", 0.0))
    means = profile["scaler"]["mean"]
    scales = profile["scaler"]["scale"]
    coefficients = profile["linear"]["coefficients"]
    for name in FEATURE_NAMES:
        standardized = (
            float(features[name]) - float(means.get(name, 0.0))
        ) / float(scales.get(name, 1.0))
        total += float(coefficients.get(name, 0.0)) * standardized
    # Numerically stable sigmoid.
    if total >= 0.0:
        decay = math.exp(-total)
        return 1.0 / (1.0 + decay)
    growth = math.exp(total)
    return growth / (1.0 + growth)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * float(cumulative[-1]), side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _box_to_latent(box: Sequence[float]) -> np.ndarray:
    x1, y1, x2, y2 = map(float, box)
    return np.asarray([
        0.5 * (x1 + x2),
        0.5 * (y1 + y2),
        math.log(max(x2 - x1, 1e-12)),
        math.log(max(y2 - y1, 1e-12)),
    ], dtype=np.float64)


def _latent_to_box(latent: np.ndarray, image_size: tuple[int, int]) -> list[float]:
    width, height = map(float, image_size)
    box_width, box_height = math.exp(float(latent[2])), math.exp(float(latent[3]))
    output = [
        float(latent[0]) - 0.5 * box_width,
        float(latent[1]) - 0.5 * box_height,
        float(latent[0]) + 0.5 * box_width,
        float(latent[1]) + 0.5 * box_height,
    ]
    output[0], output[2] = max(0.0, output[0]), min(width, output[2])
    output[1], output[3] = max(0.0, output[1]), min(height, output[3])
    return output


def guarded_robust_box_refinement(
    prediction: Mapping,
    representatives: Mapping[str, Mapping],
    image_size: tuple[int, int],
    profile: Mapping,
) -> tuple[list[float], bool, str]:
    """Apply uncertainty-gated robust box voting (UG-RBV).

    Coordinates are independently aggregated in ``cx, cy, log(w), log(h)``.
    Failure of any evidence gate or the final anchor-IoU trust region returns
    the anchor coordinates byte-for-byte as ordinary Python values.
    """
    anchor = list(prediction["bbox"])
    config = profile["refinement"]
    if not bool(config.get("enabled", False)):
        return anchor, False, "disabled"
    reps = list(representatives.values())
    models = {str(item["model"]) for item in reps}
    if models != {"a", "b"}:
        return anchor, False, "missing_independent_model_support"
    pairwise = [
        _box_iou(reps[left]["bbox"], reps[right]["bbox"])
        for left in range(len(reps))
        for right in range(left + 1, len(reps))
    ]
    dispersion = 1.0 if not pairwise else float(np.median(1.0 - np.asarray(pairwise)))
    if dispersion > float(config["max_dispersion"]):
        return anchor, False, "high_dispersion"
    non_edge_local = [
        item for item in reps
        if not bool(item.get("is_global", False))
        and not bool(item.get("touches_internal_edge", False))
    ]
    if bool(config.get("require_non_edge_local", True)) and not non_edge_local:
        return anchor, False, "no_non_edge_local_donor"
    donors = [
        item for item in reps
        if bool(item.get("is_global", False))
        or not bool(item.get("touches_internal_edge", False))
    ]
    if len(donors) < int(config["minimum_donors"]):
        return anchor, False, "insufficient_donors"
    latents = np.stack([_box_to_latent(item["bbox"]) for item in donors])
    score_power = float(config.get("score_power", 0.5))
    iou_power = float(config.get("anchor_iou_power", 1.0))
    weights = np.asarray([
        max(float(item["score"]), 1e-12) ** score_power
        * max(float(item["anchor_iou"]), 1e-12) ** iou_power
        for item in donors
    ], dtype=np.float64)
    latent = np.asarray([
        _weighted_median(latents[:, dimension], weights)
        for dimension in range(latents.shape[1])
    ])
    refined = _latent_to_box(latent, image_size)
    if not _valid_box(refined):
        return anchor, False, "invalid_refinement"
    if _box_iou(anchor, refined) < float(config["minimum_anchor_iou"]):
        return anchor, False, "outside_anchor_trust_region"
    return refined, True, "refined"


def apply_safe_r2(
    predictions: Sequence[Mapping],
    model_a_candidates: Sequence[Mapping],
    model_b_candidates: Sequence[Mapping],
    image_size: tuple[int, int],
    profile: Mapping | str | Path | None = None,
    *,
    include_diagnostics: bool = False,
) -> list[dict]:
    """Rerank and optionally refine one image while preserving membership."""
    fitted = load_profile(profile)
    output: list[tuple[int, dict]] = []
    for index, prediction in enumerate(predictions):
        if not _valid_box(prediction["bbox"]):
            raise ValueError("prediction bbox must be valid xyxy")
        representatives = select_provenance_representatives(
            prediction,
            model_a_candidates,
            model_b_candidates,
            float(fitted["support_iou"]),
        )
        features = extract_features(prediction, representatives, image_size)
        quality = quality_score(features, fitted)
        raw_base_score = float(prediction["score"])
        if not math.isfinite(raw_base_score) or not 0.0 <= raw_base_score <= 1.0:
            raise ValueError("prediction score must be finite and in [0, 1]")
        base_score = max(raw_base_score, 1e-12)
        blend = float(fitted["score_blend"])
        # Geometric blending preserves the useful dynamic range of low scores.
        reranked = (
            raw_base_score
            if blend == 0.0
            else base_score ** (1.0 - blend) * max(quality, 1e-12) ** blend
        )
        box, refined, reason = guarded_robust_box_refinement(
            prediction, representatives, image_size, fitted
        )
        item = dict(prediction)
        item["score"] = float(min(max(reranked, 0.0), 1.0))
        item["bbox"] = box
        if include_diagnostics:
            item["safe_r2"] = {
                "quality": quality,
                "features": features,
                "provenance_buckets": sorted(representatives),
                "refined": refined,
                "refinement_reason": reason,
            }
        output.append((index, item))
    # Stable original-index tie breaking makes output bitwise repeatable.
    output.sort(key=lambda pair: (-float(pair[1]["score"]), pair[0]))
    return [item for _, item in output]


__all__ = [
    "DEFAULT_PROFILE",
    "FEATURE_NAMES",
    "apply_safe_r2",
    "extract_features",
    "guarded_robust_box_refinement",
    "load_profile",
    "quality_score",
    "select_provenance_representatives",
]
