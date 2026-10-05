"""Conservative, automatically gated enhancement for grayscale industrial images."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def laplacian_variance(image: np.ndarray) -> float:
    gray = to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def tenengrad(image: np.ndarray) -> float:
    gray = to_gray(image).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def halo_fraction(original: np.ndarray, enhanced: np.ndarray, threshold: float = 18.0) -> float:
    diff = enhanced.astype(np.float32) - original.astype(np.float32)
    return float(np.mean(np.abs(diff) >= threshold))


@dataclass(frozen=True)
class DeblurResult:
    image: np.ndarray
    applied: bool
    blur_score: float
    strength: float


def adaptive_deblur(
    image: np.ndarray,
    threshold: float = 85.0,
    max_halo_fraction: float = 0.08,
) -> DeblurResult:
    """
    Apply bounded unsharp deconvolution only when the image is measurably soft.

    Blind FFT deconvolution is intentionally avoided here: an incorrect PSF can
    turn scale, water and oil textures into confident false defects.
    """
    score = laplacian_variance(image)
    if score >= threshold:
        return DeblurResult(image=image, applied=False, blur_score=score, strength=0.0)

    gray = to_gray(image)
    original_tenengrad = max(tenengrad(gray), 1e-6)
    candidates: list[tuple[float, float, np.ndarray]] = []
    for sigma, amount in ((0.8, 0.45), (1.2, 0.70), (1.6, 0.90)):
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
        enhanced = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
        halo = halo_fraction(gray, enhanced)
        if halo <= max_halo_fraction:
            quality = tenengrad(enhanced) / original_tenengrad - 2.0 * halo
            candidates.append((quality, amount, enhanced))

    if not candidates:
        return DeblurResult(image=image, applied=False, blur_score=score, strength=0.0)

    _, amount, best = max(candidates, key=lambda item: item[0])
    if image.ndim == 3:
        best = cv2.cvtColor(best, cv2.COLOR_GRAY2BGR)
    return DeblurResult(image=best, applied=True, blur_score=score, strength=amount)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, grid_size: int = 8) -> np.ndarray:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    enhanced = clahe.apply(gray)
    if image.ndim == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced
