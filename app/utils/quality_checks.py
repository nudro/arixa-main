"""Deterministic image quality checks for static registration inputs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class QualityReport:
    """Quality metrics and pass/fail signal for registration readiness."""

    blur_score: float
    face_size_ratio: float
    out_of_frame: bool
    lighting_score: float
    image_usable: bool
    warnings: list[str]


def estimate_blur_score(image_bgr: np.ndarray) -> float:
    """Estimate sharpness using the variance of Laplacian."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_size_ratio(image_shape: tuple[int, int, int], bbox_px: tuple[int, int, int, int]) -> float:
    """Compute face-box area divided by image area."""
    img_h, img_w = image_shape[:2]
    x, y, w, h = bbox_px
    face_area = max(0, w) * max(0, h)
    image_area = max(1, img_h * img_w)
    return float(face_area / image_area)


def is_out_of_frame(image_shape: tuple[int, int, int], bbox_px: tuple[int, int, int, int], margin: int = 2) -> bool:
    """Return True when face bounding box touches image edge within a margin."""
    img_h, img_w = image_shape[:2]
    x, y, w, h = bbox_px
    return x <= margin or y <= margin or (x + w) >= (img_w - margin) or (y + h) >= (img_h - margin)


def lighting_consistency_score(image_bgr: np.ndarray) -> float:
    """Return a simple [0, 1] score based on luma spread and clipping."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    spread = max(0.0, min(1.0, (p95 - p5) / 255.0))
    clipped = float(((gray <= 5) | (gray >= 250)).mean())
    return max(0.0, min(1.0, spread * (1.0 - clipped)))


def run_quality_checks(image_bgr: np.ndarray, bbox_px: tuple[int, int, int, int]) -> QualityReport:
    """Run all deterministic quality checks and aggregate warnings."""
    blur = estimate_blur_score(image_bgr)
    ratio = face_size_ratio(image_bgr.shape, bbox_px)
    out_frame = is_out_of_frame(image_bgr.shape, bbox_px)
    light = lighting_consistency_score(image_bgr)

    warnings: list[str] = []
    if blur < 30.0:
        warnings.append("blur_too_high")
    if ratio < 0.08:
        warnings.append("face_too_small")
    if out_frame:
        warnings.append("face_out_of_frame")
    if light < 0.25:
        warnings.append("lighting_inconsistent")

    return QualityReport(
        blur_score=blur,
        face_size_ratio=ratio,
        out_of_frame=out_frame,
        lighting_score=light,
        image_usable=len(warnings) == 0,
        warnings=warnings,
    )
