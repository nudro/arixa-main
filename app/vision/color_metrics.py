"""Deterministic color-space conversions and region metrics."""

from __future__ import annotations

import cv2
import numpy as np


def to_lab(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to Lab color space."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)


def to_hsv(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to HSV color space."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)


def to_ycbcr(image_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR image to YCbCr color space."""
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)


def summarize_region_color(image_space: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Return channel summary statistics inside a boolean/uint8 mask."""
    use = mask.astype(bool)
    if not np.any(use):
        return {"mean_c0": 0.0, "mean_c1": 0.0, "mean_c2": 0.0, "std_c0": 0.0, "std_c1": 0.0, "std_c2": 0.0}
    vals = image_space[use]
    return {
        "mean_c0": float(np.mean(vals[:, 0])),
        "mean_c1": float(np.mean(vals[:, 1])),
        "mean_c2": float(np.mean(vals[:, 2])),
        "std_c0": float(np.std(vals[:, 0])),
        "std_c1": float(np.std(vals[:, 1])),
        "std_c2": float(np.std(vals[:, 2])),
    }


def brownness_darkness_distribution(lab_image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute simple brownness/darkness distribution from Lab channels."""
    use = mask.astype(bool)
    if not np.any(use):
        return {"brownness_mean": 0.0, "darkness_mean": 0.0, "darkness_p90": 0.0}
    l = lab_image[..., 0][use].astype(np.float32)
    b = lab_image[..., 2][use].astype(np.float32)
    brownness = 0.5 * b + 0.5 * (255.0 - l)
    darkness = 255.0 - l
    return {
        "brownness_mean": float(np.mean(brownness)),
        "darkness_mean": float(np.mean(darkness)),
        "darkness_p90": float(np.percentile(darkness, 90.0)),
    }
