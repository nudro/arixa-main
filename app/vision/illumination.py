"""Deterministic illumination normalization utilities."""

from __future__ import annotations

import cv2
import numpy as np
try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional fallback at runtime
    ndi = None


def homomorphic_filter(gray: np.ndarray, sigma: float = 20.0) -> np.ndarray:
    """Apply homomorphic filtering to reduce multiplicative illumination effects."""
    f = gray.astype(np.float32) + 1.0
    log_i = np.log(f)
    if ndi is None:
        low = cv2.GaussianBlur(log_i, (0, 0), sigmaX=sigma, sigmaY=sigma)
    else:
        low = ndi.gaussian_filter(log_i, sigma=sigma)
    high = log_i - low
    out = np.exp(high)
    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    return out.astype(np.uint8)


def retinex_single_scale(gray: np.ndarray, sigma: float = 30.0) -> np.ndarray:
    """Simple single-scale Retinex normalization."""
    src = gray.astype(np.float32) + 1.0
    if ndi is None:
        blur = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma) + 1.0
    else:
        blur = ndi.gaussian_filter(src, sigma=sigma) + 1.0
    ret = np.log(src) - np.log(blur)
    ret = cv2.normalize(ret, None, 0, 255, cv2.NORM_MINMAX)
    return ret.astype(np.uint8)


def shading_correction(gray: np.ndarray, kernel_size: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """Estimate background illumination and return shading-corrected image."""
    background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    corrected = cv2.subtract(gray, background)
    corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
    return corrected.astype(np.uint8), background.astype(np.uint8)
