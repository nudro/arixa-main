"""Deterministic texture feature extraction utilities."""

from __future__ import annotations

import cv2
import numpy as np
try:
    import pywt
except Exception:  # pragma: no cover - optional fallback at runtime
    pywt = None

try:
    from skimage.feature import local_binary_pattern
    from skimage.filters import frangi, gabor
except Exception:  # pragma: no cover - optional fallback at runtime
    local_binary_pattern = None
    frangi = None
    gabor = None

try:
    import mahotas
except Exception:  # pragma: no cover - optional fallback at runtime
    mahotas = None


def laplacian_variance(gray: np.ndarray) -> float:
    """Return variance of Laplacian as a local texture/sharpness proxy."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def lbp_histogram(gray: np.ndarray, radius: int = 1, points: int = 8) -> np.ndarray:
    """Compute normalized LBP histogram."""
    if local_binary_pattern is None:
        return np.zeros(points + 2, dtype=np.float32)
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    bins = int(points + 2)
    hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, bins), density=True)
    return hist.astype(np.float32)


def gabor_summary(gray: np.ndarray, frequencies: tuple[float, ...] = (0.1, 0.2, 0.3)) -> dict[str, float]:
    """Compute summary statistics over a small Gabor filter bank."""
    if gabor is None:
        return {"gabor_mean": 0.0, "gabor_std": 0.0}
    vals: list[float] = []
    for freq in frequencies:
        real, _imag = gabor(gray, frequency=freq)
        vals.append(float(np.mean(real)))
        vals.append(float(np.std(real)))
    return {"gabor_mean": float(np.mean(vals)), "gabor_std": float(np.std(vals))}


def wavelet_texture_summary(gray: np.ndarray, wavelet: str = "db2", level: int = 2) -> dict[str, float]:
    """Return multi-scale wavelet energy summaries."""
    if pywt is None:
        return {"wavelet_energy_mean": 0.0, "wavelet_energy_std": 0.0}
    coeffs = pywt.wavedec2(gray.astype(np.float32), wavelet=wavelet, level=level)
    energies: list[float] = []
    for c in coeffs[1:]:
        for band in c:
            energies.append(float(np.mean(np.abs(band))))
    return {"wavelet_energy_mean": float(np.mean(energies)), "wavelet_energy_std": float(np.std(energies))}


def haralick_summary(gray: np.ndarray) -> dict[str, float]:
    """Compute Haralick features using mahotas when available."""
    if mahotas is None:
        return {"haralick_mean": 0.0, "haralick_std": 0.0}
    feats = mahotas.features.haralick(gray, ignore_zeros=False)
    return {"haralick_mean": float(np.mean(feats)), "haralick_std": float(np.std(feats))}


def optional_hessian_response(gray: np.ndarray, enable_hessian: bool = False) -> float:
    """Optional Frangi/Hessian-like response summary behind feature flag."""
    if not enable_hessian:
        return 0.0
    if frangi is None:
        return 0.0
    resp = frangi(gray.astype(np.float32) / 255.0)
    return float(np.mean(resp))
