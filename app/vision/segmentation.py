"""Deterministic region segmentation and region-level pigment metrics."""

from __future__ import annotations

import cv2
import numpy as np
try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional fallback at runtime
    ndi = None
try:
    from skimage.draw import polygon
    from skimage.segmentation import slic
except Exception:  # pragma: no cover - optional fallback at runtime
    polygon = None
    slic = None
try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover - optional fallback at runtime
    KMeans = None
    GaussianMixture = None

from app.vision.constants import PIGMENT_B_THRESHOLD, PIGMENT_L_THRESHOLD


def polygon_to_mask(shape: tuple[int, int], points: list[tuple[float, float]]) -> np.ndarray:
    """Rasterize normalized polygon points into a binary mask."""
    if not points:
        return np.zeros(shape, dtype=np.uint8)
    h, w = shape
    ys = np.array([p[1] * h for p in points], dtype=np.float32)
    xs = np.array([p[0] * w for p in points], dtype=np.float32)
    if polygon is None:
        return np.zeros(shape, dtype=np.uint8)
    rr, cc = polygon(ys, xs, shape=shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


def candidate_pigment_mask(lab_image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
    """Generate deterministic candidate pigment mask inside an ROI."""
    l = lab_image[..., 0].astype(np.float32)
    b = lab_image[..., 2].astype(np.float32)
    candidate = ((l < PIGMENT_L_THRESHOLD) & (b > PIGMENT_B_THRESHOLD) & (roi_mask.astype(bool))).astype(np.uint8)
    if ndi is not None:
        candidate = ndi.binary_opening(candidate, iterations=1)
        candidate = ndi.binary_closing(candidate, iterations=1)
    else:
        kernel = np.ones((3, 3), dtype=np.uint8)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    return candidate.astype(np.uint8)


def compute_region_metrics(lab_image: np.ndarray, roi_mask: np.ndarray, candidate_mask: np.ndarray) -> dict[str, float]:
    """Compute asymmetry, affected percentage, and darkness summaries."""
    roi_pixels = np.count_nonzero(roi_mask)
    if roi_pixels == 0:
        return {
            "affected_percent": 0.0,
            "pigment_asymmetry": 0.0,
            "darkness_mean": 0.0,
            "darkness_std": 0.0,
        }
    affected = np.count_nonzero(candidate_mask & roi_mask)
    h, w = roi_mask.shape
    mid = w // 2
    left = candidate_mask[:, :mid]
    right = candidate_mask[:, mid:]
    asym = abs(np.count_nonzero(left) - np.count_nonzero(right)) / max(1, np.count_nonzero(left) + np.count_nonzero(right))
    darkness = (255.0 - lab_image[..., 0].astype(np.float32))[roi_mask.astype(bool)]
    return {
        "affected_percent": float(100.0 * affected / roi_pixels),
        "pigment_asymmetry": float(asym),
        "darkness_mean": float(np.mean(darkness)),
        "darkness_std": float(np.std(darkness)),
    }


def kmeans_lab(lab_pixels: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """Optional k-means helper in Lab space."""
    if KMeans is None:
        return np.zeros((lab_pixels.shape[0],), dtype=np.int32)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(lab_pixels)


def gmm_lab(lab_pixels: np.ndarray, n_components: int = 3) -> np.ndarray:
    """Optional Gaussian mixture helper in Lab space."""
    if GaussianMixture is None:
        return np.zeros((lab_pixels.shape[0],), dtype=np.int32)
    return GaussianMixture(n_components=n_components, random_state=0).fit_predict(lab_pixels)


def superpixels_and_cluster(image_bgr: np.ndarray, n_segments: int = 100) -> np.ndarray:
    """Optional superpixel labels helper for deterministic region grouping."""
    if slic is None:
        h, w = image_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.int32)
    return slic(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), n_segments=n_segments, compactness=10.0, start_label=1)


def graph_region_merging_placeholder() -> None:
    """Placeholder for future graph-based region merging implementation."""
    # TODO: implement deterministic graph-based region merging if needed.
    return None
