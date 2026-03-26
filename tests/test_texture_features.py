"""Focused deterministic tests for texture features."""

import cv2
import numpy as np

from app.vision.texture_features import laplacian_variance


def test_laplacian_variance_sharp_gt_blur() -> None:
    """Sharp synthetic checkerboard has higher Laplacian variance than blurred."""
    checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(checker, (7, 7), 0)
    assert laplacian_variance(checker) > laplacian_variance(blur)
