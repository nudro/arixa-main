"""Focused deterministic tests for color metrics."""

import numpy as np

from app.vision.color_metrics import to_lab


def test_lab_conversion_shape_preserved() -> None:
    """Lab conversion preserves image shape."""
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    lab = to_lab(image)
    assert lab.shape == image.shape
