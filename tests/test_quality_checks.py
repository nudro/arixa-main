"""Unit tests for deterministic image quality checks."""

import numpy as np

from app.utils.quality_checks import face_size_ratio, lighting_consistency_score, run_quality_checks


def test_face_size_ratio_is_positive() -> None:
    """Face ratio reflects bbox area over image area."""
    ratio = face_size_ratio((100, 100, 3), (10, 10, 50, 50))
    assert ratio == 0.25


def test_lighting_consistency_score_range() -> None:
    """Lighting score remains within expected [0, 1] range."""
    img = np.full((20, 20, 3), fill_value=128, dtype=np.uint8)
    score = lighting_consistency_score(img)
    assert 0.0 <= score <= 1.0


def test_run_quality_checks_returns_metrics() -> None:
    """Aggregated checks produce stable output fields."""
    img = np.full((80, 80, 3), fill_value=120, dtype=np.uint8)
    report = run_quality_checks(img, (10, 10, 40, 40))
    assert isinstance(report.blur_score, float)
    assert isinstance(report.face_size_ratio, float)
    assert isinstance(report.image_usable, bool)
