"""Focused deterministic tests for longitudinal helpers."""

from app.vision.longitudinal import per_region_trend_slopes


def test_per_region_trend_slope_positive_for_increasing_series() -> None:
    """Increasing values produce a positive trend slope."""
    slopes = per_region_trend_slopes({"cheeks": [1.0, 2.0, 3.0, 4.0]})
    assert slopes["cheeks"] > 0.0
