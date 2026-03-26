"""Tests for deterministic registration series processing."""

from app.schemas.registration import RegistrationSeriesRequestSchema
from app.services.registration_service import register_image_series_from_samples


def test_series_single_image_accepted() -> None:
    """Series endpoint logic accepts a single selected image."""
    request = RegistrationSeriesRequestSchema(
        subject_name="bella",
        image_names=["bella_01.png"],
        start_timestamp="2026-01-01T00:00:00Z",
        step_seconds=60,
    )
    result = register_image_series_from_samples(request)
    assert result.summary.total_images == 1
    assert result.series[0].image_name == "bella_01.png"


def test_series_timestamps_increase_sequentially() -> None:
    """Synthetic timestamps increase by step_seconds in deterministic order."""
    request = RegistrationSeriesRequestSchema(
        subject_name="bella",
        image_names=["bella_01.png", "bella_02.png"],
        start_timestamp="2026-01-01T00:00:00Z",
        step_seconds=30,
    )
    result = register_image_series_from_samples(request)
    assert result.series[0].synthetic_timestamp == "2026-01-01T00:00:00Z"
    assert result.series[1].synthetic_timestamp == "2026-01-01T00:00:30Z"
