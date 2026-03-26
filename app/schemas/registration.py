"""Schemas for SessionImage registration pipeline output."""

from __future__ import annotations

from pydantic import BaseModel


class AlignmentTransformSchema(BaseModel):
    """Alignment transform metadata for downstream face normalization."""

    center_x: float
    center_y: float
    roll_degrees: float
    scale: float


class QualityMetricsSchema(BaseModel):
    """Deterministic quality metrics used for image usability checks."""

    blur_score: float
    face_size_ratio: float
    out_of_frame: bool
    lighting_score: float


class RegistrationResultSchema(BaseModel):
    """Structured deterministic registration output for one SessionImage."""

    session_image_id: int
    face_detected: bool
    landmarks_count: int
    image_usable: bool
    estimated_angle_class: str
    alignment_transform: AlignmentTransformSchema
    roi_polygons: dict[str, list[tuple[float, float]]]
    quality_metrics: QualityMetricsSchema
    warnings: list[str]


class RegistrationSeriesRequestSchema(BaseModel):
    """Request payload for multi-image series processing from samples folder."""

    subject_name: str
    image_names: list[str] | None = None
    start_timestamp: str = "2026-01-01T00:00:00Z"
    step_seconds: int = 60


class SeriesImageResultSchema(BaseModel):
    """Per-image result in a synthetic-timestamped registration series."""

    image_name: str
    synthetic_timestamp: str
    result: RegistrationResultSchema


class RegistrationSeriesSummarySchema(BaseModel):
    """Aggregate summary over a processed image series."""

    total_images: int
    usable_images: int
    warning_count: int
    angle_distribution: dict[str, int]


class RegistrationSeriesResultSchema(BaseModel):
    """Complete output for multi-image registration over sample images."""

    subject_name: str
    series: list[SeriesImageResultSchema]
    summary: RegistrationSeriesSummarySchema
