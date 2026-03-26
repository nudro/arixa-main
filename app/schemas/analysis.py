"""Schemas for deterministic single-image and multi-image analysis outputs."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.registration import RegistrationResultSchema
from app.vision.constants import DEFAULT_START_TIMESTAMP, DEFAULT_STEP_SECONDS


class AnalysisSeriesRequestSchema(BaseModel):
    """Request payload for sample-folder series analysis."""

    subject_name: str
    image_names: list[str] | None = None
    start_timestamp: str = DEFAULT_START_TIMESTAMP
    step_seconds: int = DEFAULT_STEP_SECONDS
    enable_hessian: bool = False


class ColorMetricSchema(BaseModel):
    """Color-space summaries for one ROI."""

    lab_mean_l: float
    hsv_mean_v: float
    ycbcr_mean_y: float
    brownness_mean: float
    darkness_mean: float


class TextureMetricSchema(BaseModel):
    """Texture summary values for one ROI."""

    laplacian_variance: float
    gabor_mean: float
    wavelet_energy_mean: float
    haralick_mean: float
    hessian_response: float


class RegionMetricSchema(BaseModel):
    """Region-level deterministic metrics."""

    affected_percent: float
    pigment_asymmetry: float
    darkness_mean: float
    darkness_std: float
    color: ColorMetricSchema
    texture: TextureMetricSchema


class SingleImageAnalysisResultSchema(BaseModel):
    """Analysis result for a single image/frame."""

    image_id: str
    synthetic_timestamp: str | None = None
    registration: RegistrationResultSchema
    region_metrics: dict[str, RegionMetricSchema]
    warnings: list[str]


class LongitudinalSummarySchema(BaseModel):
    """Longitudinal summary over an analyzed image series."""

    trend_slopes: dict[str, float]
    mean_abs_change: float


class AnalysisSeriesResultSchema(BaseModel):
    """Final series output for one subject set of images."""

    subject_name: str
    results: list[SingleImageAnalysisResultSchema]
    longitudinal: LongitudinalSummarySchema
