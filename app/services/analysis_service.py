"""Deterministic single-image and series facial analysis orchestration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from sqlalchemy.orm import Session

from app.db.models import SessionImage
from app.schemas.analysis import (
    AnalysisSeriesRequestSchema,
    AnalysisSeriesResultSchema,
    ColorMetricSchema,
    LongitudinalSummarySchema,
    RegionMetricSchema,
    SingleImageAnalysisResultSchema,
    TextureMetricSchema,
)
from app.schemas.registration import RegistrationSeriesRequestSchema
from app.services import registration_service
from app.vision.color_metrics import (
    brownness_darkness_distribution,
    summarize_region_color,
    to_hsv,
    to_lab,
    to_ycbcr,
)
from app.vision.longitudinal import aligned_difference_map, per_region_trend_slopes
from app.vision.segmentation import candidate_pigment_mask, compute_region_metrics, polygon_to_mask
from app.vision.texture_features import (
    gabor_summary,
    haralick_summary,
    laplacian_variance,
    optional_hessian_response,
    wavelet_texture_summary,
)


def analyze_single_session_image(
    db: Session,
    session_image_id: int,
    enable_hessian: bool = False,
) -> SingleImageAnalysisResultSchema:
    """Analyze one stored SessionImage using deterministic analysis modules."""
    row = db.get(SessionImage, session_image_id)
    if row is None:
        raise ValueError("session_image_not_found")
    registration = registration_service.register_session_image(db, session_image_id)
    full_path = registration_service._resolve_image_path(row.file_path)
    image_bgr = _read_image(full_path)
    region_metrics = _compute_region_metrics(image_bgr, registration.roi_polygons, enable_hessian)
    return SingleImageAnalysisResultSchema(
        image_id=str(session_image_id),
        registration=registration,
        region_metrics=region_metrics,
        warnings=registration.warnings,
    )


def analyze_image_series_from_samples(request: AnalysisSeriesRequestSchema) -> AnalysisSeriesResultSchema:
    """Analyze 1..N sample images in order with synthetic timestamps and longitudinal summary."""
    reg_req = RegistrationSeriesRequestSchema(
        subject_name=request.subject_name,
        image_names=request.image_names,
        start_timestamp=request.start_timestamp,
        step_seconds=request.step_seconds,
    )
    reg_series = registration_service.register_image_series_from_samples(reg_req)
    sample_dir = registration_service._resolve_sample_dir(request.subject_name)

    results: list[SingleImageAnalysisResultSchema] = []
    darkness_series: dict[str, list[float]] = {}
    gray_stack: list[np.ndarray] = []

    for item in reg_series.series:
        image_path = sample_dir / item.image_name
        image_bgr = _read_image(image_path)
        region_metrics = _compute_region_metrics(
            image_bgr,
            item.result.roi_polygons,
            request.enable_hessian,
        )
        for region_name, metrics in region_metrics.items():
            darkness_series.setdefault(region_name, []).append(metrics.darkness_mean)
        gray_stack.append(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY))
        results.append(
            SingleImageAnalysisResultSchema(
                image_id=item.image_name,
                synthetic_timestamp=item.synthetic_timestamp,
                registration=item.result,
                region_metrics=region_metrics,
                warnings=item.result.warnings,
            )
        )

    mean_abs_change = _mean_abs_change(gray_stack)
    trends = per_region_trend_slopes(darkness_series)
    return AnalysisSeriesResultSchema(
        subject_name=request.subject_name,
        results=results,
        longitudinal=LongitudinalSummarySchema(
            trend_slopes=trends,
            mean_abs_change=mean_abs_change,
        ),
    )


def _compute_region_metrics(
    image_bgr: np.ndarray,
    roi_polygons: dict[str, list[tuple[float, float]]],
    enable_hessian: bool,
) -> dict[str, RegionMetricSchema]:
    """Compute per-region deterministic color/texture/segmentation metrics."""
    lab = to_lab(image_bgr)
    hsv = to_hsv(image_bgr)
    ycbcr = to_ycbcr(image_bgr)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    out: dict[str, RegionMetricSchema] = {}
    for region_name, points in roi_polygons.items():
        mask = polygon_to_mask(gray.shape, points)
        candidate = candidate_pigment_mask(lab, mask)
        base = compute_region_metrics(lab, mask, candidate)
        lab_summary = summarize_region_color(lab, mask)
        hsv_summary = summarize_region_color(hsv, mask)
        y_summary = summarize_region_color(ycbcr, mask)
        brown_dark = brownness_darkness_distribution(lab, mask)
        region_gray = gray.copy()
        region_gray[mask == 0] = 0
        gabor = gabor_summary(region_gray)
        wave = wavelet_texture_summary(region_gray)
        har = haralick_summary(region_gray)
        texture = TextureMetricSchema(
            laplacian_variance=laplacian_variance(region_gray),
            gabor_mean=gabor["gabor_mean"],
            wavelet_energy_mean=wave["wavelet_energy_mean"],
            haralick_mean=har["haralick_mean"],
            hessian_response=optional_hessian_response(region_gray, enable_hessian=enable_hessian),
        )
        color = ColorMetricSchema(
            lab_mean_l=lab_summary["mean_c0"],
            hsv_mean_v=hsv_summary["mean_c2"],
            ycbcr_mean_y=y_summary["mean_c0"],
            brownness_mean=brown_dark["brownness_mean"],
            darkness_mean=brown_dark["darkness_mean"],
        )
        out[region_name] = RegionMetricSchema(
            affected_percent=base["affected_percent"],
            pigment_asymmetry=base["pigment_asymmetry"],
            darkness_mean=base["darkness_mean"],
            darkness_std=base["darkness_std"],
            color=color,
            texture=texture,
        )
    return out


def _mean_abs_change(gray_series: list[np.ndarray]) -> float:
    """Compute mean absolute session-to-session difference across a series."""
    if len(gray_series) < 2:
        return 0.0
    diffs: list[float] = []
    for idx in range(1, len(gray_series)):
        diff = aligned_difference_map(gray_series[idx], gray_series[idx - 1])
        diffs.append(float(np.mean(diff)))
    return float(np.mean(diffs))


def _read_image(path: Path) -> np.ndarray:
    """Read BGR image from path or raise deterministic validation error."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError("image_not_found_or_unreadable")
    return img
