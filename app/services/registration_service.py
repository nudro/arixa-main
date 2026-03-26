"""SessionImage registration pipeline orchestration service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import SessionImage
from app.schemas.registration import (
    AlignmentTransformSchema,
    QualityMetricsSchema,
    RegistrationResultSchema,
    RegistrationSeriesRequestSchema,
    RegistrationSeriesResultSchema,
    RegistrationSeriesSummarySchema,
    SeriesImageResultSchema,
)
from app.utils.quality_checks import run_quality_checks
from app.vision.face_alignment import classify_angle, estimate_alignment_transform
from app.vision.mediapipe_landmarks import extract_face_landmarks
from app.vision.roi_regions import build_roi_polygons


def register_session_image(db: Session, session_image_id: int) -> RegistrationResultSchema:
    """Run deterministic registration for one stored SessionImage."""
    row = db.get(SessionImage, session_image_id)
    if row is None:
        raise ValueError("session_image_not_found")

    full_path = _resolve_image_path(row.file_path)
    result = _register_local_image_path(full_path, session_image_id=row.id)
    _persist_registration(db, row, result)
    return result


def register_image_series_from_samples(
    request: RegistrationSeriesRequestSchema,
) -> RegistrationSeriesResultSchema:
    """Run deterministic registration across 1..N sample images with synthetic time."""
    sample_dir = _resolve_sample_dir(request.subject_name)
    image_paths = _collect_series_images(sample_dir, request.image_names)
    if not image_paths:
        raise ValueError("no_images_found")

    start_dt = _parse_timestamp(request.start_timestamp)
    series: list[SeriesImageResultSchema] = []
    angle_distribution: dict[str, int] = {}
    usable_images = 0
    warning_count = 0

    for idx, image_path in enumerate(image_paths):
        ts = start_dt + timedelta(seconds=request.step_seconds * idx)
        result = _register_local_image_path(image_path, session_image_id=0)
        if result.image_usable:
            usable_images += 1
        warning_count += len(result.warnings)
        angle_distribution[result.estimated_angle_class] = (
            angle_distribution.get(result.estimated_angle_class, 0) + 1
        )
        series.append(
            SeriesImageResultSchema(
                image_name=image_path.name,
                synthetic_timestamp=ts.isoformat().replace("+00:00", "Z"),
                result=result,
            )
        )

    summary = RegistrationSeriesSummarySchema(
        total_images=len(series),
        usable_images=usable_images,
        warning_count=warning_count,
        angle_distribution=angle_distribution,
    )
    return RegistrationSeriesResultSchema(
        subject_name=request.subject_name,
        series=series,
        summary=summary,
    )


def _register_local_image_path(image_path: Path, session_image_id: int) -> RegistrationResultSchema:
    """Run registration for a local image path and return structured result."""
    image_bgr, landmark_result = extract_face_landmarks(str(image_path))

    if not landmark_result.face_detected:
        return RegistrationResultSchema(
            session_image_id=session_image_id,
            face_detected=False,
            landmarks_count=0,
            image_usable=False,
            estimated_angle_class="unknown",
            alignment_transform=AlignmentTransformSchema(
                center_x=0.5,
                center_y=0.5,
                roll_degrees=0.0,
                scale=1.0,
            ),
            roi_polygons={},
            quality_metrics=QualityMetricsSchema(
                blur_score=0.0,
                face_size_ratio=0.0,
                out_of_frame=False,
                lighting_score=0.0,
            ),
            warnings=landmark_result.warnings,
        )

    quality = run_quality_checks(image_bgr, landmark_result.bbox_px)
    estimated_angle = classify_angle(landmark_result.landmarks)
    align = estimate_alignment_transform(landmark_result.landmarks)
    rois = build_roi_polygons(landmark_result.landmarks)

    warnings = [*landmark_result.warnings, *quality.warnings]
    return RegistrationResultSchema(
        session_image_id=session_image_id,
        face_detected=True,
        landmarks_count=landmark_result.landmarks_count,
        image_usable=quality.image_usable,
        estimated_angle_class=estimated_angle,
        alignment_transform=AlignmentTransformSchema(**align),
        roi_polygons=rois,
        quality_metrics=QualityMetricsSchema(
            blur_score=quality.blur_score,
            face_size_ratio=quality.face_size_ratio,
            out_of_frame=quality.out_of_frame,
            lighting_score=quality.lighting_score,
        ),
        warnings=warnings,
    )


def _persist_registration(db: Session, row: SessionImage, result: RegistrationResultSchema) -> None:
    """Store compact registration state on SessionImage for deterministic retrieval."""
    row.usable = result.image_usable
    row.quality_score = result.quality_metrics.blur_score
    row.landmark_status = "ok" if result.face_detected else "face_not_detected"
    row.registration_output = result.model_dump()
    db.add(row)
    db.commit()
    db.refresh(row)


def _resolve_image_path(relative_file_path: str) -> Path:
    """Resolve and validate internal upload path for a SessionImage row."""
    upload_root = Path(get_settings().upload_root).resolve()
    target = (upload_root / relative_file_path).resolve()
    if upload_root not in target.parents and target != upload_root:
        raise ValueError("invalid_image_path")
    return target


def _resolve_sample_dir(subject_name: str) -> Path:
    """Resolve samples directory by subject name with sample/samples fallback."""
    project_root = Path(__file__).resolve().parents[2]
    candidates = [
        project_root / "images" / "samples" / subject_name,
        project_root / "images" / "sample" / subject_name,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    raise ValueError("sample_folder_not_found")


def _collect_series_images(sample_dir: Path, image_names: list[str] | None) -> list[Path]:
    """Return deterministic ordered list of image paths for series processing."""
    if image_names:
        paths = [sample_dir / name for name in image_names]
    else:
        paths = [p for p in sample_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}]
    existing = [p for p in paths if p.exists() and p.is_file()]
    existing.sort(key=lambda p: p.name.lower())
    return existing


def _parse_timestamp(timestamp: str) -> datetime:
    """Parse ISO timestamp with optional Z suffix as UTC."""
    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    return datetime.fromisoformat(timestamp).astimezone(timezone.utc)
