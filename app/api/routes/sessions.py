"""Capture session CRUD, symptom upsert, and image upload routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.db.models import ImageAngle
from app.db.session import get_db
from app.schemas.capture_session import (
    CaptureSessionCreate,
    CaptureSessionListResponse,
    CaptureSessionRead,
)
from app.schemas.analysis import AnalysisSeriesRequestSchema, AnalysisSeriesResultSchema, SingleImageAnalysisResultSchema
from app.schemas.registration import RegistrationResultSchema
from app.schemas.registration import RegistrationSeriesRequestSchema, RegistrationSeriesResultSchema
from app.schemas.session_image import SessionImageUploadResponse
from app.schemas.symptom_entry import SymptomEntryCreate, SymptomEntryRead
from app.services import capture_session as capture_session_service
from app.services import analysis_service
from app.services import registration_service
from app.services import session_image as session_image_service
from app.services import symptom_entry as symptom_entry_service

router = APIRouter()


@router.post(
    "",
    response_model=CaptureSessionRead,
    status_code=status.HTTP_201_CREATED,
)
def create_session(
    body: CaptureSessionCreate,
    db: Session = Depends(get_db),
) -> CaptureSessionRead:
    """Create a new capture session."""
    row = capture_session_service.create_session(db, body)
    return CaptureSessionRead.model_validate(row)


@router.get("", response_model=CaptureSessionListResponse)
def list_sessions(db: Session = Depends(get_db)) -> CaptureSessionListResponse:
    """List capture sessions (newest first)."""
    rows = capture_session_service.list_sessions(db)
    return CaptureSessionListResponse(
        items=[CaptureSessionRead.model_validate(r) for r in rows],
    )


@router.get("/{session_id}", response_model=CaptureSessionRead)
def get_session(session_id: int, db: Session = Depends(get_db)) -> CaptureSessionRead:
    """Return one capture session by id."""
    row = capture_session_service.get_session(db, session_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return CaptureSessionRead.model_validate(row)


@router.post("/{session_id}/symptoms", response_model=SymptomEntryRead)
def add_or_update_symptoms(
    session_id: int,
    body: SymptomEntryCreate,
    db: Session = Depends(get_db),
) -> SymptomEntryRead:
    """Create or update the symptom snapshot for this session."""
    try:
        row = symptom_entry_service.upsert_symptoms_for_session(db, session_id, body)
    except ValueError as exc:
        if str(exc) == "capture_session_not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            ) from exc
        raise
    return SymptomEntryRead.model_validate(row)


@router.post("/{session_id}/images", response_model=SessionImageUploadResponse)
async def upload_session_image(
    session_id: int,
    angle: Annotated[ImageAngle, Form()],
    file: Annotated[UploadFile, File()],
    db: Session = Depends(get_db),
) -> SessionImageUploadResponse:
    """Multipart upload: store file under upload_root/session_id/ and record metadata."""
    raw = await file.read()
    try:
        row = session_image_service.save_session_image(
            db,
            session_id,
            angle,
            raw,
            file.filename,
        )
    except ValueError as exc:
        msg = str(exc)
        if msg == "capture_session_not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            ) from exc
        if msg == "invalid_image":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not read image dimensions",
            ) from exc
        raise
    return SessionImageUploadResponse(
        session_image_id=row.id,
        file_path=row.file_path,
        angle=row.angle,
        width=row.width,
        height=row.height,
        usable=row.usable,
        quality_score=row.quality_score,
        landmark_status=row.landmark_status,
    )


@router.post("/images/{session_image_id}/register", response_model=RegistrationResultSchema)
def register_image(
    session_image_id: int,
    db: Session = Depends(get_db),
) -> RegistrationResultSchema:
    """Run deterministic registration for one uploaded SessionImage."""
    try:
        return registration_service.register_session_image(db, session_image_id)
    except ValueError as exc:
        msg = str(exc)
        if msg == "session_image_not_found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session image not found") from exc
        if msg == "invalid_image_path":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid image path") from exc
        raise


@router.post("/images/register-series", response_model=RegistrationSeriesResultSchema)
def register_image_series(
    body: RegistrationSeriesRequestSchema,
) -> RegistrationSeriesResultSchema:
    """Run deterministic registration over one or more sample images in sequence."""
    try:
        return registration_service.register_image_series_from_samples(body)
    except ValueError as exc:
        msg = str(exc)
        if msg == "sample_folder_not_found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample folder not found") from exc
        if msg == "no_images_found":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No images found to process") from exc
        raise


@router.post("/images/{session_image_id}/analyze", response_model=SingleImageAnalysisResultSchema)
def analyze_single_image(
    session_image_id: int,
    enable_hessian: bool = False,
    db: Session = Depends(get_db),
) -> SingleImageAnalysisResultSchema:
    """Run deterministic analysis for one stored SessionImage."""
    try:
        return analysis_service.analyze_single_session_image(
            db,
            session_image_id=session_image_id,
            enable_hessian=enable_hessian,
        )
    except ValueError as exc:
        msg = str(exc)
        if msg == "session_image_not_found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session image not found") from exc
        if msg == "image_not_found_or_unreadable":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file unreadable") from exc
        raise


@router.post("/images/analyze-series", response_model=AnalysisSeriesResultSchema)
def analyze_image_series(body: AnalysisSeriesRequestSchema) -> AnalysisSeriesResultSchema:
    """Run deterministic analysis for a series of sample images."""
    try:
        return analysis_service.analyze_image_series_from_samples(body)
    except ValueError as exc:
        msg = str(exc)
        if msg == "sample_folder_not_found":
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Sample folder not found") from exc
        if msg == "no_images_found":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No images found to process") from exc
        if msg == "image_not_found_or_unreadable":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image file unreadable") from exc
        raise
