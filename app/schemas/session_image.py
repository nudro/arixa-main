"""Schemas for stored session images and upload responses."""

from pydantic import BaseModel, ConfigDict

from app.db.models import ImageAngle


class SessionImageRead(BaseModel):
    """One image row as returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    capture_session_id: int
    angle: ImageAngle
    file_path: str
    width: int
    height: int
    usable: bool
    quality_score: float | None
    landmark_status: str | None


class SessionImageUploadResponse(BaseModel):
    """Response after a successful multipart image upload."""

    session_image_id: int
    file_path: str
    angle: ImageAngle
    width: int
    height: int
    usable: bool
    quality_score: float | None
    landmark_status: str | None
