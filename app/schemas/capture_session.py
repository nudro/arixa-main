"""Request and response shapes for capture sessions."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import SessionGrade


class CaptureSessionCreate(BaseModel):
    """Fields accepted when creating a new capture session."""

    notes: str | None = None
    analysis_mode: bool = False
    session_grade: SessionGrade
    angle_set_completed: list = Field(default_factory=list)
    cycle_phase: str | None = None
    stress_tag: str | None = None
    sleep_tag: str | None = None
    caffeine_tag: str | None = None


class CaptureSessionRead(BaseModel):
    """Full capture session record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    notes: str | None
    analysis_mode: bool
    session_grade: SessionGrade
    angle_set_completed: list
    cycle_phase: str | None
    stress_tag: str | None
    sleep_tag: str | None
    caffeine_tag: str | None


class CaptureSessionListResponse(BaseModel):
    """List wrapper matching the earlier placeholder contract."""

    items: list[CaptureSessionRead]
