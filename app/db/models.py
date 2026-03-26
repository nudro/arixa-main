"""SQLAlchemy ORM models for capture sessions, images, and symptom intake."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class SessionGrade(str, enum.Enum):
    """How the capture session was graded for downstream use."""

    analysis_grade = "analysis_grade"
    journal_grade = "journal_grade"


class ImageAngle(str, enum.Enum):
    """Standardized head pose / camera angle for a stored frame."""

    front = "front"
    left_45 = "left_45"
    right_45 = "right_45"
    left_profile = "left_profile"
    right_profile = "right_profile"


class CaptureSession(Base):
    """A single user capture event (facial frames and optional journal context)."""

    __tablename__ = "capture_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    analysis_mode: Mapped[bool] = mapped_column(Boolean, nullable=False, insert_default=False)
    session_grade: Mapped[SessionGrade] = mapped_column(
        SAEnum(SessionGrade, native_enum=False, length=32),
        nullable=False,
    )
    angle_set_completed: Mapped[list[Any]] = mapped_column(JSON, insert_default=list, nullable=False)
    cycle_phase: Mapped[str | None] = mapped_column(String(255), nullable=True)
    stress_tag: Mapped[str | None] = mapped_column(String(255), nullable=True)
    sleep_tag: Mapped[str | None] = mapped_column(String(255), nullable=True)
    caffeine_tag: Mapped[str | None] = mapped_column(String(255), nullable=True)

    session_images: Mapped[list["SessionImage"]] = relationship(
        back_populates="capture_session",
        cascade="all, delete-orphan",
    )
    symptom_entry: Mapped["SymptomEntry | None"] = relationship(
        back_populates="capture_session",
        uselist=False,
        cascade="all, delete-orphan",
    )


class SessionImage(Base):
    """One image file linked to a capture session and camera angle."""

    __tablename__ = "session_images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_session_id: Mapped[int] = mapped_column(
        ForeignKey("capture_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    angle: Mapped[ImageAngle] = mapped_column(
        SAEnum(ImageAngle, native_enum=False, length=32),
        nullable=False,
    )
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    usable: Mapped[bool] = mapped_column(Boolean, nullable=False, insert_default=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    landmark_status: Mapped[str | None] = mapped_column(String(255), nullable=True)

    capture_session: Mapped["CaptureSession"] = relationship(back_populates="session_images")


class SymptomEntry(Base):
    """User-reported symptom snapshot associated with one capture session (at most one row per session)."""

    __tablename__ = "symptom_entries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_session_id: Mapped[int] = mapped_column(
        ForeignKey("capture_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    bloating: Mapped[int | None] = mapped_column(Integer, nullable=True)
    hot_flashes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sleep_quality: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mood: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cycle_irregularity: Mapped[int | None] = mapped_column(Integer, nullable=True)
    skin_flare: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caffeine: Mapped[str | None] = mapped_column(String(255), nullable=True)
    exercise: Mapped[str | None] = mapped_column(String(255), nullable=True)
    supplements: Mapped[str | None] = mapped_column(String(255), nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    capture_session: Mapped["CaptureSession"] = relationship(back_populates="symptom_entry")
