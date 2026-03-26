"""Create and query capture sessions."""

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.db.models import CaptureSession
from app.schemas.capture_session import CaptureSessionCreate


def create_session(db: Session, data: CaptureSessionCreate) -> CaptureSession:
    """Persist a new capture session from the create schema."""
    row = CaptureSession(
        notes=data.notes,
        analysis_mode=data.analysis_mode,
        session_grade=data.session_grade,
        angle_set_completed=list(data.angle_set_completed),
        cycle_phase=data.cycle_phase,
        stress_tag=data.stress_tag,
        sleep_tag=data.sleep_tag,
        caffeine_tag=data.caffeine_tag,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def get_session(db: Session, session_id: int) -> CaptureSession | None:
    """Return the session by primary key, or None if missing."""
    return db.get(CaptureSession, session_id)


def list_sessions(db: Session) -> list[CaptureSession]:
    """Return all sessions newest-first."""
    stmt = select(CaptureSession).order_by(desc(CaptureSession.created_at))
    return list(db.scalars(stmt))
