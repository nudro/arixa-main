"""Create or update the single symptom row for a capture session."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CaptureSession, SymptomEntry
from app.schemas.symptom_entry import SymptomEntryCreate


def upsert_symptoms_for_session(
    db: Session,
    session_id: int,
    data: SymptomEntryCreate,
) -> SymptomEntry:
    """Insert or patch SymptomEntry for ``session_id`` (fields in body only).

    Raises:
        ValueError: ``capture_session_not_found`` if the session does not exist.
    """
    if db.get(CaptureSession, session_id) is None:
        raise ValueError("capture_session_not_found")

    payload = data.model_dump(exclude_unset=True)
    stmt = select(SymptomEntry).where(SymptomEntry.capture_session_id == session_id)
    existing = db.scalars(stmt).one_or_none()

    if existing is not None:
        for key, value in payload.items():
            setattr(existing, key, value)
        db.commit()
        db.refresh(existing)
        return existing

    row = SymptomEntry(capture_session_id=session_id, **payload)
    db.add(row)
    db.commit()
    db.refresh(row)
    return row
