"""Persist uploaded images for a capture session."""

import io
import uuid
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import CaptureSession, ImageAngle, SessionImage


def save_session_image(
    db: Session,
    session_id: int,
    angle: ImageAngle,
    raw: bytes,
    original_filename: str | None,
) -> SessionImage:
    """Write bytes to disk under upload_root, validate with Pillow, insert SessionImage.

    Raises:
        ValueError: ``capture_session_not_found`` or ``invalid_image``.
    """
    if db.get(CaptureSession, session_id) is None:
        raise ValueError("capture_session_not_found")

    suffix = _safe_suffix(original_filename)
    settings = get_settings()
    session_dir = Path(settings.upload_root) / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{uuid.uuid4().hex}{suffix}"
    full_path = session_dir / filename
    full_path.write_bytes(raw)

    try:
        with Image.open(io.BytesIO(raw)) as im:
            width, height = im.size
    except Exception:
        full_path.unlink(missing_ok=True)
        raise ValueError("invalid_image") from None

    rel_path = f"{session_id}/{filename}"
    row = SessionImage(
        capture_session_id=session_id,
        angle=angle,
        file_path=rel_path,
        width=width,
        height=height,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def _safe_suffix(original_filename: str | None) -> str:
    """Derive a short file extension from the client filename, or ``.bin``."""
    if not original_filename or "." not in original_filename:
        return ".bin"
    ext = original_filename.rsplit(".", 1)[-1].lower()
    if not ext.isalnum() or len(ext) > 15:
        return ".bin"
    return f".{ext}"
