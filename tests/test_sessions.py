"""Tests for capture session persistence and HTTP routes."""

import io

from fastapi.testclient import TestClient
from PIL import Image

from app.db.base import Base
from app.db.models import CaptureSession, SessionGrade
from app.db.session import SessionLocal, engine
from app.main import create_app


def test_capture_session_model_roundtrip() -> None:
    """ORM: create a CaptureSession and read it back from SQLite."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        row = CaptureSession(
            notes="unit",
            analysis_mode=True,
            session_grade=SessionGrade.analysis_grade,
            angle_set_completed=["front"],
        )
        db.add(row)
        db.commit()
        rid = row.id
        loaded = db.get(CaptureSession, rid)
        assert loaded is not None
        assert loaded.notes == "unit"
        assert loaded.session_grade == SessionGrade.analysis_grade
        assert loaded.angle_set_completed == ["front"]
    finally:
        db.close()


def test_create_and_get_session_via_api() -> None:
    """POST /sessions then GET /sessions/{id} returns the same session."""
    app = create_app()
    client = TestClient(app)
    created = client.post(
        "/sessions",
        json={
            "session_grade": "journal_grade",
            "analysis_mode": False,
            "angle_set_completed": [],
            "notes": None,
        },
    )
    assert created.status_code == 201, created.text
    body = created.json()
    session_id = body["id"]
    fetched = client.get(f"/sessions/{session_id}")
    assert fetched.status_code == 200
    assert fetched.json()["id"] == session_id
    assert fetched.json()["session_grade"] == "journal_grade"


def test_upload_session_image_multipart() -> None:
    """POST /sessions/{id}/images stores a file and returns metadata."""
    buf = io.BytesIO()
    Image.new("RGB", (4, 5), color=(1, 2, 3)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    app = create_app()
    client = TestClient(app)
    sid = client.post(
        "/sessions",
        json={"session_grade": "analysis_grade", "angle_set_completed": []},
    ).json()["id"]

    response = client.post(
        f"/sessions/{sid}/images",
        data={"angle": "front"},
        files={"file": ("test.png", png_bytes, "image/png")},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["session_image_id"] >= 1
    assert data["angle"] == "front"
    assert data["width"] == 4
    assert data["height"] == 5
    assert str(sid) in data["file_path"]
