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


def test_register_session_image_route_returns_structured_payload() -> None:
    """POST register endpoint processes one SessionImage by id."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    app = create_app()
    client = TestClient(app)
    sid = client.post(
        "/sessions",
        json={"session_grade": "analysis_grade", "angle_set_completed": []},
    ).json()["id"]
    upload = client.post(
        f"/sessions/{sid}/images",
        data={"angle": "front"},
        files={"file": ("register.png", png_bytes, "image/png")},
    )
    session_image_id = upload.json()["session_image_id"]

    response = client.post(f"/sessions/images/{session_image_id}/register")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["session_image_id"] == session_image_id
    assert "face_detected" in payload
    assert "alignment_transform" in payload
    assert "roi_polygons" in payload
    assert "warnings" in payload


def test_register_series_route_returns_aggregate_output() -> None:
    """POST series registration returns ordered per-image and summary payload."""
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/sessions/images/register-series",
        json={
            "subject_name": "bella",
            "image_names": ["bella_01.png", "bella_02.png"],
            "start_timestamp": "2026-01-01T00:00:00Z",
            "step_seconds": 60,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["subject_name"] == "bella"
    assert payload["summary"]["total_images"] == 2
    assert payload["series"][0]["synthetic_timestamp"] == "2026-01-01T00:00:00Z"
    assert payload["series"][1]["synthetic_timestamp"] == "2026-01-01T00:01:00Z"


def test_analyze_series_route_returns_longitudinal_output() -> None:
    """POST analyze-series returns per-image results plus longitudinal summary."""
    app = create_app()
    client = TestClient(app)
    response = client.post(
        "/sessions/images/analyze-series",
        json={
            "subject_name": "bella",
            "image_names": ["bella_01.png", "bella_02.png"],
            "start_timestamp": "2026-01-01T00:00:00Z",
            "step_seconds": 60,
            "enable_hessian": False,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["subject_name"] == "bella"
    assert len(payload["results"]) == 2
    assert "longitudinal" in payload
    assert "trend_slopes" in payload["longitudinal"]
