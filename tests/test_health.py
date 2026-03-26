"""Smoke tests for application bootstrap and core routes."""

from fastapi.testclient import TestClient

from app.main import create_app


def test_health_returns_ok() -> None:
    """GET /health returns 200 and a stable status payload."""
    app = create_app()
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
