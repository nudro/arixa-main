"""Liveness and readiness style health endpoint (no database dependency)."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Return a deterministic OK payload for load balancers and smoke tests."""
    return {"status": "ok"}
