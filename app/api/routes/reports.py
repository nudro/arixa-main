"""Report routes (placeholder; report generation not implemented yet)."""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def list_reports() -> dict[str, list]:
    """Placeholder list endpoint until report services exist."""
    return {"items": []}
