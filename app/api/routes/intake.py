"""Intake routes (placeholder; structured capture not implemented yet)."""

from fastapi import APIRouter

router = APIRouter()


@router.get("")
def list_intake() -> dict[str, list]:
    """Placeholder list endpoint until intake schemas and persistence exist."""
    return {"items": []}
