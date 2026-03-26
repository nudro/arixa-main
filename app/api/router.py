"""Aggregates all API route modules into a single router."""

from fastapi import APIRouter

from app.api.routes import health, intake, reports, sessions

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["sessions"])
api_router.include_router(intake.router, prefix="/intake", tags=["intake"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
