"""FastAPI application factory and ASGI entry surface."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from app.api.router import api_router
from app.db.base import Base
import app.db.models  # noqa: F401 — register ORM metadata before create_all
from app.db.session import engine


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Create database tables on startup and dispose the engine on shutdown."""
    Base.metadata.create_all(bind=engine)
    yield
    engine.dispose()


def create_app() -> FastAPI:
    """Build and return the FastAPI application (use with uvicorn --factory)."""
    app = FastAPI(
        title="Arixa API",
        description="Layer 1 deterministic backend",
        lifespan=lifespan,
    )
    app.include_router(api_router)
    return app
