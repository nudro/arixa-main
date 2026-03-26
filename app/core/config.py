"""Application settings loaded from environment variables with sensible defaults."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the API and database."""

    model_config = SettingsConfigDict(
        env_prefix="ARIXA_",
        env_file=".env",
        extra="ignore",
    )

    #: SQLAlchemy database URL; default is a local SQLite file in the process cwd.
    database_url: str = "sqlite:///./arixa.db"

    #: Root directory for stored uploads; each session gets a subdirectory by id.
    upload_root: str = "data/uploads"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton per process)."""
    return Settings()
