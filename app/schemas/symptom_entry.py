"""User-reported symptom fields linked to a capture session."""

from pydantic import BaseModel, ConfigDict


class SymptomEntryCreate(BaseModel):
    """Partial symptom payload; omitted fields are left unchanged on upsert."""

    bloating: int | None = None
    hot_flashes: int | None = None
    sleep_quality: int | None = None
    mood: int | None = None
    cycle_irregularity: int | None = None
    skin_flare: int | None = None
    caffeine: str | None = None
    exercise: str | None = None
    supplements: str | None = None
    notes: str | None = None


class SymptomEntryRead(BaseModel):
    """Persisted symptom snapshot for one session."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    capture_session_id: int
    bloating: int | None
    hot_flashes: int | None
    sleep_quality: int | None
    mood: int | None
    cycle_irregularity: int | None
    skin_flare: int | None
    caffeine: str | None
    exercise: str | None
    supplements: str | None
    notes: str | None
