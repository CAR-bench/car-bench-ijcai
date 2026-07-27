from typing import Any, Literal
from pydantic import BaseModel, HttpUrl


class CompetitionRunContext(BaseModel):
    """Organizer-controlled context for a persisted hidden evaluation."""

    schema_version: Literal[2] = 2
    run_id: str
    team_id: str
    team_name: str
    track: Literal["track_1", "track_2"]
    run_dir: str = "/run-state"
    dataset_fingerprint: str
    base_seed: int = 10


class EvalRequest(BaseModel):
    agent_under_test: HttpUrl
    config: dict[str, Any]
    run_context: CompetitionRunContext | None = None
