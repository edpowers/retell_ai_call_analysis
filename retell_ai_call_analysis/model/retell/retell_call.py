import datetime
from typing import Any

import pytz
from pydantic import BaseModel, ConfigDict, Field


class RetellCall(BaseModel):
    """Represents a call from the Retell API with structured data."""

    # Tell Pydantic to ignore extra fields
    model_config = ConfigDict(extra="ignore")

    # Required fields
    call_id: str
    start_timestamp: int
    duration_ms: int
    call_analysis: dict[str, Any]
    transcript_with_tool_calls: list[dict[str, Any]]
    public_log_url: str
    agent_id: str

    # Added fields from the provided list
    call_status: str
    call_type: str
    disconnection_reason: str | None = None
    end_timestamp: int | None = None
    opt_in_signed_url: str | None = None
    opt_out_sensitive_data_storage: bool = False
    recording_url: str | None = None
    retell_llm_dynamic_variables: dict[str, Any] = Field(default_factory=dict)
    transcript: str | None = None
    version: str | None = None
    collected_dynamic_variables: dict[str, Any] = Field(default_factory=dict)
    from_number: str | None = None
    to_number: str | None = None
    direction: str | None = None

    @property
    def start_ts(self) -> datetime.datetime:
        """Convert millisecond timestamp to datetime object in Eastern timezone."""
        return datetime.datetime.fromtimestamp(
            self.start_timestamp / 1000, tz=datetime.UTC
        ).astimezone(pytz.timezone("US/Eastern"))

    @property
    def call_summary(self) -> str | None:
        """Extract call summary from call_analysis if available."""
        return self.call_analysis.get("call_summary")

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary for DataFrame creation."""
        return self.model_dump()
