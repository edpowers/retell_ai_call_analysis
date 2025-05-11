import datetime
from dataclasses import dataclass, field
from typing import Any

import pytz


@dataclass
class RetellCall:
    """Represents a call from the Retell API with structured data."""

    # Required fields
    call_id: str
    start_timestamp: int
    duration_ms: int
    call_analysis: dict[str, Any]
    transcript_with_tool_calls: list[dict[str, Any]]
    public_log_url: str
    access_token: str
    agent_id: str

    # Added fields from the provided list
    call_status: str
    call_type: str
    call_cost: float | None = None
    disconnection_reason: str | None = None
    end_timestamp: int | None = None
    latency: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    opt_in_signed_url: str | None = None
    opt_out_sensitive_data_storage: bool = False
    recording_url: str | None = None
    retell_llm_dynamic_variables: dict[str, Any] = field(default_factory=dict)
    transcript: str | None = None
    transcript_object: dict[str, Any] | None = None
    version: str | None = None
    collected_dynamic_variables: dict[str, Any] = field(default_factory=dict)
    from_number: str | None = None
    to_number: str | None = None
    direction: str | None = None
    telephony_identifier: str | None = None

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
        """Convert the dataclass to a dictionary for DataFrame creation."""
        return {
            "call_id": self.call_id,
            "start_timestamp": self.start_timestamp,
            "duration_ms": self.duration_ms,
            "call_analysis": self.call_analysis,
            "transcript_with_tool_calls": self.transcript_with_tool_calls,
            "public_log_url": self.public_log_url,
            "access_token": self.access_token,
            "agent_id": self.agent_id,
            "call_status": self.call_status,
            "call_type": self.call_type,
            "call_cost": self.call_cost,
            "disconnection_reason": self.disconnection_reason,
            "end_timestamp": self.end_timestamp,
            "latency": self.latency,
            "metadata": self.metadata,
            "opt_in_signed_url": self.opt_in_signed_url,
            "opt_out_sensitive_data_storage": self.opt_out_sensitive_data_storage,
            "recording_url": self.recording_url,
            "retell_llm_dynamic_variables": self.retell_llm_dynamic_variables,
            "transcript": self.transcript,
            "transcript_object": self.transcript_object,
            "version": self.version,
            "collected_dynamic_variables": self.collected_dynamic_variables,
            "from_number": self.from_number,
            "to_number": self.to_number,
            "direction": self.direction,
            "telephony_identifier": self.telephony_identifier,
            "start_ts": self.start_ts,
            "call_summary": self.call_summary,
        }
