import os
from typing import Any

import httpx
import retell
from model.retell.retell_call import RetellCall
from utils import get_timestamp_ms


class RetellClient:
    """Client for interacting with the Retell API."""

    def __init__(self, api_key: str | None = None):
        """
        Initialize the Retell client.

        Args:
            api_key: The Retell API key. If None, will try to get from RETELL_API_KEY env var.
        """
        api_key = api_key or os.getenv("RETELL_API_KEY")
        if not api_key:
            raise ValueError(
                "Retell API key must be provided or set as RETELL_API_KEY environment variable"
            )
        self.client = retell.Retell(api_key=api_key)
        self.call_responses_raw: list[Any] = []

    def get_calls(
        self,
        days_ago: int = 7,
        min_duration_ms: int = 1000,
        limit: int = 1000,
    ) -> list[RetellCall]:
        """
        Get calls from the Retell API.

        Args:
            days_ago: Number of days to look back for calls
            min_duration_ms: Minimum duration of calls in milliseconds
            limit: Maximum number of calls to return

        Returns:
            List of RetellCall objects
        """
        self.call_responses = self.client.call.list(
            filter_criteria={
                "duration_ms": {
                    "lower_threshold": min_duration_ms,
                },
                "start_timestamp": {
                    "lower_threshold": get_timestamp_ms(24 * days_ago),
                    "upper_threshold": get_timestamp_ms(0),
                },
            },
            limit=limit,
        )

        return [RetellCall(**call.model_dump()) for call in self.call_responses]

    def get_call_log(self, call_id: str) -> str | None:
        """
        Get the public log for a call.

        Args:
            call_id: The ID of the call

        Returns:
            The call log text if available
        """
        call = self.client.call.retrieve(call_id)
        if not call.public_log_url:
            return None

        response = httpx.get(call.public_log_url)
        if response.status_code == 200:
            return response.text
        return None

    def filter_calls_by_summary_content(
        self, calls: list[RetellCall], content: str | list[str]
    ) -> list[RetellCall]:
        """
        Filter calls by content in their summary.

        Args:
            calls: List of RetellCall objects
            content: String or list of strings to search for in call summaries

        Returns:
            Filtered list of RetellCall objects
        """
        if isinstance(content, str):
            content = [content]

        result = []
        for call in calls:
            if not call.call_summary:
                continue

            summary_lower = call.call_summary.lower()
            if any(term.lower() in summary_lower for term in content):
                result.append(call)

        return result
