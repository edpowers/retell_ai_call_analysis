import datetime

import pytz
from model.retell.retell_call import RetellCall
from pydantic import BaseModel


class DynamicVariables(BaseModel):
    """Model for the dynamic variables of a call."""

    contact_id: str
    contact_last: str
    phone: str
    email: str
    competitor: str

    @classmethod
    def from_retell_format(cls, retell_vars: dict) -> "DynamicVariables":
        """
        Convert from Retell's dynamic variables format to our model.

        Args:
            retell_vars: Dictionary containing dynamic variables from Retell

        Returns:
            DynamicVariables instance with normalized data
        """
        if not retell_vars:
            return DynamicVariables.create_empty()

        # Handle the different key naming in Retell's format
        return cls(
            contact_id=retell_vars.get("contactId", ""),
            contact_last=retell_vars.get("contact_last", ""),
            phone=retell_vars.get("phone", ""),
            email=retell_vars.get("email", ""),
            competitor=retell_vars.get("competitor", ""),
        )

    @classmethod
    def create_empty(cls) -> "DynamicVariables":
        """Create an empty DynamicVariables instance."""
        return cls(
            contact_id="",
            contact_last="",
            phone="",
            email="",
            competitor="",
        )


class CallAnalysisResult(BaseModel):
    """Model for the analysis result of a call."""

    call_id: str
    timestamp: datetime.datetime
    duration_ms: int
    call_url: str | None = None
    issue_type: str | None = None
    issue_description: str | None = None
    success_status: str = "failed"  # "success", "failed", "partial"
    contact_info_captured: bool = False
    booking_attempted: bool = False
    booking_successful: bool = False
    ai_detection_question: bool = False
    hang_up_early: bool = False
    dynamic_var_mismatch: bool = False
    needs_human_review: bool = False
    has_issue: bool = False
    notes: str | None = None
    transcript: str | None = None
    phone_number: str | None = None
    dynamic_variables: DynamicVariables | None = (
        None  # Use the DynamicVariables model instead of dict
    )

    @staticmethod
    def convert_format_timestamp(start_timestamp: int) -> datetime.datetime:
        """Convert the timestamp to a readable format."""
        # Convert timestamp from milliseconds to datetime
        return datetime.datetime.fromtimestamp(
            start_timestamp / 1000, tz=datetime.UTC
        ).astimezone(pytz.timezone("US/Eastern"))

    @classmethod
    def create_analysis_result(
        cls, call: RetellCall, analysis_json: dict
    ) -> "CallAnalysisResult":
        """
        Create a CallAnalysisResult model from the call and analysis JSON.

        Args:
            call: The RetellCall object
            analysis_json: The parsed analysis JSON from OpenAI

        Returns:
            CallAnalysisResult object
        """
        # Extract transcript from the call
        return CallAnalysisResult(
            call_id=call.call_id,
            timestamp=CallAnalysisResult.convert_format_timestamp(call.start_timestamp),
            duration_ms=call.duration_ms,
            call_url=call.recording_url,
            issue_type=analysis_json.get("issue_type"),
            issue_description=analysis_json.get("issue_description"),
            success_status=analysis_json.get("success_status", "failed"),
            contact_info_captured=analysis_json.get("contact_info_captured", False),
            booking_attempted=analysis_json.get("booking_attempted", False),
            booking_successful=analysis_json.get("booking_successful", False),
            ai_detection_question=analysis_json.get("ai_detection_question", False),
            hang_up_early=analysis_json.get("hang_up_early", False),
            dynamic_var_mismatch=analysis_json.get("dynamic_var_mismatch", False),
            needs_human_review=analysis_json.get("needs_human_review", False),
            has_issue=analysis_json.get("has_issue", False),
            notes=analysis_json.get("notes"),
            transcript=call.transcript,
            dynamic_variables=DynamicVariables.from_retell_format(
                call.retell_llm_dynamic_variables
            ),
        )

    @classmethod
    def create_error_instance(
        cls,
        call_id: str,
        timestamp: datetime.datetime,
        duration_ms: int,
        error: Exception,
        call_url: str | None = None,
        phone_number: str | None = None,
    ) -> "CallAnalysisResult":
        """
        Create an instance representing an error during analysis.

        Args:
            call_id: The call ID
            timestamp: The call timestamp
            duration_ms: The call duration in milliseconds
            error: The exception that occurred
            call_url: URL to the call recording

        Returns:
            A CallAnalysisResult instance with error information
        """
        return cls(
            call_id=call_id,
            timestamp=timestamp,
            duration_ms=duration_ms,
            call_url=call_url,
            issue_type="analysis_error",
            issue_description=f"Error analyzing call: {error!s}",
            success_status="failed",
            needs_human_review=True,
            has_issue=True,
            notes=f"Analysis failed with error: {error!s}",
            dynamic_variables=DynamicVariables.create_empty(),
        )

    def format_for_telegram(self) -> str:
        """
        Format the call analysis result for Telegram messages, optimized for mobile viewing.

        Returns:
            A formatted string suitable for Telegram messages
        """
        message = []

        # Add header information
        message.extend(self._format_header_section())

        # Add status section
        message.extend(self._format_status_section())

        # Add issue information if there is an issue
        if self.has_issue:
            message.extend(self._format_issues_section())

        # Add key metrics section
        message.extend(self._format_metrics_section())

        # Add flags section if any flags are true
        flags_section = self._format_flags_section()
        if flags_section:
            message.extend(flags_section)

        # Add notes if available
        if self.notes:
            message.extend(self._format_notes_section())

        # Add dynamic variables section if available
        if self.dynamic_variables:
            message.extend(self._format_dynamic_variables_section())

        # Join all parts with newlines
        return "\n".join(message)

    def _format_header_section(self) -> list[str]:
        """Format the header section of the Telegram message."""
        # Format duration from milliseconds to seconds
        duration_sec = round(self.duration_ms / 1000, 1)

        # Format timestamp to a readable format
        formatted_time = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")

        # Create emoji indicators for boolean fields
        status_emoji = {"success": "✅", "partial": "⚠️", "failed": "❌"}.get(
            self.success_status, "❓"
        )

        header = [
            f"*Call Analysis {status_emoji}*",
            f"📅 *Date:* {formatted_time}",
            f"⏱️ *Duration:* {duration_sec}s",
            f"*Call ID:* {self.call_id}",
        ]

        # Add call URL if available
        if self.call_url:
            header.append(f"🔗 [Listen to Recording]({self.call_url})")

        return header

    def _format_status_section(self) -> list[str]:
        """Format the status section of the Telegram message."""
        return ["\n*Status:*", f"📊 *Outcome:* {self.success_status.capitalize()}"]

    def _format_issues_section(self) -> list[str]:
        """Format the issues section of the Telegram message."""
        issues = ["\n*Issues:*"]

        if self.issue_type:
            issues.append(
                f"🚨 *Type:* {self.issue_type.replace('_', ' ').capitalize()}"
            )
        if self.issue_description:
            issues.append(f"📝 *Description:* {self.issue_description}")

        return issues

    def _format_metrics_section(self) -> list[str]:
        """Format the key metrics section of the Telegram message."""
        return [
            "\n*Key Metrics:*",
            f"�� *Contact Info:* {'✅ Captured' if self.contact_info_captured else '❌ Not captured'}",
            f"📅 *Booking:* {'✅ Successful' if self.booking_successful else '❌ Not successful'}",
        ]

    def _format_flags_section(self) -> list[str]:
        """Format the flags section of the Telegram message."""
        flags = []

        if self.ai_detection_question:
            flags.append("🤖 AI detection question")
        if self.hang_up_early:
            flags.append("📴 Hung up early")
        if self.dynamic_var_mismatch:
            flags.append("🔄 Dynamic variable mismatch")
        if self.needs_human_review:
            flags.append("👁️ Needs human review")

        if flags:
            return ["\n*Flags:*", *flags]
        return []

    def _format_dynamic_variables_section(self) -> list[str]:
        """Format the dynamic variables section of the Telegram message."""
        if not self.dynamic_variables:
            return []

        # Start with section header
        variables = ["\n*Dynamic Variables:*"]

        # Add each non-empty variable with an appropriate emoji
        var_emojis = {
            "contact_id": "👤",
            "contact_last": "📛",
            "phone": "📱",
            "email": "📧",
            "competitor": "🏢",
        }

        for key, value in self.dynamic_variables.model_dump().items():
            if value:  # Only include non-empty values
                emoji = var_emojis.get(key, "📌")
                formatted_key = key.replace("_", " ").title()
                variables.append(f"{emoji} *{formatted_key}:* {value}")

        return variables

    def _format_notes_section(self) -> list[str]:
        """Format the notes section of the Telegram message."""
        return ["\n*Notes:*", f"{self.notes}"]
