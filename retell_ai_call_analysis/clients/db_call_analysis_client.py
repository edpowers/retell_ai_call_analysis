"""
Client for interacting with the call analysis database.
"""

import asyncio
import datetime
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytz
from clients.db_client import SQLiteClient
from clients.openai_client import ChatCompletionRequest, ChatMessage, OpenAIClient
from clients.retell_client import RetellCall
from clients.telegram_client import TelegramClient
from pydantic import BaseModel
from rich import print as rprint


class DBCallAnalysisClient:
    """Client for managing call analysis data in the database."""

    def __init__(self, db_path: str | Path):
        """
        Initialize the call analysis client.

        Args:
            db_path: Path to SQLite database
        """
        # Ensure the path resolves to the top-level /data folder
        self.db_path = self._normalize_db_path(db_path)
        self.db_client = SQLiteClient(self.db_path)
        self.table_name = "call_analysis"

    def _normalize_db_path(self, db_path: str | Path) -> Path:
        """
        Normalize the database path to ensure it resolves to the top-level /data folder.

        Args:
            db_path: The input database path

        Returns:
            Normalized Path object pointing to the database
        """
        path = Path(db_path)

        # If path is relative, make it absolute relative to the project root
        if not path.is_absolute():
            # Get the project root (assuming this file is in seven_digit_dental/clients)
            project_root = Path(__file__).parent.parent.parent

            # If the path doesn't start with 'data', prepend it
            if not str(path).startswith("data"):
                path = Path("data") / path

            path = project_root / path

        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        return path

    def get_existing_call_ids(self) -> set[str]:
        """
        Get a set of call_ids that already exist in the database.

        Returns:
            Set of call_ids that already exist in the database
        """
        with self.db_client.connection() as conn:
            try:
                # Query existing call_ids
                query = f"SELECT call_id FROM {self.table_name}"
                # Use pandas to read from SQLite
                result = pd.read_sql_query(query, conn)
                return set(result["call_id"].tolist())
            except Exception as e:
                # Table might not exist yet
                print(f"Error querying database: {e}")
                return set()

    def filter_new_calls(self, calls: list[RetellCall]) -> list[RetellCall]:
        """
        Filter out calls that already exist in the database.

        Args:
            calls: List of RetellCall objects

        Returns:
            List of RetellCall objects that don't exist in the database
        """
        existing_call_ids = self.get_existing_call_ids()
        new_calls = [call for call in calls if call.call_id not in existing_call_ids]

        print(f"Found {len(calls)} total calls")
        print(f"Filtered out {len(calls) - len(new_calls)} existing calls")
        print(f"Processing {len(new_calls)} new calls")

        return new_calls

    def save_analysis(self, analysis_df: pd.DataFrame) -> None:
        """
        Save analysis results to SQLite database.

        Args:
            analysis_df: Pandas DataFrame with analysis results
        """
        with self.db_client.connection() as conn:
            analysis_df.to_sql(
                name=self.table_name, con=conn, if_exists="append", index=False
            )

    def get_analysis_by_call_id(self, call_id: str) -> dict | None:
        """
        Get analysis result for a specific call_id.

        Args:
            call_id: The call ID to retrieve

        Returns:
            Analysis result as a dictionary, or None if not found
        """
        with self.db_client.connection() as conn:
            query = f"SELECT * FROM {self.table_name} WHERE call_id = ?"
            result = pd.read_sql_query(query, conn, params=[call_id])

            if len(result) == 0:
                return None

            return result.iloc[0].to_dict()

    def get_all_analyses(self) -> pd.DataFrame:
        """
        Get all analysis results from the database.

        Returns:
            Pandas DataFrame with all analysis results
        """
        with self.db_client.connection() as conn:
            query = f"SELECT * FROM {self.table_name}"
            return pd.read_sql_query(query, conn)

    def delete_analysis(self, call_id: str) -> bool:
        """
        Delete analysis result for a specific call_id.

        Args:
            call_id: The call ID to delete

        Returns:
            True if deletion was successful, False otherwise
        """
        with self.db_client.connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"DELETE FROM {self.table_name} WHERE call_id = ?", [call_id]
                )
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"Error deleting analysis: {e}")
                return False

    def create_table_if_not_exists(self) -> None:
        """
        Create the call_analysis table if it doesn't exist.
        """
        with self.db_client.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    call_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    duration_ms INTEGER,
                    call_url TEXT,
                    issue_type TEXT,
                    issue_description TEXT,
                    success_status TEXT,
                    contact_info_captured BOOLEAN,
                    booking_attempted BOOLEAN,
                    booking_successful BOOLEAN,
                    ai_detection_question BOOLEAN,
                    hang_up_early BOOLEAN,
                    dynamic_var_mismatch BOOLEAN,
                    needs_human_review BOOLEAN,
                    has_issue BOOLEAN,
                    notes TEXT,
                    transcript TEXT
                )
            """)
            conn.commit()


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

    @classmethod
    def create_error_instance(
        cls,
        call_id: str,
        timestamp: datetime.datetime,
        duration_ms: int,
        error: Exception,
        call_url: str | None = None,
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
            f"📞 *Contact Info:* {'✅ Captured' if self.contact_info_captured else '❌ Not captured'}",
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

    def _format_notes_section(self) -> list[str]:
        """Format the notes section of the Telegram message."""
        return ["\n*Notes:*", f"{self.notes}"]


class CallAnalysisClient:
    def __init__(self, db_path: str, openai_client: OpenAIClient):
        self.db_client = DBCallAnalysisClient(db_path)
        self.openai_client = openai_client

        self.db_client.create_table_if_not_exists()

        self.telegram_client = TelegramClient()

    def extract_transcript(self, call: RetellCall) -> str:
        """
        Extract a readable transcript from the call data.

        Args:
            call: The RetellCall object

        Returns:
            A formatted transcript as a string
        """
        if call.transcript:
            return call.transcript

        # If no plain transcript, build one from transcript_object or transcript_with_tool_calls
        transcript_parts = []

        for message in call.transcript_with_tool_calls:
            if message.get("role") in ["user", "agent"] and "content" in message:
                role = "User" if message["role"] == "user" else "Agent"
                content = message["content"]
                if content and content.strip():
                    transcript_parts.append(f"{role}: {content}")

        return "\n".join(transcript_parts)

    def extract_tool_calls(self, call: RetellCall) -> list[dict[str, Any]]:
        """
        Extract tool calls from the call data.

        Args:
            call: The RetellCall object

        Returns:
            A list of tool call dictionaries
        """
        tool_calls = []

        for message in call.transcript_with_tool_calls:
            if message.get("role") == "tool_call_invocation":
                tool_calls.append(
                    {
                        "name": message.get("name"),
                        "arguments": message.get("arguments"),
                        "tool_call_id": message.get("tool_call_id"),
                    }
                )
            elif message.get("role") == "tool_call_result":
                tool_calls.append(
                    {
                        "result": message.get("content"),
                        "tool_call_id": message.get("tool_call_id"),
                    }
                )

        return tool_calls

    def create_analysis_prompt(
        self,
        transcript: str,
        dynamic_vars: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> str:
        """
        Create a prompt for OpenAI to analyze the call.

        Args:
            transcript: The call transcript
            dynamic_vars: Dictionary of dynamic variables
            tool_calls: List of tool call dictionaries

        Returns:
            A formatted prompt string for OpenAI
        """
        prompt = """
Please analyze this call transcript and identify any issues based on the following criteria:

1. Dynamic Variable Mismatch: Check if the details in retell_llm_dynamic_variables match what's mentioned by the user in the transcript. Only flag this if the user explicitly provided information that contradicted the dynamic variables, not if the user didn't provide information at all. Ignore any information provided by the agent.
2. Early Hang-up: Determine if the user hung up before the agent could complete their task.
3. Contact Information Issues: Check if the agent had trouble understanding the user's contact information.
4. AI Detection: Check if the user asked if the agent was an AI/robot.
5. Booking Issues: Determine if the call was nearly successful but the agent wasn't able to book properly.
6. Wrong Business Type: Check if the call was made to a non-dental business (e.g., law office, restaurant, etc.). We should only be calling dental clinics.
7. Doctor Not Found: Check if the call was made to a dental clinic but the specific doctor mentioned in the dynamic variables doesn't work there.

Dynamic Variables:
{dynamic_vars}


Tool Calls:
{tool_calls}


Transcript:
{transcript}

Provide your analysis in JSON format with the following fields:
- issue_type: The primary type of issue (one of: "dynamic_var_mismatch", "early_hang_up", "contact_info_issue", "ai_detection", "booking_issue", "wrong_business_type", "doctor_not_found", "other", or null if successful)
- issue_description: A brief description of the specific issue
- success_status: "success", "failed", or "partial"
- contact_info_captured: boolean indicating if contact information was successfully captured
- booking_attempted: boolean indicating if booking was attempted
- booking_successful: boolean indicating if booking was successful
- ai_detection_question: boolean indicating if user asked if agent was AI
- hang_up_early: boolean indicating if user hung up early
- dynamic_var_mismatch: boolean indicating if there was a mismatch between dynamic variables and user-provided information
- needs_human_review: boolean indicating if this call requires manual review by a human
- has_issue: boolean indicating if any issues were detected in the call
- notes: Any additional observations

For the issue_type field:
- Use "wrong_business_type" ONLY if there's explicit evidence that the business is not a dental clinic (e.g., they clearly state they are a law office, restaurant, etc.)
- IMPORTANT: Assume all businesses are dental clinics unless explicitly stated otherwise. Business names like "GoDent" or similar should be assumed to be dental clinics.
- Use "doctor_not_found" if the business is a dental clinic but the specific doctor mentioned in the dynamic variables doesn't work there
- Both "wrong_business_type" and "doctor_not_found" should be considered serious errors and should always set has_issue to true and needs_human_review to true

For the dynamic_var_mismatch field:
- Set to true ONLY if the user explicitly provided information that contradicted the dynamic variables
- Set to false if the user didn't provide information or if the information matches the dynamic variables
- IMPORTANT: Only consider information provided by the user, not information stated by the agent

For the needs_human_review field, set it to true ONLY if:
- There are complex issues that AI might not fully understand
- The call outcome is ambiguous and requires human judgment
- There are potential technical problems that need human verification
- The booking process had errors but contact information was captured and there's a chance for follow-up
- There's evidence of a system malfunction or unexpected behavior
- The call was made to a wrong business type (non-dental business)
- The doctor mentioned in the dynamic variables doesn't work at the dental clinic

Do NOT set needs_human_review to true for:
- Simple failed calls where the user didn't provide information
- Voicemails or messages left with no user interaction
- Calls that went to voicemail, even if there's uncertainty about whether the doctor works at the clinic
- Routine hang-ups or disconnections
- Cases where the outcome is clear (either clearly successful or clearly failed)

For the has_issue field, set it to true if any issues were detected that affected the call outcome.
"""
        return prompt.format(
            transcript=transcript,
            dynamic_vars=json.dumps(dynamic_vars, indent=2),
            tool_calls=json.dumps(tool_calls, indent=2),
        )

    async def process_calls(
        self,
        calls: list[RetellCall],
        show_verbose_output: bool = False,
        ignore_existing_calls: bool = False,
    ) -> pd.DataFrame:
        """
        Process a list of calls, analyze them, and save the results.

        Args:
            calls: List of RetellCall objects

        Returns:
            Pandas DataFrame with analysis results
        """
        # Filter out calls that already exist in the database
        new_calls = self.db_client.filter_new_calls(calls)

        if ignore_existing_calls:
            new_calls = calls

        if not new_calls:
            print("No new calls to process")
            return pd.DataFrame()

        # Create the table if it doesn't exist
        self.db_client.create_table_if_not_exists()

        # Analyze each call
        results = []
        for call in new_calls:
            try:
                analysis_json = await self.analyze_call(call)

                analysis_result = self.create_analysis_result(call, analysis_json)

                if show_verbose_output:
                    rprint(analysis_result.model_dump())

                results.append(analysis_result.model_dump())
            except Exception as e:
                print(f"Error analyzing call {call.call_id}: {e}")
                # Create timestamp from call data
                timestamp = datetime.datetime.fromtimestamp(
                    call.start_timestamp / 1000, tz=datetime.UTC
                ).astimezone(pytz.timezone("US/Eastern"))
                # Create error result
                error_result = CallAnalysisResult.create_error_instance(
                    call_id=call.call_id,
                    timestamp=timestamp,
                    duration_ms=call.duration_ms,
                    error=e,
                    call_url=call.recording_url,
                )
                results.append(error_result.model_dump())

        # Convert results to DataFrame
        if not results:
            # Then log an error message as this is probably wrong.
            rprint(
                f"No results found for {len(new_calls)} calls. This is probably wrong."
            )
            return pd.DataFrame()

        # Create pandas DataFrame from results
        analysis_df = pd.DataFrame(results)

        if not ignore_existing_calls:
            # Save results to database
            self.db_client.save_analysis(analysis_df)

        return analysis_df

    async def analyze_call(self, call: RetellCall) -> dict:
        """
        Analyze a call using OpenAI.

        Args:
            call: The RetellCall object
            openai_client: The OpenAI client

        Returns:
            Analysis result as a dictionary
        """
        transcript = self.extract_transcript(call)
        dynamic_vars = call.retell_llm_dynamic_variables
        tool_calls = self.extract_tool_calls(call)

        # Prepare the analysis prompt
        prompt = self.create_analysis_prompt(transcript, dynamic_vars, tool_calls)

        # Get analysis from OpenAI
        analysis_response = await self.openai_client.chat_completion(
            request=ChatCompletionRequest(
                model="gpt-4o-2024-08-06",
                messages=[
                    ChatMessage(
                        role="system",
                        content="You are an expert call analyzer. Analyze the call transcript and identify issues based on the criteria provided.",
                    ),
                    ChatMessage(role="user", content=prompt),
                ],
                temperature=0.1,
                max_tokens=1000,
            )
        )

        # Extract and return the analysis JSON
        return self.extract_analysis_json(
            analysis_response["choices"][0]["message"]["content"]
        )

    def extract_analysis_json(self, response_content: str) -> dict:
        """
        Extract JSON from OpenAI response content, handling different possible formats.

        Args:
            response_content: The content field from OpenAI response

        Returns:
            Parsed JSON as a dictionary
        """
        try:
            # Try to find JSON between markdown code blocks
            if (
                "```json" in response_content
                and "```" in response_content.split("```json")[1]
            ):
                json_str = response_content.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)

            # Try to find JSON between any code blocks
            if "```" in response_content:
                parts = response_content.split("```")
                for i in range(
                    1, len(parts), 2
                ):  # Check only the parts between ``` markers
                    return json.loads(parts[i].strip())

            # Try to parse the entire content as JSON
            return json.loads(response_content)

        except Exception as e:
            raise ValueError(f"Failed to parse OpenAI response: {e}") from e

    def create_analysis_result(
        self, call: RetellCall, analysis_json: dict
    ) -> CallAnalysisResult:
        """
        Create a CallAnalysisResult model from the call and analysis JSON.

        Args:
            call: The RetellCall object
            analysis_json: The parsed analysis JSON from OpenAI

        Returns:
            CallAnalysisResult object
        """
        # Convert timestamp from milliseconds to datetime
        timestamp = datetime.datetime.fromtimestamp(
            call.start_timestamp / 1000, tz=datetime.UTC
        ).astimezone(pytz.timezone("US/Eastern"))

        # Extract transcript from the call
        transcript = self.extract_transcript(call)

        return CallAnalysisResult(
            call_id=call.call_id,
            timestamp=timestamp,
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
            transcript=transcript,
        )

    async def send_analysis_to_telegram(
        self, analysis_result: CallAnalysisResult, force_send: bool = False
    ) -> bool:
        """
        Send call analysis result to the appropriate Telegram channel based on needs_human_review flag.

        Args:
            analysis_result: The CallAnalysisResult to send
            force_send: If True, send regardless of needs_human_review flag

        Returns:
            True if message was sent successfully, False otherwise
        """
        # Get channel IDs from environment variables
        review_channel_id = os.getenv("TELEGRAM_CHAT_ID")

        # If neither channel is configured, log warning and return
        if not review_channel_id:
            print("No Telegram channels configured. Set TELEGRAM_CHAT_ID")
            return False

        # Determine which channel to send to
        target_channel_id = None
        if (analysis_result.needs_human_review and review_channel_id) or (
            not analysis_result.needs_human_review and force_send
        ):
            target_channel_id = review_channel_id

        if not target_channel_id:
            print(
                f"No suitable Telegram channel found for analysis result (needs_human_review={analysis_result.needs_human_review})"
            )
            return False

        try:
            # Format the message for Telegram
            message = analysis_result.format_for_telegram()

            # Send the message
            await self.telegram_client.send_message_async(
                chat_id=target_channel_id, text=message
            )

            return True
        except Exception as e:
            print(f"Error sending analysis to Telegram: {e}")
            return False

    async def process_calls_and_notify(
        self,
        calls: list[RetellCall],
        show_verbose_output: bool = False,
        ignore_existing_calls: bool = False,
        send_to_telegram: bool = True,
    ) -> pd.DataFrame:
        """
        Process calls, analyze them, save the results, and send to Telegram.

        Args:
            calls: List of RetellCall objects
            show_verbose_output: Whether to print verbose output
            ignore_existing_calls: Whether to process calls that already exist in the database
            send_to_telegram: Whether to send results to Telegram

        Returns:
            Pandas DataFrame with analysis results
        """
        # Process the calls
        analysis_df = await self.process_calls(
            calls=calls,
            show_verbose_output=show_verbose_output,
            ignore_existing_calls=ignore_existing_calls,
        )

        if send_to_telegram and not analysis_df.empty:
            # Convert DataFrame rows to CallAnalysisResult objects and send to Telegram
            for _, row in analysis_df.iterrows():
                # Create CallAnalysisResult from row data
                analysis_result = CallAnalysisResult(
                    call_id=row.get("call_id"),
                    timestamp=row.get("timestamp"),
                    duration_ms=row.get("duration_ms"),
                    call_url=row.get("call_url"),
                    issue_type=row.get("issue_type"),
                    issue_description=row.get("issue_description"),
                    success_status=row.get("success_status", "failed"),
                    contact_info_captured=row.get("contact_info_captured", False),
                    booking_attempted=row.get("booking_attempted", False),
                    booking_successful=row.get("booking_successful", False),
                    ai_detection_question=row.get("ai_detection_question", False),
                    hang_up_early=row.get("hang_up_early", False),
                    dynamic_var_mismatch=row.get("dynamic_var_mismatch", False),
                    needs_human_review=row.get("needs_human_review", False),
                    has_issue=row.get("has_issue", False),
                    notes=row.get("notes"),
                    transcript=row.get("transcript"),
                )

                # Send to Telegram
                await self.send_analysis_to_telegram(analysis_result)

                # Add a small delay to avoid rate limiting
                await asyncio.sleep(0.5)

        return analysis_df
