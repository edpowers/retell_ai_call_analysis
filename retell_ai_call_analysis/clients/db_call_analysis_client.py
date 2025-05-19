"""
Client for interacting with the call analysis database.
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from clients.db_client import SQLiteClient
from clients.make_dot_com_client import MakeDotComClient
from clients.openai_client import ChatCompletionRequest, ChatMessage, OpenAIClient
from clients.retell_client import RetellCall
from clients.telegram_client import TelegramClient
from model.call_analysis import CallAnalysisResult
from model.prompts import CallAnalysisPrompt
from rich import print as rprint
from tqdm.auto import tqdm


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


class CallAnalysisClient:
    def __init__(self, db_path: str, openai_client: OpenAIClient):
        self.db_client = DBCallAnalysisClient(db_path)
        self.openai_client = openai_client

        self.db_client.create_table_if_not_exists()

        self.telegram_client = TelegramClient()

        self.call_analysis_prompt = CallAnalysisPrompt()

        self.make_dot_com_client = MakeDotComClient()

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
        for call in tqdm(new_calls):
            try:
                analysis_json = await self.analyze_call(call)

                analysis_result = CallAnalysisResult.create_analysis_result(
                    call, analysis_json
                )

                if show_verbose_output:
                    rprint(analysis_result.model_dump())

                results.append(analysis_result.model_dump())
            except Exception as e:
                print(f"Error analyzing call {call.call_id}: {e}")
                # Create error result
                error_result = CallAnalysisResult.create_error_instance(
                    call_id=call.call_id,
                    timestamp=CallAnalysisResult.convert_format_timestamp(
                        call.start_timestamp
                    ),
                    duration_ms=call.duration_ms,
                    error=e,
                    call_url=call.recording_url,
                    phone_number=call.retell_llm_dynamic_variables["phone"],
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
        analysis_df = pd.DataFrame(results).drop(columns=["phone_number"])

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
        prompt = self.call_analysis_prompt.create_analysis_prompt(
            transcript, dynamic_vars, tool_calls
        )

        # Get analysis from OpenAI
        analysis_response = await self.openai_client.chat_completion(
            request=ChatCompletionRequest(
                model="gpt-4o-2024-08-06",
                messages=[
                    ChatMessage(
                        role="system",
                        content=self.call_analysis_prompt.create_system_prompt(),
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
        if (analysis_result.needs_human_review and review_channel_id) or (
            not analysis_result.needs_human_review and force_send
        ):
            target_channel_id = review_channel_id
        else:
            print(f"needs_human_review={analysis_result.needs_human_review})")
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

        if analysis_df.empty:
            print("No new calls to process")
            return analysis_df

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

            # Create an opportunity in Make.com
            self.make_dot_com_client.run(
                customer_phone_number=analysis_result.phone_number,
                needs_human_review=analysis_result.needs_human_review,
            )

            # Send to Telegram
            await self.send_analysis_to_telegram(analysis_result)

            # Add a small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        return analysis_df
