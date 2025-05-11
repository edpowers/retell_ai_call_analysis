"""
Telegram client for sending messages to a Telegram channel using httpx.
"""

import asyncio
import logging
import os
import time
from typing import Any

import httpx
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class TelegramClient:
    """Client for sending messages to a Telegram channel using the Telegram Bot API."""

    BASE_URL = "https://api.telegram.org/bot"
    MAX_MESSAGE_LENGTH = 4096  # Telegram's message length limit

    def __init__(
        self,
        bot_token: str | None = None,
        chat_id: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        retry_backoff_factor: float = 2.0,
        timeout: float = 10.0,
    ):
        """
        Initialize the Telegram client.

        Args:
            bot_token: The Telegram bot token. If None, will be loaded from TELEGRAM_BOT_TOKEN env var.
            chat_id: The Telegram chat ID. If None, will be loaded from TELEGRAM_CHAT_ID env var.
            max_retries: Maximum number of retries for failed requests
            retry_delay: Initial delay between retries in seconds
            retry_backoff_factor: Factor to increase delay between retries
            timeout: Request timeout in seconds
        """
        load_dotenv()
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

        if not self.bot_token:
            raise ValueError(
                "Telegram bot token not provided and TELEGRAM_BOT_TOKEN env var not set"
            )
        if not self.chat_id:
            raise ValueError(
                "Telegram chat ID not provided and TELEGRAM_CHAT_ID env var not set"
            )

        self.api_url = f"{self.BASE_URL}{self.bot_token}"
        self.client = httpx.Client(timeout=timeout)

        # Retry configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.retry_backoff_factor = retry_backoff_factor

    def _handle_request_exception(
        self, e: Exception, retries: int, delay: float
    ) -> tuple[int, float, bool]:
        """
        Handle exceptions from Telegram API requests.

        Args:
            e: The exception that was raised
            retries: Current retry count
            delay: Current delay before next retry

        Returns:
            Tuple of (new_retries, new_delay, should_raise)

        Raises:
            The original exception if it should not be retried
        """
        # Don't retry client errors except for rate limiting (429)
        if (
            isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code != 429
            and e.response.status_code < 500
        ):
            logger.error(f"HTTP error when sending Telegram message: {e}")
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    error_content = e.response.json()
                    logger.error(f"Response content: {error_content}")
                except Exception:
                    logger.error(f"Response content (text): {e.response.text}")
            raise

        retries += 1
        if retries > self.max_retries:
            logger.error(f"Max retries reached for Telegram API request: {e}")
            if isinstance(e, httpx.HTTPStatusError):
                try:
                    error_content = e.response.json()
                    logger.error(f"Response content: {error_content}")
                except Exception:
                    logger.error(f"Response content (text): {e.response.text}")
            raise

        # For rate limiting, use the Retry-After header if available
        if (
            isinstance(e, httpx.HTTPStatusError)
            and e.response.status_code == 429
            and "Retry-After" in e.response.headers
        ):
            delay = float(e.response.headers["Retry-After"])
        else:
            delay *= self.retry_backoff_factor

        logger.warning(
            f"Retrying Telegram API request after error: {e}. "
            f"Retry {retries}/{self.max_retries} in {delay:.2f}s"
        )

        return retries, delay, False

    def send_message(
        self,
        text: str,
        chat_id: str | None = None,
        parse_mode: str | None = "HTML",
        disable_web_page_preview: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Send a message to the Telegram channel.

        Args:
            text: The message text to send
            chat_id: Override the default chat_id if needed
            parse_mode: The parse mode for the message (HTML, Markdown, MarkdownV2, or None)
            disable_web_page_preview: Whether to disable web page previews
            **kwargs: Additional parameters to pass to the Telegram API

        Returns:
            The response from the Telegram API as a dictionary

        Raises:
            httpx.HTTPError: If the HTTP request fails
            ValueError: If the Telegram API returns an error
        """
        endpoint = f"{self.api_url}/sendMessage"
        target_chat_id = chat_id or self.chat_id

        # Handle message length limit
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                f"Message exceeds Telegram's {self.MAX_MESSAGE_LENGTH} character limit. Truncating."
            )
            text = text[: self.MAX_MESSAGE_LENGTH]

        payload = {
            "chat_id": target_chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_page_preview,
            **kwargs,
        }

        if parse_mode:
            payload["parse_mode"] = parse_mode

        retries = 0
        delay = self.retry_delay

        while True:
            try:
                response = self.client.post(endpoint, json=payload)
                response.raise_for_status()
                result = response.json()

                if not result.get("ok"):
                    error_msg = f"Telegram API error: {result.get('description', 'Unknown error')} - Full response: {result}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                return result.get("result", {})

            except (httpx.HTTPError, ValueError) as e:
                retries, delay, should_raise = self._handle_request_exception(
                    e, retries, delay
                )
                if should_raise:
                    raise
                time.sleep(delay)

    async def send_message_async(
        self,
        text: str,
        chat_id: str | None = None,
        parse_mode: str | None = "HTML",
        disable_web_page_preview: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Send a message to the Telegram channel asynchronously.

        Args:
            text: The message text to send
            chat_id: Override the default chat_id if needed
            parse_mode: The parse mode for the message (HTML, Markdown, MarkdownV2, or None)
            disable_web_page_preview: Whether to disable web page previews
            **kwargs: Additional parameters to pass to the Telegram API

        Returns:
            The response from the Telegram API as a dictionary

        Raises:
            httpx.HTTPError: If the HTTP request fails
            ValueError: If the Telegram API returns an error
        """
        endpoint = f"{self.api_url}/sendMessage"
        target_chat_id = chat_id or self.chat_id

        # Handle message length limit
        if len(text) > self.MAX_MESSAGE_LENGTH:
            logger.warning(
                f"Message exceeds Telegram's {self.MAX_MESSAGE_LENGTH} character limit. Truncating."
            )
            text = text[: self.MAX_MESSAGE_LENGTH]

        payload = {
            "chat_id": target_chat_id,
            "text": text,
            "disable_web_page_preview": disable_web_page_preview,
            **kwargs,
        }

        if parse_mode:
            payload["parse_mode"] = parse_mode

        retries = 0
        delay = self.retry_delay

        while True:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(endpoint, json=payload)
                    response.raise_for_status()
                    result = response.json()

                    if not result.get("ok"):
                        error_msg = f"Telegram API error: {result.get('description', 'Unknown error')} - Full response: {result}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                    return result.get("result", {})

            except (httpx.HTTPError, ValueError) as e:
                retries, delay, should_raise = self._handle_request_exception(
                    e, retries, delay, is_async=True
                )
                if should_raise:
                    raise
                await asyncio.sleep(delay)

    def __del__(self):
        """Close the HTTP client when the object is garbage collected."""
        if hasattr(self, "client"):
            self.client.close()
