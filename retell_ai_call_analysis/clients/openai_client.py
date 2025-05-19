"""
OpenAI API client implementation for making requests to OpenAI services.
"""

import asyncio
import logging
import os
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger("root")


class OpenAIClientConfig(BaseModel):
    """Configuration for the OpenAI client."""

    api_key: str | None = None
    organization_id: str | None = None
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 60.0
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff_factor: float = 2.0


class ChatMessage(BaseModel):
    """Represents a message in a chat conversation."""

    role: str
    content: str
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    """Request model for chat completion API."""

    model: str
    messages: list[ChatMessage]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    n: int | None = 1
    stream: bool | None = False
    max_tokens: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None


class OpenAIClient:
    """
    Client for interacting with OpenAI APIs.

    This client handles authentication, request formatting, and error handling
    for OpenAI API calls.
    """

    def __init__(self, config: OpenAIClientConfig | None = None):
        """
        Initialize the OpenAI client with the provided configuration.

        Args:
            config: Configuration for the OpenAI client. If None, will try to use
                   environment variables for API key.
        """
        self.config = config or OpenAIClientConfig()

        # If API key not provided in config, try to get from environment
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")
            if not self.config.api_key:
                raise ValueError(
                    "OpenAI API key must be provided in config or set as OPENAI_API_KEY environment variable"
                )

        # If organization ID not provided in config, try to get from environment
        if not self.config.organization_id:
            self.config.organization_id = os.getenv("OPENAI_ORGANIZATION_ID")

        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers=self._get_headers(),
        )

    def _get_headers(self) -> dict[str, str]:
        """
        Get the headers for OpenAI API requests.

        Returns:
            Dictionary of HTTP headers
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        if self.config.organization_id:
            headers["OpenAI-Organization"] = self.config.organization_id

        return headers

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        await self.close()

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Make a request to the OpenAI API with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            json_data: JSON data to send in the request body
            params: Query parameters

        Returns:
            API response as a dictionary

        Raises:
            httpx.HTTPStatusError: If the request fails after retries
        """
        retries = 0
        delay = self.config.retry_delay

        while True:
            try:
                response = await self.client.request(
                    method=method,
                    url=endpoint,
                    json=json_data,
                    params=params,
                )
                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                # Don't retry client errors except for rate limiting
                if status_code < 500 and status_code != 429:
                    logger.error(f"OpenAI API error: {e.response.text}")
                    raise

                retries += 1
                if retries > self.config.max_retries:
                    logger.error(f"Max retries reached for OpenAI API request: {e}")
                    raise

                # For rate limiting, use the Retry-After header if available
                if status_code == 429 and "Retry-After" in e.response.headers:
                    delay = float(e.response.headers["Retry-After"])
                else:
                    delay *= self.config.retry_backoff_factor

                logger.warning(
                    f"Retrying OpenAI API request after error: {e}. "
                    f"Retry {retries}/{self.config.max_retries} in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

            except httpx.RequestError as e:
                retries += 1
                if retries > self.config.max_retries:
                    logger.error(f"Max retries reached for OpenAI API request: {e}")
                    raise

                delay *= self.config.retry_backoff_factor
                logger.warning(
                    f"Retrying OpenAI API request after connection error: {e}. "
                    f"Retry {retries}/{self.config.max_retries} in {delay:.2f}s"
                )
                await asyncio.sleep(delay)

    async def chat_completion(self, request: ChatCompletionRequest) -> dict[str, Any]:
        """
        Create a chat completion using the OpenAI API.

        Args:
            request: Chat completion request parameters

        Returns:
            API response containing the completion
        """
        return await self._make_request(
            method="POST",
            endpoint="/chat/completions",
            json_data=request.model_dump(exclude_none=True),
        )

    async def list_models(self) -> dict[str, Any]:
        """
        List available models from the OpenAI API.

        Returns:
            API response containing the list of models
        """
        return await self._make_request(
            method="GET",
            endpoint="/models",
        )

    async def get_model(self, model_id: str) -> dict[str, Any]:
        """
        Get information about a specific model.

        Args:
            model_id: The ID of the model to retrieve

        Returns:
            API response containing model information
        """
        return await self._make_request(
            method="GET",
            endpoint=f"/models/{model_id}",
        )

    # Convenience methods for common use cases

    async def simple_completion(
        self,
        prompt: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """
        Get a simple completion for a prompt.

        Args:
            prompt: The prompt to complete
            model: The model to use
            temperature: The temperature to use
            max_tokens: The maximum number of tokens to generate

        Returns:
            The generated completion text
        """
        request = ChatCompletionRequest(
            model=model,
            messages=[ChatMessage(role="user", content=prompt)],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        response = await self.chat_completion(request)
        return response["choices"][0]["message"]["content"]
