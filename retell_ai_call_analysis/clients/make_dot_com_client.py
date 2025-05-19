"""Make.com webhook client"""

import logging
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger("root")


class MakeCustomLookupResponse(BaseModel):
    """Model for Make.com webhook response data"""

    contact_id: str | None = Field(None, alias="contactId")
    contact_last: str | None = None
    email: str | None = None
    competitor: str | None = None
    phone: str | None = None
    # Add any other fields that might be returned


class MakeDotComClient:
    """Client for interacting with Make.com webhooks"""

    def __init__(
        self,
        customer_data_lookup_url: str | None = None,
        create_opportunity_url: str | None = None,
        timeout: int = 30,
        max_retries: int = 3,
    ):
        """
        Initialize the Make.com client

        Args:
            customer_data_lookup_url: URL for the customer data lookup webhook
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.customer_data_lookup_url = self._find_url(
            customer_data_lookup_url, "MAKE_CUSTOMER_LOOKUP_URL"
        )
        self.create_opportunity_url = self._find_url(
            create_opportunity_url, "MAKE_CREATE_OPPORTUNITY_URL"
        )
        self.timeout = timeout
        self.max_retries = max_retries

        # Configure client with retry logic
        self.client = httpx.Client(
            timeout=timeout, transport=httpx.HTTPTransport(retries=max_retries)
        )

    def _find_url(self, url: str | None, env_var_name: str) -> str:
        """
        Find a URL from either the provided parameter or environment variable

        Args:
            url: URL provided directly to the method
            env_var_name: Name of the environment variable to check if url is None

        Returns:
            The URL to use

        Raises:
            ValueError: If neither url nor environment variable is set
        """
        if url:
            return url

        if env_url := os.getenv(env_var_name):
            return env_url

        raise ValueError(f"{env_var_name} is not set")

    def lookup_customer_data(
        self,
        from_number: str,
        llm_id: str = "retell_llm_dynamic_variables",
        to_number: str | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> MakeCustomLookupResponse:
        """
        Look up customer data via Make.com webhook

        Args:
            llm_id: ID of the LLM making the request
            from_number: Customer phone number
            to_number: Optional destination phone number (defaults to from_number if None)
            additional_params: Optional additional parameters to include in the request

        Returns:
            MakeWebhookResponse: Parsed response from Make.com

        Raises:
            httpx.HTTPError: If the request fails
        """
        request_body = {
            "llm_id": llm_id,
            "from_number": from_number,
            "to_number": to_number or from_number,
        }

        # Add any additional parameters
        if additional_params:
            request_body.update(additional_params)

        try:
            logger.debug(f"Making request to Make.com webhook: {request_body}")
            response = self.client.post(
                self.customer_data_lookup_url, json=request_body
            )
            response.raise_for_status()

            # Parse the response into our Pydantic model
            return MakeCustomLookupResponse.model_validate(response.json())

        except httpx.HTTPError as e:
            logger.error(f"Error making request to Make.com: {e!s}")
            raise

    def create_opportunity(
        self,
        customer_data: MakeCustomLookupResponse,
    ):
        """Create an opportunity in Make.com"""
        response = self.client.post(
            self.create_opportunity_url, json=customer_data.model_dump()
        )
        response.raise_for_status()

    def run(
        self,
        customer_data: MakeCustomLookupResponse,
        needs_human_review: bool,
    ) -> None:
        """Run the client"""
        if not needs_human_review:
            return

        try:
            # customer_data = self.lookup_customer_data(customer_phone_number)
            self.create_opportunity(customer_data)
        except ValidationError as e:
            # Then we should make a
            logger.error(f"Error making request to Make.com: {e!s} {customer_data}")
            raise

    def close(self):
        """Close the underlying HTTP client"""
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
