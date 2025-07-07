# OpenAI-compatible wrapper for AI/ML API text completions

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import requests
from aiohttp import ClientSession
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator

from .constants import AIMLAPI_HEADERS

# Logger for this module
logger = logging.getLogger(__name__)


class AimlapiLLM(LLM):
    """
    Completion model integration for AI/ML API.

    This class wraps the AI/ML API's text completion endpoint in an
    OpenAI-compatible interface, supporting both sync and async calls,
    default parameters, validation, and error handling.
    """

    # -------------------------------------------------------------------------
    # Configuration fields (serialized by alias)
    # -------------------------------------------------------------------------

    base_url: str = Field(
        default="https://api.aimlapi.com/v1/completions",
        description="Endpoint URL for completion calls",
    )

    aimlapi_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("AIMLAPI_API_KEY", default="dummytoken"),
        description="API key for authentication; uses env var if not provided",
    )

    model: str = Field(
        ...,
        description="Name of the completion model (e.g., 'gpt-3.5-turbo')",
    )

    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature; higher makes output more random",
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Nucleus sampling probability threshold",
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling filter",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate",
    )
    repetition_penalty: Optional[float] = Field(
        default=None,
        description="Penalty for repeating tokens",
    )
    logprobs: Optional[int] = Field(
        default=None,
        description="Include log probabilities for top tokens",
    )

    # Pydantic model settings: forbid extra fields, respect aliases
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate required fields and supply defaults before instantiation.

        Warns if max_tokens is missing and sets a default of 200.
        """
        if values.get("max_tokens") is None:
            warnings.warn("'max_tokens' not set; defaulting to 200 for completions.")
            values["max_tokens"] = 200
        return values

    @property
    def _llm_type(self) -> str:
        """Unique identifier for this LLM type."""
        return "aimlapi"

    def _format_output(self, output: Dict[str, Any]) -> str:
        """
        Extracts and returns the main text from the API response payload.
        """
        # API returns list of choices; we take the first
        return output["choices"][0]["text"]

    @property
    def default_params(self) -> Dict[str, Any]:
        """
        Build default payload parameters for each call, stripping None values.
        """
        params = {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        # Remove keys where value is None
        return {k: v for k, v in params.items() if v is not None}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous completion call.

        Constructs headers, merges default and call-specific parameters,
        sends POST, handles HTTP errors, and returns the generated text.
        """
        # Build headers with API key and default tracking headers
        headers = {
            "Authorization": f"Bearer {self.aimlapi_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            **AIMLAPI_HEADERS,
        }

        # If single stop token provided, use it directly else pass list
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop

        # Merge parameters: defaults + prompt + optional overrides
        payload: Dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }
        # Strip None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        # Send HTTP request
        response = requests.post(
            url=self.base_url,
            json=payload,
            headers=headers,
        )

        # Handle server errors (5xx)
        if response.status_code >= 500:
            raise Exception(f"Server error {response.status_code}")
        # Handle client errors (4xx)
        elif response.status_code >= 400:
            raise ValueError(f"Invalid request payload: {response.text}")
        # Handle unexpected success codes
        elif response.status_code not in (200, 201):
            raise Exception(
                f"Unexpected status {response.status_code}: {response.text}"
            )

        # Parse JSON and extract text
        data = response.json()
        return self._format_output(data)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronous completion call using aiohttp session.

        Mirrors _call but leverages async IO for non-blocking behavior.
        """
        headers = {
            "Authorization": f"Bearer {self.aimlapi_api_key.get_secret_value()}",
            "Content-Type": "application/json",
            **AIMLAPI_HEADERS,
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload: Dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }
        payload = {k: v for k, v in payload.items() if v is not None}

        async with ClientSession() as session:
            async with session.post(
                self.base_url, json=payload, headers=headers
            ) as response:
                if response.status >= 500:
                    raise Exception(f"Server error {response.status}")
                elif response.status >= 400:
                    text = await response.text()
                    raise ValueError(f"Invalid payload: {text}")
                elif response.status not in (200, 201):
                    text = await response.text()
                    raise Exception(f"Unexpected status {response.status}: {text}")
                data = await response.json()
                return self._format_output(data)
