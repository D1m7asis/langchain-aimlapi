from __future__ import annotations

import os
from typing import Any, List, Optional, Literal

import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS


class AimlapiImageModel(LLM):
    """Wrapper around AI/ML API's image generation endpoint, fully OpenAI-compatible."""

    model: str = Field(default="dall-e-3")
    """Which image model to use (e.g., "dall-e-3")."""

    api_key: Optional[str] = Field(
        default=None,
        alias="api_key",
        description="Aimlapi API key; falls back to AIMLAPI_API_KEY env var.",
    )

    base_url: str = Field(
        default="https://api.aimlapi.com/v1",
        alias="base_url",
        description="Base URL for Aimlapi image service.",
    )

    timeout: Optional[float] = Field(
        default=None,
        alias="timeout",
        description="Timeout in seconds for HTTP requests.",
    )

    max_retries: int = Field(
        default=2,
        alias="max_retries",
        description="Number of retry attempts on request failure.",
    )

    @property
    def _llm_type(self) -> str:
        """Return the unique identifier for this LLM type."""
        return "aimlapi-image"

    def _client(self) -> openai.OpenAI:
        """
        Build and return an OpenAI-compatible client configured for Aimlapi.
        """
        return openai.OpenAI(
            api_key=self.api_key or os.getenv("AIMLAPI_API_KEY"),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=AIMLAPI_HEADERS,
        )

    def _async_client(self) -> openai.AsyncOpenAI:
        """Return an asynchronous OpenAI-compatible client."""
        return openai.AsyncOpenAI(
            api_key=self.api_key or os.getenv("AIMLAPI_API_KEY"),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=AIMLAPI_HEADERS,
        )

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Sync-forward call to generate a single image URL from a text prompt.

        Args:
            prompt: Textual description of desired image.
            stop: Ignored parameter for compatibility.
            run_manager: Callback manager (unused).
            **kwargs: Additional parameters for `generate_images`.

        Returns:
            URL of the generated image.
        """
        images = self.generate_images(prompt=prompt, n=1, **kwargs)
        return images[0]

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Asynchronous version of :meth:`_call`."""
        images = await self.agenerate_images(prompt=prompt, n=1, **kwargs)
        return images[0]

    def generate_images(
        self,
        prompt: str,
        n: int = 1,
        size: Literal[
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "256x256",
            "512x512",
            "1792x1024",
            "1024x1792",
        ] = "1024x1024",
        response_format: Literal["url", "b64_json"] = "url",
    ) -> List[str]:
        """
        Call the Aimlapi image generation endpoint and return results.

        Args:
            prompt: Text prompt describing the content of the image.
            n: Number of images to generate.
            size: Dimensions of generated images (OpenAI-compatible presets).
            response_format: Format of returned images: URLs or base64 strings.

        Returns:
            List of image URLs or base64 strings based on `response_format`.
        """
        client = self._client()
        resp = client.images.generate(
            model=self.model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
        )
        if response_format == "url":
            return [data.url for data in resp.data]
        return [data.b64_json for data in resp.data]

    async def agenerate_images(
        self,
        prompt: str,
        n: int = 1,
        size: Literal[
            "auto",
            "1024x1024",
            "1536x1024",
            "1024x1536",
            "256x256",
            "512x512",
            "1792x1024",
            "1024x1792",
        ] = "1024x1024",
        response_format: Literal["url", "b64_json"] = "url",
    ) -> List[str]:
        """Asynchronous counterpart to :meth:`generate_images`."""
        client = self._async_client()
        resp = await client.images.generate(
            model=self.model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
        )
        await client.aclose()
        if response_format == "url":
            return [data.url for data in resp.data]
        return [data.b64_json for data in resp.data]
