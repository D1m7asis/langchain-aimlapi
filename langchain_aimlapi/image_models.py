from __future__ import annotations

import os
from typing import Any, List, Optional, Literal

import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from openai.types import ImagesResponse
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
        resp: ImagesResponse = client.images.generate(
            model=self.model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
        )

        # Normalize resp.images into a list of items
        raw = getattr(resp, "images", None)
        if isinstance(raw, dict):
            items = [raw]
        elif isinstance(raw, list):
            items = raw
        else:
            raise RuntimeError(f"Unexpected images field type: {type(raw)}")

        result: list[str] = []
        for idx, item in enumerate(items):
            # item may come in as a string or dict
            if response_format == "url":
                if isinstance(item, dict):
                    url = item.get("url")
                    if not isinstance(url, str):
                        raise RuntimeError(
                            f"Image #{idx} missing or invalid URL: {url!r}"
                        )
                    result.append(url)
                elif isinstance(item, str):
                    result.append(item)
                else:
                    raise RuntimeError(
                        f"Image #{idx} has unsupported type: {type(item)}"
                    )
            else:  # b64_json
                if isinstance(item, dict):
                    b64 = item.get("b64_json")
                    if not isinstance(b64, str):
                        raise RuntimeError(
                            f"Image #{idx} missing or invalid b64_json: {b64!r}"
                        )
                    result.append(b64)
                elif isinstance(item, str):
                    result.append(item)
                else:
                    raise RuntimeError(
                        f"Image #{idx} has unsupported type: {type(item)}"
                    )

        return result
