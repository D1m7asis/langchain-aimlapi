from __future__ import annotations

import os
from typing import Any, List, Optional

import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS


class AimlapiImageModel(LLM):
    """Generate images using the Aimlapi service."""

    model: str = Field(default="dall-e-3")
    api_key: Optional[str] = Field(default=None, alias="api_key")
    base_url: str = Field(default="https://api.aimlapi.com/v1", alias="base_url")
    timeout: Optional[float] = Field(default=None, alias="timeout")
    max_retries: int = Field(default=2, alias="max_retries")

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        images = self.generate_images(prompt=prompt, n=1, **kwargs)
        return images[0]

    @property
    def _llm_type(self) -> str:
        return "aimlapi-image"

    def _client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.api_key or os.getenv("AIMLAPI_API_KEY"),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=AIMLAPI_HEADERS,
        )

    def generate_images(
        self,
        prompt: str,
        n: int = 1,
        size: str = "1024x1024",
        response_format: str = "url",
    ) -> List[str]:
        client = self._client()
        resp = client.images.generate(
            model=self.model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
        )
        if response_format == "url":
            return [d.url for d in resp.data]
        return [d.b64_json for d in resp.data]
