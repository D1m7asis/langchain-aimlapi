from __future__ import annotations

import os
from typing import List, Optional, Any

import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS
from langchain_core.language_models.llms import LLM


class AimlapiImageModel(LLM):
    """Generate images using the Aimlapi service."""

    def _call(self, prompt: str, stop: Optional[list[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        pass

    @property
    def _llm_type(self) -> str:
        pass

    model: str = Field(default="dall-e-3")
    api_key: Optional[str] = None
    base_url: str = "https://api.aimlapi.com/v1"
    timeout: Optional[float] = None
    max_retries: int = 2

    def __init__(
        self,
        model: str = "dall-e-3",
        api_key: Optional[str] = None,
        base_url: str = "https://api.aimlapi.com/v1",
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _client(self) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=self.api_key or os.getenv("AIMLAPI_API_KEY"),
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=AIMLAPI_HEADERS,
        )

    def generate(
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
