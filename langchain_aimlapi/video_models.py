from __future__ import annotations

import os
import time
from typing import Any, List, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS


class AimlapiVideoModel(LLM):
    """Generate videos using the Aimlapi service."""

    model: str = Field(default="google/veo3")
    provider: str = Field(default="google")
    api_key: Optional[str] = Field(default=None, alias="api_key")
    base_url: str = Field(default="https://api.aimlapi.com/v2", alias="base_url")
    timeout: Optional[float] = Field(default=None, alias="timeout")
    max_retries: int = Field(default=2, alias="max_retries")

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        videos = self.generate_videos(prompt=prompt, n=1, **kwargs)
        return videos[0]

    @property
    def _llm_type(self) -> str:
        return "aimlapi-video"

    def _client(self) -> httpx.Client:
        """Create a reusable HTTP client with basic retry logic."""
        transport = httpx.HTTPTransport(retries=self.max_retries)
        return httpx.Client(timeout=self.timeout, transport=transport)

    def generate_videos(
            self,
            prompt: str,
            n: int = 1,
            response_format: str = "url",
            poll_interval: float = 10.0,
            timeout: float = 360.0,
    ) -> List[str]:
        headers = {
            **AIMLAPI_HEADERS,
            "Authorization": f"Bearer {self.api_key or os.getenv('AIMLAPI_API_KEY')}",
        }
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "response_format": response_format,
        }

        hook_url = f"{self.base_url}/generate/video/{self.provider}/generation"

        with self._client() as client:
            # Send initial POST request to start video generation
            resp = client.post(hook_url, json=payload, headers=headers)
            resp.raise_for_status()
            jsn = resp.json()

            generation_id = jsn["id"]

            start_time = time.time()

            # Poll until the video is ready, fails, or we hit the timeout
            while True:
                resp = client.get(
                    hook_url,
                    params={"generation_id": generation_id},
                    headers=headers,
                )
                resp.raise_for_status()
                jsn = resp.json()
                status = jsn.get("status")

                if status == "completed":
                    data = jsn.get("video", [])
                    if "url" in data:
                        return [data["url"]]
                    return [data]

                if status in ("failed", "error"):
                    raise RuntimeError(f"Video generation failed: {jsn}")

                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Generation {generation_id} did not complete within {timeout} seconds"
                    )

                time.sleep(poll_interval)
