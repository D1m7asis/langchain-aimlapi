from __future__ import annotations

import os
import time
from typing import List, Optional

import httpx
from pydantic import Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS


class AimlapiVideoGenerator:
    """Generate videos using the Aimlapi service."""

    model: str = Field(default="google/veo3")
    provider: str = Field(default="google")
    api_key: Optional[str] = None
    base_url: str = "https://api.aimlapi.com/v1"
    timeout: Optional[float] = None
    max_retries: int = 2

    def __init__(
        self,
        model: str = "google/veo3",
        provider: str = "google",
        api_key: Optional[str] = None,
        base_url: str = "https://api.aimlapi.com/v2",
        timeout: Optional[float] = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def generate(
            self,
            prompt: str,
            n: int = 1,
            response_format: str = "url",
            poll_interval: float = 10.0,
            timeout: float = 360.0,
    ) -> List[str]:
        client = self._client()
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

        # Send initial POST request to start video generation
        hook_url = f"{self.base_url}/generate/video/{self.provider}/generation"
        resp = client.post(hook_url, json=payload, headers=headers)
        resp.raise_for_status()
        jsn = resp.json()

        # Extract the generation ID from the response
        generation_id = jsn["id"]

        # Prepare URL for polling (body-only identifier)
        start_time = time.time()

        # Poll until the video is ready, fails, or we hit the timeout
        while True:
            # Send generation_id in the JSON body
            resp = client.get(
                hook_url,
                params={"generation_id": generation_id},
                headers=headers,
            )
            resp.raise_for_status()
            jsn = resp.json()
            status = jsn.get("status")

            if status == "completed":
                # Once completed, return the URLs or base64 data
                data = jsn.get("video", [])
                if "url" in data:
                    return [data["url"]]
                else:
                    return [data]

            if status in ("failed", "error"):
                # Raise an error if generation fails
                raise RuntimeError(f"Video generation failed: {jsn}")

            # Check for timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(
                    f"Generation {generation_id} did not complete within {timeout} seconds"
                )

            # Wait before polling again
            time.sleep(poll_interval)
