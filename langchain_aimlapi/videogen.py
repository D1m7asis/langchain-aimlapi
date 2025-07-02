from __future__ import annotations

import os
from typing import List, Optional

import httpx
from pydantic import Field


class AimlapiVideoGenerator:
    """Generate videos using the Aimlapi service."""

    model: str = Field(default="aiml-video-001")
    api_key: Optional[str] = None
    base_url: str = "https://api.aimlapi.com/v1"
    timeout: Optional[float] = None
    max_retries: int = 2

    def __init__(self, model: str = "aiml-video-001", api_key: Optional[str] = None, base_url: str = "https://api.aimlapi.com/v1", timeout: Optional[float] = None, max_retries: int = 2) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout)

    def generate(self, prompt: str, n: int = 1, response_format: str = "url") -> List[str]:
        client = self._client()
        headers = {"Authorization": f"Bearer {self.api_key or os.getenv('AIMLAPI_API_KEY')}"}
        payload = {"model": self.model, "prompt": prompt, "n": n, "response_format": response_format}
        url = f"{self.base_url}/videos/generations"
        for _ in range(self.max_retries + 1):
            resp = client.post(url, json=payload, headers=headers)
            if resp.status_code >= 500:
                continue
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if response_format == "url":
                return [d["url"] for d in data]
            return [d["b64_json"] for d in data]
        resp.raise_for_status()
        return []
