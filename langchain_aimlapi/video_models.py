# Wrapper around AI/ML API's video generation endpoint, OpenAI-compatible interface

from __future__ import annotations

import os
import time
from typing import Any, List, Literal, Optional

import httpx
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS


class AimlapiVideoModel(LLM):
    """
    OpenAI-compatible video generation model using the AI/ML API service.

    This class implements the LangChain LLM interface to generate videos
    via a REST API. It supports simple synchronous calls, polling for
    completion, and basic retry logic.
    """

    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------

    model: str = Field(
        default="google/veo3",
        description="Name of the video model (e.g., 'google/veo3').",
    )

    provider: str = Field(
        default="google",
        description="Provider namespace for the API endpoint.",
    )

    api_key: Optional[str] = Field(
        default=None,
        alias="api_key",
        description="Aimlapi API key; falls back to AIMLAPI_API_KEY env var.",
    )

    base_url: str = Field(
        default="https://api.aimlapi.com/v2",
        alias="base_url",
        description="Base URL for the video generation service.",
    )

    timeout: Optional[float] = Field(
        default=None,
        alias="timeout",
        description="HTTP request timeout in seconds.",
    )

    max_retries: int = Field(
        default=2,
        alias="max_retries",
        description="Number of retry attempts on network failure.",
    )

    # -------------------------------------------------------------------------
    # LangChain LLM implementation
    # -------------------------------------------------------------------------

    @property
    def _llm_type(self) -> str:
        """Unique identifier for the LLM type."""
        return "aimlapi-video"

    def _call(
        self,
        prompt: str,
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronously generate a single video URL from a text prompt.

        This method satisfies the LLM interface by returning the first
        video URL produced.
        """
        # Request n=1 and return the first URL
        videos = self.generate_videos(prompt=prompt, n=1, **kwargs)
        return videos[0]

    # -------------------------------------------------------------------------
    # HTTP client setup
    # -------------------------------------------------------------------------

    def _client(self) -> httpx.Client:
        """
        Create a reusable HTTP client with retry transport.

        Uses httpx.HTTPTransport to automatically retry on transient
        network errors up to `max_retries` times.
        """
        transport = httpx.HTTPTransport(retries=self.max_retries)
        return httpx.Client(timeout=self.timeout, transport=transport)

    # -------------------------------------------------------------------------
    # Video generation logic with polling
    # -------------------------------------------------------------------------

    def generate_videos(
        self,
        prompt: str,
        n: int = 1,
        response_format: Literal["url", "b64_json"] = "url",
        poll_interval: float = 10.0,
        timeout: float = 360.0,
    ) -> List[str]:
        """
        Kick off video generation and poll until completion.

        Args:
            prompt: Text description for the desired video content.
            n: Number of videos to generate.
            response_format: 'url' to return URLs, 'b64_json' for base64 data.
            poll_interval: Seconds between status checks.
            timeout: Max seconds to wait before giving up.

        Returns:
            List of video URLs or base64 strings depending on response_format.

        Raises:
            RuntimeError: If generation fails.
            TimeoutError: If polling exceeds the timeout.
        """
        # Build headers, injecting authorization
        headers = {
            **AIMLAPI_HEADERS,
            "Authorization": f"Bearer {self.api_key or os.getenv('AIMLAPI_API_KEY')}",
        }

        # Initial payload to start generation
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": n,
            "response_format": response_format,
        }

        # Construct endpoint URL for video generation
        generation_endpoint = (
            f"{self.base_url}/generate/video/{self.provider}/generation"
        )

        # Use context manager for HTTP client
        with self._client() as client:
            # 1. POST request to initiate generation
            init_resp = client.post(generation_endpoint, json=payload, headers=headers)
            init_resp.raise_for_status()
            init_data = init_resp.json()

            generation_id = init_data.get("id")
            start_time = time.time()

            # 2. Poll loop: GET until status is 'completed'
            while True:
                status_resp = client.get(
                    generation_endpoint,
                    params={"generation_id": generation_id},
                    headers=headers,
                )
                status_resp.raise_for_status()
                status_data = status_resp.json()
                status = status_data.get("status")

                if status == "completed":
                    # Generation succeeded: extract raw video_data
                    video_data = status_data.get("video")

                    # Normalize video_data into a list of items
                    if isinstance(video_data, dict):
                        video_items = [video_data]
                    elif isinstance(video_data, list):
                        video_items = video_data
                    elif isinstance(video_data, str):
                        key = "url" if response_format == "url" else "b64_json"
                        video_items = [{key: video_data}]
                    else:
                        raise RuntimeError(
                            f"Unexpected type for video_data: {type(video_data)}"
                        )

                    # If URLs requested, extract them
                    if response_format == "url":
                        urls: list[str] = []
                        for idx, item in enumerate(video_items):
                            if (
                                isinstance(item, dict)
                                and "url" in item
                                and isinstance(item["url"], str)
                            ):
                                urls.append(item["url"])
                            elif isinstance(item, str):
                                urls.append(item)
                            else:
                                raise RuntimeError(
                                    f"Video item #{idx} has no URL "
                                    f"or wrong type: {item}"
                                )
                        return urls

                    # Otherwise, handle base64 JSON responses
                    b64_results: list[str] = []
                    for idx, item in enumerate(video_items):
                        if (
                            isinstance(item, dict)
                            and "b64_json" in item
                            and isinstance(item["b64_json"], str)
                        ):
                            b64_results.append(item["b64_json"])
                        elif isinstance(item, str):
                            b64_results.append(item)
                        else:
                            raise RuntimeError(
                                f"Video item #{idx} missing "
                                f"'b64_json' or wrong type: {item}"
                            )
                    return b64_results

                if status in ("failed", "error"):
                    # Stop immediately on failure
                    raise RuntimeError(f"Video generation failed: {status_data}")

                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Generation {generation_id} timed out after {timeout} seconds"
                    )

                # Wait before next poll
                time.sleep(poll_interval)
