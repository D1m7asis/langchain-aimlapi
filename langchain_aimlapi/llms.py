"""Completion models for Aimlapi."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import Generation, GenerationChunk, LLMResult
from openai import APIConnectionError, OpenAIError
from pydantic import Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS

logger = logging.getLogger(__name__)


class AimlapiLLM(LLM):
    """Wrapper for the OpenAI-compatible Aimlapi completion API.

    Additional options can be supplied via ``model_kwargs`` with conflicting
    keys removed automatically. Requests are retried with exponential backoff
    and the model falls back to parroting when no API key is provided.
    """

    model_name: str = Field(alias="model")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    api_key: Optional[str] = None
    base_url: str = "https://api.aimlapi.com/v1"
    parrot_buffer_length: int = 0
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "aimlapi-llm"

    def _client(self) -> Optional[openai.OpenAI]:
        api_key = self.api_key or os.getenv("AIMLAPI_API_KEY")
        if api_key is None:
            return None
        return openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=AIMLAPI_HEADERS,
        )

    def _filter_model_kwargs(self) -> Dict[str, Any]:
        """Remove keys that collide with explicit parameters."""
        return {
            k: v
            for k, v in self.model_kwargs.items()
            if k not in {"model", "temperature", "max_tokens", "stop"}
        }

    def _execute_with_retry(self, fn, *args, **kwargs):
        """Synchronously run ``fn`` with retry logic."""
        return self._execute_with_retry_sync(fn, *args, **kwargs)

    def _execute_with_retry_sync(self, fn, *args, **kwargs):
        """Execute a client call with retries and exponential backoff."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (OpenAIError, APIConnectionError, TimeoutError) as err:  # noqa: PERF203
                last_err = err
                backoff = min(2**attempt, 8)
                logger.warning("AimlapiLLM error on attempt %s: %s", attempt + 1, err)
                time.sleep(backoff)
        raise last_err  # type: ignore[misc]

    async def _execute_with_retry_async(self, fn, *args, **kwargs):
        """Async version of :meth:`_execute_with_retry_sync`."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except (OpenAIError, APIConnectionError, TimeoutError) as err:  # noqa: PERF203
                last_err = err
                backoff = min(2**attempt, 8)
                logger.warning("AimlapiLLM error on attempt %s: %s", attempt + 1, err)
                await asyncio.sleep(backoff)
        raise last_err

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = self._client()
        if client is None:
            text = prompt[: self.parrot_buffer_length or 50]
            # fallback to parroting the prompt
            generation = Generation(text=text)
            return LLMResult(generations=[[generation]])
        # prepare parameters for single completion request
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop if isinstance(stop, list) else [stop] if stop else None,
        }
        params.update(self._filter_model_kwargs())
        params.update(kwargs)
        # issue request with retry
        response = self._execute_with_retry_sync(client.completions.create, **params)
        choice = response.choices[0]
        usage = response.usage
        message = AIMessage(
            content=choice.text or "",
            usage_metadata={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            response_metadata={
                "model_name": self.model_name,
                "finish_reason": choice.finish_reason,
            },
        )
        generation = Generation(
            text=choice.text or "", generation_info={"message": message}
        )
        return LLMResult(generations=[[generation]])

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = self._client()
        if client is None:
            text = prompt[: self.parrot_buffer_length or 50]
            generation = Generation(text=text)
            return LLMResult(generations=[[generation]])
        # prepare parameters for async completion request
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop if isinstance(stop, list) else [stop] if stop else None,
        }
        params.update(self._filter_model_kwargs())
        params.update(kwargs)
        # issue async request with retry
        response = await self._execute_with_retry_async(  # type: ignore[call-arg]
            client.completions.create, **params
        )
        choice = response.choices[0]
        usage = response.usage
        message = AIMessage(
            content=choice.text or "",
            usage_metadata={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            response_metadata={
                "model_name": self.model_name,
                "finish_reason": choice.finish_reason,
            },
        )
        generation = Generation(
            text=choice.text or "", generation_info={"message": message}
        )
        return LLMResult(generations=[[generation]])
