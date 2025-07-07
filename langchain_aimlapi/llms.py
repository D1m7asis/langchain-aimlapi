"""Completion models for Aimlapi.

Setup:
    Install ``langchain-aimlapi`` and set ``AIMLAPI_API_KEY``.

    .. code-block:: bash

        pip install -U langchain-aimlapi
        export AIMLAPI_API_KEY="your-key"
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, ClassVar, Dict, List, Optional

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.messages import AIMessage
from langchain_core.outputs import Generation, LLMResult
from langchain_core.utils import secret_from_env
from openai import OpenAIError
from openai.error import APIConnectionError
from pydantic import ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_aimlapi.constants import AIMLAPI_HEADERS

logger = logging.getLogger(__name__)


class AimlapiLLM(LLM):
    """Wrapper for the OpenAI-compatible Aimlapi completion API.

    Extra model parameters can be provided via ``model_kwargs``. Both sync and
    async calls retry on ``OpenAIError`` or connection errors using bounded
    exponential backoff.
    """

    model_name: str = Field(alias="model")
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    api_key: Optional[str] = Field(
        default_factory=secret_from_env("AIMLAPI_API_KEY", default=None)
    )
    base_url: str = "https://api.aimlapi.com/v1"
    parrot_buffer_length: int = 0
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    client: Optional[openai.OpenAI] = Field(default=None, exclude=True)
    async_client: Optional[openai.AsyncOpenAI] = Field(default=None, exclude=True)

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Create API clients if a key is available."""
        client_params = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "default_headers": AIMLAPI_HEADERS,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries
        if self.api_key:
            if self.client is None:
                self.client = openai.OpenAI(**client_params).completions
            if self.async_client is None:
                self.async_client = openai.AsyncOpenAI(**client_params).completions
        return self

    @property
    def _llm_type(self) -> str:
        return "aimlapi-llm"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Mapping of secret fields for LangChain serialization."""
        return {"api_key": "AIMLAPI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Namespace for LangChain object."""
        return ["langchain", "llms", "aimlapi"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """Attributes to include during serialization."""
        attrs: Dict[str, Any] = {}
        if self.base_url:
            attrs["base_url"] = self.base_url
        return attrs

    def _client(self) -> Optional[openai.OpenAI]:
        """Return the cached sync OpenAI client."""
        return self.client

    MAX_BACKOFF: ClassVar[int] = 8

    def _execute_with_retry_sync(self, fn, *args, **kwargs):
        """Execute a client call with retries and exponential backoff."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (OpenAIError, APIConnectionError, TimeoutError) as err:  # noqa: PERF203
                last_err = err
                backoff = min(2**attempt, self.MAX_BACKOFF)
                logger.warning("AimlapiLLM error on attempt %s: %s", attempt + 1, err)
                time.sleep(backoff)
        raise last_err  # type: ignore[misc]

    async def _execute_with_retry_async(self, fn, *args, **kwargs):
        """Asynchronously execute a client call with retries and exponential backoff."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except (OpenAIError, APIConnectionError, TimeoutError) as err:  # noqa: PERF203
                last_err = err
                backoff = min(2**attempt, self.MAX_BACKOFF)
                logger.warning("AimlapiLLM error on attempt %s: %s", attempt + 1, err)
                await asyncio.sleep(backoff)
        raise last_err  # type: ignore[misc]

    def _build_params(
        self,
        prompt: str,
        stop: Optional[List[str]],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        filtered = {
            k: v
            for k, v in self.model_kwargs.items()
            if k not in {"model", "temperature", "max_tokens", "stop"}
        }
        if isinstance(stop, str):
            stop = [stop]
        params = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
            **filtered,
            **kwargs,
        }
        if stream:
            params["stream"] = True
        return params

    def _llm_from_response(self, choice: Any, usage: Any) -> LLMResult:
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
        generation = Generation(text=choice.text or "", message=message)
        return LLMResult(generations=[[generation]])

    def _parrot_result(self, prompt: str) -> LLMResult:
        buffer_len = self.parrot_buffer_length or 50
        text = prompt[-buffer_len:]
        message = AIMessage(content=text)
        return LLMResult(generations=[[Generation(text=text, message=message)]])

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = self._client()
        if client is None:
            return self._parrot_result(prompt)

        params = self._build_params(prompt, stop, **kwargs)
        response = self._execute_with_retry_sync(client.create, **params)
        choice = response.choices[0]
        return self._llm_from_response(choice, response.usage)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        client = self.async_client
        if client is None:
            return self._parrot_result(prompt)

        params = self._build_params(prompt, stop, **kwargs)
        response = await self._execute_with_retry_async(client.create, **params)
        choice = response.choices[0]
        return self._llm_from_response(choice, response.usage)
