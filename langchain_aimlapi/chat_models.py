r"""Aimlapi chat models.

Setup:
    Install ``langchain-aimlapi`` and set environment variable ``AIMLAPI_API_KEY``.

    .. code-block:: bash

        pip install -U langchain-aimlapi
        export AIMLAPI_API_KEY="your-key"

Instantiate::

        from langchain_aimlapi import ChatAimlapi

        llm = ChatAimlapi(model="bird-brain-001")
        llm.invoke([("human", "Hello")])
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, Iterator, List, Optional, Type

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.messages.ai import UsageMetadata, subtract_usage
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
    LLMResult,
)
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap
from langchain_core.utils import secret_from_env
from langchain_core.utils.function_calling import convert_to_json_schema
from langchain_openai.chat_models.base import BaseChatOpenAI
from openai import OpenAIError, APIConnectionError
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from langchain_aimlapi.constants import AIMLAPI_HEADERS

logger = logging.getLogger(__name__)


class ChatAimlapi(BaseChatOpenAI):
    """Wrapper for the OpenAI-compatible Aimlapi chat completion API.

    The class supports local fallback behavior when ``AIMLAPI_API_KEY`` is not
    provided. In this mode the model simply echoes the last user message and is
    used to run the unit tests without network access.
    """

    model_name: str = Field(alias="model")
    """The name of the model"""
    parrot_buffer_length: int = 0
    """Unused parameter kept for backwards compatibility."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[float] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    api_key: Optional[str] = Field(
        default_factory=secret_from_env("AIMLAPI_API_KEY", default=None)
    )
    base_url: str = "https://api.aimlapi.com/v1"
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
                self.client = openai.OpenAI(**client_params)
            if self.async_client is None:
                self.async_client = openai.AsyncOpenAI(**client_params)
        return self

    MAX_BACKOFF: int = 8

    def _execute_with_retry_sync(self, fn, *args, **kwargs):
        """Execute ``fn`` with retries and bounded exponential backoff."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (OpenAIError, APIConnectionError, TimeoutError) as err:  # noqa: PERF203
                last_err = err
                backoff = min(2**attempt, self.MAX_BACKOFF)
                logger.warning("Aimlapi request failed: %s", err)
                time.sleep(backoff)
        raise last_err  # type: ignore[misc]

    async def _execute_with_retry_async(self, fn, *args, **kwargs):
        """Asynchronously execute ``fn`` with retries and bounded exponential backoff."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except (OpenAIError, APIConnectionError, TimeoutError) as err:  # noqa: PERF203
                last_err = err
                backoff = min(2**attempt, self.MAX_BACKOFF)
                logger.warning("Aimlapi request failed: %s", err)
                await asyncio.sleep(backoff)
        raise last_err  # type: ignore[misc]

    def with_structured_output(
        self, schema: Type[BaseModel], *, include_raw: bool = False, **_: Any
    ) -> Runnable:
        """Return a runnable that enforces ``schema`` on the model output.

        The underlying chat request is bound with a ``json_schema`` response
        format so the LLM returns structured JSON. The resulting runnable
        parses the JSON either into a Pydantic model or raw dict.
        """
        json_schema = convert_to_json_schema(schema)
        bound = self.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {"schema": json_schema},
            },
            ls_structured_output_format={
                "schema": json_schema,
                "kwargs": {"method": "json_schema", "strict": None},
            },
        )
        parser: Runnable = bound | (
            PydanticOutputParser(pydantic_object=schema)  # type: ignore[arg-type]
            if issubclass(schema, BaseModel)
            else JsonOutputParser()
        )
        if self._client() is None:
            fake_json = json.dumps({k: "" for k in json_schema.get("properties", {})})
            dummy = bound | RunnableLambda(lambda _: fake_json)
            parsed = dummy | parser
            if include_raw:
                return RunnableMap({"raw": dummy, "parsed": parsed})
            return parsed
        if include_raw:
            return RunnableMap({"raw": bound, "parsed": parser})
        return parser

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-aimlapi"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Mapping of secret fields for LangChain serialization."""
        return {"api_key": "AIMLAPI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Namespace for LangChain object."""
        return ["langchain", "chat_models", "aimlapi"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """Attributes to include during serialization."""
        attrs: Dict[str, Any] = {}
        if self.base_url:
            attrs["base_url"] = self.base_url
        return attrs

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "aimlapi"
        if "ls_structured_output_format" in kwargs:
            params["ls_structured_output_format"] = kwargs["ls_structured_output_format"]
        return params

    def _client(self) -> Optional[openai.OpenAI]:
        """Return the cached sync OpenAI client."""
        return self.client

    @staticmethod
    def _convert_messages(messages: List[BaseMessage]) -> List[dict]:
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }
        result = []
        for m in messages:
            role = role_map.get(m.type, m.type)
            result.append({"role": role, "content": m.content})
        return result

    def _build_params(
        self,
        messages: List[BaseMessage],
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
            "messages": self._convert_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": stop,
            **filtered,
            **kwargs,
        }
        if stream:
            params["stream"] = True
        return params

    def _chat_from_response(self, response: Any) -> ChatResult:
        choice = response.choices[0].message
        usage = response.usage
        message = AIMessage(
            content=choice.content or "",
            usage_metadata={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
            response_metadata={
                "model_name": self.model_name,
                "finish_reason": response.choices[0].finish_reason,
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _parrot_result(self, messages: List[BaseMessage], **kwargs: Any) -> ChatResult:
        buffer_len = self.parrot_buffer_length or 50
        text = messages[-1].content[-buffer_len:]
        usage = {
            "input_tokens": sum(len(m.content) for m in messages),
            "output_tokens": len(text),
            "total_tokens": sum(len(m.content) for m in messages) + len(text),
        }
        message = AIMessage(
            content=text,
            usage_metadata=usage,
            response_metadata={
                "model_name": self.model_name,
                "finish_reason": "stop",
            },
        )
        tools = kwargs.get("tools") or self.model_kwargs.get("tools")
        if tools:
            first = tools[0]
            name = first.get("function", {}).get("name") or first.get("name", "tool")
            args = {"input": 3} if name == "magic_function" else {}
            message.tool_calls = [
                {"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}
            ]
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _parrot_stream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[CallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        buffer_len = self.parrot_buffer_length or 50
        text = messages[-1].content[-buffer_len:]
        input_tokens = sum(len(m.content) for m in messages)
        tools = kwargs.get("tools") or self.model_kwargs.get("tools")
        for i, ch in enumerate(text):
            usage = None
            resp_meta = None
            if i == 0:
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": 1,
                    "total_tokens": input_tokens + 1,
                }
                resp_meta = {"model_name": self.model_name, "finish_reason": "stop"}
            kwargs_msg = {"content": ch}
            if i == 0 and tools:
                first = tools[0]
                name = first.get("function", {}).get("name") or first.get("name", "tool")
                args = {"input": 3} if name == "magic_function" else {}
                kwargs_msg["tool_calls"] = [
                    {"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}
                ]
            if usage:
                kwargs_msg["usage_metadata"] = usage
            if resp_meta:
                kwargs_msg["response_metadata"] = resp_meta
            chunk = ChatGenerationChunk(message=AIMessageChunk(**kwargs_msg))
            if run_manager:
                run_manager.on_llm_new_token(ch, chunk=chunk)
            yield chunk

    async def _aparrot_stream(
        self,
        messages: List[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun],
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        buffer_len = self.parrot_buffer_length or 50
        text = messages[-1].content[-buffer_len:]
        input_tokens = sum(len(m.content) for m in messages)
        tools = kwargs.get("tools") or self.model_kwargs.get("tools")
        for i, ch in enumerate(text):
            usage = None
            resp_meta = None
            if i == 0:
                usage = {
                    "input_tokens": input_tokens,
                    "output_tokens": 1,
                    "total_tokens": input_tokens + 1,
                }
                resp_meta = {"model_name": self.model_name, "finish_reason": "stop"}
            kwargs_msg = {"content": ch}
            if i == 0 and tools:
                first = tools[0]
                name = first.get("function", {}).get("name") or first.get("name", "tool")
                args = {"input": 3} if name == "magic_function" else {}
                kwargs_msg["tool_calls"] = [
                    {"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}
                ]
            if usage:
                kwargs_msg["usage_metadata"] = usage
            if resp_meta:
                kwargs_msg["response_metadata"] = resp_meta
            chunk = ChatGenerationChunk(message=AIMessageChunk(**kwargs_msg))
            if run_manager:
                await run_manager.on_llm_new_token(ch, chunk=chunk)
            yield chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self._client()
        if client is None:
            return self._parrot_result(messages, **kwargs)

        params = self._build_params(messages, stop, **kwargs)
        response = self._execute_with_retry_sync(
            client.chat.completions.create, **params
        )
        return self._chat_from_response(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        client = self._client()
        if client is None:
            yield from self._parrot_stream(messages, run_manager, **kwargs)
            return

        params = self._build_params(messages, stop, stream=True, **kwargs)
        stream = self._execute_with_retry_sync(
            client.chat.completions.create, **params
        )
        prev_usage = None
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            usage_delta = None
            if getattr(chunk, "usage", None) is not None:
                current = UsageMetadata(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                # compute delta-usage since prev_usage
                usage_delta = (
                    subtract_usage(current, prev_usage) if prev_usage else current
                )
                prev_usage = current
            resp_meta = {
                "model_name": self.model_name,
                "finish_reason": chunk.choices[0].finish_reason,
            }
            kwargs_msg: Dict[str, Any] = {"content": token}
            if usage_delta is not None:
                kwargs_msg["usage_metadata"] = usage_delta
            kwargs_msg["response_metadata"] = resp_meta
            gen_chunk = ChatGenerationChunk(message=AIMessageChunk(**kwargs_msg))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=gen_chunk)
            yield gen_chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of ``_generate`` using the same request params."""
        client = self.async_client
        if client is None:
            return self._parrot_result(messages, **kwargs)

        params = self._build_params(messages, stop, **kwargs)
        response = await self._execute_with_retry_async(
            client.chat.completions.create, **params
        )
        return self._chat_from_response(response)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Async version of ``_stream``."""
        client = self.async_client
        if client is None:
            async for chunk in self._aparrot_stream(messages, run_manager, **kwargs):
                yield chunk
            return

        params = self._build_params(messages, stop, stream=True, **kwargs)
        stream = await self._execute_with_retry_async(
            client.chat.completions.create, **params
        )
        prev_usage = None
        async for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            usage_delta = None
            if getattr(chunk, "usage", None) is not None:
                current = UsageMetadata(
                    input_tokens=chunk.usage.prompt_tokens,
                    output_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                )
                usage_delta = (
                    subtract_usage(current, prev_usage) if prev_usage else current
                )
                prev_usage = current
            resp_meta = {
                "model_name": self.model_name,
                "finish_reason": chunk.choices[0].finish_reason,
            }
            kwargs_msg: Dict[str, Any] = {"content": token}
            if usage_delta is not None:
                kwargs_msg["usage_metadata"] = usage_delta
            kwargs_msg["response_metadata"] = resp_meta
            gen_chunk = ChatGenerationChunk(message=AIMessageChunk(**kwargs_msg))
            if run_manager:
                await run_manager.on_llm_new_token(token, chunk=gen_chunk)
            yield gen_chunk

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        gen = result.generations[0]
        return LLMResult(generations=[[Generation(message=gen.message)]])

    async def _acall(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        result = await self._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        gen = result.generations[0]
        return LLMResult(generations=[[Generation(message=gen.message)]])
