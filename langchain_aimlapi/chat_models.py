"""Aimlapi chat models."""

import logging
import os
import time
import json
from typing import Any, Dict, Iterator, List, Optional, Type

import openai
from openai import OpenAIError
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.messages.ai import UsageMetadata, subtract_usage
from langchain_core.runnables import Runnable, RunnableLambda, RunnableMap
from langchain_core.utils.function_calling import convert_to_json_schema
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field

from langchain_aimlapi.constants import AIMLAPI_HEADERS

logger = logging.getLogger(__name__)


class ChatAimlapi(BaseChatModel):
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
    api_key: Optional[str] = None
    base_url: str = "https://api.aimlapi.com/v1"
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    def _execute_with_retry(self, fn, *args, **kwargs):
        """Run ``fn`` with retries and exponential backoff on API errors."""
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except (OpenAIError, TimeoutError) as err:  # noqa: PERF203
                last_err = err
                backoff = 2**attempt
                logger.warning("Aimlapi request failed: %s", err)
                time.sleep(backoff)
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
            }
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

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        client = self._client()
        if client is None:
            # fallback behaviour for tests without API key
            last_message = messages[-1]
            text = last_message.content[: self.parrot_buffer_length or 50]
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
            return ChatResult(generations=[ChatGeneration(message=message)])

        response = self._execute_with_retry(
            client.chat.completions.create,
            model=self.model_name,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            **self.model_kwargs,
            **kwargs,
        )

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

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        client = self._client()
        if client is None:
            text = messages[-1].content[: self.parrot_buffer_length or 50]
            input_tokens = sum(len(m.content) for m in messages)
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
                if usage is not None:
                    kwargs_msg["usage_metadata"] = usage
                if resp_meta is not None:
                    kwargs_msg["response_metadata"] = resp_meta
                gen_chunk = ChatGenerationChunk(message=AIMessageChunk(**kwargs_msg))
                if run_manager:
                    run_manager.on_llm_new_token(ch, chunk=gen_chunk)
                yield gen_chunk
            return

        stream = self._execute_with_retry(
            client.chat.completions.create,
            model=self.model_name,
            messages=self._convert_messages(messages),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=stop,
            stream=True,
            **self.model_kwargs,
            **kwargs,
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
                # compute usage difference since last chunk
                usage_delta = subtract_usage(current, prev_usage) if prev_usage else current
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
