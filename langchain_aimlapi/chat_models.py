# Wrapper around AI/ML API chat completions, OpenAI-compatible interface

from __future__ import annotations

import hashlib
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
)

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, model_validator
from typing_extensions import Self

from .constants import AIMLAPI_HEADERS


class ChatAimlapi(BaseChatOpenAI):
    """
    Chat model powered by AI/ML API, fully OpenAI-compatible.

    This class wraps an OpenAI-like chat completion endpoint provided by the
    AI/ML API service. Supports both real API calls and a mock mode (parrot) for
    offline testing.
    """

    # Pydantic configuration to read fields by their alias names
    model_config = ConfigDict(populate_by_name=True)

    # Core parameters for chat invocation
    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """Name of the chat model to use (e.g., "gpt-3.5-turbo")."""

    # Internal flag to switch to mock mode when using dummy key
    _use_mock: bool = PrivateAttr(default=False)

    # Headers used for all HTTP requests to the API
    default_headers: Dict[str, str] = Field(
        default_factory=lambda: AIMLAPI_HEADERS.copy()
    )

    # API key, read from environment if not provided explicitly
    aimlapi_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("AIMLAPI_API_KEY", default="dummytoken"),
    )
    """AI/ML API key; if not set, mock mode is enabled."""

    # Base URL for the chat completion endpoint
    aimlapi_api_base: str = Field(
        alias="base_url",
        default_factory=from_env(
            "AIMLAPI_API_BASE", default="https://api.aimlapi.com/v1/"
        ),
    )
    """API base URL for chat completions."""

    # -------------------------------------------------------------------------
    # LangChain-specific serialization hooks
    # -------------------------------------------------------------------------

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Mapping for hiding secret fields in serialized output."""
        return {"aimlapi_api_key": "AIMLAPI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Namespace for LangChain serialization (e.g., for LangSmith)."""
        return ["langchain", "chat_models", "aimlapi"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """Additional non-secret fields to include in serialized output."""
        attrs: Dict[str, Any] = {}
        if self.aimlapi_api_base:
            attrs["aimlapi_api_base"] = self.aimlapi_api_base
        return attrs

    def _get_ls_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """
        Extend LangSmith params with provider info.
        Called internally by LangChain for logging/monitoring.
        """
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "aimlapi"
        return params

    # Inherit tool binding and structured output handling from BaseChatModel
    bind_tools = BaseChatModel.bind_tools  # type: ignore[assignment]
    with_structured_output = BaseChatModel.with_structured_output  # type: ignore[assignment]

    # -------------------------------------------------------------------------
    # Environment validation and client initialization
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """
        After Pydantic validation, configure clients or enable mock mode.

        - If API key is dummy, switch to mock mode.
        - Otherwise, instantiate OpenAI-compatible sync/async clients.
        """
        # Validate 'n' and streaming constraints
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        # Check for dummy API key to enable mock (parrot) mode
        key = self.aimlapi_api_key.get_secret_value() if self.aimlapi_api_key else None
        if key == "dummytoken":
            self._use_mock = True
            return self

        # Build parameters for real API clients
        client_params: dict = {
            "api_key": key,
            "base_url": self.aimlapi_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        # Instantiate synchronous client for chat completions
        if not getattr(self, "client", None):
            sync_opts = {"http_client": self.http_client} if self.http_client else {}
            # chat.completions is a Completions object (with .create method)
            self.client = openai.OpenAI(**client_params, **sync_opts).chat.completions

        # Instantiate asynchronous client for chat completions
        if not getattr(self, "async_client", None):
            async_opts = (
                {"http_client": self.http_async_client}
                if self.http_async_client
                else {}
            )
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_opts
            ).chat.completions

        return self

    # -------------------------------------------------------------------------
    # Mock (parrot) answer generator for testing without API calls
    # -------------------------------------------------------------------------

    def _fake_answer(self, messages: Sequence[BaseMessage]) -> str:
        """
        Generate a deterministic mock response based on message contents.
        Returns an AIMessage with or without tool_calls depending on input.
        """
        text = " ".join(getattr(m, "content", "") for m in messages)
        digest = hashlib.sha1(text.encode()).hexdigest()
        return f"mock-{digest[:8]}"

    # -------------------------------------------------------------------------
    # Core generation logic: override sync and stream methods
    # -------------------------------------------------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Synchronous chat completion:
        - If mock mode is on, return a fake AIMessage.
        - Otherwise defer to BaseChatOpenAI._generate.
        """
        if self._use_mock:
            content = self._fake_answer(messages)
            msg = AIMessage(content=content)
            # ChatResult with one generation and metadata
            return ChatResult(
                generations=[ChatGeneration(message=msg)],
                llm_output={"model_name": self.model_name},
            )
        # Real API call
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Synchronous streaming chat completion:
        - If mock mode is on, yield chunks of the fake answer.
        - Otherwise defer to BaseChatOpenAI._stream.
        """
        if self._use_mock:
            text = self._fake_answer(messages)
            # Yield each token as a separate chunk
            for token in text.split():
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
            return
        yield from super()._stream(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        *,
        stream_usage: Optional[bool] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """
        Asynchronous streaming chat completion, analogous to _stream().
        """
        if self._use_mock:
            text = self._fake_answer(messages)
            for token in text.split():
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
            return
        async for chunk in super()._astream(
            messages,
            stop=stop,
            run_manager=run_manager,
            stream_usage=stream_usage,
            **kwargs,
        ):
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Asynchronous chat generation:
        - If mock mode is on, reuse synchronous fake generation logic.
        - Otherwise defer to BaseChatOpenAI._agenerate.
        """
        if self._use_mock:
            # Leverage synchronous fake answer inside async context
            return self._generate(messages, stop=stop, run_manager=None, **kwargs)
        return await super()._agenerate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
