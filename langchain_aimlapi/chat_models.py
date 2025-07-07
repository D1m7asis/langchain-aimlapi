"""Wrapper around AI/ML API chat completions."""

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Iterable,
    AsyncIterable,
    Mapping,
)

import hashlib
import openai
from langchain_core.language_models.chat_models import LangSmithParams, BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
    AsyncCallbackManagerForLLMRun,
)
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI

from .constants import AIMLAPI_HEADERS
from pydantic import ConfigDict, Field, SecretStr, PrivateAttr, model_validator
from typing_extensions import Self


class ChatAimlapi(BaseChatOpenAI):
    """Chat model powered by AI/ML API."""

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"aimlapi_api_key": "AIMLAPI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return ["langchain", "chat_models", "aimlapi"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        attributes: Dict[str, Any] = {}
        if self.aimlapi_api_base:
            attributes["aimlapi_api_base"] = self.aimlapi_api_base
        return attributes

    @property
    def _llm_type(self) -> str:
        return "aimlapi-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "aimlapi"
        return params

    bind_tools = BaseChatModel.bind_tools
    with_structured_output = BaseChatModel.with_structured_output

    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """Model name to use."""

    _use_mock: bool = PrivateAttr(default=False)

    default_headers: Optional[Mapping[str, str]] = Field(
        default_factory=lambda: AIMLAPI_HEADERS.copy()
    )

    aimlapi_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("AIMLAPI_API_KEY", default="dummytoken"),
    )
    """AI/ML API key."""

    aimlapi_api_base: str = Field(
        default_factory=from_env(
            "AIMLAPI_API_BASE", default="https://api.aimlapi.com/v1/"
        ),
        alias="base_url",
    )

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        if self.aimlapi_api_key and self.aimlapi_api_key.get_secret_value() == "dummytoken":
            self._use_mock = True
            return self

        client_params: dict = {
            "api_key": self.aimlapi_api_key.get_secret_value() if self.aimlapi_api_key else None,
            "base_url": self.aimlapi_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return self

    def _fake_answer(self, messages: Sequence[BaseMessage]) -> str:
        joined = " ".join(getattr(m, "content", "") for m in messages)
        digest = hashlib.sha1(joined.encode()).hexdigest()
        return f"mock-{digest[:8]}"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if getattr(self, "_use_mock", False):
            content = self._fake_answer(messages)
            msg = AIMessage(content=content)
            return ChatResult(
                generations=[ChatGeneration(message=msg)],
                llm_output={"model_name": self.model_name},
            )
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterable[ChatGenerationChunk]:
        if getattr(self, "_use_mock", False):
            text = self._fake_answer(messages)
            for token in text.split():
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
            return
        yield from super()._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterable[ChatGenerationChunk]:
        if getattr(self, "_use_mock", False):
            text = self._fake_answer(messages)
            for token in text.split():
                yield ChatGenerationChunk(message=AIMessageChunk(content=token))
            return
        async for chunk in super()._astream(messages, stop=stop, run_manager=run_manager, **kwargs):
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if getattr(self, "_use_mock", False):
            return self._generate(messages, stop=stop, run_manager=None, **kwargs)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
