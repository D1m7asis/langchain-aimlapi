"""Wrapper around AI/ML API chat completions."""

from typing import Any, Dict, List, Optional

import openai
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
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

    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    """Model name to use."""

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

        client_params: dict = {
            "api_key": self.aimlapi_api_key.get_secret_value()
            if self.aimlapi_api_key
            else None,
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
