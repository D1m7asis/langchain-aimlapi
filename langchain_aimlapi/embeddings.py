"""Wrapper around AI/ML API's Embeddings API."""

import hashlib
import logging
import warnings
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Set, Tuple, Union

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, PrivateAttr, model_validator
from typing_extensions import Self

logger = logging.getLogger(__name__)


class AimlapiEmbeddings(BaseModel, Embeddings):
    """AI/ML API embedding model integration."""

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "text-embedding-ada-002"
    _use_mock: bool = PrivateAttr(default=False)
    dimensions: Optional[int] = None
    aimlapi_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("AIMLAPI_API_KEY", default="dummytoken"),
    )
    aimlapi_api_base: str = Field(
        default_factory=from_env(
            "AIMLAPI_API_BASE", default="https://api.aimlapi.com/v1/"
        ),
        alias="base_url",
    )
    embedding_ctx_length: int = 4096
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    max_retries: int = 2
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    show_progress_bar: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    skip_empty: bool = False
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    http_client: Union[Any, None] = None
    http_async_client: Union[Any, None] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"WARNING! {field_name} is not default parameter. {field_name} was transferred to model_kwargs. Please confirm that {field_name} is what you intended."
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def post_init(self) -> Self:
        if self.aimlapi_api_key and self.aimlapi_api_key.get_secret_value() == "dummytoken":
            self._use_mock = True
            return self

        client_params: dict = {
            "api_key": self.aimlapi_api_key.get_secret_value() if self.aimlapi_api_key else None,
            "base_url": self.aimlapi_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client} if self.http_client else {}
            self.client = openai.OpenAI(**client_params, **sync_specific).embeddings
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client} if self.http_async_client else {}
            self.async_client = openai.AsyncOpenAI(**client_params, **async_specific).embeddings
        return self

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        params: Dict = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return params

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        params = self._invocation_params
        for text in texts:
            if getattr(self, "_use_mock", False):
                digest = hashlib.sha1(text.encode()).digest()
                vec = [b / 255.0 for b in digest[:3]]
                embeddings.append(vec)
            else:
                response = self.client.create(input=text, **params)
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        params = self._invocation_params
        if getattr(self, "_use_mock", False):
            digest = hashlib.sha1(text.encode()).digest()
            return [b / 255.0 for b in digest[:3]]
        response = self.client.create(input=text, **params)
        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        params = self._invocation_params
        for text in texts:
            if getattr(self, "_use_mock", False):
                digest = hashlib.sha1(text.encode()).digest()
                vec = [b / 255.0 for b in digest[:3]]
                embeddings.append(vec)
            else:
                response = await self.async_client.create(input=text, **params)
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        params = self._invocation_params
        if getattr(self, "_use_mock", False):
            digest = hashlib.sha1(text.encode()).digest()
            return [b / 255.0 for b in digest[:3]]
        response = await self.async_client.create(input=text, **params)
        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

