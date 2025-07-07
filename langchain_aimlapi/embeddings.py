"""Wrapper around AI/ML API's Embeddings API integration for LangChain."""

import hashlib
import logging
import warnings
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    PrivateAttr,
    model_validator,
)
from typing_extensions import Self

from .constants import AIMLAPI_HEADERS

# Configure logger for this module
logger = logging.getLogger(__name__)


# OpenAI-compatible Embeddings model: implements LangChain Embeddings API over an OpenAI-like interface
class AimlapiEmbeddings(BaseModel, Embeddings):
    """
    AI/ML API embedding model integration for generating vector embeddings from text.

    This class wraps the OpenAI-compatible embeddings endpoint provided by the
    Aimlapi service. It supports synchronous and asynchronous embedding,
    automatic retry logic, and a mock mode for testing purposes.

    Attributes:
        client: Sync API client for embedding requests (excluded from serialization).
        async_client: Async API client for embedding requests (excluded from serialization).
        model: Name of the embedding model to use.
        _use_mock: Flag to indicate mock mode (no real API calls).
        dimensions: Optional override for embedding vector size.
        aimlapi_api_key: Secret API key for authentication.
        aimlapi_api_base: Base URL for the Aimlapi service.
        embedding_ctx_length: Context window size for chunking input text.
        allowed_special: Characters/tokens to keep in inputs.
        disallowed_special: Characters/tokens to strip from inputs.
        chunk_size: Length of text chunks when splitting large inputs.
        max_retries: Number of retry attempts on API failure.
        request_timeout: Timeout for HTTP requests (seconds).
        show_progress_bar: Toggle embedding progress bar.
        model_kwargs: Extra provider-specific parameters.
        skip_empty: Skip empty strings in embedding operations.
        default_headers: HTTP headers used for requests.
        default_query: Default query parameters for HTTP client.
        http_client: Custom HTTP client for sync requests.
        http_async_client: Custom HTTP client for async requests.
    """

    # Private attributes excluded from serialization
    client: Any = Field(default=None, exclude=True)
    async_client: Any = Field(default=None, exclude=True)

    # Model configuration
    model: str = Field(default="text-embedding-ada-002")
    _use_mock: bool = PrivateAttr(default=False)
    dimensions: Optional[int] = None

    # Secret and environment configuration
    aimlapi_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("AIMLAPI_API_KEY", default="dummytoken"),
    )

    aimlapi_api_base: str = Field(
        alias="base_url",
        default_factory=from_env(
            "AIMLAPI_API_BASE", default="https://api.aimlapi.com/v1/"
        ),
    )

    # Embedding parameters
    embedding_ctx_length: int = 4096
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000

    # Retry and timeout settings
    max_retries: int = 2
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        alias="timeout",
        default=None,
    )

    show_progress_bar: bool = False
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    skip_empty: bool = False

    # HTTP client overrides
    default_headers: Union[Mapping[str, str], None] = Field(
        default_factory=lambda: AIMLAPI_HEADERS.copy()
    )
    default_query: Union[Mapping[str, object], None] = None
    http_client: Union[Any, None] = None
    http_async_client: Union[Any, None] = None

    # Pydantic strict model config
    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
        """
        Pre-process unknown fields into model_kwargs with warnings.
        """
        all_fields = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})

        for key in list(values):
            if key in extra:
                raise ValueError(f"Duplicate field: {key}")
            if key not in all_fields:
                warnings.warn(f"Unknown field '{key}' moved to model_kwargs.")
                extra[key] = values.pop(key)

        conflicts = all_fields.intersection(extra.keys())
        if conflicts:
            raise ValueError(
                f"Explicit params {conflicts} should not appear in model_kwargs."
            )

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def post_init(self) -> Self:
        """
        Initialize clients or enable mock mode based on API key.
        """
        key = self.aimlapi_api_key.get_secret_value() if self.aimlapi_api_key else None
        if key == "dummytoken":
            self._use_mock = True
            return self

        params = {
            "api_key": key,
            "base_url": self.aimlapi_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        # Sync client
        if not self.client:
            sync_opts = {"http_client": self.http_client} if self.http_client else {}
            self.client = openai.OpenAI(**params, **sync_opts).embeddings

        # Async client
        if not self.async_client:
            async_opts = (
                {"http_client": self.http_async_client}
                if self.http_async_client
                else {}
            )
            self.async_client = openai.AsyncOpenAI(**params, **async_opts).embeddings

        return self

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """
        Build base params for embedding API calls.
        """
        p = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            p["dimensions"] = self.dimensions
        return p

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents; returns deterministic mock or real embeddings.
        """
        results: List[List[float]] = []
        for text in texts:
            if self.skip_empty and not text:
                continue
            if self._use_mock:
                digest = hashlib.sha1(text.encode()).digest()
                results.append([b / 255.0 for b in digest[:3]])
            else:
                resp = self.client.create(input=text, **self._invocation_params)
                data = resp.model_dump() if hasattr(resp, "model_dump") else resp
                results.extend([item["embedding"] for item in data["data"]])
        return results

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query; returns first embedding.
        """
        if self._use_mock:
            digest = hashlib.sha1(text.encode()).digest()
            return [b / 255.0 for b in digest[:3]]
        resp = self.client.create(input=text, **self._invocation_params)
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        return data["data"][0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of embed_documents with similar mock vs real logic.
        """
        results: List[List[float]] = []
        for text in texts:
            if self.skip_empty and not text:
                continue
            if self._use_mock:
                digest = hashlib.sha1(text.encode()).digest()
                results.append([b / 255.0 for b in digest[:3]])
            else:
                resp = await self.async_client.create(
                    input=text, **self._invocation_params
                )
                data = resp.model_dump() if hasattr(resp, "model_dump") else resp
                results.extend([item["embedding"] for item in data["data"]])
        return results

    async def aembed_query(self, text: str) -> List[float]:
        """
        Async version of embed_query; returns first embedding.
        """
        if self._use_mock:
            digest = hashlib.sha1(text.encode()).digest()
            return [b / 255.0 for b in digest[:3]]
        resp = await self.async_client.create(input=text, **self._invocation_params)
        data = resp.model_dump() if hasattr(resp, "model_dump") else resp
        return data["data"][0]["embedding"]
