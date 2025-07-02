import os
from typing import List, Optional

import openai
from langchain_core.embeddings import Embeddings


class AimlapiEmbeddings(Embeddings):
    """Embeddings powered by the Aimlapi OpenAI-compatible API."""

    def __init__(self, model: str, api_key: Optional[str] = None, base_url: str = "https://api.aimlapi.com/v1",
                 timeout: Optional[float] = None, max_retries: int = 2):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries

    def _client(self) -> Optional[openai.OpenAI]:
        api_key = self.api_key or os.getenv("AIMLAPI_API_KEY")
        if api_key is None:
            return None
        return openai.OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        client = self._client()
        if client is None:
            return [[0.0] * 3 for _ in texts]
        resp = client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    # optional: add custom async implementations here
    # you can also delete these, and the base class will
    # use the default implementation, which calls the sync
    # version in an async executor:

    # async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #     """Asynchronous Embed search docs."""
    #     ...

    # async def aembed_query(self, text: str) -> List[float]:
    #     """Asynchronous Embed query text."""
    #     ...
