"""Aimlapi retrievers."""

from typing import Any, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class AimlapiRetriever(BaseRetriever):
    """Simple example retriever returning dummy documents."""

    k: int = 3

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> List[Document]:
        k = kwargs.get("k", self.k)
        return [
            Document(page_content=f"Result {i} for query: {query}") for i in range(k)
        ]

    # optional: add custom async implementations here
    # async def _aget_relevant_documents(
    #     self,
    #     query: str,
    #     *,
    #     run_manager: AsyncCallbackManagerForRetrieverRun,
    #     **kwargs: Any,
    # ) -> List[Document]: ...
