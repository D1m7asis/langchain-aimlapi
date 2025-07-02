"""Aimlapi vector stores."""

from __future__ import annotations

import uuid
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import _cosine_similarity as cosine_similarity

VST = TypeVar("VST", bound=VectorStore)


class AimlapiVectorStore(VectorStore):
    """In-memory vector store using cosine similarity."""

    def __init__(self, embedding: Embeddings) -> None:
        """Initialize with the given embedding function.

        Args:
            embedding: embedding function to use.
        """
        self._database: dict[str, dict[str, Any]] = {}
        self.embedding = embedding

    @classmethod
    def from_texts(
        cls: Type[AimlapiVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> AimlapiVectorStore:
        store = cls(
            embedding=embedding,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
        return store

    # optional: add custom async implementations
    # @classmethod
    # async def afrom_texts(
    #     cls: Type[VST],
    #     texts: List[str],
    #     embedding: Embeddings,
    #     metadatas: Optional[List[dict]] = None,
    #     **kwargs: Any,
    # ) -> VST:
    #     return await asyncio.get_running_loop().run_in_executor(
    #         None, partial(cls.from_texts, **kwargs), texts, embedding, metadatas
    #     )

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        vectors = self.embedding.embed_documents(texts)

        if ids and len(ids) != len(texts):
            msg = (
                f"ids must be the same length as texts. "
                f"Got {len(ids)} ids and {len(texts)} texts."
            )
            raise ValueError(msg)

        id_iterator: Iterator[Optional[str]] = (
            iter(ids) if ids else iter(doc.id for doc in documents)
        )

        ids_ = []

        for doc, vector in zip(documents, vectors):
            doc_id = next(id_iterator)
            doc_id_ = doc_id if doc_id else str(uuid.uuid4())
            ids_.append(doc_id_)
            self._database[doc_id_] = {
                "id": doc_id_,
                "vector": vector,
                "text": doc.page_content,
                "metadata": doc.metadata,
            }

        return ids_

    # optional: add custom async implementations
    # async def aadd_documents(
    #     self,
    #     documents: List[Document],
    #     ids: Optional[List[str]] = None,
    #     **kwargs: Any,
    # ) -> List[str]:
    #     raise NotImplementedError

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        if ids:
            for _id in ids:
                self._database.pop(_id, None)

    # optional: add custom async implementations
    # async def adelete(
    #     self, ids: Optional[List[str]] = None, **kwargs: Any
    # ) -> None:
    #     raise NotImplementedError

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        documents = []

        for doc_id in ids:
            doc = self._database.get(doc_id)
            if doc:
                documents.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=doc["metadata"],
                    )
                )
        return documents

    # optional: add custom async implementations
    # async def aget_by_ids(self, ids: Sequence[str], /) -> list[Document]:
    #     raise NotImplementedError

    # NOTE: the below helper method implements similarity search for in-memory
    # storage. It is optional and not a part of the vector store interface.
    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float, List[float]]]:
        # get all docs with fixed order in list
        docs = list(self._database.values())

        if filter is not None:
            docs = [
                doc
                for doc in docs
                if filter(Document(page_content=doc["text"], metadata=doc["metadata"]))
            ]

        if not docs:
            return []

        similarity = cosine_similarity([embedding], [doc["vector"] for doc in docs])[0]

        # get the indices ordered by similarity score
        top_k_idx = similarity.argsort()[::-1][:k]

        return [
            (
                # Document
                Document(
                    id=doc_dict["id"],
                    page_content=doc_dict["text"],
                    metadata=doc_dict["metadata"],
                ),
                # Score
                float(similarity[idx].item()),
                # Embedding vector
                doc_dict["vector"],
            )
            for idx in top_k_idx
            # Assign using walrus operator to avoid multiple lookups
            if (doc_dict := docs[idx])
        ]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, **kwargs
            )
        ]

    # optional: add custom async implementations
    # async def asimilarity_search(
    #     self, query: str, k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(self.similarity_search, query, k=k, **kwargs)
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, **kwargs
            )
        ]

    # optional: add custom async implementations
    # async def asimilarity_search_with_score(
    #     self, *args: Any, **kwargs: Any
    # ) -> List[Tuple[Document, float]]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(self.similarity_search_with_score, *args, **kwargs)
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    ### ADDITIONAL OPTIONAL SEARCH METHODS BELOW ###

    # def similarity_search_by_vector(
    #     self, embedding: List[float], k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     raise NotImplementedError

    # optional: add custom async implementations
    # async def asimilarity_search_by_vector(
    #     self, embedding: List[float], k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(self.similarity_search_by_vector, embedding, k=k, **kwargs)
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    # def max_marginal_relevance_search(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError

    # optional: add custom async implementations
    # async def amax_marginal_relevance_search(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     # This is a temporary workaround to make the similarity search
    #     # asynchronous. The proper solution is to make the similarity search
    #     # asynchronous in the vector store implementations.
    #     func = partial(
    #         self.max_marginal_relevance_search,
    #         query,
    #         k=k,
    #         fetch_k=fetch_k,
    #         lambda_mult=lambda_mult,
    #         **kwargs,
    #     )
    #     return await asyncio.get_event_loop().run_in_executor(None, func)

    # def max_marginal_relevance_search_by_vector(
    #     self,
    #     embedding: List[float],
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError

    # optional: add custom async implementations
    # async def amax_marginal_relevance_search_by_vector(
    #     self,
    #     embedding: List[float],
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError
