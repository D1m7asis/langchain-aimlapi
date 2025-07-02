"""Aimlapi document loader."""

from typing import Iterator

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class AimlapiLoader(BaseLoader):
    """Placeholder loader for the Aimlapi docs."""

    def lazy_load(self) -> Iterator[Document]:
        raise NotImplementedError()
