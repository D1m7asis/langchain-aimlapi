"""Aimlapi document loader."""

from typing import Iterator, Optional

from langchain_core.document_loaders.base import BaseLoader
from langchain_core.documents import Document


class AimlapiLoader(BaseLoader):
    """Simple loader that reads a single file into a document."""

    def __init__(self, path: str, encoding: Optional[str] = "utf-8") -> None:
        self.path = path
        self.encoding = encoding

    def lazy_load(self) -> Iterator[Document]:
        with open(self.path, "r", encoding=self.encoding) as f:
            text = f.read()
        yield Document(page_content=text)
