from typing import Generator

import pytest
from langchain_aimlapi.vectorstores import AimlapiVectorStore
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests


class TestAimlapiVectorStore(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = AimlapiVectorStore(self.get_embeddings())
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass
