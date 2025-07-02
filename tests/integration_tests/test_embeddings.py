"""Test Aimlapi embeddings."""

from typing import Type

from langchain_aimlapi.embeddings import AimlapiEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[AimlapiEmbeddings]:
        return AimlapiEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
