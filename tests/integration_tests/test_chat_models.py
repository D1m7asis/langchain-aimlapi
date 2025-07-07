"""Test ChatAimlapi chat model."""

from typing import Type

from langchain_aimlapi.chat_models import ChatAimlapi
from langchain_tests.integration_tests import ChatModelIntegrationTests


class TestChatParrotLinkIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatAimlapi]:
        return ChatAimlapi

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model": "bird-brain-001",
            "temperature": 0,
        }

    @property
    def returns_usage_metadata(self) -> bool:
        return False
