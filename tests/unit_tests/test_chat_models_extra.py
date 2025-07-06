from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import HumanMessage
from openai import OpenAIError
from pydantic import BaseModel

from langchain_aimlapi.chat_models import ChatAimlapi


class DummyError(OpenAIError):
    pass


class Person(BaseModel):
    name: str
    age: str


def test_retry_backoff(monkeypatch):
    chat = ChatAimlapi(model="bird-brain-001", api_key="sk")
    count = 0

    def fail(*args: Any, **kwargs: Any):
        nonlocal count
        count += 1
        if count < 3:
            raise DummyError("fail")

        class Resp:
            choices = [MagicMock(message=MagicMock(content="hi", finish_reason="stop"))]
            usage = MagicMock(prompt_tokens=0, completion_tokens=0, total_tokens=0)

        return Resp()

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    result = chat._execute_with_retry_sync(fail)
    assert result.choices[0].message.content == "hi"
    assert sleeps == [1, 2]


def test_parrot_fallback():
    chat = ChatAimlapi(model="bird-brain-001", parrot_buffer_length=4)
    res = chat._generate([HumanMessage(content="abcdef")])
    assert res.generations[0].message.content == "abcd"


def test_filtering_stop(monkeypatch):
    recorded = {}

    class Client:
        class Chat:
            class Completions:
                def create(self, **kwargs: Any):
                    recorded.update(kwargs)

                    class Resp:
                        choices = [
                            MagicMock(
                                message=MagicMock(content="hi", finish_reason="stop")
                            )
                        ]
                        usage = MagicMock(
                            prompt_tokens=0, completion_tokens=0, total_tokens=0
                        )

                    return Resp()

            completions = Completions()

        chat = Chat()

    chat_model = ChatAimlapi(
        model="bird-brain-001",
        api_key="sk",
        model_kwargs={"stop": ["bad"], "temperature": 2, "foo": "bar"},
    )
    monkeypatch.setattr(chat_model, "_client", lambda: Client())
    chat_model._generate([HumanMessage(content="hi")], stop="THE END")
    assert recorded["stop"] == ["THE END"]
    assert recorded.get("foo") == "bar"
    assert recorded.get("model") == "bird-brain-001"


def test_structured_output():
    chat = ChatAimlapi(model="bird-brain-001")
    runnable = chat.with_structured_output(Person)
    result = runnable.invoke("tell me")
    assert isinstance(result, Person)
