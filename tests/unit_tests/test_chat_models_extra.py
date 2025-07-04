import types
from unittest import mock

from langchain_core.messages import HumanMessage
from openai import OpenAIError
from pydantic import BaseModel

from langchain_aimlapi.chat_models import ChatAimlapi


class SampleSchema(BaseModel):
    foo: str


def test_with_structured_output_local():
    model = ChatAimlapi(model="bird-brain-001")
    runnable = model.with_structured_output(SampleSchema)
    result = runnable.invoke("hi")
    assert isinstance(result, SampleSchema)


def test_execute_with_retry():
    model = ChatAimlapi(model="bird-brain-001", max_retries=1)
    calls = []

    def fn():
        if not calls:
            calls.append(1)
            raise OpenAIError("boom")
        return "ok"

    with mock.patch("time.sleep", return_value=None):
        assert model._execute_with_retry(fn) == "ok"
    assert calls == [1]


def test_stream_usage_deltas():
    model = ChatAimlapi(model="bird-brain-001")

    class Usage:
        def __init__(self, p, c, t):
            self.prompt_tokens = p
            self.completion_tokens = c
            self.total_tokens = t

    class FakeChunk:
        def __init__(self, token, usage, finish_reason=None):
            self.choices = [
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=token),
                    finish_reason=finish_reason,
                )
            ]
            self.usage = usage

    stream = iter(
        [
            FakeChunk("h", Usage(1, 1, 2), "stop"),
            FakeChunk("i", Usage(1, 2, 3), None),
        ]
    )

    class FakeClient:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_: stream)
            )

    with mock.patch.object(model, "_client", return_value=FakeClient()):
        chunks = list(model._stream([HumanMessage(content="hi")]))

    assert chunks[0].message.usage_metadata == {
        "input_tokens": 1,
        "output_tokens": 1,
        "total_tokens": 2,
    }
    assert chunks[1].message.usage_metadata == {
        "input_tokens": 0,
        "output_tokens": 1,
        "total_tokens": 1,
    }
