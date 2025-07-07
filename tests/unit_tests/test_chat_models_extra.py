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


def test_execute_with_retry_sync_backoff():
    model = ChatAimlapi(model="bird-brain-001", max_retries=2)
    calls = []

    def fn():
        if len(calls) < 2:
            calls.append(1)
            raise OpenAIError("boom")
        return "ok"

    with mock.patch("time.sleep", return_value=None) as sleep_mock:
        assert model._execute_with_retry_sync(fn) == "ok"
    # ensure two backoff attempts were made with exponential delays
    assert [c.args[0] for c in sleep_mock.call_args_list] == [1, 2]


def test_parrot_fallback_chat():
    model = ChatAimlapi(model="bird-brain-001", parrot_buffer_length=3)
    result = model._generate([HumanMessage(content="hello world")])
    assert result.generations[0].message.content == "rld"


def test_stop_filtering_and_normalization():
    model = ChatAimlapi(
        model="bird-brain-001",
        api_key="key",
        model_kwargs={"stop": ["x"], "echo": True},
    )
    dummy_resp = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content="ok"),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    mock_create = mock.MagicMock(return_value=dummy_resp)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=mock_create)))
    with mock.patch.object(model, "_client", return_value=client):
        model._generate([HumanMessage(content="hi")], stop="end")
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "bird-brain-001"
    assert call_kwargs["stop"] == ["end"]
    assert call_kwargs["echo"] is True


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
