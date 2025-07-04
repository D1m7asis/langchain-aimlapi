from unittest.mock import MagicMock

from openai import OpenAIError

from langchain_aimlapi.llms import AimlapiLLM


class _DummyResponse:
    def __init__(self, text: str = "ok"):
        self.choices = [MagicMock(text=text, finish_reason="stop")]
        self.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)


def test_retry_and_model_kwargs(monkeypatch):
    llm = AimlapiLLM(
        model="bird-brain-001",
        api_key="key",
        max_retries=2,
        model_kwargs={"model": "wrong", "temperature": 0.2, "echo": True},
    )

    side_effects = [OpenAIError("fail1"), OpenAIError("fail2"), _DummyResponse("hi")]
    mock_create = MagicMock(side_effect=side_effects)
    client = MagicMock()
    client.completions.create = mock_create
    monkeypatch.setattr(llm, "_client", lambda: client)

    result = llm._call("hello")
    assert mock_create.call_count == 3
    assert result.generations[0][0].text == "hi"

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["model"] == "bird-brain-001"
    assert call_kwargs["temperature"] is None
    assert call_kwargs["echo"] is True

