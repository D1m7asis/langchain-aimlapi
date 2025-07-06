from typing import Any
from unittest.mock import MagicMock

import pytest
from openai import OpenAIError

from langchain_aimlapi.llms import AimlapiLLM


class DummyError(OpenAIError):
    pass


def test_retry_backoff(monkeypatch):
    llm = AimlapiLLM(model="bird-brain-001", api_key="sk")
    call_times = []

    def fail_first(*args: Any, **kwargs: Any):
        call_times.append("call")
        if len(call_times) < 3:
            raise DummyError("fail")
        return "ok"

    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    result = llm._execute_with_retry_sync(fail_first)
    assert result == "ok"
    assert sleeps == [1, 2]


def test_fallback_no_key():
    llm = AimlapiLLM(model="bird-brain-001", parrot_buffer_length=5)
    result = llm._call("hello world")
    assert result.generations[0][0].text == "hello"  # parroted output


def test_kwargs_filtering_and_stop(monkeypatch):
    recorded = {}

    class Client:
        class Completions:
            def create(self, **kwargs: Any):
                recorded.update(kwargs)

                class Resp:
                    choices = [MagicMock(text="hi", finish_reason="stop")]
                    usage = MagicMock(
                        prompt_tokens=0, completion_tokens=0, total_tokens=0
                    )

                return Resp()

        completions = Completions()

    llm = AimlapiLLM(
        model="bird-brain-001",
        api_key="sk",
        model_kwargs={"model": "bad", "temperature": 2, "foo": 1},
    )
    monkeypatch.setattr(llm, "_client", lambda: Client())
    result = llm._call("test", stop="END")
    assert recorded["stop"] == ["END"]
    assert (
        "foo" in recorded
        and "model" in recorded
        and recorded["model"] == "bird-brain-001"
    )
    assert recorded.get("temperature") == llm.temperature
    assert result.generations[0][0].text == "hi"
