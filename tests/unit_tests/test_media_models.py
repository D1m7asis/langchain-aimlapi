import pytest
from langchain_aimlapi.image_models import AimlapiImageModel
from langchain_aimlapi.video_models import AimlapiVideoModel


def test_image_model_call(monkeypatch):
    model = AimlapiImageModel()
    monkeypatch.setattr(AimlapiImageModel, "_call", lambda self, *a, **k: "image")
    assert model.invoke("a cat") == "image"


def test_video_model_call(monkeypatch):
    model = AimlapiVideoModel()
    monkeypatch.setattr(AimlapiVideoModel, "_call", lambda self, *a, **k: "video")
    assert model.invoke("a dog") == "video"
