import pytest

from langchain_aimlapi.image_models import AimlapiImageModel
from langchain_aimlapi.video_models import AimlapiVideoModel


# --- Fixtures ---
@pytest.fixture(autouse=True)
def set_env_api_key(monkeypatch):
    # Ensure consistent API key in environment for tests
    monkeypatch.setenv("AIMLAPI_API_KEY", "testtoken")
    yield


@pytest.fixture
def dummy_httpx(monkeypatch):
    """
    Monkeypatch httpx.Client.post and get for video model to return controlled responses.
    """

    class DummyResponse:
        def __init__(self, json_data, status_code=200):
            self._json = json_data
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP Error: {self.status_code}")

        def json(self):
            return self._json

    client_cls = __import__(
        "langchain_aimlapi.video_models", fromlist=["httpx"]
    ).httpx.Client
    monkeypatch.setattr(
        client_cls,
        "post",
        lambda self, url, json, headers: DummyResponse({"id": "gen123"}, 200),
    )
    monkeypatch.setattr(
        client_cls,
        "get",
        lambda self, url, params, headers: DummyResponse(
            {"status": "completed", "video": {"url": "https://example.com/video.mp4"}},
            200,
        ),
    )
    yield


# --- Tests for AimlapiImageModel ---


def test_generate_images_url(monkeypatch):
    """Test that generate_images returns URLs when response_format='url'."""
    model = AimlapiImageModel(api_key="key", base_url="https://api.test/v1")

    # Create dummy response with two images
    class DummyData:
        def __init__(self, url=None, b64_json=None):
            self.url = url
            self.b64_json = b64_json

    dummy_resp = type("Resp", (), {"data": [DummyData(url="u1"), DummyData(url="u2")]})
    # Monkeypatch client.images.generate
    monkeypatch.setattr(
        model,
        "_client",
        lambda: type(
            "C", (), {"images": type("Img", (), {"generate": lambda **kw: dummy_resp})}
        )(),
    )
    urls = model.generate_images("prompt", n=2, response_format="url")
    assert urls == ["u1", "u2"]


def test_generate_images_b64(monkeypatch):
    """Test that generate_images returns base64 strings when response_format='b64_json'."""
    model = AimlapiImageModel(api_key="key")

    class DummyData:
        def __init__(self, url=None, b64_json=None):
            self.url = url
            self.b64_json = b64_json

    dummy_resp = type("Resp", (), {"data": [DummyData(b64_json="imgdata")]})
    monkeypatch.setattr(
        model,
        "_client",
        lambda: type(
            "C", (), {"images": type("Img", (), {"generate": lambda **kw: dummy_resp})}
        )(),
    )
    result = model.generate_images("prompt", response_format="b64_json")
    assert result == ["imgdata"]


# --- Tests for AimlapiVideoModel ---


def test_generate_videos_success(dummy_httpx):
    """Test successful video generation flow returns correct URL."""
    model = AimlapiVideoModel()
    videos = model.generate_videos(
        "test", n=1, response_format="url", poll_interval=0.01
    )
    assert videos == ["https://example.com/video.mp4"]


def test_video_model_call(monkeypatch, dummy_httpx):
    """Test that invoke returns the first URL via generate_videos override."""
    model = AimlapiVideoModel()
    # Monkeypatch generate_videos to simulate lower-level logic covered elsewhere
    monkeypatch.setattr(
        AimlapiVideoModel,
        "generate_videos",
        lambda self, prompt, **kw: ["url1", "url2"],
    )
    # invoke() uses _call which returns the first element
    assert model.invoke("prompt") == "url1"
