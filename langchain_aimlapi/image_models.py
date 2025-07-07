"""Placeholder image model integration for Aimlapi."""

class AimlapiImageModel:
    """Simple stub for future image generation models."""

    def __init__(self, model: str = "image-model-001") -> None:
        self.model = model

    def generate(self, prompt: str) -> str:
        """Return a mock image URL for the provided prompt."""
        return f"https://example.com/{self.model}/image.png"
