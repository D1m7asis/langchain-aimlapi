from importlib import metadata

from langchain_aimlapi.chat_models import ChatAimlapi
from langchain_aimlapi.llms import AimlapiLLM
from langchain_aimlapi.embeddings import AimlapiEmbeddings
from langchain_aimlapi.image_models import AimlapiImageModel
from langchain_aimlapi.video_models import AimlapiVideoModel

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatAimlapi",
    "AimlapiLLM",
    "AimlapiEmbeddings",
    "AimlapiImageModel",
    "AimlapiVideoModel",
    "__version__",
]
