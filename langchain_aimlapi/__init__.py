from importlib import metadata

from langchain_aimlapi.chat_models import ChatAimlapi
from langchain_aimlapi.document_loaders import AimlapiLoader
from langchain_aimlapi.embeddings import AimlapiEmbeddings
from langchain_aimlapi.imagegen import AimlapiImageGenerator
from langchain_aimlapi.retrievers import AimlapiRetriever
from langchain_aimlapi.toolkits import AimlapiToolkit
from langchain_aimlapi.tools import AimlapiTool
from langchain_aimlapi.vectorstores import AimlapiVectorStore
from langchain_aimlapi.videogen import AimlapiVideoGenerator

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatAimlapi",
    "AimlapiVectorStore",
    "AimlapiEmbeddings",
    "AimlapiLoader",
    "AimlapiRetriever",
    "AimlapiToolkit",
    "AimlapiTool",
    "AimlapiImageGenerator",
    "AimlapiVideoGenerator",
    "__version__",
]
