# langchain-aimlapi

This package contains the LangChain integration with Aimlapi

## Installation

```bash
pip install -U langchain-aimlapi
```

And set the environment variable `AIMLAPI_API_KEY` with your API token.
## Chat Models

`ChatAimlapi` class exposes chat models from Aimlapi.

```python
from langchain_aimlapi import ChatAimlapi

llm = ChatAimlapi()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`AimlapiEmbeddings` class exposes embeddings from Aimlapi.

```python
from langchain_aimlapi import AimlapiEmbeddings

embeddings = AimlapiEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## Image Generation

`AimlapiImageGenerator` wraps the image generation endpoint.

```python
from langchain_aimlapi import AimlapiImageGenerator

image_gen = AimlapiImageGenerator()
image_gen.generate("a cute kitten")
```

## Video Generation

`AimlapiVideoGenerator` exposes video generation.

```python
from langchain_aimlapi import AimlapiVideoGenerator

video_gen = AimlapiVideoGenerator()
video_gen.generate("a dancing robot")
```
