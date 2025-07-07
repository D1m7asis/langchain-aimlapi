# langchain-aimlapi

This package contains the LangChain integration with Aimlapi. AI/ML API provides
over **300** models including Deepseek, Gemini and ChatGPT. All models are
served with enterprise-grade rate limits and uptimes via
[Aimlapi](https://aimlapi.com/app/?utm_source=langchain&utm_medium=github&utm_campaign=integration).

## Installation

```bash
pip install -U langchain-aimlapi
```

And you should configure credentials by setting the following environment variable:

* `AIMLAPI_API_KEY` – your AI/ML API key

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

## LLMs
`AimlapiLLM` class exposes LLMs from Aimlapi.

```python
from langchain_aimlapi import AimlapiLLM

llm = AimlapiLLM()
llm.invoke("The meaning of life is")
```
