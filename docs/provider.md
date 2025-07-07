# Aimlapi Integration

The **langchain-aimlapi** package provides convenient wrappers for using [Aimlapi](https://api.aimlapi.com/) models with LangChain.

Install the package with:

```bash
pip install -U langchain-aimlapi
```

Set your API key in the environment:

```bash
export AIMLAPI_API_KEY="your-api-key"
```

Available components:

- `ChatAimlapi` – OpenAI compatible chat model.
- `AimlapiLLM` – text completion model.
- `AimlapiEmbeddings` – embedding model.
- `AimlapiImageModel` – image generation model.
- `AimlapiVideoModel` – video generation model.

See the following pages for usage examples of each class.
