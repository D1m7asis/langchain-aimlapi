# Aimlapi Integration

The **langchain-aimlapi** package provides convenient wrappers for using
[Aimlapi](https://api.aimlapi.com/) models with LangChain. Aimlapi hosts over
300 models including Deepseek, Gemini and ChatGPT with enterprise-grade rate
limits and uptime.

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
- `AIMLAPI_HEADERS` – default request headers used for analytics.

Each component mirrors the corresponding OpenAI class so you can switch from the
OpenAI provider to Aimlapi with minimal code changes. See the following pages
for concrete examples of each class in action.
