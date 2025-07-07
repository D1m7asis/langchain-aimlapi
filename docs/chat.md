# ChatAimlapi

`ChatAimlapi` wraps the Aimlapi chat completion endpoint. The API is compatible with `langchain_openai.ChatOpenAI`.

```python
from langchain_aimlapi import ChatAimlapi

llm = ChatAimlapi(model="gpt-3.5-turbo", temperature=0)
response = llm.invoke("Hello!")
print(response)
```

The model falls back to a mock "parrot" mode when `AIMLAPI_API_KEY` is set to the dummy value `dummytoken`.
