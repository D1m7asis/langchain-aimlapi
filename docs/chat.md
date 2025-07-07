# ChatAimlapi

`ChatAimlapi` wraps the Aimlapi chat completion endpoint. The API is fully
compatible with `langchain_openai.ChatOpenAI` so it can be used as a drop-in
replacement in existing LangChain applications.

```python
from langchain_aimlapi import ChatAimlapi

llm = ChatAimlapi(
    model="gpt-3.5-turbo",
    temperature=0.0,
    api_key="YOUR_API_KEY",  # or set AIMLAPI_API_KEY
)
messages = [
    ("system", "You are a helpful translator. Translate to Russian."),
    ("human", "I love programming."),
]
response = llm.invoke(messages)
print("Chat →", response)
```

### Async usage

```python
async def async_chat():
    out = await llm.ainvoke(messages)
    print("Async Chat →", out)

asyncio.run(async_chat())
```

The model falls back to a deterministic mock "parrot" mode when
`AIMLAPI_API_KEY` is set to the dummy value `dummytoken`. This is useful for
offline testing without real API calls.
