# AimlapiLLM

`AimlapiLLM` exposes the text completion models provided by Aimlapi. Usage mirrors other LangChain LLM classes and is compatible with prompts used for OpenAI text completion models.

```python
from langchain_aimlapi import AimlapiLLM

llm = AimlapiLLM(
    model="gpt-3.5-turbo-instruct",
    temperature=0.0,
    max_tokens=None,  # defaults to 200
    api_key="YOUR_API_KEY",
)
response = llm.invoke("I love programming.")
print("Completion →", response)
```

The same interface is available asynchronously with `acall`/`ainvoke`.
