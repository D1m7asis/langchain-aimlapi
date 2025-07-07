# AimlapiLLM

`AimlapiLLM` exposes the text completion models provided by Aimlapi. Usage mirrors other LangChain LLM classes.

```python
from langchain_aimlapi import AimlapiLLM

llm = AimlapiLLM(model="bird-brain-001", temperature=0)
llm.invoke("The meaning of life is")
```
