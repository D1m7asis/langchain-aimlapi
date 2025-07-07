# AimlapiEmbeddings

`AimlapiEmbeddings` provides access to embedding models.

```python
from langchain_aimlapi import AimlapiEmbeddings

embeddings = AimlapiEmbeddings(model="text-embedding-ada-002")
vector = embeddings.embed_query("What is LangChain?")
print(vector[:3])
```
