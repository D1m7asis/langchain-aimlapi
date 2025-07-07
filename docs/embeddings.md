# AimlapiEmbeddings

`AimlapiEmbeddings` provides access to the wide range of embedding models
available on Aimlapi. The interface mirrors `langchain_openai.OpenAIEmbeddings`.

```python
from langchain_aimlapi import AimlapiEmbeddings

embedder = AimlapiEmbeddings(
    model="text-embedding-3-large",
    api_key="YOUR_API_KEY",
)
text = "AI/ML API is awesome"
vector = embedder.embed_query(text)
print("Embeddings shape:", len(vector))
```

Embeddings can also be computed asynchronously using `aembed_query` and
`aembed_documents`.
