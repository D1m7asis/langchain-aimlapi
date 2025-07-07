# AimlapiImageModel

`AimlapiImageModel` is an OpenAI-compatible wrapper over the Aimlapi image
generation API. It can produce images from text prompts using the same
parameters as the OpenAI SDK.

```python
from langchain_aimlapi import AimlapiImageModel

img = AimlapiImageModel(
    model="stable-diffusion-v3-medium",
    size="512x512",
    n=1,
    api_key="YOUR_API_KEY",
)
md_img = img.invoke("A serene mountain lake at sunset")
print("Markdown Image →", md_img)
```

Async generation is available via `agenerate_images` and `ainvoke`.
