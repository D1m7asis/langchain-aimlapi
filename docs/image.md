# AimlapiImageModel

`AimlapiImageModel` is an OpenAI compatible wrapper over the Aimlapi image generation API.

```python
from langchain_aimlapi import AimlapiImageModel

img_model = AimlapiImageModel(model="dall-e-3")
url = img_model.invoke("a robot reading a book")
print(url)
```
