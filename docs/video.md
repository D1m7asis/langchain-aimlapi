# AimlapiVideoModel

`AimlapiVideoModel` generates short video clips using Aimlapi.

```python
from langchain_aimlapi import AimlapiVideoModel

video_model = AimlapiVideoModel(model="google/veo3")
video_url = video_model.invoke("a cat surfing")
print(video_url)
```
