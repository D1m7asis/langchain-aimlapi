# AimlapiVideoModel

`AimlapiVideoModel` generates short video clips using Aimlapi. The class
implements the same interface as other LangChain LLMs.

```python
from langchain_aimlapi import AimlapiVideoModel

vid = AimlapiVideoModel(
    model="veo2",
    api_key="YOUR_API_KEY",
)
md_vid = vid("A timelapse of city lights at night")
print("Markdown Video →", md_vid)
```
