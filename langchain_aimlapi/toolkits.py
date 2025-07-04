"""Aimlapi toolkits."""

from typing import List, Optional, Any

from langchain_core.tools import BaseTool, BaseToolkit

from langchain_aimlapi.tools import AimlapiTool


class AimlapiToolkit(BaseToolkit):
    """Toolkit bundling a set of Aimlapi tools."""

    def __init__(self, /, tools: Optional[List[BaseTool]] = None, **data: Any) -> None:
        super().__init__(**data)
        self.tools = tools or [AimlapiTool()]

    def get_tools(self) -> List[BaseTool]:
        return self.tools
