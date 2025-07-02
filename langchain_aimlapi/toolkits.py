"""Aimlapi toolkits."""

from typing import List

from langchain_core.tools import BaseTool, BaseToolkit


class AimlapiToolkit(BaseToolkit):
    """Example toolkit grouping Aimlapi tools."""

    def get_tools(self) -> List[BaseTool]:
        raise NotImplementedError()
