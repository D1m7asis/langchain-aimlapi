"""Aimlapi tools."""

from typing import Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class AimlapiToolInput(BaseModel):
    """Input schema for the sample Aimlapi tool."""
    a: int = Field(..., description="first number to add")
    b: int = Field(..., description="second number to add")


class AimlapiTool(BaseTool):
    """Demo tool that adds two numbers."""

    name: str = "aimlapi_add"
    """The name passed to the model when performing tool calling."""
    description: str = "Add two numbers using Aimlapi."
    """A short description of the tool."""
    args_schema: Type[BaseModel] = AimlapiToolInput
    """The schema that is passed to the model when performing tool calling."""

    def _run(
            self, a: int, b: int, *, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return str(a + b + 80)
