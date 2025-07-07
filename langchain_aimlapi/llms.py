"""Wrapper around AI/ML API's completion API."""

import logging
import warnings
from typing import Any, Dict, List, Optional

import requests
from aiohttp import ClientSession
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.utils import secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator

logger = logging.getLogger(__name__)


class AimlapiLLM(LLM):
    """Completion models from AI/ML API."""

    base_url: str = "https://api.aimlapi.com/v1/completions"
    aimlapi_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("AIMLAPI_API_KEY", default="dummytoken"),
    )
    model: str
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: Optional[int] = None
    repetition_penalty: Optional[float] = None
    logprobs: Optional[int] = None

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        if values.get("max_tokens") is None:
            warnings.warn(
                "The completions endpoint, has 'max_tokens' as required argument. "
                "The default value is being set to 200 "
                "Consider setting this value, when initializing LLM"
            )
            values["max_tokens"] = 200
        return values

    @property
    def _llm_type(self) -> str:
        return "aimlapi"

    def _format_output(self, output: dict) -> str:
        return output["choices"][0]["text"]

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.aimlapi_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload: Dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        response = requests.post(url=self.base_url, json=payload, headers=headers)

        if response.status_code >= 500:
            raise Exception(f"Aimlapi Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"Aimlapi received an invalid payload: {response.text}")
        elif response.status_code not in (200, 201):
            raise Exception(
                f"Aimlapi returned an unexpected response with status {response.status_code}: {response.text}"
            )

        data = response.json()
        return self._format_output(data)

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.aimlapi_api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        stop_to_use = stop[0] if stop and len(stop) == 1 else stop
        payload: Dict[str, Any] = {
            **self.default_params,
            "prompt": prompt,
            "stop": stop_to_use,
            **kwargs,
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        async with ClientSession() as session:
            async with session.post(self.base_url, json=payload, headers=headers) as response:
                if response.status >= 500:
                    raise Exception(f"Aimlapi Server: Error {response.status}")
                elif response.status >= 400:
                    raise ValueError(
                        f"Aimlapi received an invalid payload: {await response.text()}"
                    )
                elif response.status != 200:
                    raise Exception(
                        f"Aimlapi returned an unexpected response with status {response.status}: {await response.text()}"
                    )
                response_json = await response.json()
                return self._format_output(response_json)
