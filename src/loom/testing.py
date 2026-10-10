"""Deterministic, offline model adapter for application and framework tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from loom.providers.base import ModelAdapter, ModelCapabilities, ModelRequest, ModelResponse


class ScriptedModel(ModelAdapter):
    def __init__(
        self,
        responses: Sequence[ModelResponse | Exception] | Callable[[ModelRequest], ModelResponse],
        *,
        local: bool = True,
        native_tools: bool = True,
    ) -> None:
        self.responses = responses
        self.requests: list[ModelRequest] = []
        self._capabilities = ModelCapabilities(native_tool_calling=native_tools, local=local)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def generate(self, request: ModelRequest) -> ModelResponse:
        index = len(self.requests)
        self.requests.append(request)
        if callable(self.responses):
            return self.responses(request)
        if index >= len(self.responses):
            raise AssertionError("ScriptedModel exhausted its response script")
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self) -> None:
        return None
