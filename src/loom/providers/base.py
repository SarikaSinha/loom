from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import Field

from loom.contracts import (
    JSONValue,
    LoomConfigurationError,
    LoomError,
    Message,
    Record,
    Schema,
    ToolCall,
    ToolSpec,
)

SchemaMode = Literal["json_schema", "json_object", "prompt_json"]


class ModelCapabilities(Record):
    native_tool_calling: bool
    structured_output: SchemaMode = "prompt_json"
    local: bool = False


class ModelRequest(Record):
    instructions: str
    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...]
    output_schema: Schema | None = None
    output_mode: SchemaMode = "prompt_json"
    max_output_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)


class ModelResponse(Record):
    output: JSONValue = None
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: Literal["tool_calls", "completed", "refused", "output_limit"] = "completed"
    continuation: dict[str, JSONValue] = Field(default_factory=dict)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class ModelFailure(LoomError):
    def __init__(
        self,
        code: str,
        *,
        transient: bool = False,
        retry_after: float | None = None,
    ) -> None:
        self.transient = transient
        self.retry_after = retry_after
        super().__init__(code)


class ModelAdapter(ABC):
    concurrency_safe: bool = False

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError

    @abstractmethod
    async def generate(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError


_guard = threading.Lock()
_active: dict[int, ModelAdapter] = {}


def _after_fork() -> None:
    global _guard, _active
    _guard, _active = threading.Lock(), {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def claim_adapter(adapter: ModelAdapter) -> None:
    if adapter.concurrency_safe:
        return
    with _guard:
        if id(adapter) in _active:
            raise LoomConfigurationError("adapter_in_use")
        _active[id(adapter)] = adapter


def release_adapter(adapter: ModelAdapter) -> None:
    with _guard:
        _active.pop(id(adapter), None)


def native_schema(schema: Schema) -> bool:
    keywords = {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "anyOf",
        "description",
        "title",
    }

    def valid(node: JSONValue) -> bool:
        if not isinstance(node, dict) or set(node) - keywords:
            return False
        if node.get("type") == "object":
            properties = node.get("properties")
            required = node.get("required")
            if (
                not isinstance(properties, dict)
                or not isinstance(required, list)
                or node.get("additionalProperties") is not False
                or set(required) != set(properties)
            ):
                return False
            if not all(valid(child) for child in properties.values()):
                return False
        if "items" in node and not valid(node["items"]):
            return False
        unions = node.get("anyOf")
        return unions is None or isinstance(unions, list) and all(valid(child) for child in unions)

    return schema.get("type") == "object" and "anyOf" not in schema and valid(schema)
