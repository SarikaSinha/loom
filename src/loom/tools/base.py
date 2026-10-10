from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Literal

from pydantic import BaseModel

from loom.contracts import (
    JSONValue,
    Schema,
    ToolContext,
    ToolFailure,
    ToolSpec,
    validate_schema,
)
from loom.security.transport import origin

_live_guard = threading.Lock()
_live: dict[str, tuple[str, str]] = {}


def _after_fork() -> None:
    global _live_guard, _live
    _live_guard, _live = threading.Lock(), {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


class Tool(ABC):
    def __init__(self, spec: ToolSpec) -> None:
        validate_schema(spec.argument_schema)
        validate_schema(spec.output_schema)
        if any(origin(url) != url for url in spec.egress_origins):
            raise ValueError("tool origins must be canonical HTTPS origins")
        self._spec = spec.model_copy(deep=True)
        self._claim_guard = threading.Lock()
        self._claim: tuple[str, str, str] | None = None

    @property
    def spec(self) -> ToolSpec:
        return self._spec.model_copy(deep=True)

    def claim(self, call_id: str, run_id: str) -> None:
        with self._claim_guard, _live_guard:
            if self._claim is not None:
                call, run, phase = self._claim
                raise ToolFailure(
                    "tool_busy",
                    recoverable=True,
                    details={
                        "blocking_call_id": call,
                        "blocking_run_id": run,
                        "blocking_phase": phase,
                    },
                )
            if call_id in _live:
                run, phase = _live[call_id]
                raise ToolFailure(
                    "tool_busy",
                    recoverable=True,
                    details={
                        "blocking_call_id": call_id,
                        "blocking_run_id": run,
                        "blocking_phase": phase,
                    },
                )
            self._claim = (call_id, run_id, "approval")
            _live[call_id] = (run_id, "approval")

    def phase(self, call_id: str, phase: str) -> None:
        with self._claim_guard, _live_guard:
            if self._claim and self._claim[0] == call_id:
                self._claim = (call_id, self._claim[1], phase)
                _live[call_id] = (self._claim[1], phase)

    def release(self, call_id: str) -> None:
        with self._claim_guard, _live_guard:
            if self._claim and self._claim[0] == call_id:
                self._claim = None
                _live.pop(call_id, None)

    def is_active(self, call_id: str) -> bool:
        with _live_guard:
            return call_id in _live

    @abstractmethod
    async def execute(
        self, arguments: Mapping[str, JSONValue], context: ToolContext
    ) -> JSONValue | BaseModel:
        raise NotImplementedError

    @staticmethod
    def from_callable(
        fn: Callable[..., object],
        *,
        name: str | None = None,
        description: str | None = None,
        version: str = "1",
        argument_schema: Schema | type[BaseModel] | None = None,
        output_schema: Schema | type[BaseModel] | None = None,
        sensitive: bool = True,
        retry_safe: bool = False,
        timeout_seconds: float | None = None,
        egress: Literal["none", "fixed_network", "unknown"] = "unknown",
        egress_origins: tuple[str, ...] = (),
    ) -> Tool:
        from loom.tools.function import FunctionTool

        return FunctionTool(
            fn,
            name=name,
            description=description,
            version=version,
            argument_schema=argument_schema,
            output_schema=output_schema,
            sensitive=sensitive,
            retry_safe=retry_safe,
            timeout_seconds=timeout_seconds,
            egress=egress,
            egress_origins=egress_origins,
        )
