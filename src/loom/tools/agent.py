from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from loom.contracts import (
    CallerIdentity,
    CancellationToken,
    ConversationState,
    JSONValue,
    LoomConfigurationError,
    RunResult,
    Schema,
    ToolContext,
    ToolFailure,
    ToolSpec,
    json_value,
    schema_for,
)
from loom.tools.base import Tool

if TYPE_CHECKING:
    from loom.agent import Agent
    from loom.tools.http import HTTPTool


class _HandoffSignal(BaseException):
    """Internal sentinel raised by AgentTool when handoff=True and Agent B succeeds.

    Inherits from BaseException (not Exception) so the ToolRunner's broad
    ``except Exception`` handler does not swallow it — it must be caught
    explicitly in _Run.execute_tools.
    """

    def __init__(self, result: RunResult, output: JSONValue) -> None:
        super().__init__()
        self.result = result
        self.output = output


_DEFAULT_ARGUMENT_SCHEMA: Schema = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
    "additionalProperties": False,
}

_DEFAULT_OUTPUT_SCHEMA: Schema = {"type": "string"}


class AgentTool(Tool):
    """A Tool whose callable is another Agent.

    Supports two execution modes controlled by ``handoff``:

    * ``handoff=False`` (default): Agent A suspends, Agent B runs to completion,
      and the result is returned as a normal tool observation.
    * ``handoff=True``: Agent A terminates immediately after Agent B completes;
      Agent B's output becomes Agent A's output with an ``agent_handoff``
      diagnostic attached.
    """

    def __init__(
        self,
        agent: Agent,
        *,
        name: str,
        description: str,
        version: str = "1",
        handoff: bool = False,
        argument_schema: Schema | type[BaseModel] | None = None,
        output_schema: Schema | type[BaseModel] | None = None,
        caller_override: CallerIdentity | None = None,
        context_selector: Callable[[ToolContext], Mapping[str, object]] | None = None,
        state_selector: Callable[[ToolContext], ConversationState | None] | None = None,
        output_selector: Callable[[RunResult], JSONValue] | None = None,
        egress_policy: Literal["deny_all_http"] | None = None,
        sensitive: bool = False,
        retry_safe: bool = False,
        timeout_seconds: float | None = None,
    ) -> None:
        from loom.tools.http import HTTPTool

        if not description:
            raise LoomConfigurationError("missing_tool_description")

        resolved_arg_schema = (
            schema_for(argument_schema) if argument_schema is not None else _DEFAULT_ARGUMENT_SCHEMA
        )
        if resolved_arg_schema is None:
            raise LoomConfigurationError("invalid_tool_schema")

        # Output schema: explicit > mirror Agent B's schema > string fallback
        if output_schema is not None:
            resolved_out_schema = schema_for(output_schema)
        elif agent.schema is not None:
            resolved_out_schema = agent.schema
        else:
            resolved_out_schema = _DEFAULT_OUTPUT_SCHEMA
        if resolved_out_schema is None:
            raise LoomConfigurationError("invalid_tool_schema")

        # Derive egress from Agent B's HTTP tools
        http_tools: list[HTTPTool] = [t for t in agent.tools if isinstance(t, HTTPTool)]
        if not http_tools:
            egress: Literal["none", "fixed_network", "unknown"] = "none"
            egress_origins: tuple[str, ...] = ()
        elif all(t.spec.egress == "fixed_network" for t in http_tools):
            egress = "fixed_network"
            egress_origins = tuple(
                sorted({o for t in http_tools for o in t.spec.egress_origins})
            )
        else:
            egress = "unknown"
            egress_origins = ()

        super().__init__(
            ToolSpec(
                name=name,
                description=description,
                version=version,
                argument_schema=resolved_arg_schema,
                output_schema=resolved_out_schema,
                sensitive=sensitive,
                retry_safe=retry_safe,
                timeout_seconds=timeout_seconds,
                egress=egress,
                egress_origins=egress_origins,
            )
        )
        self._agent = agent
        self._handoff = handoff
        self._caller_override = caller_override
        self._context_selector = context_selector
        self._state_selector = state_selector
        self._output_selector = output_selector
        self._egress_policy = egress_policy

    async def execute(
        self,
        arguments: Mapping[str, JSONValue],
        context: ToolContext,
    ) -> JSONValue | BaseModel:
        from loom.tools.http import HTTPTool

        agent = self._agent
        ctx = context.context

        # 1. Cycle detection
        agent_chain: tuple[str, ...] = ctx.get("_agent_chain", ())  # type: ignore[assignment]
        if agent.name in agent_chain:
            raise ToolFailure("agent_cycle_detected", recoverable=True)

        # 2. Depth check
        if len(agent_chain) >= agent.limits.max_agent_depth:
            raise ToolFailure("agent_depth_exceeded", recoverable=True)

        # 3. Egress policy
        if self._egress_policy == "deny_all_http":
            if any(isinstance(t, HTTPTool) for t in agent.tools):
                raise ToolFailure("egress_policy_denied", recoverable=True)

        # 4. Deadline intersection
        now = datetime.now(UTC)
        child_deadline = min(
            context.deadline,
            now + timedelta(seconds=agent.limits.deadline_seconds),
        )
        if self.spec.timeout_seconds is not None:
            child_deadline = min(child_deadline, now + timedelta(seconds=self.spec.timeout_seconds))
        remaining = max(0.001, (child_deadline - now).total_seconds())

        # 5. Caller identity
        if self._caller_override is not None:
            caller = self._caller_override
        else:
            parent_name: str = ctx.get("_parent_agent_name", "unknown")  # type: ignore[assignment]
            caller = CallerIdentity(subject=f"agent:{parent_name}")

        # 6. Build child context (extends chain with Agent B's name)
        child_context: dict[str, object] = {
            **(self._context_selector(context) if self._context_selector else {}),
            "_agent_chain": (*agent_chain, agent.name),
            "_parent_agent_name": agent.name,
        }

        # 7. State for Agent B
        child_state = self._state_selector(context) if self._state_selector else None

        # 8. Cascading cancellation: cancel child if parent is cancelled
        child_cancellation = CancellationToken()

        async def _watch_cancellation() -> None:
            while not context.cancellation.is_cancelled():
                await asyncio.sleep(0.05)
            child_cancellation.cancel()

        watcher = asyncio.ensure_future(_watch_cancellation())
        try:
            # Build the input string — use "input" key by default, or JSON-encode full args
            input_str = str(arguments.get("input", ""))
            if set(arguments.keys()) - {"input"}:
                input_str = json.dumps(dict(arguments), allow_nan=False)

            result = await agent.arun(
                input=input_str,
                state=child_state,
                context=child_context,
                caller=caller,
                cancellation=child_cancellation,
            )
        finally:
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass

        # 9. Handle result
        if result.status != "completed":
            code = (
                "sub_agent_cancelled"
                if result.status == "cancelled"
                else "sub_agent_failed"
            )
            outcome_unknown = (
                result.error.outcome_unknown if result.error is not None else False
            )
            raise ToolFailure(code, recoverable=True, outcome_unknown=outcome_unknown)

        output_value: JSONValue = (
            self._output_selector(result) if self._output_selector else result.output
        )

        if self._handoff:
            raise _HandoffSignal(result, output_value)

        return json_value(output_value)
