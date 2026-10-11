from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TypeVar

from pydantic import ValidationError

from loom.config import Config
from loom.contracts import (
    ApprovalRequest,
    Diagnostic,
    JSONValue,
    Omission,
    OutputTooLarge,
    Provenance,
    RunError,
    ToolCall,
    ToolContext,
    ToolFailure,
    ToolObservation,
    ToolSpec,
    canonical,
    validate_output,
)
from loom.runtime import OwnershipLease, Permit, pool, wait_work
from loom.security.egress import check_tool_egress
from loom.tools.approval import ApprovalCallback, approve
from loom.tools.base import Tool
from loom.tools.function import FunctionTool
from loom.tools.http import HTTPTool

T = TypeVar("T")


def effect_key(spec: ToolSpec, arguments: bytes) -> tuple[str, str, str]:
    return spec.name, spec.version, hashlib.sha256(arguments).hexdigest()


def run_error(error: ToolFailure, origin: str = "tool") -> RunError:
    return RunError(
        code=error.code,
        message=error.message,
        origin=origin,
        retryable=error.transient,
        outcome_unknown=error.outcome_unknown,
        details=error.details,
    )


async def bounded(
    operation: Awaitable[T],
    *,
    deadline: float,
    context: ToolContext,
    finished: Callable[[], None],
) -> T:
    task = asyncio.ensure_future(operation)

    def done(completed: asyncio.Future[T]) -> None:
        finished()
        if not completed.cancelled():
            completed.exception()

    task.add_done_callback(done)
    try:
        while True:
            if context.cancellation.is_cancelled():
                raise ToolFailure("operation_cancelled", recoverable=True)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            ready, _ = await asyncio.wait({task}, timeout=min(remaining, 0.05))
            if ready:
                return task.result()
    finally:
        if not task.done():
            task.cancel()


class ToolRunner:
    def __init__(
        self,
        *,
        tools: Mapping[str, Tool],
        approval: ApprovalCallback | None = None,
        policies: Mapping[str, str] | None = None,
        grants: Mapping[str, Sequence[str]] | None = None,
        lease: OwnershipLease | None = None,
        on_event: Callable[[str, dict[str, JSONValue]], Awaitable[None]] | None = None,
        config: Config | None = None,
    ) -> None:
        self.tools = tools
        self.approval = approval
        self.policies = policies or {}
        self.grants = grants or {}
        self.lease = lease
        self.on_event = on_event
        self.config = config
        self.diagnostics: list[Diagnostic] = []
        self.omitted_effects: set[tuple[str, str, str]] = set()
        self.approved_calls: set[str] = set()

    async def call(
        self,
        call: ToolCall,
        context: ToolContext,
        *,
        argument_bytes: int,
        approval_seconds: float,
        reserve_attempt: Callable[[], Awaitable[None]],
        tool_seconds: float = 30.0,
    ) -> ToolObservation:
        tool = self.tools.get(call.name)
        if tool is None:
            return self.failure(call, ToolFailure("unknown_tool", recoverable=True))
        spec = tool.spec
        try:
            arguments = canonical(call.arguments, max_bytes=argument_bytes)
            validate_output(call.arguments, spec.argument_schema)
            if isinstance(tool, FunctionTool):
                tool.arguments_model.model_validate_json(arguments)
        except (ToolFailure, ValidationError):
            return self.failure(call, ToolFailure("invalid_tool_arguments", recoverable=True))
        effect = effect_key(spec, arguments)
        if not spec.retry_safe and effect in self.omitted_effects:
            return self.failure(call, ToolFailure("duplicate_effect_prevented", recoverable=True))
        claimed = False
        dispatched = False
        permit: Permit | None = None
        release: Callable[[], None] | None = None

        def finished() -> None:
            tool.release(call.call_id)
            if release is not None:
                release()

        try:
            check_tool_egress(spec, context.provenance, self.policies, self.grants)
            tool.claim(call.call_id, context.run_id)
            claimed = True
            if spec.sensitive and call.call_id not in self.approved_calls:
                if self.on_event is not None:
                    await self.on_event("approval.requested", {"call_id": call.call_id})
                decision, diagnostic = await approve(
                    ApprovalRequest(
                        tool_name=spec.name,
                        tool_version=spec.version,
                        call_id=call.call_id,
                        arguments=call.arguments,
                        caller=context.caller,
                        egress_origins=spec.egress_origins,
                        provenance=context.provenance,
                        contains_document_content=bool(context.provenance),
                    ),
                    context,
                    self.approval,
                    approval_seconds,
                    lease=self.lease,
                )
                if diagnostic:
                    self.diagnostics.append(
                        Diagnostic(
                            code=diagnostic, message=diagnostic.replace("_", " "), origin="approval"
                        )
                    )
                if self.on_event is not None:
                    await self.on_event(
                        "approval.decided",
                        {"call_id": call.call_id, "approved": decision.approved},
                    )
                if context.cancellation.is_cancelled():
                    raise ToolFailure("operation_cancelled", recoverable=True)
                if not decision.approved:
                    return self.failure(call, ToolFailure("approval_denied", recoverable=True))
                self.approved_calls.add(call.call_id)
            if isinstance(tool, FunctionTool) and not tool.is_async:
                permit = pool("user_code").reserve()
            await reserve_attempt()
            from datetime import UTC, datetime, timedelta

            remaining = (context.deadline - datetime.now(UTC)).total_seconds()
            if remaining <= 0 or context.cancellation.is_cancelled():
                raise ToolFailure("operation_cancelled", recoverable=True)
            remaining = min(remaining, spec.timeout_seconds or tool_seconds)
            context = context.model_copy(
                update={
                    "deadline": datetime.now(UTC) + timedelta(seconds=remaining),
                }
            )
            deadline = time.monotonic() + remaining
            check_tool_egress(spec, context.provenance, self.policies, self.grants)
            tool.phase(call.call_id, "running")
            release = self.lease.retain() if self.lease is not None else None
            if isinstance(tool, FunctionTool) and not tool.is_async:
                work = pool("user_code").submit(
                    lambda: tool.invoke(call.arguments, context),
                    permit=permit,
                    deadline=deadline,
                    cancellation=context.cancellation,
                    finished=finished,
                )
                dispatched = True
                value = await wait_work(work, deadline=deadline, cancellation=context.cancellation)
            else:
                dispatched = True
                value = await bounded(
                    tool.execute(call.arguments, context, config=self.config)
                    if isinstance(tool, HTTPTool)
                    else tool.execute(call.arguments, context),
                    deadline=deadline,
                    context=context,
                    finished=finished,
                )
            from loom.contracts import SearchResult
            from loom.documents import SearchTool

            provenance: tuple[Provenance, ...] = ()
            if isinstance(tool, SearchTool) and isinstance(value, SearchResult):
                provenance = tool.provenance(value)
                cap = min(context.observation_bytes, 4096)
                while (
                    len(
                        canonical(
                            ToolObservation(
                                call_id=call.call_id,
                                execution_status="succeeded",
                                output=value.model_dump(mode="json"),
                                provenance=provenance,
                            )
                        )
                    )
                    > cap
                ):
                    if not value.chunks:
                        raise ToolFailure("observation_too_large", recoverable=True)
                    value = tool.clip(value)
                    provenance = tool.provenance(value)
                if value.clipped and not value.chunks:
                    raise ToolFailure("observation_too_large", recoverable=True)
            observation = self.output(call, value, spec.output_schema, context, provenance)
            if observation.omission is not None and not spec.retry_safe:
                self.omitted_effects.add(effect)
            return observation
        except TimeoutError:
            tool.phase(call.call_id, "timed_out_running")
            if not spec.retry_safe:
                raise ToolFailure("unknown_tool_outcome", outcome_unknown=True) from None
            return self.failure(
                call,
                ToolFailure("tool_timeout", recoverable=True, transient=True),
                status="unknown",
            )
        except OutputTooLarge:
            if not spec.retry_safe:
                self.omitted_effects.add(effect)
            return ToolObservation(
                call_id=call.call_id,
                execution_status="succeeded",
                error=RunError(
                    code="observation_too_large",
                    message="Completed output exceeds the limit.",
                    origin="tool",
                    details={"operation_executed": True},
                ),
                omission=Omission(byte_count=None, sha256=None, validation="not_performed"),
            )
        except ToolFailure as error:
            if error.code == "operation_cancelled":
                if dispatched:
                    tool.phase(call.call_id, "timed_out_running")
                if dispatched and not spec.retry_safe:
                    raise ToolFailure("unknown_tool_outcome", outcome_unknown=True) from None
                raise
            if dispatched and error.outcome_unknown and not spec.retry_safe:
                raise ToolFailure("unknown_tool_outcome", outcome_unknown=True) from None
            if error.recoverable or (error.transient and spec.retry_safe):
                return self.failure(call, error, status="failed" if dispatched else "not_started")
            raise
        except asyncio.CancelledError:
            if dispatched:
                tool.phase(call.call_id, "timed_out_running")
            raise
        except BaseException as exc:
            # _HandoffSignal inherits from BaseException, not Exception, so it
            # would not be caught by a plain `except Exception` — but be explicit.
            from loom.tools.agent import _HandoffSignal

            if isinstance(exc, _HandoffSignal):
                raise
            # All other non-Exception BaseExceptions (e.g. KeyboardInterrupt) propagate.
            if not isinstance(exc, Exception):
                raise
            # Regular Exception subclasses land here.
            raise ToolFailure(
                "unknown_tool_outcome" if dispatched and not spec.retry_safe else "tool_failed",
                outcome_unknown=dispatched and not spec.retry_safe,
            ) from None
        finally:
            if not dispatched:
                if permit is not None:
                    permit.release()
                if claimed:
                    tool.release(call.call_id)
                if release is not None:
                    release()

    @staticmethod
    def failure(
        call: ToolCall,
        error: ToolFailure,
        *,
        status: str = "not_started",
    ) -> ToolObservation:
        return ToolObservation.model_validate(
            {
                "call_id": call.call_id,
                "execution_status": status,
                "error": run_error(error),
            }
        )

    @staticmethod
    def output(
        call: ToolCall,
        value: object,
        schema: dict[str, JSONValue],
        context: ToolContext,
        provenance: tuple[Provenance, ...] = (),
    ) -> ToolObservation:
        try:
            encoded = canonical(value, max_bytes=1048576)
        except ValidationError:
            raise ToolFailure("invalid_json_output") from None
        except ToolFailure as error:
            if error.code != "payload_too_large":
                raise
            encoded = None
        if encoded is not None:
            value = validate_output(value, schema)
            observation = ToolObservation(
                call_id=call.call_id,
                execution_status="succeeded",
                output=value,
                provenance=provenance,
            )
            if len(canonical(observation)) <= context.observation_bytes:
                return observation
        return ToolObservation(
            call_id=call.call_id,
            execution_status="succeeded",
            output=None,
            error=RunError(
                code="observation_too_large",
                message="Completed output exceeds the observation limit.",
                origin="tool",
                details={"operation_executed": True},
            ),
            omission=Omission(
                byte_count=len(encoded) if encoded is not None else None,
                sha256=hashlib.sha256(encoded).hexdigest() if encoded is not None else None,
                validation="validated" if encoded is not None else "not_performed",
            ),
            provenance=provenance,
        )
