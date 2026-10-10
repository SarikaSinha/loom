from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Literal

from loom.agent import Agent, _Run
from loom.config import Config, StoreConfig
from loom.contracts import (
    NAME,
    CallerIdentity,
    CancellationToken,
    ConversationState,
    Diagnostic,
    JSONValue,
    LoomConfigurationError,
    LoomStorageError,
    Message,
    Record,
    RunResult,
    ToolCall,
    ToolContext,
    ToolFailure,
    Usage,
    WorkflowLimits,
    fingerprint,
    finite_seconds,
    json_value,
)
from loom.events import Events, EventSink, resolved_secrets
from loom.providers.base import claim_adapter, release_adapter
from loom.recovery import resolve
from loom.runtime import OwnershipLease, leased_ownership, offload, retry_pause, sync
from loom.security.egress import merge_provenance
from loom.security.maintenance import Managed, protected
from loom.storage import Checkpoint, JSONStorage, PendingOperation, SQLiteStorage, Storage
from loom.tools.approval import ApprovalCallback
from loom.tools.base import Tool
from loom.tools.runner import ToolRunner, run_error


class WorkflowView(Record):
    input: JSONValue
    outputs: dict[str, JSONValue]
    last_step: str | None


def _name(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(NAME, value):
        raise LoomConfigurationError("invalid_workflow_name")


def _control(function: object) -> None:
    if (
        not callable(function)
        or inspect.iscoroutinefunction(function)
        or inspect.iscoroutinefunction(type(function).__call__)
    ):
        raise LoomConfigurationError("workflow_control_must_be_synchronous")


@dataclass(frozen=True)
class Case:
    name: str
    predicate: Callable[[WorkflowView], bool]
    target: str

    def __post_init__(self) -> None:
        _name(self.name)
        _name(self.target)
        _control(self.predicate)


@dataclass(frozen=True)
class Branch:
    cases: Sequence[Case]
    default: str | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))
        if not self.cases or any(not isinstance(case, Case) for case in self.cases):
            raise LoomConfigurationError("invalid_branch")
        if len({case.name for case in self.cases}) != len(self.cases):
            raise LoomConfigurationError("duplicate_case")
        if self.default is not None:
            _name(self.default)


@dataclass(frozen=True)
class Step:
    name: str
    target: Agent | Tool
    input_selector: Callable[[WorkflowView], JSONValue] = field(kw_only=True)
    next: str | Branch | None = field(default=None, kw_only=True)
    version: str = field(default="1", kw_only=True)

    def __post_init__(self) -> None:
        _name(self.name)
        _control(self.input_selector)
        if not isinstance(self.target, (Agent, Tool)) or not self.version:
            raise LoomConfigurationError("invalid_workflow_step")
        if isinstance(self.next, str):
            _name(self.next)
        elif self.next is not None and not isinstance(self.next, Branch):
            raise LoomConfigurationError("invalid_workflow_transition")
        if isinstance(self.target, Agent) and self.target.storage is not None:
            raise LoomConfigurationError("workflow_owns_child_storage")


class Workflow:
    def __init__(
        self,
        *,
        name: str,
        steps: Sequence[Step],
        entry: str,
        output_selector: Callable[[WorkflowView], JSONValue],
        version: str = "1",
        limits: WorkflowLimits | None = None,
        storage: Storage | None = None,
        approval: ApprovalCallback | None = None,
        event_sinks: Sequence[EventSink] = (),
        tool_document_access: Mapping[str, Sequence[str]] | None = None,
        config: Config | str | Path | None = None,
    ) -> None:
        _name(name)
        _control(output_selector)
        if not version or not steps or any(not isinstance(step, Step) for step in steps):
            raise LoomConfigurationError("invalid_workflow")
        self.name, self.version, self.entry = name, version, entry
        self.steps = {step.name: step for step in steps}
        if len(self.steps) != len(steps) or entry not in self.steps:
            raise LoomConfigurationError("invalid_workflow_graph")
        self._validate_graph()
        self.output_selector = output_selector
        self.config = Config.load(config)
        self.config.validate_paths()
        self.approval, self.sinks = approval, tuple(event_sinks)
        self.grants = {name: tuple(stores) for name, stores in (tool_document_access or {}).items()}
        self.policies: dict[str, str] = {}
        bindings: dict[str, tuple[Path, Path]] = {}
        for step in self.steps.values():
            if not isinstance(step.target, Agent):
                continue
            for store in step.target.data:
                assert store.index_path is not None
                binding = (store.path, store.index_path)
                if store.name in bindings and bindings[store.name] != binding:
                    raise LoomConfigurationError("workflow_store_binding_conflict")
                bindings[store.name] = binding
                policy = step.target.document_access.get(store.name, "local_only")
                previous = self.policies.get(store.name, "allow_hosted")
                self.policies[store.name] = (
                    "allow_hosted" if previous == policy == "allow_hosted" else "local_only"
                )
        for step_name, stores in self.grants.items():
            if step_name not in self.steps or not isinstance(self.steps[step_name].target, Tool):
                raise LoomConfigurationError("invalid_tool_document_grant")
            if any(self.policies.get(store) != "allow_hosted" for store in stores):
                raise LoomConfigurationError("invalid_tool_document_grant")
        derived: dict[str, int | float] = {
            "max_step_activations": len(steps),
            "max_model_turns": 0,
            "max_tool_attempts": 0,
            "deadline_seconds": 0.0,
        }
        for step in steps:
            if isinstance(step.target, Agent):
                derived["max_model_turns"] += step.target.limits.max_model_turns
                derived["max_tool_attempts"] += step.target.limits.max_tool_attempts
                derived["deadline_seconds"] += step.target.limits.deadline_seconds
            else:
                derived["max_tool_attempts"] += 3 if step.target.spec.retry_safe else 1
                derived["deadline_seconds"] += 120.0
        if limits is not None and not isinstance(limits, WorkflowLimits):
            raise LoomConfigurationError("invalid_workflow_limits")
        for key, value in (limits or WorkflowLimits()).model_dump(exclude_none=True).items():
            if value > derived[key]:
                raise LoomConfigurationError("workflow_limit_exceeds_derived")
            derived[key] = value
        self.ceilings = derived
        self.storage = storage.bound(self.config) if storage is not None else None
        if self.storage is None and self.config.storage.enabled:
            settings = self.config.storage
            assert settings.path is not None
            backend = SQLiteStorage if settings.backend == "sqlite" else JSONStorage
            self.storage = backend(path=self.config.path(settings.path), config=self.config)
        self.config.validate_paths(
            stores={
                name: StoreConfig(path=str(source), index_path=str(index))
                for name, (source, index) in bindings.items()
            },
            storage_path=self.storage.path if self.storage is not None else None,
        )

    def _validate_graph(self) -> None:
        edges: dict[str, set[str]] = {}
        for name, step in self.steps.items():
            following = step.next
            targets = (
                {following}
                if isinstance(following, str)
                else {case.target for case in following.cases}
                | ({following.default} if following.default is not None else set())
                if isinstance(following, Branch)
                else set()
            )
            if targets - self.steps.keys():
                raise LoomConfigurationError("unknown_workflow_target")
            edges[name] = targets
        reached: set[str] = set()
        pending = [self.entry]
        while pending:
            name = pending.pop()
            if name not in reached:
                reached.add(name)
                pending.extend(edges[name])
        if reached != self.steps.keys():
            raise LoomConfigurationError("unreachable_workflow_step")
        degrees = {name: 0 for name in self.steps}
        for targets in edges.values():
            for target in targets:
                degrees[target] += 1
        pending = [name for name, degree in degrees.items() if degree == 0]
        visited = 0
        while pending:
            visited += 1
            for target in edges[pending.pop()]:
                degrees[target] -= 1
                if degrees[target] == 0:
                    pending.append(target)
        if visited != len(self.steps):
            raise LoomConfigurationError("workflow_cycle")

    def descriptor(self) -> dict[str, JSONValue]:
        steps: dict[str, JSONValue] = {}
        for name, step in self.steps.items():
            following = step.next
            steps[name] = {
                "version": step.version,
                "kind": "agent" if isinstance(step.target, Agent) else "tool",
                "target": (
                    step.target.descriptor()
                    if isinstance(step.target, Agent)
                    else json_value(step.target.spec)
                ),
                "next": {
                    "cases": [
                        {"name": case.name, "target": case.target} for case in following.cases
                    ],
                    "default": following.default,
                }
                if isinstance(following, Branch)
                else following,
            }
        return {
            "name": self.name,
            "version": self.version,
            "entry": self.entry,
            "steps": steps,
            "limits": json_value(self.ceilings),
            "document_access": json_value(self.policies),
            "tool_document_access": json_value(self.grants),
            "binding_paths": {
                "storage": str(self.storage.path) if self.storage else None,
                "trace_prefix": str(self.config.path(self.config.observability.jsonl_path))
                if self.config.observability.jsonl_path
                else None,
            },
        }

    def run(
        self,
        *,
        input: JSONValue,
        context: Mapping[str, object] | None = None,
        caller: CallerIdentity | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        return sync(
            lambda: self.arun(
                input=input,
                context=context,
                caller=caller,
                cancellation=cancellation,
            )
        )

    async def arun(
        self,
        *,
        input: JSONValue,
        context: Mapping[str, object] | None = None,
        caller: CallerIdentity | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        caller = caller or CallerIdentity()
        run_id = str(uuid.uuid4())
        descriptor = self.descriptor()
        checkpoint = Checkpoint(
            kind="workflow",
            root_run_id=run_id,
            run_id=run_id,
            configuration=descriptor,
            configuration_fingerprint=fingerprint(descriptor),
            workflow_input=json_value(input),
            active_step=self.entry,
            state=ConversationState(
                conversation_id=str(uuid.uuid4()),
                caller_subject=caller.subject,
                continuation_eligible=False,
            ),
            deadline=datetime.now(UTC) + timedelta(seconds=self.ceilings["deadline_seconds"]),
        )
        return await self._execute(
            checkpoint, context or {}, caller, cancellation or CancellationToken(), resume=False
        )

    def resume(
        self,
        run_id: str,
        *,
        extend_deadline_seconds: float | None = None,
        context: Mapping[str, object] | None = None,
        caller: CallerIdentity | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        return sync(
            lambda: self.aresume(
                run_id,
                extend_deadline_seconds=extend_deadline_seconds,
                context=context,
                caller=caller,
                cancellation=cancellation,
            )
        )

    async def aresume(
        self,
        run_id: str,
        *,
        extend_deadline_seconds: float | None = None,
        context: Mapping[str, object] | None = None,
        caller: CallerIdentity | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        if self.storage is None:
            raise LoomConfigurationError("durability_required")
        checkpoint = await offload(
            "internal_io", partial(self.storage.load_run, run_id), seconds=30
        )
        return await self._execute(
            checkpoint,
            context or {},
            caller or CallerIdentity(),
            cancellation or CancellationToken(),
            resume=True,
            extension=extend_deadline_seconds,
        )

    async def _execute(
        self,
        checkpoint: Checkpoint,
        context: Mapping[str, object],
        caller: CallerIdentity,
        cancellation: CancellationToken,
        *,
        resume: bool,
        extension: float | None = None,
    ) -> RunResult:
        ownership = (
            self.storage.ownership(checkpoint.run_id, checkpoint.state.conversation_id)
            if self.storage
            else nullcontext()
        )
        resources = []
        if self.storage is not None:
            resources.append(Managed(self.storage.config, self.storage.path, self.storage.kind))
        if self.config.observability.jsonl_path:
            resources.append(
                Managed(
                    self.config,
                    self.config.path(self.config.observability.jsonl_path),
                    "trace_manifest",
                )
            )
        for step in self.steps.values():
            if isinstance(step.target, Agent):
                resources.extend(step.target._managed())
        async with leased_ownership(protected(resources, ownership)) as lease:
            if resume:
                assert self.storage is not None
                checkpoint = await offload(
                    "internal_io",
                    partial(self.storage.load_run, checkpoint.run_id),
                    seconds=30,
                )
            if checkpoint.kind != "workflow" or (
                checkpoint.configuration_fingerprint != fingerprint(self.descriptor())
            ):
                raise LoomConfigurationError("checkpoint_incompatible")
            if checkpoint.state.caller_subject != caller.subject:
                raise LoomConfigurationError("caller_mismatch")
            if checkpoint.result is not None:
                return checkpoint.result
            if extension is not None:
                proposed = datetime.now(UTC) + timedelta(seconds=finite_seconds(extension))
                child = checkpoint.child
                checkpoint = checkpoint.model_copy(
                    update={
                        "deadline": max(proposed, checkpoint.deadline),
                        "step_deadline": max(proposed, checkpoint.step_deadline)
                        if checkpoint.step_deadline
                        else None,
                        "child": child.model_copy(
                            update={"deadline": max(proposed, child.deadline)}
                        )
                        if child
                        else None,
                        "diagnostics": (
                            *checkpoint.diagnostics,
                            Diagnostic(
                                code="deadline_extended",
                                message="Operator extended the workflow deadline.",
                                origin="operator",
                            ),
                        ),
                    }
                )
            return await _WorkflowRun(
                self, checkpoint, context, caller, cancellation, lease
            ).execute(resume)

    def resolve_tool_outcome(
        self,
        run_id: str,
        call_id: str,
        *,
        resolution: Literal["succeeded", "failed", "abandoned"],
        output: JSONValue = None,
        error: ToolFailure | None = None,
        operator_note: str,
    ) -> None:
        if self.storage is None:
            raise LoomConfigurationError("durability_required")
        with self.storage.ownership(run_id):
            checkpoint = self.storage.load_run(run_id)
            if checkpoint.configuration_fingerprint != fingerprint(self.descriptor()):
                raise LoomConfigurationError("checkpoint_incompatible")
            if checkpoint.active_step is None or checkpoint.phase != "needs_resolution":
                raise LoomConfigurationError("resolution_not_required")
            target = self.steps[checkpoint.active_step].target
            child = checkpoint.child
            if isinstance(target, Agent):
                if child is None:
                    raise LoomStorageError("workflow_child_missing")
                tool = next(
                    (
                        tool
                        for tool in target.tools
                        if any(
                            op.call.call_id == call_id and op.call.name == tool.spec.name
                            for op in child.pending
                        )
                    ),
                    None,
                )
                if tool is None:
                    raise LoomConfigurationError("unknown_tool_call")
                resolved = resolve(
                    child,
                    tool,
                    call_id,
                    resolution=resolution,
                    output=output,
                    error=error,
                    operator_note=operator_note,
                    observation_bytes=target.limits.max_observation_bytes,
                )
                checkpoint = checkpoint.model_copy(update={"child": resolved, "phase": "ready"})
            else:
                checkpoint = resolve(
                    checkpoint,
                    target,
                    call_id,
                    resolution=resolution,
                    output=output,
                    error=error,
                    operator_note=operator_note,
                    observation_bytes=8192,
                )
            self.storage.save(checkpoint)


class _WorkflowRun:
    def __init__(
        self,
        workflow: Workflow,
        checkpoint: Checkpoint,
        context: Mapping[str, object],
        caller: CallerIdentity,
        cancellation: CancellationToken,
        lease: OwnershipLease,
    ) -> None:
        self.workflow, self.cp, self.context = workflow, checkpoint, context
        self.caller, self.cancellation = caller, cancellation
        self.lease = lease
        self.started = time.monotonic()
        from loom.tools.http import HTTPTool

        secret_env = [
            reference["env"]
            for step in workflow.steps.values()
            if isinstance(step.target, HTTPTool)
            for reference in step.target.auth_env.values()
        ]
        self.events = Events(
            checkpoint.run_id,
            workflow.sinks,
            sequence=checkpoint.event_sequence,
            capture_payloads=workflow.config.observability.capture_payloads,
            source_paths=workflow.config.observability.capture_source_paths,
            extra_keys=workflow.config.observability.redact_keys,
            secrets=resolved_secrets(workflow.config, secret_env),
            lease=lease,
            deadline=checkpoint.deadline,
        )
        if workflow.config.observability.jsonl_path:
            from loom.trace import TraceSink

            self.events.sinks.append(TraceSink(workflow.config, self.events.diagnostics.append))

    def check(self) -> float:
        if self.cancellation.is_cancelled():
            raise ToolFailure("operation_cancelled")
        remaining = (self.cp.deadline - datetime.now(UTC)).total_seconds()
        if self.cp.step_deadline is not None:
            remaining = min(remaining, (self.cp.step_deadline - datetime.now(UTC)).total_seconds())
        if remaining <= 0:
            raise ToolFailure("deadline_exhausted")
        return remaining

    async def save(self) -> None:
        self.cp = self.cp.model_copy(update={"event_sequence": self.events.sequence + 1})
        if self.workflow.storage is not None:
            try:
                self.cp = await offload(
                    "internal_io",
                    partial(self.workflow.storage.save, self.cp),
                    seconds=30,
                )
            except (LoomStorageError, ToolFailure, TimeoutError) as error:
                raise ToolFailure(
                    "storage_write_failed",
                    details={"cause": getattr(error, "code", "timeout")},
                ) from None

    async def event(self, kind: str, metadata: dict[str, JSONValue] | None = None) -> None:
        await self.save()
        await self.events.emit(kind, metadata={"step_id": self.cp.active_step, **(metadata or {})})

    def charge(self, previous: Usage, current: Usage) -> None:
        counters: dict[str, int] = {}
        fields = (
            "logical_model_turns",
            "model_requests",
            "tool_attempts",
            "retries",
            "corrections",
        )
        for name in fields:
            counters[name] = (
                getattr(self.cp.usage, name) + getattr(current, name) - getattr(previous, name)
            )
        for name, ceiling, code in (
            ("logical_model_turns", "max_model_turns", "workflow_model_turn_limit"),
            ("tool_attempts", "max_tool_attempts", "workflow_tool_attempt_limit"),
        ):
            if counters[name] > self.workflow.ceilings[ceiling]:
                raise ToolFailure(code)
        changes: dict[str, object] = dict(counters)
        for name in ("input_tokens", "output_tokens"):
            value = getattr(current, name)
            total = getattr(self.cp.usage, name)
            changes[name] = (
                (total or 0) + value - (getattr(previous, name) or 0)
                if value is not None
                else total
            )
        changes["complete"] = self.cp.usage.complete and current.complete
        self.cp = self.cp.model_copy(update={"usage": self.cp.usage.model_copy(update=changes)})

    async def control(
        self,
        function: Callable[[WorkflowView], JSONValue],
        *,
        predicate: bool = False,
    ) -> JSONValue:
        view = WorkflowView(
            input=self.cp.workflow_input,
            outputs=self.cp.step_outputs,
            last_step=self.cp.last_step,
        )

        def evaluate() -> JSONValue:
            try:
                value = function(view)
                if predicate and type(value) is not bool:
                    raise ValueError("predicate must return a boolean")
                return json_value(value)
            except Exception:
                raise ToolFailure(
                    "branch_evaluation_failed" if predicate else "workflow_control_failed",
                ) from None

        try:
            return await self.lease.offload(
                "user_code",
                evaluate,
                seconds=min(30, self.check()),
                cancellation=self.cancellation,
            )
        except TimeoutError:
            raise ToolFailure("workflow_control_timeout") from None

    async def persist_child(self, child: Checkpoint) -> Checkpoint:
        self.cp = self.cp.model_copy(
            update={
                "child": child,
                "state": self.cp.state.model_copy(
                    update={
                        "snapshots": self.cp.state.snapshots | child.state.snapshots,
                        "provenance": merge_provenance(
                            self.cp.state.provenance, child.state.provenance
                        ),
                    }
                ),
            }
        )
        await self.save()
        return child

    async def agent(self, step: Step, target: Agent) -> RunResult:
        child = self.cp.child
        if child is None:
            value = await self.control(step.input_selector)
            if not isinstance(value, str):
                raise ToolFailure("invalid_step_input")
            run_id = str(uuid.uuid5(uuid.UUID(self.cp.run_id), step.name))
            descriptor = target.descriptor()
            child = Checkpoint(
                root_run_id=self.cp.run_id,
                run_id=run_id,
                configuration=descriptor,
                configuration_fingerprint=fingerprint(descriptor),
                deadline=self.cp.step_deadline or self.cp.deadline,
                state=ConversationState(
                    conversation_id=str(uuid.uuid4()),
                    caller_subject=self.caller.subject,
                    snapshots=self.cp.state.snapshots,
                    provenance=self.cp.state.provenance,
                    continuation_eligible=False,
                    messages=(
                        Message(role="user", content=value, provenance=self.cp.state.provenance),
                    ),
                ),
            )
            await self.persist_child(child)
        if child.result is not None:
            return child.result
        adapter, owned = target._adapter()
        claim_adapter(adapter)
        runner: _Run | None = None
        try:
            runner = _Run(
                target,
                child,
                adapter,
                self.context,
                self.caller,
                self.cancellation,
                persist=self.persist_child,
                charge=self.charge,
                approval=self.workflow.approval,
                step_id=step.name,
                lease=self.lease,
            )
            return await runner.execute(resume=child.phase != "ready")
        finally:
            if (
                runner is not None
                and runner.model_task is not None
                and not runner.model_task.done()
            ):
                runner.model_task.add_done_callback(lambda _: release_adapter(adapter))
            else:
                release_adapter(adapter)
            if owned:
                await adapter.aclose()

    async def tool(self, step: Step, tool: Tool) -> JSONValue:
        if not self.cp.pending:
            value = await self.control(step.input_selector)
            if not isinstance(value, dict):
                raise ToolFailure("invalid_step_input")
            call = ToolCall(
                call_id=f"tc_{self.cp.run_id}_0_{len(self.cp.activated_steps)}",
                name=tool.spec.name,
                arguments=value,
                ordinal=len(self.cp.activated_steps),
            )
            self.cp = self.cp.model_copy(update={"pending": (PendingOperation(call=call),)})
            await self.save()
        operation = self.cp.pending[0]
        if operation.status in {"started", "unknown"} and (
            not tool.spec.retry_safe or tool.is_active(operation.call.call_id)
        ):
            raise ToolFailure("unknown_tool_outcome", outcome_unknown=True)
        runner = ToolRunner(
            tools={tool.spec.name: tool},
            approval=self.workflow.approval,
            policies=self.workflow.policies,
            grants={tool.spec.name: self.workflow.grants.get(step.name, ())},
            lease=self.lease,
            on_event=self.event,
            config=self.workflow.config,
        )
        runner.omitted_effects.update(self.cp.omitted_effects)
        observation = operation.observation
        step_deadline = self.cp.step_deadline or self.cp.deadline
        while observation is None:
            operation = self.cp.pending[0]
            if operation.attempts >= (3 if tool.spec.retry_safe else 1):
                raise ToolFailure("tool_attempts_exhausted")
            context = ToolContext(
                root_run_id=self.cp.run_id,
                run_id=self.cp.run_id,
                conversation_id=self.cp.state.conversation_id,
                step_id=step.name,
                call_id=operation.call.call_id,
                attempt=operation.attempts + 1,
                deadline=step_deadline,
                cancellation=self.cancellation,
                caller=self.caller,
                context=self.context,
                provenance=self.cp.state.provenance,
                observation_bytes=8192,
            )

            async def reserve() -> None:
                self.check()
                previous = self.cp.usage
                current = previous.model_copy(
                    update={
                        "tool_attempts": previous.tool_attempts + 1,
                        "retries": previous.retries + bool(self.cp.pending[0].attempts),
                    }
                )
                self.charge(previous, current)
                pending = self.cp.pending[0].model_copy(
                    update={
                        "status": "started",
                        "attempts": self.cp.pending[0].attempts + 1,
                    }
                )
                self.cp = self.cp.model_copy(update={"pending": (pending,)})
                await self.event("tool.started")

            observation = await runner.call(
                operation.call,
                context,
                argument_bytes=8192,
                approval_seconds=min(60, self.check()),
                reserve_attempt=reserve,
            )
            self.cp = self.cp.model_copy(update={"omitted_effects": tuple(runner.omitted_effects)})
            if (
                observation.error
                and observation.error.retryable
                and tool.spec.retry_safe
                and (not tool.is_active(operation.call.call_id) and self.cp.pending[0].attempts < 3)
            ):
                self.cp = self.cp.model_copy(
                    update={
                        "pending": (self.cp.pending[0].model_copy(update={"status": "pending"}),)
                    }
                )
                await self.event("retry.scheduled")
                await retry_pause(
                    context.attempt,
                    check=self.check,
                    retry_after=observation.error.details.get("retry_after_seconds"),
                )
                observation = None
        self.cp = self.cp.model_copy(
            update={
                "pending": (
                    self.cp.pending[0].model_copy(
                        update={
                            "status": "succeeded"
                            if observation.execution_status == "succeeded"
                            else "failed",
                            "observation": observation,
                        }
                    ),
                ),
                "diagnostics": (*self.cp.diagnostics, *runner.diagnostics),
                "state": self.cp.state.model_copy(
                    update={
                        "provenance": merge_provenance(
                            self.cp.state.provenance, observation.provenance
                        ),
                    }
                ),
            }
        )
        await self.event("tool.finished")
        if observation.error:
            raise ToolFailure(
                observation.error.code,
                outcome_unknown=observation.error.outcome_unknown,
                details=observation.error.details,
            )
        return observation.output

    async def execute(self, resume: bool) -> RunResult:
        try:
            await self.event("checkpoint.saved")
            await self.event("run.resumed" if resume else "run.started")
            if self.cp.phase == "needs_resolution":
                raise ToolFailure("unknown_tool_outcome", outcome_unknown=True)
            while self.cp.active_step is not None:
                self.check()
                step = self.workflow.steps[self.cp.active_step]
                if step.name not in self.cp.step_outputs:
                    if step.name not in self.cp.activated_steps:
                        if (
                            len(self.cp.activated_steps)
                            >= self.workflow.ceilings["max_step_activations"]
                        ):
                            raise ToolFailure("workflow_step_limit")
                        self.cp = self.cp.model_copy(
                            update={
                                "activated_steps": (*self.cp.activated_steps, step.name),
                                "step_deadline": min(
                                    self.cp.deadline,
                                    datetime.now(UTC)
                                    + timedelta(
                                        seconds=step.target.limits.deadline_seconds
                                        if isinstance(step.target, Agent)
                                        else 120,
                                    ),
                                ),
                            }
                        )
                        await self.event("workflow.step_started")
                    if isinstance(step.target, Agent):
                        result = await self.agent(step, step.target)
                        if result.status != "completed":
                            assert result.error is not None
                            return await self.finish(
                                result.status,
                                error=ToolFailure(
                                    result.error.code,
                                    outcome_unknown=result.error.outcome_unknown,
                                    details={"step_id": step.name, "child_run_id": result.run_id},
                                ),
                            )
                        output = result.output
                        self.cp = self.cp.model_copy(
                            update={
                                "diagnostics": (*self.cp.diagnostics, *result.diagnostics),
                            }
                        )
                    else:
                        output = await self.tool(step, step.target)
                    self.cp = self.cp.model_copy(
                        update={
                            "step_outputs": self.cp.step_outputs | {step.name: output},
                            "last_step": step.name,
                        }
                    )
                    await self.event("workflow.step_finished")
                following = step.next
                if isinstance(following, Branch):
                    matches = []
                    for case in following.cases:
                        if await self.control(case.predicate, predicate=True):
                            matches.append(case.target)
                    if len(matches) > 1:
                        raise ToolFailure("branch_ambiguous")
                    selected = matches[0] if matches else following.default
                    if selected is None:
                        raise ToolFailure("branch_no_match")
                else:
                    selected = following
                self.cp = self.cp.model_copy(
                    update={
                        "active_step": selected,
                        "child": None,
                        "pending": (),
                        "cursor": 0,
                        "phase": "ready",
                        "step_deadline": None,
                    }
                )
                await self.event(
                    "workflow.branch_selected"
                    if isinstance(following, Branch)
                    else "checkpoint.saved"
                )
            output = await self.control(self.workflow.output_selector)
            return await self.finish("completed", output=output)
        except asyncio.CancelledError:
            self.cancellation.cancel()
            uncertain = self.cp.child is not None and self.cp.child.phase == "needs_resolution"
            if self.cp.active_step is not None:
                target = self.workflow.steps[self.cp.active_step].target
                uncertain |= (
                    isinstance(target, Tool)
                    and not target.spec.retry_safe
                    and any(op.status == "started" for op in self.cp.pending)
                )
            await self.finish(
                "cancelled",
                error=ToolFailure(
                    "unknown_tool_outcome" if uncertain else "operation_cancelled",
                    outcome_unknown=uncertain,
                ),
            )
            raise
        except (ToolFailure, LoomConfigurationError) as error:
            failure = error if isinstance(error, ToolFailure) else ToolFailure(error.code)
            status: Literal["failed", "cancelled", "limit_exhausted"] = (
                "cancelled"
                if self.cancellation.is_cancelled() or error.code == "operation_cancelled"
                else "limit_exhausted"
                if error.code
                in {
                    "deadline_exhausted",
                    "workflow_step_limit",
                    "workflow_model_turn_limit",
                    "workflow_tool_attempt_limit",
                }
                else "failed"
            )
            return await self.finish(
                status, error=failure, persist=error.code != "storage_write_failed"
            )

    async def finish(
        self,
        status: Literal["completed", "failed", "cancelled", "limit_exhausted"],
        *,
        output: JSONValue = None,
        error: ToolFailure | None = None,
        persist: bool = True,
    ) -> RunResult:
        state = self.cp.state.model_copy(
            update={
                "continuation_eligible": status == "completed",
                "version": 1 if status == "completed" else 0,
            }
        )
        state = ConversationState.model_validate_json(state.model_dump_json())
        usage = self.cp.usage.model_copy(
            update={
                "elapsed_ms": self.cp.usage.elapsed_ms + (time.monotonic() - self.started) * 1000,
            }
        )
        result = RunResult(
            run_id=self.cp.run_id,
            conversation_id=state.conversation_id,
            status=status,
            output=output,
            error=run_error(error, "workflow") if error else None,
            state=state,
            snapshots=state.snapshots,
            usage=usage,
            diagnostics=(*self.cp.diagnostics, *self.events.diagnostics),
        )
        phase = "needs_resolution" if error and error.outcome_unknown else status
        self.cp = self.cp.model_copy(
            update={
                "phase": phase,
                "result": None if phase == "needs_resolution" else result,
                "usage": usage,
                "state": state,
            }
        )
        if persist:
            if error is not None:
                await self.event("error", {"code": error.code})
            await self.event("run.finished", {"status": status})
            diagnostics = (*self.cp.diagnostics, *self.events.diagnostics)
            if result.diagnostics != diagnostics:
                result = result.model_copy(update={"diagnostics": diagnostics})
                self.cp = self.cp.model_copy(
                    update={"result": None if phase == "needs_resolution" else result}
                )
                await self.save()
        return result
