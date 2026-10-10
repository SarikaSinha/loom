from __future__ import annotations

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError

from loom.config import AgentProfile, Config
from loom.contracts import (
    NAME,
    AgentSpec,
    CallerIdentity,
    CancellationToken,
    ConversationState,
    Diagnostic,
    JSONValue,
    Limits,
    LoomConfigurationError,
    LoomStorageError,
    Message,
    RunResult,
    Schema,
    ToolContext,
    ToolFailure,
    Usage,
    canonical,
    decode_json,
    fingerprint,
    finite_seconds,
    json_value,
    schema_for,
    validate_output,
)
from loom.events import Events, EventSink, resolved_secrets
from loom.providers.base import (
    ModelAdapter,
    ModelFailure,
    ModelRequest,
    ModelResponse,
    claim_adapter,
    native_schema,
    release_adapter,
)
from loom.runtime import OwnershipLease, leased_ownership, offload, retry_pause, sync
from loom.security.egress import merge_provenance
from loom.security.maintenance import Managed, protected
from loom.storage import Checkpoint, JSONStorage, PendingOperation, SQLiteStorage, Storage
from loom.tools.approval import ApprovalCallback
from loom.tools.base import Tool
from loom.tools.runner import ToolRunner, run_error

if TYPE_CHECKING:
    from loom.documents import DocumentStore


class Agent:
    def __init__(
        self,
        *,
        name: str,
        instructions: str,
        version: str = "1",
        model: str | ModelAdapter | None = None,
        tools: Sequence[Tool] = (),
        data: Sequence[DocumentStore] = (),
        output_schema: type[BaseModel] | Schema | str | None = None,
        limits: Limits | Mapping[str, JSONValue] | None = None,
        config: Config | str | Path | None = None,
        storage: Storage | None = None,
        approval: ApprovalCallback | None = None,
        event_sinks: Sequence[EventSink] = (),
        document_access: Mapping[str, Literal["local_only", "allow_hosted"]] | None = None,
        tool_document_access: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if not re.fullmatch(NAME, name) or not instructions.strip() or not version:
            raise LoomConfigurationError("invalid_agent_definition")
        self.name, self.instructions, self.version = name, instructions, version
        self.config = Config.load(config)
        self.config.validate_paths()
        profile = self.config.agents.get(name, AgentProfile())
        self.model = (
            model if model is not None else profile.provider or self.config.defaults.provider
        )
        if (
            self.model is None
            or isinstance(self.model, str)
            and self.model not in self.config.providers
        ):
            raise LoomConfigurationError("model_unavailable")
        chosen_schema = (
            output_schema
            if output_schema is not None
            else (profile.output_schema or self.config.defaults.output_schema)
        )
        if isinstance(chosen_schema, str):
            chosen_schema = self.config.output_schema(chosen_schema)
        self.schema = schema_for(chosen_schema)
        overrides = (
            limits.model_dump(exclude_unset=True)
            if isinstance(limits, Limits)
            else dict(limits or {})
        )
        try:
            self.limits = Limits.model_validate(
                self.config.defaults.limits | profile.limits | overrides
            )
        except ValidationError:
            raise LoomConfigurationError("invalid_limits") from None
        self.data = tuple(store.bind(self.config) for store in data)
        self.tools = tuple(tools)
        self.allowlist = profile.allowlist
        attached = {tool.spec.name for tool in tools} | {
            f"search_{store.name}" for store in self.data
        }
        if len(attached) != len(tools) + len(self.data):
            raise LoomConfigurationError("duplicate_tool_name")
        if self.allowlist is not None and set(self.allowlist) - attached:
            raise LoomConfigurationError("unknown_allowlist_tool")
        stores = {store.name for store in self.data}
        self.document_access = profile.document_access | dict(document_access or {})
        if set(self.document_access) - stores - self.config.stores.keys():
            raise LoomConfigurationError("unattached_document_permission")
        self.grants: dict[str, Sequence[str]] = {
            **profile.tool_document_access,
            **dict(tool_document_access or {}),
        }
        for tool, granted in self.grants.items():
            if (
                tool not in attached
                or set(granted) - stores
                or any(
                    self.document_access.get(store, "local_only") != "allow_hosted"
                    for store in granted
                )
            ):
                raise LoomConfigurationError("invalid_tool_document_grant")
        self.approval = approval
        self.sinks = tuple(event_sinks)
        self.storage: Storage | None
        if storage is not None:
            self.storage = storage.bound(self.config)
        elif self.config.storage.enabled:
            settings = self.config.storage
            assert settings.path is not None
            backend = SQLiteStorage if settings.backend == "sqlite" else JSONStorage
            self.storage = backend(path=self.config.path(settings.path), config=self.config)
        else:
            self.storage = None
        from loom.config import StoreConfig

        self.config.validate_paths(
            stores={
                store.name: StoreConfig(path=str(store.path), index_path=str(store.index_path))
                for store in self.data
            },
            storage_path=self.storage.path if self.storage is not None else None,
        )

    def _adapter(self) -> tuple[ModelAdapter, bool]:
        if isinstance(self.model, ModelAdapter):
            return self.model, False
        assert isinstance(self.model, str)
        from loom.providers.factory import create_adapter

        return create_adapter(self.config.providers[self.model], self.config), True

    def _managed(self) -> list[Managed]:
        resources = []
        if self.storage is not None:
            resources.append(Managed(self.storage.config, self.storage.path, self.storage.kind))
        for store in self.data:
            assert store.index_path is not None
            resources.append(Managed(self.config, store.index_path, "index_current"))
        if self.config.observability.jsonl_path:
            resources.append(
                Managed(
                    self.config,
                    self.config.path(self.config.observability.jsonl_path),
                    "trace_manifest",
                )
            )
        return resources

    @property
    def spec(self) -> AgentSpec:
        return AgentSpec.model_validate_json(canonical(self.descriptor()))

    def descriptor(self) -> dict[str, JSONValue]:
        assert self.model is not None
        model = (
            json_value(self.config.providers[self.model])
            if isinstance(self.model, str)
            else {
                "adapter": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
                "capabilities": json_value(self.model.capabilities),
            }
        )
        bindings: dict[str, JSONValue] = {
            "storage": str(self.storage.path) if self.storage else None,
            "embedding_cache": str(self.config.path(self.config.embeddings.cache_path)),
            "trace_prefix": (
                str(self.config.path(self.config.observability.jsonl_path))
                if self.config.observability.jsonl_path
                else None
            ),
            "stores": {
                store.name: {"source": str(store.path), "index": str(store.index_path)}
                for store in self.data
            },
        }
        return {
            "name": self.name,
            "instructions": self.instructions,
            "version": self.version,
            "model": model,
            "tools": [json_value(tool.spec) for tool in self.tools],
            "output_schema": self.schema,
            "limits": json_value(self.limits),
            "document_access": json_value(self.document_access),
            "tool_document_access": json_value(self.grants),
            "allowlist": json_value(list(self.allowlist)) if self.allowlist is not None else None,
            "binding_paths": bindings,
        }

    def run(
        self,
        *,
        input: str,
        state: ConversationState | None = None,
        context: Mapping[str, object] | None = None,
        caller: CallerIdentity | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        return sync(
            lambda: self.arun(
                input=input, state=state, context=context, caller=caller, cancellation=cancellation
            )
        )

    async def arun(
        self,
        *,
        input: str,
        state: ConversationState | None = None,
        context: Mapping[str, object] | None = None,
        caller: CallerIdentity | None = None,
        cancellation: CancellationToken | None = None,
    ) -> RunResult:
        if not isinstance(input, str):
            raise LoomConfigurationError("invalid_input")
        caller = caller or CallerIdentity()
        if state is not None:
            state = ConversationState.model_validate_json(state.model_dump_json())
            if not state.continuation_eligible:
                raise LoomConfigurationError("resume_required")
            if state.caller_subject != caller.subject:
                raise LoomConfigurationError("caller_mismatch")
        else:
            state = ConversationState(
                conversation_id=str(uuid.uuid4()), caller_subject=caller.subject
            )
        state = state.model_copy(
            update={
                "messages": (*state.messages, Message(role="user", content=input)),
                "continuation_eligible": False,
            }
        )
        run_id = str(uuid.uuid4())
        descriptor = self.descriptor()
        checkpoint = Checkpoint(
            root_run_id=run_id,
            run_id=run_id,
            state=state,
            base_conversation_version=state.version,
            configuration=descriptor,
            configuration_fingerprint=fingerprint(descriptor),
            deadline=datetime.now(UTC) + timedelta(seconds=self.limits.deadline_seconds),
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
        adapter: ModelAdapter | None = None
        owned = claimed = False
        ownership = (
            self.storage.ownership(checkpoint.run_id, checkpoint.state.conversation_id)
            if self.storage is not None
            else nullcontext()
        )
        runner: _Run | None = None
        try:
            async with leased_ownership(protected(self._managed(), ownership)) as lease:
                if resume:
                    assert self.storage is not None
                    checkpoint = await offload(
                        "internal_io", partial(self.storage.load_run, checkpoint.run_id), seconds=30
                    )
                if checkpoint.configuration_fingerprint != fingerprint(self.descriptor()):
                    raise LoomConfigurationError("checkpoint_incompatible")
                if checkpoint.state.caller_subject != caller.subject:
                    raise LoomConfigurationError("caller_mismatch")
                if checkpoint.result is not None:
                    return checkpoint.result
                if extension is not None:
                    proposed = datetime.now(UTC) + timedelta(seconds=finite_seconds(extension))
                    checkpoint = checkpoint.model_copy(
                        update={
                            "deadline": max(checkpoint.deadline, proposed),
                            "diagnostics": (
                                *checkpoint.diagnostics,
                                Diagnostic(
                                    code="deadline_extended",
                                    message="Operator extended the run deadline.",
                                    origin="runtime",
                                ),
                            ),
                        }
                    )
                adapter, owned = self._adapter()
                claim_adapter(adapter)
                claimed = True
                runner = _Run(self, checkpoint, adapter, context, caller, cancellation, lease=lease)
                return await runner.execute(resume)
        finally:
            if adapter is not None and claimed:
                if (
                    runner is not None
                    and runner.model_task is not None
                    and not runner.model_task.done()
                ):

                    def release_completed(
                        task: asyncio.Task[ModelResponse], adapter: ModelAdapter = adapter
                    ) -> None:
                        release_adapter(adapter)

                    runner.model_task.add_done_callback(release_completed)
                else:
                    release_adapter(adapter)
                if owned:
                    await adapter.aclose()

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
        from loom.recovery import resolve

        with self.storage.ownership(run_id):
            checkpoint = self.storage.load_run(run_id)
            if checkpoint.configuration_fingerprint != fingerprint(self.descriptor()):
                raise LoomConfigurationError("checkpoint_incompatible")
            operation = next((op for op in checkpoint.pending if op.call.call_id == call_id), None)
            if operation is None:
                raise LoomConfigurationError("unknown_tool_call")
            tool = next((t for t in self.tools if t.spec.name == operation.call.name), None)
            if tool is None:
                raise LoomConfigurationError("tool_unavailable_or_active")
            self.storage.save(
                resolve(
                    checkpoint,
                    tool,
                    call_id,
                    resolution=resolution,
                    output=output,
                    error=error,
                    operator_note=operator_note,
                    observation_bytes=self.limits.max_observation_bytes,
                )
            )


class _Run:
    def __init__(
        self,
        agent: Agent,
        checkpoint: Checkpoint,
        adapter: ModelAdapter,
        context: Mapping[str, object],
        caller: CallerIdentity,
        cancellation: CancellationToken,
        *,
        persist: Callable[[Checkpoint], Awaitable[Checkpoint]] | None = None,
        charge: Callable[[Usage, Usage], None] | None = None,
        approval: ApprovalCallback | None = None,
        step_id: str | None = None,
        lease: OwnershipLease | None = None,
    ) -> None:
        self.agent, self.cp, self.adapter = agent, checkpoint, adapter
        self.context = MappingProxyType(dict(context))
        self.caller, self.cancellation = caller, cancellation
        self.limits = agent.limits
        self.persist, self.charge, self.step_id = persist, charge, step_id
        self.lease = lease
        self.started = time.monotonic()
        self.model_task: asyncio.Task[ModelResponse] | None = None
        from loom.tools.http import HTTPTool

        secret_env = [
            reference["env"]
            for tool in agent.tools
            if isinstance(tool, HTTPTool)
            for reference in tool.auth_env.values()
        ]
        self.events = Events(
            checkpoint.run_id,
            agent.sinks,
            root_run_id=checkpoint.root_run_id,
            sequence=checkpoint.event_sequence,
            capture_payloads=agent.config.observability.capture_payloads,
            source_paths=agent.config.observability.capture_source_paths,
            extra_keys=agent.config.observability.redact_keys,
            secrets=resolved_secrets(agent.config, secret_env),
            deadline=checkpoint.deadline,
            lease=lease,
            step_id=step_id,
        )
        if agent.config.observability.jsonl_path:
            from loom.trace import TraceSink

            self.events.sinks.append(TraceSink(agent.config, self.events.diagnostics.append))
        self.tools = {tool.spec.name: tool for tool in agent.tools}
        self.tool_runner = ToolRunner(
            tools=self.tools,
            approval=agent.approval if agent.approval is not None else approval,
            policies=agent.document_access,
            grants=agent.grants,
            lease=lease,
            on_event=self.event,
            config=agent.config,
        )
        self.tool_runner.omitted_effects.update(checkpoint.omitted_effects)

    def remaining(self) -> float:
        return (self.cp.deadline - datetime.now(UTC)).total_seconds()

    async def event(self, kind: str, metadata: dict[str, JSONValue]) -> None:
        await self.save()
        await self.events.emit(kind, metadata=metadata)

    def check(self) -> None:
        if self.cancellation.is_cancelled():
            raise ToolFailure("operation_cancelled")
        if self.remaining() <= 0:
            raise ToolFailure("deadline_exhausted")

    def usage(self, **changes: object) -> None:
        usage = self.cp.usage.model_copy(update=changes)
        if self.charge is not None:
            self.charge(self.cp.usage, usage)
        self.cp = self.cp.model_copy(update={"usage": usage})

    async def save(self) -> None:
        self.cp = self.cp.model_copy(
            update={
                "event_sequence": self.events.sequence + 1,
                "omitted_effects": tuple(sorted(self.tool_runner.omitted_effects)),
            }
        )
        storage = self.agent.storage
        if self.persist is not None:
            self.cp = await self.persist(self.cp)
        elif storage is not None:
            try:
                self.cp = await offload("internal_io", partial(storage.save, self.cp), seconds=30)
            except (ToolFailure, LoomStorageError, TimeoutError) as error:
                raise ToolFailure(
                    "storage_write_failed", details={"cause": getattr(error, "code", "timeout")}
                ) from None

    def append(self, message: Message) -> None:
        self.cp = self.cp.model_copy(
            update={
                "state": self.cp.state.model_copy(
                    update={
                        "messages": (*self.cp.state.messages, message),
                        "provenance": merge_provenance(
                            self.cp.state.provenance, message.provenance
                        ),
                    }
                ),
            }
        )

    def request(self) -> ModelRequest:
        for source in self.cp.state.provenance:
            attached = any(store.name == source.store for store in self.agent.data)
            policy = self.agent.document_access.get(source.store)
            if (not attached and policy is None) or (
                not self.adapter.capabilities.local and policy != "allow_hosted"
            ):
                raise ToolFailure("document_transfer_denied")
        mode = self.adapter.capabilities.structured_output
        if self.agent.schema is not None and not native_schema(self.agent.schema):
            mode = "prompt_json"
        instructions = self.agent.instructions
        if self.agent.schema is not None and mode == "prompt_json":
            instructions += "\nReturn only JSON conforming to this schema:\n" + canonical(
                self.agent.schema
            ).decode("utf-8")
        return ModelRequest(
            instructions=instructions,
            messages=self.cp.state.messages,
            tools=tuple(tool.spec for tool in self.tools.values()),
            output_schema=self.agent.schema,
            output_mode=mode,
            max_output_tokens=self.limits.max_output_tokens,
            timeout_seconds=max(0.001, min(self.limits.model_timeout_seconds, self.remaining())),
        )

    def context_bytes(self) -> int:
        request = self.request()
        return len(
            canonical(
                {
                    "instructions": request.instructions,
                    "messages": [json_value(m) for m in request.messages],
                    "tools": [json_value(tool) for tool in request.tools],
                    "output_schema": request.output_schema,
                }
            )
        )

    async def execute(self, resume: bool) -> RunResult:
        if self.tools and not self.adapter.capabilities.native_tool_calling:
            raise LoomConfigurationError("model_capability_unsupported")
        for store in self.agent.data:
            if not self.adapter.capabilities.local and (
                self.agent.document_access.get(store.name, "local_only") != "allow_hosted"
            ):
                raise LoomConfigurationError("document_access_denied")
            snapshot = self.cp.state.snapshots.get(store.name)
            snapshot = await offload("internal_io", partial(store.pin, snapshot), seconds=30)
            self.cp = self.cp.model_copy(
                update={
                    "state": self.cp.state.model_copy(
                        update={
                            "snapshots": self.cp.state.snapshots | {store.name: snapshot},
                        }
                    )
                }
            )
            search_tool = store.tool(snapshot)
            self.tools[search_tool.spec.name] = search_tool
        if self.agent.allowlist is not None:
            self.tools = {
                name: tool for name, tool in self.tools.items() if name in self.agent.allowlist
            }
            self.tool_runner.tools = self.tools
        if self.tools and not self.adapter.capabilities.native_tool_calling:
            raise LoomConfigurationError("model_capability_unsupported")
        if isinstance(self.agent.model, str):
            profile = self.agent.config.providers[self.agent.model]
            if profile.kind == "ollama" and (
                profile.num_ctx is None
                or profile.num_ctx
                < (self.limits.max_context_bytes + self.limits.max_output_tokens + 4096)
                or len(self.tools) > 16
            ):
                raise LoomConfigurationError("ollama_context_budget")
        if self.agent.schema is not None and self.request().output_mode == "prompt_json":
            self.cp = self.cp.model_copy(
                update={
                    "diagnostics": (
                        *self.cp.diagnostics,
                        Diagnostic(
                            code="native_schema_not_used",
                            message="Using explicit JSON instructions and local validation.",
                            origin="model",
                        ),
                    )
                }
            )
        try:
            await self.event("checkpoint.saved", {})
            await self.save()
            await self.events.emit("run.resumed" if resume else "run.started")
            while True:
                self.check()
                if self.cp.phase == "needs_resolution":
                    return await self.finish(
                        "failed", error=ToolFailure("unknown_tool_outcome", outcome_unknown=True)
                    )
                if self.cp.phase == "tools_pending":
                    await self.execute_tools(resume)
                    resume = False
                if self.context_bytes() > self.limits.max_context_bytes:
                    raise ToolFailure("context_limit")
                if (
                    len(self.cp.state.messages) > 64
                    and isinstance(self.agent.model, str)
                    and (self.agent.config.providers[self.agent.model].kind == "ollama")
                ):
                    raise ToolFailure("context_limit")
                response = await self.model_turn()
                if response.finish_reason == "refused":
                    raise ToolFailure("model_refusal")
                if response.finish_reason == "output_limit":
                    raise ToolFailure("output_truncated")
                if response.tool_calls:
                    ids = [
                        call.provider_call_id
                        for call in response.tool_calls
                        if call.provider_call_id
                    ]
                    if len(ids) != len(set(ids)):
                        raise ToolFailure("provider_protocol_error")
                    calls = tuple(
                        call.model_copy(
                            update={
                                "call_id": (
                                    f"tc_{self.cp.run_id}_"
                                    f"{self.cp.usage.logical_model_turns}_{ordinal}"
                                ),
                                "turn": self.cp.usage.logical_model_turns,
                                "ordinal": ordinal,
                            }
                        )
                        for ordinal, call in enumerate(response.tool_calls, 1)
                    )
                    self.append(
                        Message(
                            role="assistant",
                            tool_calls=calls,
                            continuation=response.continuation,
                            provenance=self.cp.state.provenance,
                        )
                    )
                    self.cp = self.cp.model_copy(
                        update={
                            "pending": tuple(PendingOperation(call=call) for call in calls),
                            "cursor": 0,
                            "phase": "tools_pending",
                            "model_response": None,
                        }
                    )
                    await self.save()
                    continue
                try:
                    value = self.validate_answer(response.output)
                except (ToolFailure, ValueError):
                    self.check()
                    if self.cp.usage.logical_model_turns >= self.limits.max_model_turns:
                        raise ToolFailure("model_turn_limit") from None
                    if self.cp.usage.corrections >= self.limits.max_corrections:
                        raise ToolFailure("output_validation_failed") from None
                    self.usage(corrections=self.cp.usage.corrections + 1)
                    self.append(
                        Message(
                            role="user",
                            content=(
                                "The final answer did not satisfy the output/evidence contract. "
                                "Search attached documents if present, then return a valid answer "
                                "using only verified retrieved citations and the unchanged "
                                "output schema."
                            ),
                            provenance=self.cp.state.provenance,
                        )
                    )
                    self.cp = self.cp.model_copy(update={"phase": "ready", "model_response": None})
                    await self.save()
                    await self.events.emit("output.correction_requested")
                    continue
                self.append(
                    Message(
                        role="assistant",
                        content=value if isinstance(value, str) else canonical(value).decode(),
                        provenance=self.cp.state.provenance,
                        continuation=response.continuation,
                    )
                )
                return await self.finish("completed", output=value)
        except asyncio.CancelledError:
            self.cancellation.cancel()
            uncertain = any(
                op.status == "started"
                and op.call.name in self.tools
                and not self.tools[op.call.name].spec.retry_safe
                for op in self.cp.pending
            )
            await self.finish(
                "cancelled",
                error=ToolFailure(
                    "unknown_tool_outcome" if uncertain else "operation_cancelled",
                    outcome_unknown=uncertain,
                ),
            )
            raise
        except (ToolFailure, ModelFailure) as error:
            code = error.code
            status: Literal["failed", "cancelled", "limit_exhausted"] = (
                "cancelled"
                if code == "operation_cancelled" or self.cancellation.is_cancelled()
                else "limit_exhausted"
                if code
                in {
                    "context_limit",
                    "deadline_exhausted",
                    "model_turn_limit",
                    "tool_attempt_limit",
                    "workflow_model_turn_limit",
                    "workflow_tool_attempt_limit",
                }
                else "failed"
            )
            failure = error if isinstance(error, ToolFailure) else ToolFailure(code)
            return await self.finish(status, error=failure, persist=code != "storage_write_failed")

    async def model_turn(self) -> ModelResponse:
        if self.cp.model_response is not None:
            return self.cp.model_response
        if self.cp.phase != "model_pending":
            if self.cp.usage.logical_model_turns >= self.limits.max_model_turns:
                raise ToolFailure("model_turn_limit")
            self.usage(logical_model_turns=self.cp.usage.logical_model_turns + 1)
            self.cp = self.cp.model_copy(update={"phase": "model_pending", "model_attempt": 0})
        retry_after = None
        for attempt in range(self.cp.model_attempt, self.limits.max_retries + 1):
            self.check()
            if attempt:
                await self.event("retry.scheduled", {"attempt": attempt + 1})
                await retry_pause(attempt, check=self.check, retry_after=retry_after)
                self.check()
            if self.model_task is not None and not self.model_task.done():
                raise ModelFailure("model_timeout")
            self.usage(
                model_requests=self.cp.usage.model_requests + 1,
                retries=self.cp.usage.retries + bool(attempt),
            )
            self.cp = self.cp.model_copy(update={"model_attempt": attempt + 1})
            await self.save()
            request = self.request()
            await self.events.emit(
                "model.started",
                payload={
                    "instructions": request.instructions,
                    "messages": [
                        json_value(message.model_dump(mode="json", exclude={"continuation"}))
                        for message in request.messages
                    ],
                },
            )
            try:
                response = await self.generate(request)
            except TimeoutError:
                failure = ModelFailure("model_timeout", transient=True)
            except ModelFailure as error:
                failure = error
            else:
                canonical(response, max_bytes=262144)
                previous = self.cp.usage
                self.usage(
                    input_tokens=(previous.input_tokens or 0) + response.input_tokens
                    if response.input_tokens is not None
                    else previous.input_tokens or None,
                    output_tokens=(previous.output_tokens or 0) + response.output_tokens
                    if response.output_tokens is not None
                    else previous.output_tokens or None,
                    complete=previous.complete
                    and response.input_tokens is not None
                    and response.output_tokens is not None,
                )
                self.cp = self.cp.model_copy(update={"phase": "ready", "model_response": response})
                await self.save()
                await self.events.emit(
                    "model.finished",
                    usage=self.cp.usage,
                    payload=json_value(response.model_dump(mode="json", exclude={"continuation"})),
                )
                return response
            self.usage(complete=False)
            if not failure.transient or attempt == self.limits.max_retries:
                raise failure
            retry_after = failure.retry_after
        raise ModelFailure("model_attempts_exhausted")

    async def generate(self, request: ModelRequest) -> ModelResponse:
        release = self.lease.retain() if self.lease is not None else lambda: None
        try:
            task = asyncio.create_task(self.adapter.generate(request))
        except BaseException:
            release()
            raise
        task.add_done_callback(lambda _: release())
        self.model_task = task
        deadline = time.monotonic() + request.timeout_seconds
        try:
            while True:
                self.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError
                done, _ = await asyncio.wait({task}, timeout=min(0.05, remaining))
                if done:
                    return task.result()
        finally:
            if not task.done():
                task.cancel()

            def consume(completed: asyncio.Task[ModelResponse]) -> None:
                if not completed.cancelled():
                    completed.exception()

            task.add_done_callback(consume)

    async def execute_tools(self, resume: bool) -> None:
        from loom.contracts import SearchResult

        for index in range(self.cp.cursor, len(self.cp.pending)):
            self.check()
            operation = self.cp.pending[index]
            tool = self.tools.get(operation.call.name)
            if operation.status in {"started", "unknown"}:
                if (
                    tool is None
                    or not tool.spec.retry_safe
                    or tool.is_active(operation.call.call_id)
                ):
                    self.cp = self.cp.model_copy(update={"phase": "needs_resolution"})
                    await self.save()
                    raise ToolFailure("unknown_tool_outcome", outcome_unknown=True)
            observation = operation.observation
            if observation is None:
                if operation.attempts >= self.limits.max_retries + 1:
                    raise ToolFailure("tool_attempts_exhausted")
                remaining_calls = len(self.cp.pending) - index
                available = self.limits.max_context_bytes - self.context_bytes() - 2048
                if available < 1024 * remaining_calls:
                    raise ToolFailure("context_limit")
                allowance = min(
                    self.limits.max_observation_bytes, available - 1024 * (remaining_calls - 1)
                )
                context = ToolContext(
                    root_run_id=self.cp.root_run_id,
                    run_id=self.cp.run_id,
                    conversation_id=self.cp.state.conversation_id,
                    step_id=self.step_id,
                    call_id=operation.call.call_id,
                    attempt=operation.attempts + 1,
                    deadline=self.cp.deadline,
                    cancellation=self.cancellation,
                    caller=self.caller,
                    context=self.context,
                    provenance=self.cp.state.provenance,
                    observation_bytes=allowance,
                )

                async def reserve(
                    index: int = index,
                    name: str = operation.call.name,
                ) -> None:
                    self.check()
                    if self.cp.usage.tool_attempts >= self.limits.max_tool_attempts:
                        raise ToolFailure("tool_attempt_limit")
                    self.usage(
                        tool_attempts=self.cp.usage.tool_attempts + 1,
                        retries=self.cp.usage.retries + bool(self.cp.pending[index].attempts),
                    )
                    pending = list(self.cp.pending)
                    pending[index] = pending[index].model_copy(
                        update={
                            "status": "started",
                            "attempts": pending[index].attempts + 1,
                        }
                    )
                    self.cp = self.cp.model_copy(update={"pending": tuple(pending)})
                    await self.save()
                    await self.events.emit(
                        "tool.started", metadata={"call_id": self.cp.pending[index].call.call_id}
                    )

                while True:
                    observation = await self.tool_runner.call(
                        operation.call,
                        context,
                        argument_bytes=self.limits.max_tool_arguments_bytes,
                        approval_seconds=min(
                            self.limits.approval_timeout_seconds, self.remaining()
                        ),
                        reserve_attempt=reserve,
                        tool_seconds=self.limits.tool_timeout_seconds,
                    )
                    attempts = self.cp.pending[index].attempts
                    if (
                        observation.error is None
                        or not observation.error.retryable
                        or tool is None
                        or not tool.spec.retry_safe
                        or tool.is_active(operation.call.call_id)
                    ):
                        break
                    if attempts > self.limits.max_retries:
                        raise ToolFailure(observation.error.code)
                    pending = list(self.cp.pending)
                    pending[index] = pending[index].model_copy(update={"status": "pending"})
                    self.cp = self.cp.model_copy(update={"pending": tuple(pending)})
                    await self.event("retry.scheduled", {"call_id": operation.call.call_id})
                    await retry_pause(
                        attempts,
                        check=self.check,
                        retry_after=observation.error.details.get("retry_after_seconds"),
                    )
                    self.check()
                    context = context.model_copy(update={"attempt": attempts + 1})
            from loom.documents import SearchTool

            if isinstance(tool, SearchTool) and observation.error is None:
                result = SearchResult.model_validate_json(canonical(observation.output))
                self.cp = self.cp.model_copy(
                    update={
                        "searches": (*self.cp.searches, result),
                        "diagnostics": (
                            *self.cp.diagnostics,
                            *(
                                item
                                for item in result.diagnostics
                                if item not in self.cp.diagnostics
                            ),
                        ),
                    }
                )
                await self.event(
                    "retrieval.finished",
                    {"snapshot_id": result.snapshot_id, "chunks": len(result.chunks)},
                )
            self.append(
                Message(
                    role="tool",
                    observation=observation,
                    provenance=merge_provenance(self.cp.state.provenance, observation.provenance),
                )
            )
            pending = list(self.cp.pending)
            pending[index] = pending[index].model_copy(
                update={
                    "status": "succeeded"
                    if observation.execution_status == "succeeded"
                    else "failed",
                    "observation": observation,
                }
            )
            self.cp = self.cp.model_copy(update={"pending": tuple(pending), "cursor": index + 1})
            await self.save()
            await self.events.emit(
                "tool.finished",
                metadata={"call_id": operation.call.call_id},
                payload=json_value(observation),
            )
        self.cp = self.cp.model_copy(update={"phase": "ready", "pending": (), "cursor": 0})

    def validate_answer(self, value: JSONValue) -> JSONValue:
        if self.agent.schema is not None:
            if isinstance(value, str):
                value = decode_json(value)
            value = validate_output(value, self.agent.schema)
        elif not isinstance(value, str):
            raise ToolFailure("output_validation_failed")
        if self.agent.data:
            from loom.documents.citations import validate_citations

            value = validate_citations(value, self.cp.searches)
        return value

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
                "version": self.cp.base_conversation_version + (1 if status == "completed" else 0),
                "continuation_eligible": status == "completed",
            }
        )
        state = ConversationState.model_validate_json(canonical(state))
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
            error=run_error(error, "runtime") if error else None,
            usage=usage,
            snapshots=state.snapshots,
            state=state,
            diagnostics=(
                *self.cp.diagnostics,
                *self.tool_runner.diagnostics,
                *self.events.diagnostics,
            ),
        )
        phase = "needs_resolution" if error and error.outcome_unknown else status
        self.cp = self.cp.model_copy(
            update={
                "phase": phase,
                "result": None if phase == "needs_resolution" else result,
                "state": state,
                "usage": usage,
            }
        )
        if persist:
            if error is not None:
                await self.event("error", {"code": error.code})
            await self.save()
            await self.events.emit("run.finished", metadata={"status": status})
            diagnostics = (
                *self.cp.diagnostics,
                *self.tool_runner.diagnostics,
                *self.events.diagnostics,
            )
            if result.diagnostics != diagnostics:
                result = result.model_copy(update={"diagnostics": diagnostics})
                self.cp = self.cp.model_copy(
                    update={"result": None if phase == "needs_resolution" else result}
                )
                await self.save()
        return result
