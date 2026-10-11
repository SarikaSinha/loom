from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import Field

from loom.contracts import (
    ConversationState,
    Diagnostic,
    JSONValue,
    LoomStorageError,
    Record,
    RunResult,
    SearchResult,
    ToolCall,
    ToolObservation,
    Usage,
)
from loom.providers.base import ModelResponse


def identifier(value: str, *, child: bool = False) -> str:
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        raise LoomStorageError("invalid_identifier") from None
    if str(parsed) != value or parsed.version not in ((4, 5) if child else (4,)):
        raise LoomStorageError("invalid_identifier")
    return value


class PendingOperation(Record):
    call: ToolCall
    status: Literal["pending", "approved", "started", "succeeded", "failed", "unknown"] = "pending"
    attempts: int = Field(default=0, ge=0)
    observation: ToolObservation | None = None
    operator_note: str | None = None


class Checkpoint(Record):
    schema_version: Literal[1] = 1
    kind: Literal["agent", "workflow"] = "agent"
    root_run_id: str
    run_id: str
    revision: int = Field(default=0, ge=0)
    phase: Literal[
        "ready",
        "model_pending",
        "tools_pending",
        "finalizing",
        "needs_resolution",
        "completed",
        "failed",
        "cancelled",
        "limit_exhausted",
    ] = "ready"
    configuration: dict[str, JSONValue]
    configuration_fingerprint: str
    state: ConversationState
    base_conversation_version: int = Field(default=0, ge=0)
    usage: Usage = Field(default_factory=Usage)
    deadline: datetime
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model_attempt: int = Field(default=0, ge=0)
    pending: tuple[PendingOperation, ...] = ()
    cursor: int = Field(default=0, ge=0)
    step_outputs: dict[str, JSONValue] = Field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    event_sequence: int = Field(default=0, ge=0)
    result: RunResult | None = None
    searches: tuple[SearchResult, ...] = ()
    model_response: ModelResponse | None = None
    omitted_effects: tuple[tuple[str, str, str], ...] = ()
    workflow_input: JSONValue = None
    active_step: str | None = None
    last_step: str | None = None
    activated_steps: tuple[str, ...] = ()
    child: Checkpoint | None = None
    step_deadline: datetime | None = None


class ConversationHead(Record):
    head_run_id: str
    state: ConversationState


class Image(Record):
    schema_version: Literal[1] = 1
    revision: int = Field(default=0, ge=0)
    runs: dict[str, Checkpoint] = Field(default_factory=dict)
    conversations: dict[str, ConversationHead] = Field(default_factory=dict)
