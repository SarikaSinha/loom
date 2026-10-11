from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Literal, Never, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PrivateAttr,
    TypeAdapter,
    ValidationError,
    model_validator,
)

JSONValue = JsonValue
Schema = dict[str, JSONValue]
NAME = r"^[a-z][a-z0-9_]{0,47}$"
TOOL_NAME = r"^[a-z][a-z0-9_]{0,63}$"
PositiveSeconds = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class LoomError(Exception):
    def __init__(self, code: str, message: str | None = None, **details: JSONValue) -> None:
        self.code = code
        self.message = message or code.replace("_", " ")
        self.details = details
        super().__init__(self.message)


class LoomConfigurationError(LoomError):
    pass


class LoomStorageError(LoomError):
    pass


class ToolFailure(LoomError):
    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        recoverable: bool = False,
        transient: bool = False,
        outcome_unknown: bool = False,
        details: dict[str, JSONValue] | None = None,
    ) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", code):
            raise LoomConfigurationError("invalid_error_code")
        self.recoverable = recoverable
        self.transient = transient
        self.outcome_unknown = outcome_unknown
        super().__init__(code, message, **(details or {}))


class OutputTooLarge(ToolFailure):
    def __init__(self) -> None:
        super().__init__("observation_too_large", recoverable=True)


def _immutable(*args: object, **kwargs: object) -> Never:
    raise TypeError("Loom records are immutable")


class _FrozenDict(dict[str, object]):
    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenDict:
        return self


class _FrozenList(list[object]):
    __setitem__ = __delitem__ = __iadd__ = __imul__ = _immutable
    append = clear = extend = insert = pop = remove = reverse = sort = _immutable

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenList:
        return self


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return _FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze(item) for item in value)
    return value


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    @model_validator(mode="after")
    def freeze_fields(self) -> Self:
        for name in type(self).model_fields:
            if name != "context":
                object.__setattr__(self, name, _freeze(getattr(self, name)))
        return self


JSON_ADAPTER: TypeAdapter[JSONValue] = TypeAdapter(
    JSONValue, config=ConfigDict(strict=True, allow_inf_nan=False)
)


def json_value(value: object) -> JSONValue:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return JSON_ADAPTER.validate_python(value)


def canonical(value: object, *, max_bytes: int | None = None) -> bytes:
    value = json_value(value)
    encoder = json.JSONEncoder(
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    result = bytearray()
    for part in encoder.iterencode(value):
        # iterencode can yield a single large string; encode it in bounded pieces.
        for offset in range(0, len(part), 4096):
            piece = part[offset : offset + 4096].encode("utf-8")
            if max_bytes is not None and len(result) + len(piece) > max_bytes:
                raise ToolFailure("payload_too_large", recoverable=True)
            result.extend(piece)
    return bytes(result)


def fingerprint(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def decode_json(data: bytes | str) -> JSONValue:
    return JSON_ADAPTER.validate_json(data)


class Limits(Record):
    max_model_turns: int = Field(default=10, gt=0)
    max_tool_attempts: int = Field(default=20, gt=0)
    deadline_seconds: PositiveSeconds = 120.0
    model_timeout_seconds: PositiveSeconds = 30.0
    tool_timeout_seconds: PositiveSeconds = 30.0
    approval_timeout_seconds: PositiveSeconds = 60.0
    max_retries: int = Field(default=2, ge=0, le=2)
    max_corrections: int = Field(default=1, ge=0, le=1)
    max_context_bytes: int = Field(default=65536, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    max_observation_bytes: int = Field(default=8192, ge=1024, le=1048576)
    max_tool_arguments_bytes: int = Field(default=8192, gt=0, le=1048576)
    max_agent_calls: int = Field(default=20, gt=0)
    max_agent_depth: int = Field(default=5, gt=0)


class WorkflowLimits(Record):
    max_step_activations: int | None = Field(default=None, gt=0)
    max_model_turns: int | None = Field(default=None, ge=0)
    max_tool_attempts: int | None = Field(default=None, ge=0)
    deadline_seconds: PositiveSeconds | None = None


class CallerIdentity(Record):
    subject: str = Field(default="anonymous", min_length=1)
    roles: tuple[str, ...] = ()
    attributes: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_roles(self) -> Self:
        if len(set(self.roles)) != len(self.roles):
            raise ValueError("roles must be unique")
        return self


class CancellationToken(Record):
    _event: threading.Event = PrivateAttr(default_factory=threading.Event)

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class Provenance(Record):
    store: str
    snapshot_id: str
    document_ids: tuple[str, ...] = ()


class ToolCall(Record):
    call_id: str
    provider_call_id: str | None = None
    name: str
    arguments: dict[str, JSONValue]
    turn: int = Field(default=0, ge=0)
    ordinal: int = Field(default=0, ge=0)


class RunError(Record):
    code: str
    message: str
    origin: str
    retryable: bool = False
    outcome_unknown: bool = False
    details: dict[str, JSONValue] = Field(default_factory=dict)


class Omission(Record):
    byte_count: int | None
    sha256: str | None
    validation: Literal["validated", "not_performed"]


class ToolObservation(Record):
    call_id: str
    execution_status: Literal["not_started", "succeeded", "failed", "unknown"]
    output: JSONValue = None
    error: RunError | None = None
    omission: Omission | None = None
    provenance: tuple[Provenance, ...] = ()


class Message(Record):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    observation: ToolObservation | None = None
    provenance: tuple[Provenance, ...] = ()
    continuation: dict[str, JSONValue] = Field(default_factory=dict)


class ConversationState(Record):
    schema_version: Literal[1] = 1
    conversation_id: str
    version: int = Field(default=0, ge=0)
    caller_subject: str = "anonymous"
    messages: tuple[Message, ...] = ()
    application_state: dict[str, JSONValue] = Field(default_factory=dict)
    snapshots: dict[str, str] = Field(default_factory=dict)
    provenance: tuple[Provenance, ...] = ()
    continuation_eligible: bool = True


class Usage(Record):
    model_requests: int = 0
    logical_model_turns: int = 0
    tool_attempts: int = 0
    retries: int = 0
    corrections: int = 0
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    complete: bool = True
    elapsed_ms: float = 0.0
    agent_calls: int = 0


class Diagnostic(Record):
    code: str
    message: str
    origin: str
    severity: Literal["info", "warning", "error"] = "warning"
    details: dict[str, JSONValue] = Field(default_factory=dict)


class RunResult(Record):
    schema_version: Literal[1] = 1
    run_id: str
    conversation_id: str
    status: Literal["completed", "failed", "cancelled", "limit_exhausted"]
    output: JSONValue = None
    error: RunError | None = None
    usage: Usage
    snapshots: dict[str, str] = Field(default_factory=dict)
    diagnostics: tuple[Diagnostic, ...] = ()
    state: ConversationState

    @model_validator(mode="after")
    def valid_outcome(self) -> Self:
        if self.status == "completed":
            if self.error is not None:
                raise ValueError("completed results cannot contain an error")
        elif self.output is not None or self.error is None:
            raise ValueError("non-completed results require an error and null output")
        return self

    def to_dict(self, include_state: bool = False) -> dict[str, JSONValue]:
        data = json_value(
            self.model_dump(mode="json", exclude=None if include_state else {"state"})
        )
        assert isinstance(data, dict)
        return data

    def to_json(self, include_state: bool = False, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(include_state), ensure_ascii=False, allow_nan=False, indent=indent
        )

    def __str__(self) -> str:
        return self.to_json()


class ToolSpec(Record):
    name: str = Field(pattern=TOOL_NAME)
    description: str = Field(min_length=1)
    version: str = Field(default="1", min_length=1)
    argument_schema: Schema
    output_schema: Schema
    sensitive: bool = True
    retry_safe: bool = False
    timeout_seconds: PositiveSeconds | None = None
    egress: Literal["none", "fixed_network", "unknown"] = "unknown"
    egress_origins: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_egress(self) -> Self:
        if bool(self.egress_origins) != (self.egress == "fixed_network"):
            raise ValueError("only fixed_network tools require egress origins")
        return self


class AgentSpec(Record):
    name: str = Field(pattern=NAME)
    version: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    model: dict[str, JSONValue]
    tools: tuple[ToolSpec, ...]
    output_schema: Schema | None
    limits: Limits
    document_access: dict[str, Literal["local_only", "allow_hosted"]]
    tool_document_access: dict[str, tuple[str, ...]]
    allowlist: tuple[str, ...] | None
    binding_paths: dict[str, JSONValue]


class ToolContext(Record):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, arbitrary_types_allowed=True
    )
    root_run_id: str
    run_id: str
    conversation_id: str
    step_id: str | None = None
    call_id: str
    attempt: int = Field(ge=1)
    deadline: datetime
    cancellation: CancellationToken
    caller: CallerIdentity
    context: Mapping[str, object] = Field(default_factory=dict)
    provenance: tuple[Provenance, ...] = ()
    observation_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def immutable_context(self) -> Self:
        if self.deadline.tzinfo is None or self.deadline.utcoffset() is None:
            raise ValueError("tool deadline must have a UTC offset")
        object.__setattr__(self, "context", MappingProxyType(dict(self.context)))
        return self


class ApprovalRequest(Record):
    tool_name: str
    tool_version: str
    call_id: str
    arguments: dict[str, JSONValue]
    caller: CallerIdentity
    egress_origins: tuple[str, ...]
    provenance: tuple[Provenance, ...]
    contains_document_content: bool


class ApprovalDecision(Record):
    approved: bool
    reason: str = ""

    @model_validator(mode="after")
    def denial_reason(self) -> Self:
        if not self.approved and not self.reason.strip():
            raise ValueError("denial requires a reason")
        return self


class PDFLocation(Record):
    kind: Literal["pdf"] = "pdf"
    page: int = Field(ge=1)


class MarkdownLocation(Record):
    kind: Literal["markdown"] = "markdown"
    section: str | None
    line_start: int = Field(ge=1)


class DOCXLocation(Record):
    kind: Literal["docx"] = "docx"
    section: str | None
    block: int = Field(ge=1)


Location = Annotated[PDFLocation | MarkdownLocation | DOCXLocation, Field(discriminator="kind")]


class Citation(Record):
    store: str
    snapshot_id: str
    chunk_id: Digest
    path: str
    location: Location
    excerpt: str = Field(min_length=1, max_length=1200)


class DocumentAnswer(Record):
    status: Literal["answered", "abstained", "conflict"]
    answer: str = Field(min_length=1)
    sources: list[Citation]

    @model_validator(mode="after")
    def evidence_shape(self) -> Self:
        if self.status == "abstained" and self.sources:
            raise ValueError("abstention cannot contain sources")
        if self.status == "answered" and not self.sources:
            raise ValueError("an answer requires sources")
        if self.status == "conflict":
            locations = {(s.path, canonical(s.location)) for s in self.sources}
            if len(locations) < 2:
                raise ValueError("conflict requires distinct source locations")
        return self


class SearchChunk(Record):
    chunk_id: Digest
    document_id: Digest
    path: str
    location: Location
    text: str
    score: float


class SearchResult(Record):
    store: str
    snapshot_id: str
    query: str
    chunks: tuple[SearchChunk, ...]
    clipped: bool = False
    diagnostics: tuple[Diagnostic, ...] = ()


def finite_seconds(value: float) -> float:
    if not math.isfinite(value) or value <= 0:
        raise LoomConfigurationError("invalid_timeout")
    return value


def schema_for(value: type[BaseModel] | Schema | str) -> Schema | None:
    if value == "text":
        return None
    if value == "document_qa":
        value = DocumentAnswer
    if isinstance(value, type) and issubclass(value, BaseModel):
        result = json_value(value.model_json_schema())
    elif isinstance(value, dict):
        result = json_value(value)
    else:
        raise LoomConfigurationError("invalid_output_schema")
    if not isinstance(result, dict):
        raise LoomConfigurationError("invalid_output_schema")
    validate_schema(result)
    return result


def validate_schema(schema: Schema) -> None:
    from jsonschema import Draft202012Validator, SchemaError
    from referencing import Registry, Resource
    from referencing.exceptions import Unresolvable
    from referencing.jsonschema import DRAFT202012

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError:
        raise LoomConfigurationError("invalid_schema") from None
    resource = Resource.from_contents(schema, default_specification=DRAFT202012)
    resolver = Registry().with_resource("", resource).crawl().resolver()
    pending = [resource]
    while pending:
        current = pending.pop()
        value = current.contents
        if isinstance(value, dict):
            for key in ("$ref", "$dynamicRef", "$recursiveRef", "$id"):
                item = value.get(key)
                if item is None:
                    continue
                if not isinstance(item, str) or not item.startswith("#"):
                    raise LoomConfigurationError("external_schema_reference")
                if key != "$id":
                    try:
                        resolver.lookup(item)
                    except Unresolvable:
                        raise LoomConfigurationError("invalid_schema_reference") from None
        pending.extend(current.subresources())


def validate_output(value: object, schema: Schema) -> JSONValue:
    from jsonschema import Draft202012Validator
    from referencing import Registry

    try:
        result = json_value(value)
    except ValidationError:
        raise ToolFailure("invalid_json_output") from None
    if (
        next(Draft202012Validator(schema, registry=Registry()).iter_errors(result), None)
        is not None
    ):
        raise ToolFailure("schema_validation_failed")
    return result
