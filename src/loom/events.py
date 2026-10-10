from __future__ import annotations

import inspect
import os
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Literal

from pydantic import Field

from loom.config import Config
from loom.contracts import Diagnostic, JSONValue, Record, Usage
from loom.runtime import OwnershipLease, offload, wait_async

SECRET_KEYS = frozenset(
    """
authorization proxy_authorization api_key apikey x_api_key x_goog_api_key
token access_token refresh_token id_token session_token bearer_token x_auth_token
password passwd secret client_secret private_key cookie set_cookie
aws_secret_access_key aws_session_token
""".split()
)
SOURCE_KEYS = frozenset({"path", "source_path", "filename", "file_path", "source_name"})


def resolved_secrets(config: Config, extra_env: Sequence[str] = ()) -> tuple[str, ...]:
    names = {"OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"}
    names.update(extra_env)
    names.update(
        profile.api_key_env for profile in config.providers.values() if profile.api_key_env
    )
    if config.encryption is not None:
        names.update(config.encryption.keys.values())
    return tuple(
        sorted({value for name in names if (value := os.environ.get(name))}, key=len, reverse=True)
    )


def normalize_key(key: str) -> str:
    key = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key.strip())
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"_+", "_", re.sub(r"[\s-]+", "_", key)).lower()


def redact(
    value: JSONValue,
    *,
    extra_keys: Sequence[str] = (),
    secrets: Sequence[str] = (),
    source_paths: bool = False,
) -> JSONValue:
    keys = SECRET_KEYS | {normalize_key(key) for key in extra_keys}

    def visit(item: JSONValue) -> JSONValue:
        if isinstance(item, dict):
            return {
                key: "[REDACTED]" if normalize_key(key) in keys else visit(child)
                for key, child in item.items()
                if source_paths or normalize_key(key) not in SOURCE_KEYS
            }
        if isinstance(item, list):
            return [visit(child) for child in item]
        if isinstance(item, str):
            for secret in secrets:
                if secret:
                    item = item.replace(secret, "[REDACTED]")
        return item

    return visit(value)


class Event(Record):
    schema_version: Literal[1] = 1
    event_id: str
    root_run_id: str
    run_id: str
    step_id: str | None = None
    sequence: int = Field(ge=1)
    timestamp: str
    event_type: str
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    payload: JSONValue = None
    duration_ms: float | None = Field(default=None, ge=0)
    usage: Usage | None = None


EventSink = Callable[[Event], None | Awaitable[None]]


class Events:
    def __init__(
        self,
        run_id: str,
        sinks: Sequence[EventSink] = (),
        *,
        root_run_id: str | None = None,
        sequence: int = 0,
        capture_payloads: bool = False,
        source_paths: bool = False,
        extra_keys: Sequence[str] = (),
        secrets: Sequence[str] = (),
        lease: OwnershipLease | None = None,
        deadline: datetime | None = None,
        step_id: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.root_run_id = root_run_id or run_id
        self.sequence = sequence
        self.sinks = list(sinks)
        self.diagnostics: list[Diagnostic] = []
        self.capture = capture_payloads
        self.source_paths = source_paths
        self.extra_keys = extra_keys
        self.secrets = secrets
        self.lease, self.deadline, self.step_id = lease, deadline, step_id

    async def emit(
        self,
        kind: str,
        *,
        metadata: dict[str, JSONValue] | None = None,
        payload: JSONValue = None,
        duration_ms: float | None = None,
        usage: Usage | None = None,
    ) -> None:
        self.sequence += 1
        safe_meta = redact(
            metadata or {},
            extra_keys=self.extra_keys,
            secrets=self.secrets,
            source_paths=self.source_paths,
        )
        assert isinstance(safe_meta, dict)
        step_id = safe_meta.get("step_id", self.step_id)
        event = Event(
            event_id=f"{self.run_id}:{self.sequence}",
            root_run_id=self.root_run_id,
            run_id=self.run_id,
            step_id=step_id if isinstance(step_id, str) else None,
            sequence=self.sequence,
            timestamp=datetime.now(UTC).isoformat(),
            event_type=kind,
            metadata=safe_meta,
            payload=redact(
                payload,
                extra_keys=self.extra_keys,
                secrets=self.secrets,
                source_paths=self.source_paths,
            )
            if self.capture
            else None,
            duration_ms=duration_ms,
            usage=usage,
        )
        for sink in tuple(self.sinks):
            try:
                seconds = (
                    min(
                        0.5,
                        (self.deadline - datetime.now(UTC)).total_seconds(),
                    )
                    if self.deadline is not None
                    else 0.5
                )
                if seconds <= 0:
                    raise TimeoutError
                if inspect.iscoroutinefunction(sink) or inspect.iscoroutinefunction(
                    type(sink).__call__
                ):
                    result = sink(event)
                    if not inspect.isawaitable(result):
                        raise TypeError("asynchronous event sink returned a non-awaitable")
                    await wait_async(
                        result,
                        seconds=seconds,
                        finished=self.lease.retain() if self.lease is not None else None,
                    )
                else:
                    result = await (self.lease.offload if self.lease is not None else offload)(
                        "user_code",
                        partial(sink, event),
                        seconds=seconds,
                    )
                    if inspect.isawaitable(result):
                        await wait_async(
                            result,
                            seconds=seconds,
                            finished=self.lease.retain() if self.lease is not None else None,
                        )
            except Exception:
                self.sinks.remove(sink)
                self.diagnostics.append(
                    Diagnostic(
                        code="event_sink_failed",
                        message="An event sink failed and was disabled.",
                        origin="events",
                    )
                )
                print("loom: event_sink_failed", file=sys.stderr)
