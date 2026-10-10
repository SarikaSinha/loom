from __future__ import annotations

import base64
import os
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import Field

from loom.config import Config
from loom.contracts import Diagnostic, LoomStorageError, Record, canonical, decode_json
from loom.events import Event
from loom.runtime import offload
from loom.security.crypto import Crypto, resource_id
from loom.security.files import (
    check_permissions,
    file_lock,
    private_open,
    read_private,
    secure_directory,
    sync_directory,
)
from loom.security.maintenance import Managed
from loom.storage import identifier

SEGMENT_BYTES = 16 * 1024**2
EVENT_BYTES = 1024**2


class Segment(Record):
    number: int = Field(ge=1)
    uuid: str
    closed: bool = False


class Manifest(Record):
    schema_version: int = 1
    revision: int = 0
    next_segment_number: int = 1
    current_segment_uuid: str | None = None
    segments: tuple[Segment, ...] = ()
    pending_delete: Segment | None = None


class TraceSink:
    def __init__(
        self,
        config: Config,
        on_diagnostic: Callable[[Diagnostic], None] | None = None,
    ) -> None:
        prefix = config.observability.jsonl_path
        if not prefix.endswith(".enc.jsonl"):
            raise LoomStorageError("invalid_trace_prefix")
        self.config, self.crypto = config, Crypto(config)
        self.prefix = config.path(prefix)
        self.directory = self.prefix.with_name(self.prefix.name + ".segments")
        self.manifest_path = self.directory / "manifest.enc"
        self.intent_path = self.directory / "rollover-intent.enc"
        self.marker = self.prefix.with_name(self.prefix.name + ".maintenance.enc")
        self.append_lock = self.prefix.with_name(self.prefix.name + ".append.lock")
        self.on_diagnostic = on_diagnostic

    def path(self, segment: Segment) -> Path:
        identifier(segment.uuid)
        return self.directory / f"{segment.number}-{segment.uuid}.enc.jsonl"

    @contextmanager
    def locked(self) -> Iterator[None]:
        with Managed(self.config, self.prefix, "trace_manifest").shared():
            with file_lock(self.append_lock):
                secure_directory(self.directory)
                yield

    def manifest(self) -> Manifest:
        if not self.manifest_path.exists():
            return Manifest()
        header, plaintext = self.crypto.read(self.manifest_path, "trace_manifest", 65536)
        try:
            manifest = Manifest.model_validate_json(plaintext)
            numbers = [segment.number for segment in manifest.segments]
            identifiers = [identifier(segment.uuid) for segment in manifest.segments]
            if (
                manifest.schema_version != 1
                or manifest.revision != header.revision
                or not 1 <= len(numbers) <= 5
                or numbers != sorted(set(numbers))
                or len(set(identifiers)) != len(identifiers)
                or manifest.current_segment_uuid != identifiers[-1]
                or manifest.segments[-1].closed
                or any(not s.closed for s in manifest.segments[:-1])
                or manifest.next_segment_number <= numbers[-1]
            ):
                raise ValueError
            return manifest
        except ValueError:
            raise LoomStorageError("trace_corrupt") from None

    def publish(self, manifest: Manifest) -> None:
        self.crypto.write(
            self.manifest_path,
            "trace_manifest",
            canonical(manifest, max_bytes=65536),
            manifest.revision,
        )

    def recover(self, manifest: Manifest) -> Manifest:
        if self.intent_path.exists():
            _, intent = self.crypto.read_json(self.intent_path, "trace_intent", 65536)
            if (
                not isinstance(intent, dict)
                or set(intent)
                != {
                    "schema_version",
                    "revision",
                    "base_manifest_revision",
                    "new_segment_number",
                    "new_segment_uuid",
                }
                or intent["schema_version"] != 1
            ):
                raise LoomStorageError("trace_corrupt")
            segment = Segment.model_validate(
                {
                    "number": intent["new_segment_number"],
                    "uuid": intent["new_segment_uuid"],
                }
            )
            if manifest.revision == intent["base_manifest_revision"]:
                path = self.path(segment)
                if path.exists():
                    check_permissions(path)
                    path.unlink()
            elif not any(
                s.uuid == segment.uuid and s.number == segment.number for s in manifest.segments
            ):
                raise LoomStorageError("trace_corrupt")
            self.intent_path.unlink()
            sync_directory(self.directory)
        if manifest.pending_delete is not None:
            path = self.path(manifest.pending_delete)
            if path.exists():
                check_permissions(path)
                path.unlink()
                sync_directory(self.directory)
            if self.on_diagnostic is not None:
                self.on_diagnostic(
                    Diagnostic(
                        code="trace_retention_applied",
                        message="The oldest closed trace segment expired.",
                        origin="trace",
                        details={"segment_id": manifest.pending_delete.uuid},
                    )
                )
            manifest = manifest.model_copy(
                update={
                    "pending_delete": None,
                    "revision": manifest.revision + 1,
                }
            )
            self.publish(manifest)
        return manifest

    def rollover(self, manifest: Manifest) -> Manifest:
        segment = Segment(number=manifest.next_segment_number, uuid=str(uuid.uuid4()))
        self.crypto.write(
            self.intent_path,
            "trace_intent",
            canonical(
                {
                    "schema_version": 1,
                    "revision": manifest.revision + 1,
                    "base_manifest_revision": manifest.revision,
                    "new_segment_number": segment.number,
                    "new_segment_uuid": segment.uuid,
                }
            ),
            manifest.revision + 1,
        )
        fd = private_open(self.path(segment), exclusive=True)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        segments = [old.model_copy(update={"closed": True}) for old in manifest.segments]
        segments.append(segment)
        pending = segments.pop(0) if len(segments) > 5 else None
        updated = Manifest(
            revision=manifest.revision + 1,
            next_segment_number=segment.number + 1,
            current_segment_uuid=segment.uuid,
            segments=tuple(segments),
            pending_delete=pending,
        )
        self.publish(updated)
        return self.recover(updated)

    def line(self, event: Event, segment: Segment) -> bytes:
        if event.event_id != f"{event.run_id}:{event.sequence}":
            raise LoomStorageError("trace_identity_invalid")
        identifier(event.run_id, child=True)
        envelope = self.crypto.seal(
            canonical(event, max_bytes=EVENT_BYTES),
            kind="trace",
            identity=resource_id(self.path(segment), "trace"),
            revision=event.sequence,
            run_id=event.run_id,
            event_id=event.event_id,
        )
        return canonical({"envelope": base64.b64encode(envelope).decode("ascii")}) + b"\n"

    def append(self, event: Event) -> None:
        with self.locked():
            manifest = self.recover(self.manifest())
            if not manifest.segments:
                manifest = self.rollover(manifest)
            segment = manifest.segments[-1]
            encoded = self.line(event, segment)
            if len(encoded) > SEGMENT_BYTES:
                raise LoomStorageError("trace_event_too_large")
            if self.path(segment).stat().st_size + len(encoded) > SEGMENT_BYTES:
                manifest = self.rollover(manifest)
                segment = manifest.segments[-1]
                encoded = self.line(event, segment)
            fd = private_open(self.path(segment))
            with os.fdopen(fd, "ab") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())

    async def __call__(self, event: Event) -> None:
        await offload("internal_io", lambda: self.append(event), seconds=0.5)

    def read(self, run_id: str | None = None) -> Iterator[Event]:
        if run_id is not None:
            identifier(run_id, child=True)
        with self.locked():
            segments = self.recover(self.manifest()).segments
        for segment in segments:
            with self.locked():
                try:
                    data = read_private(self.path(segment), SEGMENT_BYTES)
                except FileNotFoundError:
                    raise LoomStorageError("trace_segment_expired") from None
            if data and not data.endswith(b"\n"):
                raise LoomStorageError("trace_corrupt")
            for line in data.splitlines():
                try:
                    value = decode_json(line)
                    if not isinstance(value, dict) or set(value) != {"envelope"}:
                        raise ValueError
                    encoded = value["envelope"]
                    if not isinstance(encoded, str):
                        raise ValueError
                    header, plaintext = self.crypto.open(
                        base64.b64decode(encoded, validate=True),
                        kind="trace",
                        identity=resource_id(self.path(segment), "trace"),
                        maximum=EVENT_BYTES,
                    )
                    event = Event.model_validate_json(plaintext)
                    if (
                        header.run_id != event.run_id
                        or header.event_id != event.event_id
                        or header.revision != event.sequence
                        or event.event_id != f"{event.run_id}:{event.sequence}"
                    ):
                        raise ValueError
                except ValueError:
                    raise LoomStorageError("trace_corrupt") from None
                if run_id is None or event.run_id == run_id:
                    yield event
