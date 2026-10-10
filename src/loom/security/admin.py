from __future__ import annotations

import base64
import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

from loom.config import Config
from loom.contracts import (
    JSONValue,
    LoomStorageError,
    canonical,
    decode_json,
    fingerprint,
    json_value,
)
from loom.events import Event
from loom.security.crypto import OVERHEAD, Crypto, Header, ResourceType, resource_id
from loom.security.files import (
    atomic_write,
    canonical_path,
    check_permissions,
    file_lock,
    read_private,
    reject_links,
)
from loom.security.maintenance import Managed, configured
from loom.storage import Checkpoint, JSONStorage, SQLiteStorage, identifier
from loom.trace import EVENT_BYTES, SEGMENT_BYTES, Segment, TraceSink

MIB = 1024**2


@dataclass(frozen=True)
class Artifact:
    path: Path
    kind: ResourceType
    maximum: int


@contextmanager
def locked(config: Config, *, exclusive: bool) -> Iterator[dict[str, Managed]]:
    roots = configured(config)
    with ExitStack() as stack:
        for root in sorted(roots.values(), key=lambda root: canonical_path(root.lock)):
            stack.enter_context(file_lock(root.lock, shared=not exclusive, blocking=False))
            if root.marker.exists():
                marker = root.read()
                if exclusive and marker.status in {"pending", "aborted"}:
                    raise LoomStorageError("artifact_maintenance_required")
            elif root.initialized():
                raise LoomStorageError("maintenance_marker_missing")
        yield roots


def inventory(config: Config, *, journals: bool = True) -> list[Artifact]:
    artifacts: list[Artifact] = []
    for role, root in configured(config).items():
        if root.marker.exists():
            root.read()
            artifacts.append(Artifact(root.marker, "maintenance", 65536))
        elif root.initialized():
            raise LoomStorageError("maintenance_marker_missing")
        if role == "storage":
            if root.path.exists():
                maximum = 16 * MIB if root.kind == "storage_json" else 128 * MIB
                artifacts.append(Artifact(root.path, root.kind, maximum))
        elif role.startswith("store:"):
            current = root.path / "CURRENT.enc"
            if current.exists():
                artifacts.append(Artifact(current, "index_current", 65536))
            snapshots = root.path / "snapshots"
            if snapshots.exists():
                check_permissions(snapshots, directory=True)
                for directory in sorted(snapshots.iterdir()):
                    reject_links(directory)
                    if directory.name.startswith(".building-"):
                        raise LoomStorageError("index_generation_incomplete")
                    identifier(directory.name)
                    check_permissions(directory, directory=True)
                    expected: dict[str, tuple[ResourceType, int]] = {
                        "index.sqlite.enc": ("index_sqlite", 512 * MIB),
                        "manifest.json.enc": ("index_manifest", 16 * MIB),
                        "ingestion-report.json.enc": ("index_report", 16 * MIB),
                    }
                    if {path.name for path in directory.iterdir()} != set(expected):
                        raise LoomStorageError("index_generation_incomplete")
                    for name, (kind, maximum) in expected.items():
                        artifacts.append(Artifact(directory / name, kind, maximum))
        elif role == "trace":
            trace = TraceSink(config)
            manifest = trace.manifest()
            known: set[Path] = set()
            if trace.manifest_path.exists():
                artifacts.append(Artifact(trace.manifest_path, "trace_manifest", 65536))
                known.add(trace.manifest_path)
            for segment in manifest.segments:
                path = trace.path(segment)
                if not path.exists():
                    raise LoomStorageError("trace_corrupt")
                artifacts.append(Artifact(path, "trace", SEGMENT_BYTES))
                known.add(path)
            if manifest.pending_delete is not None:
                path = trace.path(manifest.pending_delete)
                if path.exists() and path not in known:
                    artifacts.append(Artifact(path, "trace", SEGMENT_BYTES))
                    known.add(path)
            if trace.intent_path.exists():
                artifacts.append(Artifact(trace.intent_path, "trace_intent", 65536))
                known.add(trace.intent_path)
                _, intent = Crypto(config).read_json(trace.intent_path, "trace_intent", 65536)
                try:
                    if not isinstance(intent, dict):
                        raise ValueError
                    segment = Segment.model_validate(
                        {
                            "number": intent["new_segment_number"],
                            "uuid": intent["new_segment_uuid"],
                        }
                    )
                    path = trace.path(segment)
                except (ValueError, KeyError):
                    raise LoomStorageError("trace_corrupt") from None
                if path.exists() and path not in known:
                    artifacts.append(Artifact(path, "trace", SEGMENT_BYTES))
                    known.add(path)
            if trace.directory.exists() and set(trace.directory.iterdir()) - known:
                raise LoomStorageError("untracked_trace_artifact")
    if journals:
        directory = config.path(".loom/relocations")
        if directory.exists():
            check_permissions(directory, directory=True)
            for transaction in sorted(directory.iterdir()):
                identifier(transaction.name)
                check_permissions(transaction, directory=True)
                journal = transaction / "journal.enc"
                if not journal.is_file():
                    raise LoomStorageError("relocation_journal_missing")
                artifacts.append(Artifact(journal, "relocation_journal", 16 * MIB))
    return sorted(artifacts, key=lambda artifact: canonical_path(artifact.path))


def trace_records(config: Config, artifact: Artifact) -> Iterator[tuple[Header, bytes]]:
    data = read_private(artifact.path, SEGMENT_BYTES)
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
            header, plaintext = Crypto(config).open(
                base64.b64decode(encoded, validate=True),
                kind="trace",
                identity=resource_id(artifact.path, "trace"),
                maximum=EVENT_BYTES,
            )
            event = Event.model_validate_json(plaintext)
            identifier(event.run_id)
            if (
                header.run_id != event.run_id
                or header.event_id != event.event_id
                or header.revision != event.sequence
                or event.event_id != f"{event.run_id}:{event.sequence}"
            ):
                raise ValueError
            yield header, plaintext
        except ValueError:
            raise LoomStorageError("trace_corrupt") from None


def validate_checkpoint(checkpoint: Checkpoint) -> None:
    identifier(checkpoint.run_id)
    identifier(checkpoint.root_run_id)
    identifier(checkpoint.state.conversation_id)
    if fingerprint(checkpoint.configuration) != checkpoint.configuration_fingerprint:
        raise LoomStorageError("checkpoint_fingerprint_invalid")
    if checkpoint.child is not None:
        if checkpoint.kind != "workflow" or checkpoint.child.root_run_id != checkpoint.run_id:
            raise LoomStorageError("workflow_child_invalid")
        validate_checkpoint(checkpoint.child)


def read(config: Config, artifact: Artifact) -> tuple[Header, bytes]:
    header, plaintext = Crypto(config).read(artifact.path, artifact.kind, artifact.maximum)
    if artifact.kind in {"storage_json", "storage_sqlite"}:
        backend = JSONStorage if artifact.kind == "storage_json" else SQLiteStorage
        image = backend(path=artifact.path, config=config).decode(plaintext)
        if image.revision != header.revision:
            raise LoomStorageError("storage_payload_invalid")
        for checkpoint in image.runs.values():
            validate_checkpoint(checkpoint)
    elif artifact.kind == "index_sqlite":
        if header.revision != 1:
            raise LoomStorageError("index_payload_invalid")
        database = sqlite3.connect(":memory:")
        try:
            database.deserialize(plaintext)
            database.execute("PRAGMA temp_store=MEMORY")
            database.execute("PRAGMA query_only=ON")
            if database.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise LoomStorageError("index_payload_invalid")
            database.execute("SELECT chunk_id, document_id, embedding FROM chunks LIMIT 1")
            database.execute("SELECT chunk_id FROM chunks_fts LIMIT 1")
        except sqlite3.Error:
            raise LoomStorageError("index_payload_invalid") from None
        finally:
            database.close()
    else:
        decode_json(plaintext)
    return header, plaintext


def verify(config: Config) -> dict[str, JSONValue]:
    results: list[JSONValue] = []
    with locked(config, exclusive=False):
        for artifact in inventory(config):
            headers = (
                (header for header, _ in trace_records(config, artifact))
                if artifact.kind == "trace"
                else iter((read(config, artifact)[0],))
            )
            keys = sorted({header.key_id for header in headers})
            results.append(
                {
                    "resource_id": resource_id(artifact.path, artifact.kind),
                    "key_ids": json_value(keys),
                }
            )
    return {"status": "completed", "operation": "verify", "artifacts": results}


def reseal(
    config: Config,
    destination: Config,
    artifact: Artifact,
    target: Path,
    plaintext: bytes | None = None,
    revision: int | None = None,
) -> bytes:
    crypto = Crypto(destination)
    if artifact.kind == "trace":
        data = bytearray()
        for header, event in trace_records(config, artifact):
            envelope = crypto.seal(
                event,
                kind="trace",
                identity=resource_id(target, "trace"),
                revision=header.revision,
                run_id=header.run_id,
                event_id=header.event_id,
            )
            data.extend(canonical({"envelope": base64.b64encode(envelope).decode("ascii")}) + b"\n")
            if len(data) > SEGMENT_BYTES:
                raise LoomStorageError("trace_rekey_capacity_exceeded")
        return bytes(data)
    header, original = read(config, artifact)
    return crypto.seal(
        original if plaintext is None else plaintext,
        kind=artifact.kind,
        identity=resource_id(target, artifact.kind),
        revision=header.revision if revision is None else revision,
    )


def rekey(config: Config) -> dict[str, JSONValue]:
    if config.encryption is None:
        raise LoomStorageError("encryption_key_unavailable")
    Crypto(config).key(config.encryption.active_key_id)
    results: list[JSONValue] = []
    with locked(config, exclusive=True):
        artifacts = inventory(config)
        for artifact in artifacts:
            if artifact.kind == "trace":
                for _ in trace_records(config, artifact):
                    pass
            else:
                _, payload = read(config, artifact)
                if artifact.kind == "relocation_journal":
                    journal = decode_json(payload)
                    if not isinstance(journal, dict) or journal.get("state") not in {
                        "committed",
                        "aborted",
                    }:
                        raise LoomStorageError("artifact_maintenance_required")
        for artifact in artifacts:
            atomic_write(artifact.path, reseal(config, config, artifact, artifact.path))
            if artifact.kind == "trace":
                headers = [header for header, _ in trace_records(config, artifact)]
            else:
                headers = [read(config, artifact)[0]]
            if any(header.key_id != config.encryption.active_key_id for header in headers):
                raise LoomStorageError("rekey_verification_failed")
            results.append(
                {
                    "resource_id": resource_id(artifact.path, artifact.kind),
                    "key_id": config.encryption.active_key_id,
                }
            )
    return {"status": "completed", "operation": "rekey", "artifacts": results}


def ciphertext_hash(artifact: Artifact) -> str:
    maximum = artifact.maximum if artifact.kind == "trace" else artifact.maximum + OVERHEAD
    return hashlib.sha256(read_private(artifact.path, maximum)).hexdigest()
