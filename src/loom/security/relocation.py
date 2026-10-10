from __future__ import annotations

import hashlib
import shutil
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Literal

from pydantic import Field

from loom.config import Config
from loom.contracts import (
    JSONValue,
    LoomConfigurationError,
    LoomStorageError,
    Record,
    canonical,
    fingerprint,
    json_value,
)
from loom.security.admin import (
    Artifact,
    ciphertext_hash,
    inventory,
    read,
    reseal,
    trace_records,
)
from loom.security.crypto import Crypto, ResourceType
from loom.security.files import (
    atomic_write,
    canonical_path,
    check_permissions,
    file_lock,
    read_private,
    reject_links,
    secure_directory,
)
from loom.security.maintenance import Managed, configured
from loom.storage import Checkpoint, JSONStorage, SQLiteStorage, identifier

MAXIMUM = 16 * 1024**2


class Entry(Record):
    source: str
    destination: str
    kind: ResourceType
    maximum: int = Field(gt=0)
    source_digest: str
    destination_digest: str | None = None


class Journal(Record):
    schema_version: Literal[1] = 1
    relocation_id: str
    revision: int = Field(ge=1)
    state: Literal["prepared", "copying", "verified", "retiring", "committed", "aborted"]
    old_hash: str
    new_hash: str
    bindings: dict[str, str]
    entries: tuple[Entry, ...]


def bindings(old: Config, new: Config) -> dict[str, str]:
    pairs: list[tuple[Path, Path]] = [
        (old.path(old.embeddings.cache_path), new.path(new.embeddings.cache_path)),
    ]
    old_roots, new_roots = configured(old), configured(new)
    if old_roots.keys() != new_roots.keys():
        raise LoomConfigurationError("relocation_configuration_mismatch")
    for role, root in old_roots.items():
        target_root = new_roots[role]
        if root.kind != target_root.kind:
            raise LoomConfigurationError("relocation_configuration_mismatch")
        pairs.extend(((root.path, target_root.path), (root.marker, target_root.marker)))
        if role == "trace":
            pairs.append(
                (
                    root.path.with_name(root.path.name + ".segments"),
                    target_root.path.with_name(target_root.path.name + ".segments"),
                )
            )
    for name, store in old.stores.items():
        pairs.append((old.path(store.path), new.path(new.stores[name].path)))
    mapped: dict[str, str] = {}
    for source, target in pairs:
        key = canonical_path(source)
        if key in mapped and canonical_path(Path(mapped[key])) != canonical_path(target):
            raise LoomConfigurationError("relocation_binding_ambiguous")
        mapped[key] = str(target)
    return mapped


def semantics(config: Config) -> JSONValue:
    data = config.model_dump(mode="json", exclude={"encryption"})
    data["embeddings"]["cache_path"] = None
    data["storage"]["path"] = None
    data["observability"]["jsonl_path"] = bool(config.observability.jsonl_path)
    for store in data["stores"].values():
        store["path"] = store["index_path"] = None
    data["defaults"]["output_schema"] = config.output_schema(config.defaults.output_schema)
    for name, agent in config.agents.items():
        if agent.output_schema is not None:
            data["agents"][name]["output_schema"] = config.output_schema(agent.output_schema)
    return json_value(data)


def config_hash(config: Config) -> str:
    return fingerprint({"config": json_value(config), "base": canonical_path(config.path("."))})


def mapped_path(path: Path, mapping: dict[str, str]) -> Path:
    normalized = Path(canonical_path(path))
    matches = [
        (Path(source), Path(destination))
        for source, destination in mapping.items()
        if normalized == Path(source) or normalized.is_relative_to(Path(source))
    ]
    if not matches:
        raise LoomStorageError("unconfigured_programmatic_binding")
    source, destination = max(matches, key=lambda pair: len(pair[0].parts))
    return destination / normalized.relative_to(source)


def descriptor(value: dict[str, JSONValue], mapping: dict[str, str]) -> dict[str, JSONValue]:
    result = dict(value)
    paths = result.get("binding_paths")
    if not isinstance(paths, dict):
        raise LoomStorageError("unconfigured_programmatic_binding")

    def binding(value: JSONValue) -> JSONValue:
        if value is None:
            return None
        if isinstance(value, str):
            key = canonical_path(Path(value))
            if key not in mapping:
                raise LoomStorageError("unconfigured_programmatic_binding")
            return mapping[key]
        if isinstance(value, dict):
            return {key: binding(child) for key, child in value.items()}
        raise LoomStorageError("checkpoint_binding_invalid")

    result["binding_paths"] = binding(paths)
    steps = result.get("steps")
    if isinstance(steps, dict):
        transformed: dict[str, JSONValue] = {}
        for name, step in steps.items():
            if not isinstance(step, dict):
                raise LoomStorageError("checkpoint_binding_invalid")
            copied = dict(step)
            if step.get("kind") == "agent":
                target = step.get("target")
                if not isinstance(target, dict):
                    raise LoomStorageError("checkpoint_binding_invalid")
                copied["target"] = descriptor(target, mapping)
            transformed[name] = copied
        result["steps"] = transformed
    return result


def checkpoint(value: Checkpoint, mapping: dict[str, str]) -> Checkpoint:
    changed = descriptor(value.configuration, mapping)
    return value.model_copy(
        update={
            "configuration": changed,
            "configuration_fingerprint": fingerprint(changed),
            "child": checkpoint(value.child, mapping) if value.child is not None else None,
        }
    )


def relocated(
    old: Config,
    new: Config,
    artifact: Artifact,
    target: Path,
    mapping: dict[str, str],
) -> bytes:
    if artifact.kind not in {"storage_json", "storage_sqlite"}:
        return reseal(old, new, artifact, target)
    backend = JSONStorage if artifact.kind == "storage_json" else SQLiteStorage
    _, plaintext = read(old, artifact)
    storage = backend(path=artifact.path, config=old)
    image = storage.decode(plaintext)
    runs = {key: checkpoint(value, mapping) for key, value in image.runs.items()}
    updated = image.model_copy(
        update={
            "runs": runs,
            "revision": image.revision + int(runs != image.runs),
        }
    )
    data = storage.encode(updated)
    if len(data) > artifact.maximum:
        raise LoomStorageError("storage_capacity_exceeded")
    return reseal(old, new, artifact, target, data, updated.revision)


def write_journal(old: Config, path: Path, journal: Journal, **updates: object) -> Journal:
    updated = journal.model_copy(update={"revision": journal.revision + 1, **updates})
    Crypto(old).write(
        path,
        "relocation_journal",
        canonical(updated, max_bytes=MAXIMUM),
        updated.revision,
    )
    return updated


def journals(old: Config) -> list[tuple[Path, Journal]]:
    directory = old.path(".loom/relocations")
    if not directory.exists():
        return []
    check_permissions(directory, directory=True)
    found = []
    for transaction in sorted(directory.iterdir()):
        identifier(transaction.name)
        check_permissions(transaction, directory=True)
        path = transaction / "journal.enc"
        header, data = Crypto(old).read(path, "relocation_journal", MAXIMUM)
        try:
            journal = Journal.model_validate_json(data)
            if journal.revision != header.revision or journal.relocation_id != transaction.name:
                raise ValueError
        except ValueError:
            raise LoomStorageError("relocation_journal_invalid") from None
        found.append((path, journal))
    return found


def root_for(path: Path, roots: dict[str, Managed]) -> Managed:
    for role, root in roots.items():
        if path == root.path or path == root.marker:
            return root
        if root.kind == "index_current" and path.is_relative_to(root.path):
            return root
        if role == "trace" and path.is_relative_to(
            root.path.with_name(root.path.name + ".segments")
        ):
            return root
    raise LoomStorageError("relocation_inventory_invalid")


def absent(root: Managed) -> bool:
    if root.marker.exists() or root.initialized():
        return False
    return not (root.kind == "index_current" and root.path.exists())


def compatible_roots(old_roots: dict[str, Managed], new_roots: dict[str, Managed]) -> None:
    def paths(roots: dict[str, Managed]) -> list[Path]:
        result: list[Path] = []
        for role, root in roots.items():
            result.extend((root.path, root.marker))
            if role == "trace":
                result.append(root.path.with_name(root.path.name + ".segments"))
        return result

    for source in paths(old_roots):
        for target in paths(new_roots):
            reject_links(source)
            reject_links(target)
            old_path, new_path = Path(canonical_path(source)), Path(canonical_path(target))
            if (
                old_path == new_path
                or old_path.is_relative_to(new_path)
                or new_path.is_relative_to(old_path)
            ):
                raise LoomConfigurationError("relocation_roots_overlap")


def verify_models(config: Config) -> None:
    from loom.documents.embeddings import ARTIFACTS, model_path

    for name, (size, digest) in ARTIFACTS.items():
        data = read_private(model_path(config) / name, size)
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise LoomConfigurationError("embedding_artifact_mismatch")


def check_capacity(allocations: list[tuple[Path, int]]) -> None:
    volumes: dict[int, tuple[Path, int]] = {}
    for path, size in allocations:
        parent = path.parent
        while not parent.exists():
            parent = parent.parent
        device = parent.stat().st_dev
        _, previous = volumes.get(device, (parent, 0))
        volumes[device] = parent, previous + size
    for parent, required in volumes.values():
        if shutil.disk_usage(parent).free < required:
            raise LoomStorageError("relocation_capacity_exceeded")


def relocate(old: Config, new: Config, *, abort: bool = False) -> dict[str, JSONValue]:
    old.validate_paths()
    new.validate_paths()
    if semantics(old) != semantics(new):
        raise LoomConfigurationError("relocation_configuration_mismatch")
    if old.encryption is None or new.encryption is None:
        raise LoomConfigurationError("encryption_key_unavailable")
    Crypto(old).key(old.encryption.active_key_id)
    Crypto(new).key(new.encryption.active_key_id)
    old_roots, new_roots = configured(old), configured(new)
    compatible_roots(old_roots, new_roots)
    mapping = bindings(old, new)
    old_hash, new_hash = config_hash(old), config_hash(new)
    transaction_lock = old.path(".loom/relocations.lock")
    with file_lock(transaction_lock, blocking=False), ExitStack() as locks:
        candidates = [
            (path, journal)
            for path, journal in journals(old)
            if journal.old_hash == old_hash
            and journal.new_hash == new_hash
            and (journal.state != "aborted" or abort)
        ]
        if len(candidates) > 1:
            raise LoomStorageError("relocation_journal_ambiguous")
        existing = candidates[0] if candidates else None
        if existing is None:
            if abort:
                raise LoomStorageError("relocation_not_found")
            if any(not absent(root) for root in new_roots.values()):
                raise LoomStorageError("relocation_destination_exists")
        for root in sorted(
            (*old_roots.values(), *new_roots.values()),
            key=lambda root: canonical_path(root.lock),
        ):
            locks.enter_context(file_lock(root.lock, blocking=False))
        if existing is None:
            for root in old_roots.values():
                if not root.marker.exists() and not root.initialized():
                    root.write("ready")
                if root.read().status != "ready":
                    raise LoomStorageError("artifact_maintenance_required")
            if any(root.marker.exists() or root.initialized() for root in new_roots.values()):
                raise LoomStorageError("relocation_destination_exists")
            entries = []
            allocations = [(old.path(".loom/relocations/journal.enc"), MAXIMUM * 3 + 65536)]
            for artifact in inventory(old, journals=False):
                if artifact.kind == "maintenance":
                    continue
                target_path = mapped_path(artifact.path, mapping)
                # Validate every transformation before blocking normal access.
                source_digest = ciphertext_hash(artifact)
                data = relocated(old, new, artifact, target_path, mapping)
                if ciphertext_hash(artifact) != source_digest:
                    raise LoomStorageError("relocation_source_changed")
                allocations.append((target_path, len(data) * 2))
                entries.append(
                    Entry(
                        source=str(artifact.path),
                        destination=str(target_path),
                        kind=artifact.kind,
                        maximum=artifact.maximum,
                        source_digest=source_digest,
                    )
                )
            check_capacity(allocations)
            if any(entry.kind == "index_sqlite" for entry in entries):
                verify_models(new)
            relocation_id = str(uuid.uuid4())
            journal = Journal(
                relocation_id=relocation_id,
                revision=1,
                state="prepared",
                old_hash=old_hash,
                new_hash=new_hash,
                bindings=mapping,
                entries=tuple(entries),
            )
            canonical(journal, max_bytes=MAXIMUM)
            path = old.path(f".loom/relocations/{relocation_id}/journal.enc")
            Crypto(old).write(path, "relocation_journal", canonical(journal), journal.revision)
        else:
            path, journal = existing
            if journal.bindings != mapping:
                raise LoomStorageError("relocation_configuration_mismatch")
        for root in (*old_roots.values(), *new_roots.values()):
            if root.marker.exists():
                marker = root.read()
                if marker.relocation_id not in {None, journal.relocation_id}:
                    raise LoomStorageError("relocation_owner_mismatch")
        if abort:
            if journal.state == "committed":
                raise LoomStorageError("relocation_already_committed")
            journal = write_journal(old, path, journal, state="aborted")
            for root in old_roots.values():
                root.write("ready")
            for root in new_roots.values():
                root.write("aborted", journal.relocation_id)
            return report(journal)
        if journal.state == "aborted":
            raise LoomStorageError("artifact_maintenance_required")
        if journal.state != "committed":
            for root in old_roots.values():
                marker = root.read()
                if marker.status != "retired":
                    root.write("pending", journal.relocation_id)
            for root in new_roots.values():
                root.write("pending", journal.relocation_id)
            if journal.state == "prepared":
                journal = write_journal(old, path, journal, state="copying")
            for index, entry in enumerate(journal.entries):
                source = Artifact(Path(entry.source), entry.kind, entry.maximum)
                target = Artifact(Path(entry.destination), entry.kind, entry.maximum)
                if mapped_path(source.path, mapping) != target.path:
                    raise LoomStorageError("relocation_inventory_invalid")
                if ciphertext_hash(source) != entry.source_digest:
                    raise LoomStorageError("relocation_source_changed")
                if entry.destination_digest is not None:
                    if ciphertext_hash(target) != entry.destination_digest:
                        raise LoomStorageError("relocation_destination_changed")
                else:
                    secure_directory(target.path.parent)
                    atomic_write(target.path, relocated(old, new, source, target.path, mapping))
                if entry.kind == "trace":
                    for _ in trace_records(new, target):
                        pass
                else:
                    read(new, target)
                updated = list(journal.entries)
                updated[index] = entry.model_copy(
                    update={"destination_digest": ciphertext_hash(target)}
                )
                journal = write_journal(old, path, journal, entries=tuple(updated))
            journal = write_journal(old, path, journal, state="verified")
            journal = write_journal(old, path, journal, state="retiring")
            for root in old_roots.values():
                root.write("retired", journal.relocation_id)
            journal = write_journal(old, path, journal, state="committed")
        for root in old_roots.values():
            marker = root.read()
            if marker.status != "retired" or marker.relocation_id != journal.relocation_id:
                raise LoomStorageError("relocation_retirement_invalid")
        for root in new_roots.values():
            marker = root.read()
            if marker.status == "ready":
                continue
            if marker.status != "pending" or marker.relocation_id != journal.relocation_id:
                raise LoomStorageError("relocation_activation_invalid")
            for entry in journal.entries:
                target = Artifact(Path(entry.destination), entry.kind, entry.maximum)
                if (
                    root_for(target.path, new_roots) == root
                    and ciphertext_hash(target) != entry.destination_digest
                ):
                    raise LoomStorageError("relocation_destination_changed")
            root.write("ready", journal.relocation_id)
        return report(journal)


def report(journal: Journal) -> dict[str, JSONValue]:
    return {
        "status": "completed",
        "operation": "relocate",
        "relocation_id": journal.relocation_id,
        "state": journal.state,
        "artifact_count": len(journal.entries),
        "verified": journal.state == "committed",
    }
