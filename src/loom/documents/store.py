from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path

from loom.config import Config, StoreConfig, absolute
from loom.contracts import (
    NAME,
    Diagnostic,
    JSONValue,
    LoomConfigurationError,
    LoomStorageError,
    Provenance,
    SearchResult,
    ToolContext,
    ToolFailure,
    ToolSpec,
    json_value,
)
from loom.documents.index import MAXIMUM, MIB, PIPELINE
from loom.documents.index import build as build_index
from loom.documents.index import search as search_index
from loom.runtime import Work, offload, pool, sync, wait_work
from loom.security.crypto import OVERHEAD, Crypto
from loom.security.files import (
    canonical_path,
    check_permissions,
    file_lock,
    reject_links,
    secure_directory,
)
from loom.security.maintenance import Managed
from loom.storage import identifier
from loom.tools.base import Tool


class _Snapshot:
    def __init__(
        self,
        database: sqlite3.Connection,
        size: int,
        key: bytes,
        key_id: str,
        diagnostics: tuple[Diagnostic, ...],
    ) -> None:
        self.database, self.size, self.key = database, size, key
        self.key_id = key_id
        self.diagnostics = diagnostics
        self.leases = 0
        self.lock = threading.Lock()


_cache_guard = threading.Lock()
_cache: OrderedDict[tuple[str, str], _Snapshot] = OrderedDict()
_charged = 0


class _Load:
    def __init__(self, reserved: int) -> None:
        self.work: Work[_Snapshot] | None = None
        self.entry: _Snapshot | None = None
        self.waiters = 1
        self.reserved = reserved


_loads: dict[tuple[str, str], _Load] = {}


def _after_fork() -> None:
    global _cache_guard, _cache, _loads, _charged
    for entry in _cache.values():
        entry.database.close()
    _cache_guard, _cache, _loads, _charged = threading.Lock(), OrderedDict(), {}, 0


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _coverage(manifest: JSONValue) -> tuple[Diagnostic, ...]:
    if not isinstance(manifest, dict):
        raise LoomStorageError("index_manifest_invalid")
    counts: dict[str, JSONValue] = {}
    for name in ("indexed", "excluded", "partially_extracted"):
        value = manifest.get(name)
        if type(value) is not int or value < 0:
            raise LoomStorageError("index_manifest_invalid")
        counts[name] = value
    if not counts["excluded"] and not counts["partially_extracted"]:
        return ()
    return (
        Diagnostic(
            code="index_coverage_incomplete",
            message="Some source content was excluded from this snapshot.",
            origin="index",
            details={
                "store": manifest.get("store"),
                "snapshot_id": manifest.get("snapshot_id"),
                **counts,
            },
        ),
    )


def _space(requested: int, *, protect: tuple[str, str], adding: bool = False) -> None:
    global _charged
    while _charged + requested > 3 * 1024**3 or (
        len(_cache.keys() | _loads.keys()) + int(adding) > 2
    ):
        victim = next(
            (
                key
                for key, value in _cache.items()
                if value.leases == 0 and key != protect and key not in _loads
            ),
            None,
        )
        if victim is None:
            raise ToolFailure(
                "query_memory_exhausted",
                recoverable=True,
                details={
                    "requested_bytes": requested,
                    "available_bytes": max(0, 3 * 1024**3 - _charged),
                    "resident_snapshots": len(_cache.keys() | _loads.keys()),
                },
            )
        removed = _cache.pop(victim)
        removed.database.close()
        _charged -= 2 * removed.size + 64 * MIB


class DocumentStore:
    def __init__(
        self,
        *,
        path: str | Path,
        name: str | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        if index_path is not None and name is None:
            raise LoomConfigurationError("store_name_required")
        if name is not None and not re.fullmatch(NAME, name):
            raise LoomConfigurationError("invalid_store_name")
        self.path = absolute(path)
        self.name = name or ""
        self.index_path = absolute(index_path) if index_path is not None else None
        self.config: Config | None = None

    def bind(self, config: Config) -> DocumentStore:
        if self.name:
            binding = config.stores.get(self.name)
            if self.index_path is None and binding is None:
                raise LoomConfigurationError("store_binding_missing")
            if binding is not None and absolute(config.path(binding.path)) != self.path:
                raise LoomConfigurationError("store_path_mismatch")
            index = self.index_path or (config.path(binding.index_path) if binding else None)
            name = self.name
        else:
            matches = [
                (name, item)
                for name, item in config.stores.items()
                if canonical_path(config.path(item.path)) == canonical_path(self.path)
            ]
            if len(matches) != 1:
                raise LoomConfigurationError("store_binding_ambiguous")
            name, binding = matches[0]
            index = config.path(binding.index_path)
        assert index is not None
        if index == self.path or index.is_relative_to(self.path):
            raise LoomConfigurationError("generated_path_in_source")
        config.validate_paths(
            stores={name: StoreConfig(path=str(self.path), index_path=str(index))}
        )
        bound = DocumentStore(path=self.path, name=name, index_path=index)
        bound.config = config
        return bound

    def _bound(self) -> DocumentStore:
        return self if self.config is not None else self.bind(Config.load())

    def pin(self, snapshot_id: str | None = None) -> str:
        store = self._bound()
        assert store.index_path is not None and store.config is not None
        crypto = Crypto(store.config)
        with Managed(store.config, store.index_path, "index_current").shared():
            if snapshot_id is None:
                _, current = crypto.read_json(store.index_path / "CURRENT.enc", "index_current")
                if not isinstance(current, dict) or set(current) != {
                    "schema_version",
                    "snapshot_id",
                }:
                    raise LoomStorageError("index_pointer_invalid")
                value = current.get("snapshot_id")
                if not isinstance(value, str):
                    raise LoomStorageError("index_pointer_invalid")
                snapshot_id = value
            identifier(snapshot_id)
            directory = store.index_path / "snapshots" / snapshot_id
            if not (directory / "index.sqlite.enc").exists():
                raise LoomConfigurationError("snapshot_missing")
            _, manifest = crypto.read_json(directory / "manifest.json.enc", "index_manifest")
            if not isinstance(manifest, dict) or manifest.get("pipeline_fingerprint") != PIPELINE:
                raise LoomConfigurationError("index_incompatible")
            if manifest.get("snapshot_id") != snapshot_id or manifest.get("store") != store.name:
                raise LoomStorageError("index_identity_mismatch")
        return snapshot_id

    def search(self, query: str, *, snapshot_id: str | None = None) -> SearchResult:
        return sync(lambda: self.asearch(query, snapshot_id=snapshot_id))

    async def asearch(self, query: str, *, snapshot_id: str | None = None) -> SearchResult:
        global _charged
        store = self._bound()
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            raise ToolFailure("invalid_search_query", recoverable=True)
        deadline = time.monotonic() + 30
        snapshot_id = await offload("internal_io", lambda: store.pin(snapshot_id), seconds=30)
        assert store.config is not None and store.index_path is not None
        key = (canonical_path(store.index_path), snapshot_id)
        permit = pool("inference").reserve()
        entry: _Snapshot | None = None
        charged = False
        dispatched = False

        def release() -> None:
            global _charged
            with _cache_guard:
                if entry is not None:
                    entry.leases -= 1
                if charged:
                    _charged -= 64 * MIB

        try:
            with _cache_guard:
                _space(64 * MIB, protect=key)
                _charged += 64 * MIB
                charged = True
            entry = await store._snapshot(snapshot_id, deadline)

            def search() -> SearchResult:
                assert (
                    store.config is not None and store.index_path is not None and entry is not None
                )
                with Managed(store.config, store.index_path, "index_current").shared():
                    return store._search(query, snapshot_id, entry)

            work = pool("inference").submit(
                search,
                deadline=deadline,
                permit=permit,
                finished=release,
            )
            dispatched = True
            return await wait_work(work, deadline=deadline)
        finally:
            if not dispatched:
                permit.release()
                release()

    def tool(self, snapshot_id: str) -> Tool:
        return SearchTool(self, snapshot_id)

    async def _snapshot(self, snapshot_id: str, deadline: float) -> _Snapshot:
        global _charged
        assert self.index_path is not None and self.config is not None
        path = self.index_path / "snapshots" / snapshot_id / "index.sqlite.enc"

        def size_of() -> int:
            check_permissions(path)
            return path.stat().st_size

        size = await offload("internal_io", size_of, seconds=max(0, deadline - time.monotonic()))
        if size > MAXIMUM + OVERHEAD:
            raise ToolFailure("index_capacity_exceeded")
        cache_key = (canonical_path(self.index_path), snapshot_id)
        crypto = Crypto(self.config)
        with _cache_guard:
            entry = _cache.get(cache_key)
            if entry is not None:
                entry.leases += 1
                _cache.move_to_end(cache_key)
            load = _loads.get(cache_key)
            if entry is None and load is not None:
                load.waiters += 1
            if entry is None and load is None:
                requested = 4 * size + 64 * MIB
                _space(requested, protect=cache_key, adding=True)
                load = _Load(requested)
                _loads[cache_key] = load
                _charged += requested
                ticket = load

                def finished() -> None:
                    global _charged
                    with _cache_guard:
                        _loads.pop(cache_key, None)
                        _charged -= ticket.reserved
                        ticket.reserved = 0

                def decrypt() -> _Snapshot:
                    global _charged
                    assert self.config is not None and self.index_path is not None
                    with _cache_guard:
                        if ticket.waiters == 0:
                            raise ToolFailure("operation_cancelled")
                    with Managed(self.config, self.index_path, "index_current").shared():
                        _, manifest = crypto.read_json(
                            path.with_name("manifest.json.enc"), "index_manifest", 16 * MIB
                        )
                        diagnostics = _coverage(manifest)
                        header, plaintext = crypto.read(path, "index_sqlite", MAXIMUM)
                        key = crypto.key(header.key_id)
                        database = sqlite3.connect(":memory:", check_same_thread=False)
                        try:
                            if header.revision != 1:
                                raise LoomStorageError("index_payload_invalid")
                            database.deserialize(plaintext)
                            database.execute("PRAGMA temp_store=MEMORY")
                            database.execute("PRAGMA journal_mode=MEMORY")
                            database.execute("PRAGMA cache_size=-32768")
                            database.execute("PRAGMA query_only=ON")
                        except (sqlite3.Error, LoomStorageError):
                            database.close()
                            raise LoomStorageError("index_payload_invalid") from None
                        del plaintext
                        loaded = _Snapshot(database, size, key, header.key_id, diagnostics)
                        with _cache_guard:
                            loaded.leases = ticket.waiters
                            ticket.entry = loaded
                            _cache[cache_key] = loaded
                            _charged -= 2 * size
                            ticket.reserved = 0
                        return loaded

                try:
                    load.work = pool("internal_io").submit(
                        decrypt, deadline=deadline, finished=finished
                    )
                except BaseException:
                    _loads.pop(cache_key, None)
                    _charged -= requested
                    raise
        if entry is None:
            assert load is not None and load.work is not None
            try:
                # Cancellation abandons this waiter, not a load shared by other queries.
                entry = await wait_work(
                    Work(load.work.future, threading.Event()),
                    deadline=deadline,
                )
            except BaseException:
                with _cache_guard:
                    if load.entry is not None:
                        load.entry.leases -= 1
                    else:
                        load.waiters -= 1
                raise
        try:
            if crypto.key(entry.key_id) != entry.key:
                raise LoomConfigurationError("encryption_key_unavailable")
            return entry
        except BaseException:
            with _cache_guard:
                entry.leases -= 1
            raise

    def _search(self, query: str, snapshot_id: str, entry: _Snapshot) -> SearchResult:
        return search_index(self, query, snapshot_id, entry)

    def build(self, *, refresh: bool = False) -> dict[str, JSONValue]:
        store = self._bound()
        assert store.index_path is not None and store.config is not None
        if not store.path.is_dir():
            raise LoomConfigurationError("source_root_unavailable")
        reject_links(store.path)
        secure_directory(store.index_path)
        crypto = Crypto(store.config)
        with (
            Managed(store.config, store.index_path, "index_current").shared(),
            file_lock(store.index_path / "writer.lock"),
        ):
            exists = (store.index_path / "CURRENT.enc").exists()
            if exists != refresh:
                raise LoomConfigurationError("index_exists" if exists else "index_missing")
            return store._build(crypto)

    def _build(self, crypto: Crypto) -> dict[str, JSONValue]:
        return build_index(self, crypto)


class SearchTool(Tool):
    def __init__(self, store: DocumentStore, snapshot: str) -> None:
        self.store, self.snapshot = store, snapshot
        schema = json_value(SearchResult.model_json_schema())
        assert isinstance(schema, dict)
        super().__init__(
            ToolSpec(
                name=f"search_{store.name}",
                description=f"Search the indexed {store.name} documents.",
                argument_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 4096}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                output_schema=schema,
                sensitive=False,
                retry_safe=True,
                egress="none",
            )
        )

    async def execute(
        self, arguments: Mapping[str, JSONValue], context: ToolContext
    ) -> SearchResult:
        query = arguments["query"]
        if not isinstance(query, str):
            raise ToolFailure("invalid_search_query", recoverable=True)
        return await self.store.asearch(query, snapshot_id=self.snapshot)

    @staticmethod
    def clip(result: SearchResult) -> SearchResult:
        diagnostics = result.diagnostics
        if not result.clipped:
            diagnostics = (
                *diagnostics,
                Diagnostic(
                    code="retrieval_context_clipped",
                    message="Lowest-ranked chunks were omitted to fit the observation budget.",
                    origin="retrieval",
                ),
            )
        return result.model_copy(
            update={"chunks": result.chunks[:-1], "clipped": True, "diagnostics": diagnostics}
        )

    @staticmethod
    def provenance(result: SearchResult) -> tuple[Provenance, ...]:
        return (
            Provenance(
                store=result.store,
                snapshot_id=result.snapshot_id,
                document_ids=tuple(sorted({chunk.document_id for chunk in result.chunks})),
            ),
        )
