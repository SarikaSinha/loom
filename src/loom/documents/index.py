from __future__ import annotations

import hashlib
import os
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from loom.contracts import (
    Diagnostic,
    JSONValue,
    Location,
    LoomConfigurationError,
    LoomStorageError,
    SearchChunk,
    SearchResult,
    ToolFailure,
    canonical,
    fingerprint,
    json_value,
)
from loom.documents import embeddings
from loom.documents.chunks import windows
from loom.documents.loaders import blocks, source_bytes
from loom.security.crypto import Crypto, resource_id
from loom.security.files import (
    atomic_write,
    secure_directory,
    sync_directory,
    temporary_directory,
)
from loom.storage import identifier

if TYPE_CHECKING:
    from loom.documents.store import DocumentStore, _Snapshot

MAXIMUM = 512 * 1024**2

MIB = 1024**2

PIPELINE = fingerprint(
    {"version": 1, "embedding": embeddings.REVISION, "chunk": 400, "overlap": 50}
)


def search(store: DocumentStore, query: str, snapshot_id: str, entry: _Snapshot) -> SearchResult:
    from loom.documents.store import SearchTool

    assert store.config is not None
    try:
        with entry.lock:
            vector = embeddings.encoder(store.config).embed([query], query=True)[0]
            database = entry.database
            terms = sorted(set(re.findall(r"[^\W_]+", query.lower(), flags=re.UNICODE)))
            lexical = (
                [
                    row[0]
                    for row in database.execute(
                        "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
                        "ORDER BY bm25(chunks_fts), chunk_id LIMIT 20",
                        (" OR ".join(f'"{term}"' for term in terms),),
                    )
                ]
                if terms
                else []
            )
            semantic: list[tuple[float, str]] = []
            for chunk_id, blob in database.execute("SELECT chunk_id, embedding FROM chunks"):
                embedded = np.frombuffer(blob, dtype="<f4")
                if embedded.shape != (384,) or not np.isfinite(embedded).all():
                    raise LoomStorageError("index_vector_invalid")
                semantic.append((float(embedded @ vector), chunk_id))
                if len(semantic) >= 256:
                    semantic = sorted(semantic, key=lambda item: (-item[0], item[1]))[:20]
            semantic = sorted(semantic, key=lambda item: (-item[0], item[1]))[:20]
            scores: dict[str, float] = {}
            for ordering in (lexical, [key for _, key in semantic]):
                for rank, chunk_id in enumerate(ordering, 1):
                    scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
            chunks: list[SearchChunk] = []
            for chunk_id in sorted(scores, key=lambda key: (-scores[key], key))[:5]:
                row = database.execute(
                    "SELECT document_id,source_path,location_json,text "
                    "FROM chunks WHERE chunk_id=?",
                    (chunk_id,),
                ).fetchone()
                if row is None:
                    raise LoomStorageError("index_chunk_missing")
                from pydantic import TypeAdapter

                chunks.append(
                    SearchChunk(
                        chunk_id=chunk_id,
                        document_id=row[0],
                        path=row[1],
                        location=TypeAdapter(Location).validate_json(row[2]),
                        text=row[3],
                        score=scores[chunk_id],
                    )
                )
            result = SearchResult(
                store=store.name,
                snapshot_id=snapshot_id,
                query=query,
                chunks=tuple(chunks),
                diagnostics=entry.diagnostics,
            )
            while len(canonical(result)) > 4096 and result.chunks:
                result = SearchTool.clip(result)
            if (chunks and not result.chunks) or len(canonical(result)) > 4096:
                raise ToolFailure("observation_too_large", recoverable=True)
            return result
    except sqlite3.Error:
        raise LoomStorageError("index_payload_invalid") from None


def build(store: DocumentStore, crypto: Crypto) -> dict[str, JSONValue]:
    assert store.config is not None and store.index_path is not None
    encoder = embeddings.encoder(store.config)
    database = sqlite3.connect(":memory:")
    database.executescript("""
        PRAGMA temp_store=MEMORY;
        PRAGMA journal_mode=MEMORY;
        CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
            source_path TEXT NOT NULL, location_json TEXT NOT NULL, text TEXT NOT NULL,
            embedding BLOB NOT NULL);
        CREATE VIRTUAL TABLE chunks_fts
        USING fts5(chunk_id UNINDEXED,text,tokenize='unicode61');
    """)
    page_size = database.execute("PRAGMA page_size").fetchone()[0]
    database.execute(f"PRAGMA max_page_count={MAXIMUM // page_size}")
    previous: sqlite3.Connection | None = None
    report: list[JSONValue] = []
    hashes: dict[str, JSONValue] = {}
    indexed = 0
    chunk_count = 0
    reused_vectors = 0
    try:
        current = store.index_path / "CURRENT.enc"
        if current.exists():
            _, pointer = crypto.read_json(current, "index_current", 65536)
            if not isinstance(pointer, dict) or not isinstance(pointer.get("snapshot_id"), str):
                raise LoomStorageError("index_pointer_invalid")
            value = pointer["snapshot_id"]
            assert isinstance(value, str)
            snapshot_id = identifier(value)
            directory = store.index_path / "snapshots" / snapshot_id
            _, old_manifest = crypto.read_json(directory / "manifest.json.enc", "index_manifest")
            if not isinstance(old_manifest, dict) or old_manifest.get("snapshot_id") != snapshot_id:
                raise LoomStorageError("index_identity_mismatch")
            if old_manifest.get("pipeline_fingerprint") == PIPELINE:
                _, old_image = crypto.read(directory / "index.sqlite.enc", "index_sqlite", MAXIMUM)
                previous = sqlite3.connect(":memory:")
                previous.deserialize(old_image)
                del old_image
                previous.execute("PRAGMA temp_store=MEMORY")
                previous.execute("PRAGMA cache_size=-32768")
                previous.execute("PRAGMA query_only=ON")
        files: list[Path] = []

        def unreadable(error: OSError) -> None:
            raise LoomConfigurationError("source_root_unavailable") from None

        for parent, directories, names in os.walk(
            store.path, followlinks=False, onerror=unreadable
        ):
            directories[:] = [
                name
                for name in directories
                if not (Path(parent) / name).is_symlink()
                and not (getattr((Path(parent) / name).lstat(), "st_file_attributes", 0) & 0x400)
            ]
            files.extend(
                Path(parent) / name
                for name in names
                if not (Path(parent) / name).is_symlink() and (Path(parent) / name).is_file()
            )
        for path in sorted(files, key=lambda path: path.relative_to(store.path).as_posix()):
            relative = path.relative_to(store.path).as_posix()
            diagnostics: list[Diagnostic] = []
            rows: list[tuple[str, str, str, str, str, bytes]] = []
            batch: list[tuple[str, str, str, str, str]] = []
            reused = 0

            def flush(
                batch: list[tuple[str, str, str, str, str]],
                rows: list[tuple[str, str, str, str, str, bytes]],
            ) -> None:
                vectors = encoder.embed([row[4] for row in batch])
                rows.extend(
                    (*row, vector.astype("<f4", copy=False).tobytes())
                    for row, vector in zip(batch, vectors, strict=True)
                )
                batch.clear()

            try:
                data = source_bytes(path)
                hashes[relative] = hashlib.sha256(data).hexdigest()
                document_id = hashlib.sha256(f"{store.name}\n{relative}".encode()).hexdigest()
                characters = 0
                for text, location in blocks(path, diagnostics, data=data):
                    characters += len(text.encode("utf-8"))
                    if characters > 10 * MIB:
                        raise ToolFailure("extracted_text_limit")
                    for start_char, end_char, content in windows(text, encoder.offsets(text)):
                        location_json = canonical(location).decode()
                        block_hash = hashlib.sha256(text.encode()).hexdigest()
                        chunk_id = hashlib.sha256(
                            f"{document_id}\n{block_hash}\n{location_json}\n{start_char}\n{end_char}".encode()
                        ).hexdigest()
                        row = (chunk_id, document_id, relative, location_json, content)
                        saved = (
                            previous.execute(
                                "SELECT embedding FROM chunks WHERE chunk_id=?",
                                (chunk_id,),
                            ).fetchone()
                            if previous is not None
                            else None
                        )
                        if saved is not None:
                            vector = np.frombuffer(saved[0], dtype="<f4")
                            if vector.shape != (384,) or not np.isfinite(vector).all():
                                raise LoomStorageError("index_vector_invalid")
                            rows.append((*row, saved[0]))
                            reused += 1
                        else:
                            batch.append(row)
                            if len(batch) == 32:
                                flush(batch, rows)
                if batch:
                    flush(batch, rows)
                if not rows:
                    raise ToolFailure("empty_document")
            except (ToolFailure, OSError, ValueError) as error:
                report.append(
                    {
                        "path": relative,
                        "status": "excluded",
                        "reason": getattr(error, "code", "parse_failed"),
                    }
                )
                continue
            database.executemany("INSERT OR IGNORE INTO chunks VALUES(?,?,?,?,?,?)", rows)
            database.executemany("INSERT INTO chunks_fts VALUES(?,?)", [(r[0], r[4]) for r in rows])
            indexed += 1
            chunk_count += len(rows)
            reused_vectors += reused
            report.append(
                {
                    "path": relative,
                    "status": "partial" if diagnostics else "indexed",
                    "diagnostics": [json_value(diagnostic) for diagnostic in diagnostics],
                }
            )
        if report and not indexed:
            raise ToolFailure("no_indexable_documents")
        database.commit()
        image = database.serialize()
        if len(image) > MAXIMUM:
            raise ToolFailure("index_capacity_exceeded")
        snapshot = str(uuid.uuid4())
        destination = store.index_path / "snapshots" / snapshot
        manifest: dict[str, JSONValue] = {
            "schema_version": 1,
            "snapshot_id": snapshot,
            "store": store.name,
            "created_at": datetime.now(UTC).isoformat(),
            "pipeline_fingerprint": PIPELINE,
            "embedding_revision": embeddings.REVISION,
            "embedding_dimension": 384,
            "input_file_hashes": hashes,
            "indexed": indexed,
            "excluded": len(report) - indexed,
            "partially_extracted": sum(
                isinstance(item, dict) and item.get("status") == "partial" for item in report
            ),
            "chunk_count": chunk_count,
        }
        from loom.security.crypto import ResourceType

        components: tuple[tuple[str, ResourceType, bytes], ...] = (
            ("index.sqlite.enc", "index_sqlite", image),
            ("manifest.json.enc", "index_manifest", canonical(manifest, max_bytes=16 * MIB)),
            (
                "ingestion-report.json.enc",
                "index_report",
                canonical(report, max_bytes=16 * MIB),
            ),
        )
        secure_directory(destination.parent)
        with temporary_directory(destination.parent, prefix=".building-") as staging:
            for filename, kind, plaintext in components:
                atomic_write(
                    staging / filename,
                    crypto.seal(
                        plaintext,
                        kind=kind,
                        identity=resource_id(destination / filename, kind),
                        revision=1,
                    ),
                )
            sync_directory(staging)
            os.replace(staging, destination)
            sync_directory(destination.parent)
        current = store.index_path / "CURRENT.enc"
        revision = (
            crypto.read(current, "index_current", 65536)[0].revision + 1 if current.exists() else 1
        )
        crypto.write(
            current,
            "index_current",
            canonical(
                {
                    "schema_version": 1,
                    "snapshot_id": snapshot,
                }
            ),
            revision,
        )
        snapshot_bytes = sum(component.stat().st_size for component in destination.iterdir())
        retained_bytes = sum(
            component.stat().st_size
            for generation in destination.parent.iterdir()
            if generation.is_dir() and not generation.name.startswith(".")
            for component in generation.iterdir()
            if component.is_file()
        )
        return {
            **manifest,
            "snapshot_bytes": snapshot_bytes,
            "retained_bytes": retained_bytes,
            "reused_vectors": reused_vectors,
            "files": report,
        }
    except ToolFailure as error:
        if error.code == "payload_too_large":
            raise ToolFailure("index_capacity_exceeded") from None
        raise
    except sqlite3.Error as error:
        raise ToolFailure(
            "index_capacity_exceeded"
            if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_FULL
            else "index_build_failed",
        ) from None
    finally:
        database.close()
        if previous is not None:
            previous.close()
