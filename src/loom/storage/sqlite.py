from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from pydantic import ValidationError

from loom.contracts import (
    ConversationState,
    LoomStorageError,
    canonical,
)
from loom.security.crypto import ResourceType
from loom.storage.base import Storage
from loom.storage.checkpoint import (
    Checkpoint,
    ConversationHead,
    Image,
)


class SQLiteStorage(Storage):
    kind: ResourceType = "storage_sqlite"
    maximum = 128 * 1024**2

    @staticmethod
    def database() -> sqlite3.Connection:
        database = sqlite3.connect(":memory:")
        database.execute("PRAGMA temp_store=MEMORY")
        database.execute("PRAGMA journal_mode=MEMORY")
        database.execute("PRAGMA foreign_keys=ON")
        return database

    def decode(self, data: bytes) -> Image:
        database = self.database()
        try:
            database.deserialize(data)
            meta = database.execute("SELECT schema_version, revision FROM store_meta").fetchall()
            if len(meta) != 1 or meta[0][0] != 1:
                raise LoomStorageError("storage_schema_unsupported")
            runs = {
                key: Checkpoint.model_validate_json(payload)
                for key, payload in database.execute("SELECT run_id, checkpoint_json FROM runs")
            }
            conversations = {
                key: ConversationHead(
                    head_run_id=head, state=ConversationState.model_validate_json(payload)
                )
                for key, head, payload in database.execute(
                    "SELECT conversation_id, head_run_id, state_json FROM conversations"
                )
            }
            return Image(revision=meta[0][1], runs=runs, conversations=conversations)
        except ValidationError:
            raise LoomStorageError("storage_payload_invalid") from None
        finally:
            database.close()

    def encode(self, image: Image) -> bytes:
        database = self.database()
        try:
            database.executescript("""
                CREATE TABLE store_meta(schema_version INTEGER NOT NULL, revision INTEGER NOT NULL);
                CREATE TABLE runs(
                    run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, revision INTEGER NOT NULL,
                    phase TEXT NOT NULL, checkpoint_json TEXT NOT NULL,
                    result_json TEXT, updated_at TEXT NOT NULL
                );
                CREATE TABLE conversations(
                    conversation_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
                    head_run_id TEXT NOT NULL, state_json TEXT NOT NULL,
                    FOREIGN KEY(head_run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
                );
            """)
            database.execute("INSERT INTO store_meta VALUES(1, ?)", (image.revision,))
            for checkpoint in image.runs.values():
                database.execute(
                    "INSERT INTO runs VALUES(?,?,?,?,?,?,?)",
                    (
                        checkpoint.run_id,
                        checkpoint.kind,
                        checkpoint.revision,
                        checkpoint.phase,
                        canonical(checkpoint).decode("utf-8"),
                        canonical(checkpoint.result).decode("utf-8") if checkpoint.result else None,
                        datetime.now(UTC).isoformat(),
                    ),
                )
            for key, head in image.conversations.items():
                database.execute(
                    "INSERT INTO conversations VALUES(?,?,?,?)",
                    (
                        key,
                        head.state.version,
                        head.head_run_id,
                        canonical(head.state).decode("utf-8"),
                    ),
                )
            database.commit()
            return database.serialize()
        finally:
            database.close()
