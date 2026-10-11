from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from loom.config import Config, absolute
from loom.contracts import (
    ConversationState,
    JSONValue,
    LoomStorageError,
    canonical,
    decode_json,
    fingerprint,
)
from loom.security.crypto import Crypto, ResourceType, resource_id
from loom.security.files import file_lock
from loom.security.maintenance import Managed
from loom.storage.checkpoint import (
    Checkpoint,
    ConversationHead,
    Image,
    identifier,
)

TERMINAL = frozenset({"completed", "failed", "cancelled", "limit_exhausted"})


class Storage:
    kind: ResourceType
    maximum: int

    def __init__(self, *, path: str | Path, config: Config | str | Path | None = None) -> None:
        self.path = absolute(path)
        self._config = Config.load(config) if config is not None else None
        self.lock_directory = self.path.with_name(self.path.name + ".locks")
        self.marker = self.path.with_name(self.path.name + ".maintenance.enc")

    @property
    def config(self) -> Config:
        return self._config or Config.load()

    def bound(self, config: Config) -> Storage:
        return self if self._config is not None else type(self)(path=self.path, config=config)

    @contextmanager
    def maintenance(self) -> Iterator[None]:
        with Managed(self.config, self.path, self.kind).shared():
            yield

    @contextmanager
    def ownership(self, run_id: str, conversation_id: str | None = None) -> Iterator[None]:
        names = [(f"run-{identifier(run_id)}.lock", "run_in_use")]
        if conversation_id is not None:
            names.append(
                (f"conversation-{identifier(conversation_id)}.lock", "conversation_in_use")
            )
        with ExitStack() as stack:
            stack.enter_context(self.maintenance())
            for name, code in sorted(names):
                try:
                    stack.enter_context(file_lock(self.lock_directory / name, blocking=False))
                except LoomStorageError as error:
                    if error.code == "resource_in_use":
                        raise LoomStorageError(code) from None
                    raise
            yield

    def _read(self) -> Image:
        if not self.path.exists():
            return Image()
        header, data = Crypto(self.config).read(self.path, self.kind, self.maximum)
        try:
            image = self.decode(data)
            if image.revision != header.revision:
                raise ValueError
            for key, checkpoint in image.runs.items():
                identifier(key)
                if key != checkpoint.run_id or fingerprint(checkpoint.configuration) != (
                    checkpoint.configuration_fingerprint
                ):
                    raise ValueError
            return image
        except (ValueError, sqlite3.Error):
            raise LoomStorageError("storage_payload_invalid") from None

    def _write(self, image: Image) -> None:
        data = self.encode(image)
        reserved = sum(
            max(0, 4 * 1024**2 - len(canonical(checkpoint)))
            for checkpoint in image.runs.values()
            if checkpoint.phase not in TERMINAL
        )
        if len(data) + reserved > self.maximum:
            raise LoomStorageError("storage_capacity_exceeded")
        Crypto(self.config).write(self.path, self.kind, data, image.revision)

    def decode(self, data: bytes) -> Image:
        raise NotImplementedError

    def encode(self, image: Image) -> bytes:
        raise NotImplementedError

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        identifier(checkpoint.run_id)
        canonical(checkpoint, max_bytes=4 * 1024**2)
        canonical(checkpoint.state, max_bytes=256 * 1024)
        with self.maintenance(), file_lock(self.lock_directory / "storage.lock"):
            image = self._read()
            previous = image.runs.get(checkpoint.run_id)
            expected = previous.revision if previous is not None else 0
            if checkpoint.revision != expected:
                raise LoomStorageError("stale_checkpoint")
            head = image.conversations.get(checkpoint.state.conversation_id)
            if (
                head
                and head.head_run_id != checkpoint.run_id
                and (head.state.version != checkpoint.base_conversation_version)
            ):
                raise LoomStorageError("stale_conversation")
            saved = checkpoint.model_copy(update={"revision": expected + 1})
            runs = image.runs | {saved.run_id: saved}
            conversations = dict(image.conversations)
            if saved.phase == "completed":
                conversations[saved.state.conversation_id] = ConversationHead(
                    head_run_id=saved.run_id, state=saved.state
                )
            self._write(Image(revision=image.revision + 1, runs=runs, conversations=conversations))
            return saved

    def load_run(self, run_id: str) -> Checkpoint:
        identifier(run_id)
        with self.maintenance(), file_lock(self.lock_directory / "storage.lock", shared=True):
            checkpoint = self._read().runs.get(run_id)
        if checkpoint is None:
            raise LoomStorageError("run_not_found")
        return checkpoint

    def load_conversation(self, conversation_id: str) -> ConversationState:
        identifier(conversation_id)
        with self.maintenance(), file_lock(self.lock_directory / "storage.lock", shared=True):
            head = self._read().conversations.get(conversation_id)
        if head is None:
            raise LoomStorageError("conversation_not_found")
        return head.state

    def export_run(self, run_id: str) -> bytes:
        checkpoint = self.load_run(run_id)
        source = resource_id(self.path, self.kind)
        identity = hashlib.sha256(
            f"loom-v1\nexport\n{source}\n{checkpoint.root_run_id}".encode()
        ).hexdigest()
        return Crypto(self.config).seal(
            canonical(
                {
                    "schema_version": 1,
                    "storage_resource_id": source,
                    "root_run_id": checkpoint.root_run_id,
                    "checkpoint": checkpoint.model_dump(mode="json"),
                }
            ),
            kind="export",
            identity=identity,
            revision=checkpoint.revision,
        )

    def read_export(self, encrypted_envelope: bytes) -> JSONValue:
        header, plaintext = Crypto(self.config).open(
            encrypted_envelope, kind="export", identity=None, maximum=4 * 1024**2
        )
        try:
            payload = decode_json(plaintext)
            if not isinstance(payload, dict):
                raise ValueError
            source = resource_id(self.path, self.kind)
            root = payload.get("root_run_id")
            if not isinstance(root, str) or payload.get("storage_resource_id") != source:
                raise ValueError
            expected = hashlib.sha256(f"loom-v1\nexport\n{source}\n{root}".encode()).hexdigest()
            if header.resource_id != expected:
                raise ValueError
            return payload
        except ValueError:
            raise LoomStorageError("invalid_export") from None

    def delete_run(self, run_id: str) -> None:
        with self.ownership(run_id), file_lock(self.lock_directory / "storage.lock"):
            image = self._read()
            if run_id not in image.runs:
                raise LoomStorageError("run_not_found")
            if any(head.head_run_id == run_id for head in image.conversations.values()):
                raise LoomStorageError("record_referenced")
            self._write(
                Image(
                    revision=image.revision + 1,
                    runs={key: value for key, value in image.runs.items() if key != run_id},
                    conversations=image.conversations,
                )
            )

    def delete_conversation(self, conversation_id: str) -> None:
        identifier(conversation_id)
        with (
            self.maintenance(),
            file_lock(self.lock_directory / f"conversation-{conversation_id}.lock", blocking=False),
            file_lock(self.lock_directory / "storage.lock"),
        ):
            image = self._read()
            if conversation_id not in image.conversations:
                raise LoomStorageError("conversation_not_found")
            self._write(
                Image(
                    revision=image.revision + 1,
                    runs=image.runs,
                    conversations={
                        key: value
                        for key, value in image.conversations.items()
                        if key != conversation_id
                    },
                )
            )
