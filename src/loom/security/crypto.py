from __future__ import annotations

import base64
import binascii
import hashlib
import os
import struct
from pathlib import Path
from typing import Literal

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import Field, ValidationError

from loom.config import Config
from loom.contracts import (
    JSONValue,
    LoomConfigurationError,
    LoomStorageError,
    Record,
    canonical,
    decode_json,
)
from loom.security.files import atomic_write, canonical_path, read_private

MAGIC = b"LOOMENC1"
OVERHEAD = 12 + 4096 + 16
ResourceType = Literal[
    "storage_sqlite",
    "storage_json",
    "index_sqlite",
    "index_manifest",
    "index_report",
    "index_current",
    "trace",
    "trace_manifest",
    "trace_intent",
    "maintenance",
    "relocation_journal",
    "export",
]


class Header(Record):
    version: Literal[1] = 1
    algorithm: Literal["AES-256-GCM"] = "AES-256-GCM"
    key_id: str
    resource_type: ResourceType
    resource_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=0)
    nonce: str
    run_id: str | None = None
    event_id: str | None = None


def resource_id(path: Path, kind: ResourceType) -> str:
    return hashlib.sha256(f"loom-v1\n{kind}\n{canonical_path(path)}".encode()).hexdigest()


class Crypto:
    def __init__(self, config: Config) -> None:
        self.config = config

    def key(self, key_id: str) -> bytes:
        encryption = self.config.encryption
        if encryption is None or key_id not in encryption.keys:
            raise LoomConfigurationError("encryption_key_unavailable")
        encoded = os.environ.get(encryption.keys[key_id])
        if encoded is None:
            raise LoomConfigurationError("encryption_key_unavailable")
        try:
            value = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise LoomConfigurationError("invalid_encryption_key") from None
        if len(value) != 32:
            raise LoomConfigurationError("invalid_encryption_key")
        return value

    def _derived(self, header: Header) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=bytes.fromhex(header.resource_id),
            info=f"loom-v1/{header.resource_type}".encode(),
        ).derive(self.key(header.key_id))

    def seal(
        self,
        plaintext: bytes,
        *,
        kind: ResourceType,
        identity: str,
        revision: int,
        run_id: str | None = None,
        event_id: str | None = None,
    ) -> bytes:
        if self.config.encryption is None:
            raise LoomConfigurationError("encryption_key_unavailable")
        if (kind == "trace" and (not run_id or not event_id)) or (
            kind != "trace" and (run_id is not None or event_id is not None)
        ):
            raise LoomStorageError("invalid_artifact_identity")
        nonce = os.urandom(12)
        header = Header(
            key_id=self.config.encryption.active_key_id,
            resource_type=kind,
            resource_id=identity,
            revision=revision,
            nonce=base64.b64encode(nonce).decode("ascii"),
            run_id=run_id,
            event_id=event_id,
        )
        encoded = canonical(header)
        aad = MAGIC + struct.pack(">I", len(encoded)) + encoded
        return aad + AESGCM(self._derived(header)).encrypt(nonce, plaintext, aad)

    def open(
        self,
        envelope: bytes,
        *,
        kind: ResourceType,
        identity: str | None,
        maximum: int,
    ) -> tuple[Header, bytes]:
        try:
            if len(envelope) < 28 or len(envelope) > maximum + OVERHEAD:
                raise ValueError
            if envelope[:8] != MAGIC:
                raise ValueError
            size = struct.unpack(">I", envelope[8:12])[0]
            if not 1 <= size <= 4096 or len(envelope) < 12 + size + 16:
                raise ValueError
            header = Header.model_validate_json(envelope[12 : 12 + size])
            if header.resource_type != kind or (
                identity is not None and header.resource_id != identity
            ):
                raise ValueError
            if kind == "trace":
                if not header.run_id or not header.event_id:
                    raise ValueError
            elif header.run_id is not None or header.event_id is not None:
                raise ValueError
            nonce = base64.b64decode(header.nonce, validate=True)
            if len(nonce) != 12 or len(envelope) - (12 + size + 16) > maximum:
                raise ValueError
            plaintext = AESGCM(self._derived(header)).decrypt(
                nonce, envelope[12 + size :], envelope[: 12 + size]
            )
            return header, plaintext
        except (ValueError, ValidationError, InvalidTag, binascii.Error):
            raise LoomStorageError("artifact_authentication_failed") from None

    def read(self, path: Path, kind: ResourceType, maximum: int) -> tuple[Header, bytes]:
        return self.open(
            read_private(path, maximum + OVERHEAD),
            kind=kind,
            identity=resource_id(path, kind),
            maximum=maximum,
        )

    def write(self, path: Path, kind: ResourceType, plaintext: bytes, revision: int) -> None:
        atomic_write(
            path,
            self.seal(plaintext, kind=kind, identity=resource_id(path, kind), revision=revision),
        )

    def read_json(
        self,
        path: Path,
        kind: ResourceType,
        maximum: int = 16 * 1024**2,
    ) -> tuple[Header, JSONValue]:
        header, plaintext = self.read(path, kind, maximum)
        try:
            return header, decode_json(plaintext)
        except (ValueError, ValidationError):
            raise LoomStorageError("artifact_payload_invalid") from None
