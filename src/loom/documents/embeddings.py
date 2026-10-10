from __future__ import annotations

import hashlib
import json
import os
import threading
from importlib.resources import files
from pathlib import Path

import httpx
import numpy as np
from numpy.typing import NDArray

from loom.config import Config
from loom.contracts import LoomConfigurationError, LoomStorageError, ToolFailure
from loom.security.files import (
    atomic_write,
    canonical_path,
    file_lock,
    read_private,
    secure_directory,
    sync_directory,
    temporary_directory,
)
from loom.security.transport import ssl_context, validate_url

_manifest = json.loads(
    files("loom.documents").joinpath("embedding-manifest.json").read_text(encoding="utf-8")
)
MODEL: str = _manifest["model"]
REVISION: str = _manifest["revision"]
REPOSITORY: str = _manifest["repository"]
ARTIFACTS: dict[str, tuple[int, str]] = {
    name: (item["bytes"], item["sha256"]) for name, item in _manifest["artifacts"].items()
}
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def model_path(config: Config) -> Path:
    return config.path(config.embeddings.cache_path) / REVISION


def setup(config: Config) -> None:
    config.validate_paths()
    destination = model_path(config)
    secure_directory(destination.parent)
    with file_lock(destination.parent / "setup.lock"):
        if destination.exists():
            verify(config)
            return
        with temporary_directory(destination.parent, prefix=".model-") as staging:
            _download(config, staging)
            sync_directory(staging)
            os.replace(staging, destination)
            sync_directory(destination.parent)


def verify(config: Config) -> None:
    directory = model_path(config)
    for name, (size, digest) in ARTIFACTS.items():
        try:
            data = read_private(directory / name, size)
        except FileNotFoundError:
            raise LoomConfigurationError("embedding_artifact_missing") from None
        except LoomStorageError as error:
            if error.code == "artifact_too_large":
                raise LoomConfigurationError("embedding_artifact_corrupt") from None
            raise
        if len(data) != size or hashlib.sha256(data).hexdigest() != digest:
            raise LoomConfigurationError("embedding_artifact_corrupt")


def _download(config: Config, destination: Path) -> None:
    with httpx.Client(
        verify=ssl_context(config), trust_env=False, follow_redirects=False, timeout=120
    ) as client:
        for name, (size, digest) in ARTIFACTS.items():
            path = destination / name
            url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}"
            content = bytearray()
            for _ in range(6):
                validate_url(url)
                with client.stream("GET", url, headers={"Accept-Encoding": "identity"}) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise LoomConfigurationError("model_download_failed")
                        url = str(response.url.join(location))
                        continue
                    response.raise_for_status()
                    for chunk in response.iter_bytes(65536):
                        if len(content) + len(chunk) > size:
                            raise LoomConfigurationError("embedding_artifact_mismatch")
                        content.extend(chunk)
                    break
            else:
                raise LoomConfigurationError("model_download_redirect_limit")
            if len(content) != size or hashlib.sha256(content).hexdigest() != digest:
                raise LoomConfigurationError("embedding_artifact_mismatch")
            atomic_write(path, bytes(content))


class Encoder:
    def __init__(self, config: Config) -> None:
        from fastembed import TextEmbedding
        from tokenizers import Tokenizer

        directory = model_path(config)
        verify(config)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.model = TextEmbedding(
            model_name=MODEL,
            specific_model_path=str(directory),
            local_files_only=True,
            providers=["CPUExecutionProvider"],
            threads=4,
        )
        self.lock = threading.RLock()

    def offsets(self, text: str) -> list[tuple[int, int]]:
        with self.lock:
            return self.tokenizer.encode(text, add_special_tokens=False).offsets

    def embed(self, texts: list[str], *, query: bool = False) -> NDArray[np.float32]:
        with self.lock:
            for text in texts:
                value = QUERY_PREFIX + text if query else text
                if len(self.tokenizer.encode(value, add_special_tokens=True).ids) > 512:
                    raise ToolFailure(
                        "query_too_long" if query else "chunk_too_long", recoverable=True
                    )
            values = (
                self.model.query_embed(texts, batch_size=32, parallel=None)
                if query
                else self.model.passage_embed(texts, batch_size=32, parallel=None)
            )
            array = np.asarray(list(values), dtype=np.float32)
            if array.shape != (len(texts), 384) or not np.isfinite(array).all():
                raise ToolFailure("embedding_invalid")
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ToolFailure("embedding_invalid")
            return np.asarray(array / norms, dtype=np.float32)


_guard = threading.Lock()
_encoder: Encoder | None = None
_verified_paths: set[str] = set()


def encoder(config: Config) -> Encoder:
    global _encoder
    with _guard:
        path = canonical_path(model_path(config))
        if _encoder is None:
            _encoder = Encoder(config)
            _verified_paths.add(path)
        elif path not in _verified_paths:
            verify(config)
            _verified_paths.add(path)
        return _encoder


def _after_fork() -> None:
    global _guard, _encoder, _verified_paths
    _guard, _encoder, _verified_paths = threading.Lock(), None, set()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
