from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field

from loom.config import Config
from loom.contracts import LoomStorageError, Record, canonical
from loom.security.crypto import Crypto, ResourceType, resource_id
from loom.security.files import canonical_path, file_lock


class Marker(Record):
    schema_version: Literal[1] = 1
    revision: int = Field(ge=1)
    status: Literal["ready", "pending", "retired", "aborted"]
    relocation_id: str | None = None
    source_id: str


@dataclass(frozen=True)
class Managed:
    config: Config
    path: Path
    kind: ResourceType

    @property
    def marker(self) -> Path:
        return (
            self.path / "maintenance.enc"
            if self.kind == "index_current"
            else self.path.with_name(self.path.name + ".maintenance.enc")
        )

    @property
    def lock(self) -> Path:
        return self.marker.with_suffix(".lock")

    @property
    def identity(self) -> str:
        return resource_id(self.path, self.kind)

    def initialized(self) -> bool:
        if self.kind == "index_current":
            return (self.path / "CURRENT.enc").exists() or (self.path / "snapshots").exists()
        if self.kind == "trace_manifest":
            return self.path.with_name(self.path.name + ".segments").exists()
        return self.path.exists()

    def read(self) -> Marker:
        if not self.marker.exists():
            raise LoomStorageError("maintenance_marker_missing")
        header, plaintext = Crypto(self.config).read(self.marker, "maintenance", 65536)
        try:
            marker = Marker.model_validate_json(plaintext)
            if marker.revision != header.revision or marker.source_id != self.identity:
                raise ValueError
            return marker
        except ValueError:
            raise LoomStorageError("maintenance_marker_invalid") from None

    def write(
        self,
        status: Literal["ready", "pending", "retired", "aborted"],
        relocation_id: str | None = None,
    ) -> Marker:
        previous = self.read() if self.marker.exists() else None
        marker = Marker(
            revision=previous.revision + 1 if previous else 1,
            status=status,
            relocation_id=relocation_id,
            source_id=self.identity,
        )
        Crypto(self.config).write(self.marker, "maintenance", canonical(marker), marker.revision)
        return marker

    @contextmanager
    def shared(self) -> Iterator[None]:
        if not self.marker.exists():
            with file_lock(self.lock):
                if not self.marker.exists():
                    if self.initialized():
                        raise LoomStorageError("maintenance_marker_missing")
                    self.write("ready")
        with file_lock(self.lock, shared=True):
            marker = self.read()
            if marker.status != "ready":
                raise LoomStorageError(
                    "artifact_relocated"
                    if marker.status == "retired"
                    else "artifact_maintenance_required"
                )
            yield


def configured(config: Config) -> dict[str, Managed]:
    resources: dict[str, Managed] = {}
    if config.storage.enabled:
        assert config.storage.path is not None
        resources["storage"] = Managed(
            config,
            config.path(config.storage.path),
            "storage_sqlite" if config.storage.backend == "sqlite" else "storage_json",
        )
    for name, store in config.stores.items():
        resources[f"store:{name}"] = Managed(config, config.path(store.index_path), "index_current")
    if config.observability.jsonl_path:
        resources["trace"] = Managed(
            config,
            config.path(config.observability.jsonl_path),
            "trace_manifest",
        )
    return resources


@contextmanager
def protected(
    resources: Sequence[Managed],
    lifecycle: AbstractContextManager[object],
) -> Iterator[None]:
    unique = {canonical_path(resource.lock): resource for resource in resources}
    with ExitStack() as stack:
        for _, resource in sorted(unique.items()):
            stack.enter_context(resource.shared())
        stack.enter_context(lifecycle)
        yield
