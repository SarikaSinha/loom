from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, PrivateAttr, ValidationError, model_validator

from loom.contracts import NAME, JSONValue, Limits, LoomConfigurationError, Record, decode_json


def absolute(path: str | Path, base: Path | None = None) -> Path:
    return Path(os.path.abspath((base or Path.cwd()) / path))


class ProviderConfig(Record):
    kind: Literal["openai", "azure_openai", "anthropic", "gemini", "ollama"]
    model: str = Field(min_length=1)
    native_tool_calling: bool
    structured_output: Literal["json_schema", "json_object", "prompt_json"] = "prompt_json"
    base_url: str | None = None
    endpoint: str | None = None
    api_version: str | None = None
    api_key_env: str | None = None
    model_digest: str | None = None
    server_version: str | None = None
    num_ctx: int | None = Field(default=None, gt=0)
    num_gpu: int = Field(default=0, ge=0)
    num_thread: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def provider_fields(self) -> Self:
        specific = {
            "openai": {"base_url"},
            "azure_openai": {"endpoint", "api_version"},
            "anthropic": {"base_url"},
            "gemini": set(),
            "ollama": {
                "base_url",
                "model_digest",
                "server_version",
                "num_ctx",
                "num_gpu",
                "num_thread",
            },
        }
        common = {"kind", "model", "native_tool_calling", "structured_output", "api_key_env"}
        if self.model_fields_set - common - specific[self.kind]:
            raise ValueError("unsupported provider fields")
        if self.kind == "azure_openai" and not (self.endpoint and self.api_version):
            raise ValueError("Azure requires endpoint and api_version")
        if self.kind == "anthropic" and self.structured_output == "json_object":
            raise ValueError("Anthropic does not support json_object mode")
        if self.kind == "ollama":
            if not (self.model_digest and self.server_version and self.num_ctx):
                raise ValueError("Ollama requires model_digest, server_version and num_ctx")
            if not re.fullmatch(r"(sha256:)?[0-9a-fA-F]{64}", self.model_digest):
                raise ValueError("invalid model digest")
        from loom.security.transport import validate_url

        validate_url(self.url, ollama=self.kind == "ollama")
        return self

    @property
    def url(self) -> str:
        defaults = {
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com",
            "gemini": "https://generativelanguage.googleapis.com",
            "ollama": "http://127.0.0.1:11434",
        }
        if self.kind == "azure_openai":
            assert self.endpoint
            return self.endpoint
        return self.base_url or defaults[self.kind]


class Defaults(Record):
    provider: str | None = None
    output_schema: str = "text"
    limits: dict[str, JSONValue] = Field(default_factory=dict)


class AgentProfile(Record):
    provider: str | None = None
    output_schema: str | None = None
    limits: dict[str, JSONValue] = Field(default_factory=dict)
    allowlist: tuple[str, ...] | None = None
    document_access: dict[str, Literal["local_only", "allow_hosted"]] = Field(default_factory=dict)
    tool_document_access: dict[str, tuple[str, ...]] = Field(default_factory=dict)


class StoreConfig(Record):
    path: str
    index_path: str


class EmbeddingConfig(Record):
    backend: Literal["fastembed"] = "fastembed"
    model: Literal["BAAI/bge-small-en-v1.5"] = "BAAI/bge-small-en-v1.5"
    cache_path: str = ".loom/models"


class StorageConfig(Record):
    enabled: bool = False
    backend: Literal["sqlite", "json"] = "sqlite"
    path: str | None = None

    @model_validator(mode="after")
    def required_path(self) -> Self:
        if self.enabled and not self.path:
            raise ValueError("enabled storage requires a path")
        return self


class EncryptionConfig(Record):
    active_key_id: str = Field(pattern=NAME)
    keys: dict[str, str]

    @model_validator(mode="after")
    def valid_keys(self) -> Self:
        if self.active_key_id not in self.keys:
            raise ValueError("active key ID is not in the keyring")
        for key, env in self.keys.items():
            if not re.fullmatch(NAME, key) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env):
                raise ValueError("invalid key reference")
        return self


class TransportConfig(Record):
    ca_bundle_env: str | None = None


class ObservabilityConfig(Record):
    capture_payloads: bool = False
    capture_source_paths: bool = False
    jsonl_path: str = ""
    redact_keys: tuple[str, ...] = ()


class Config(Record):
    schema_version: Literal[1] = 1
    defaults: Defaults = Field(default_factory=Defaults)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    agents: dict[str, AgentProfile] = Field(default_factory=dict)
    stores: dict[str, StoreConfig] = Field(default_factory=dict)
    embeddings: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    transport: TransportConfig = Field(default_factory=TransportConfig)
    encryption: EncryptionConfig | None = None
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    _base: Path = PrivateAttr(default_factory=Path.cwd)

    @classmethod
    def load(cls, value: Config | str | Path | None = None) -> Config:
        if isinstance(value, Config):
            return value
        path = absolute(value or "loom.toml")
        if value is None and not path.exists():
            return cls()
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
            # JSON input preserves strict scalar validation while accepting TOML arrays as tuples.
            import json

            result = cls.model_validate_json(json.dumps(data, allow_nan=False))
            result._base = path.parent
            result.validate_paths()
            return result
        except (OSError, ValueError, ValidationError):
            raise LoomConfigurationError("invalid_configuration") from None

    @model_validator(mode="after")
    def names_and_limits(self) -> Self:
        for group in (self.providers, self.agents, self.stores):
            if any(not re.fullmatch(NAME, name) for name in group):
                raise ValueError("invalid profile name")
        Limits.model_validate(self.defaults.limits)
        for agent in self.agents.values():
            Limits.model_validate(self.defaults.limits | agent.limits)
            if agent.provider and agent.provider not in self.providers:
                raise ValueError("unknown provider")
        if self.defaults.provider and self.defaults.provider not in self.providers:
            raise ValueError("unknown default provider")
        if self.observability.jsonl_path and not self.observability.jsonl_path.endswith(
            ".enc.jsonl"
        ):
            raise ValueError("trace path must end in .enc.jsonl")
        return self

    def path(self, value: str | Path) -> Path:
        return absolute(value, self._base)

    def validate_paths(
        self,
        *,
        stores: Mapping[str, StoreConfig] | None = None,
        storage_path: Path | None = None,
    ) -> None:
        from loom.security.files import canonical_path

        bindings = self.stores | dict(stores or {})
        sources = [self.path(s.path) for s in bindings.values()]
        generated = [self.path(s.index_path) for s in bindings.values()]
        generated.append(self.path(self.embeddings.cache_path))
        if storage_path is not None:
            generated.append(storage_path)
        elif self.storage.path is not None:
            generated.append(self.path(self.storage.path))
        if self.observability.jsonl_path:
            generated.append(self.path(self.observability.jsonl_path))
        sources = [Path(canonical_path(path)) for path in sources]
        generated = [Path(canonical_path(path)) for path in generated]
        for target in generated:
            if any(target == source or target.is_relative_to(source) for source in sources):
                raise LoomConfigurationError("generated_path_in_source")
        for index, target in enumerate(generated):
            if any(
                target == other or target.is_relative_to(other) or other.is_relative_to(target)
                for other in generated[index + 1 :]
            ):
                raise LoomConfigurationError("artifact_paths_overlap")

    def output_schema(self, value: str) -> str | dict[str, JSONValue]:
        if value in {"text", "document_qa"}:
            return value
        try:
            schema = decode_json(self.path(value).read_bytes())
        except (OSError, ValueError):
            raise LoomConfigurationError("output_schema_unavailable") from None
        if not isinstance(schema, dict):
            raise LoomConfigurationError("invalid_output_schema")
        return schema
