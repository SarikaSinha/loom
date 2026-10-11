from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from string import Formatter
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import httpx
from pydantic import BaseModel

from loom.config import Config
from loom.contracts import (
    JSONValue,
    LoomConfigurationError,
    OutputTooLarge,
    Schema,
    ToolContext,
    ToolFailure,
    ToolSpec,
    canonical,
    decode_json,
    json_value,
    schema_for,
)
from loom.events import SECRET_KEYS, normalize_key
from loom.runtime import offload
from loom.security.transport import client, origin, retry_after, validate_url
from loom.tools.base import Tool

RequestBuilder = Callable[[Mapping[str, JSONValue], ToolContext], Mapping[str, JSONValue]]
BLOCKED_HEADERS = {"host", "content_length", "transfer_encoding", "connection"}


class HTTPTool(Tool):
    def __init__(
        self,
        *,
        name: str,
        description: str,
        method: str,
        endpoint: str,
        request_builder: RequestBuilder,
        argument_schema: Schema | type[BaseModel],
        output_schema: Schema | type[BaseModel],
        version: str = "1",
        headers: Mapping[str, str] | None = None,
        auth_env: Mapping[str, Mapping[str, str]] | None = None,
        sensitive: bool = True,
        retry_safe: bool = False,
        timeout_seconds: float | None = None,
    ) -> None:
        validate_url(endpoint)
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise LoomConfigurationError("invalid_http_method")
        parsed = urlsplit(endpoint)
        if "{" in parsed.netloc + parsed.query or "}" in parsed.netloc + parsed.query:
            raise LoomConfigurationError("invalid_endpoint_template")
        self.fields: set[str] = set()
        try:
            for _, field, format_spec, conversion in Formatter().parse(parsed.path):
                if field is not None:
                    if (
                        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", field)
                        or format_spec
                        or conversion
                    ):
                        raise ValueError
                    self.fields.add(field)
        except ValueError:
            raise LoomConfigurationError("invalid_endpoint_template") from None
        self.headers = dict(headers or {})
        self.auth_env = {key: dict(value) for key, value in (auth_env or {}).items()}
        for header, value in self.headers.items():
            self._header(header, value)
            if normalize_key(header) in SECRET_KEYS | BLOCKED_HEADERS:
                raise LoomConfigurationError("literal_sensitive_header")
        for header, reference in self.auth_env.items():
            self._header(header, reference.get("prefix", ""))
            if (
                set(reference) != {"env", "prefix"}
                or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", reference["env"])
                or normalize_key(header) in BLOCKED_HEADERS
            ):
                raise LoomConfigurationError("invalid_auth_reference")
        if {key.lower() for key in self.headers} & {key.lower() for key in self.auth_env}:
            raise LoomConfigurationError("duplicate_auth_header")
        arguments, output = schema_for(argument_schema), schema_for(output_schema)
        assert arguments is not None and output is not None
        super().__init__(
            ToolSpec(
                name=name,
                version=version,
                description=description,
                argument_schema=arguments,
                output_schema=output,
                sensitive=sensitive,
                retry_safe=retry_safe,
                timeout_seconds=timeout_seconds,
                egress="fixed_network",
                egress_origins=(origin(endpoint),),
            )
        )
        self.method, self.endpoint, self.builder = method, endpoint, request_builder

    @staticmethod
    def _header(name: str, value: str) -> None:
        if (
            not re.fullmatch(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", name)
            or "\r" in value
            or "\n" in value
        ):
            raise LoomConfigurationError("invalid_http_header")

    def request(self, value: object) -> tuple[str, JSONValue]:
        built = json_value(value)
        if not isinstance(built, dict) or set(built) != {"path_params", "query", "json_body"}:
            raise ToolFailure("invalid_request_builder_result")
        parameters, query = built["path_params"], built["query"]
        if (
            not isinstance(parameters, dict)
            or set(parameters) != self.fields
            or not isinstance(query, dict)
        ):
            raise ToolFailure("invalid_request_builder_result")

        def scalar(item: JSONValue) -> str:
            if isinstance(item, (dict, list)):
                raise ToolFailure("invalid_request_builder_result")
            return (
                "" if item is None else str(item).lower() if isinstance(item, bool) else str(item)
            )

        path_values = {
            key: quote(scalar(item), safe="").replace(".", "%2E")
            for key, item in parameters.items()
        }
        parts = urlsplit(self.endpoint)
        path = parts.path.format_map(path_values)
        pairs = [
            (key, scalar(item))
            for key, values in query.items()
            for item in (values if isinstance(values, list) else [values])
        ]
        encoded = urlencode(pairs)
        combined = "&".join(part for part in (parts.query, encoded) if part)
        endpoint = urlunsplit((parts.scheme, parts.netloc, path, combined, ""))
        if origin(endpoint) != self.spec.egress_origins[0]:
            raise ToolFailure("http_origin_mismatch")
        if self.method == "GET" and built["json_body"] is not None:
            raise ToolFailure("get_body_forbidden")
        canonical(built["json_body"], max_bytes=1048576)
        return endpoint, built["json_body"]

    async def execute(
        self,
        arguments: Mapping[str, JSONValue],
        context: ToolContext,
        *,
        config: Config | None = None,
    ) -> JSONValue:
        config = Config.load(config)
        remaining = max(0.001, (context.deadline - datetime.now(UTC)).total_seconds())
        built = await offload(
            "user_code",
            lambda: self.builder(arguments, context),
            seconds=remaining,
            cancellation=context.cancellation,
        )
        endpoint, body = self.request(built)
        headers = self.headers | {"Accept-Encoding": "identity"}
        if body is not None and not any(key.lower() == "content-type" for key in headers):
            headers["Content-Type"] = "application/json"
        for name, reference in self.auth_env.items():
            value = os.environ.get(reference["env"])
            if not value:
                raise ToolFailure("tool_credentials_unavailable")
            headers[name] = reference["prefix"] + value
            self._header(name, headers[name])
        remaining = max(0.001, (context.deadline - datetime.now(UTC)).total_seconds())
        try:
            async with client(config, timeout=remaining) as session:
                async with session.stream(
                    self.method,
                    endpoint,
                    headers=headers,
                    content=canonical(body) if body is not None else None,
                ) as response:
                    status = response.status_code
                    if not 200 <= status < 300:
                        raise ToolFailure(
                            "http_not_found"
                            if status == 404
                            else "http_authentication_failed"
                            if status in {401, 403}
                            else "http_request_failed",
                            recoverable=status in {404, 408, 429} or status >= 500,
                            transient=status in {408, 429} or status >= 500,
                            outcome_unknown=not self.spec.retry_safe
                            and (status == 408 or status >= 500),
                            details={
                                "status_code": status,
                                "retry_after_seconds": retry_after(
                                    response.headers.get("retry-after")
                                ),
                            },
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes(chunk_size=16384):
                        if len(content) + len(chunk) > 1048576:
                            raise OutputTooLarge()
                        content.extend(chunk)
            try:
                return decode_json(bytes(content))
            except ValueError:
                raise ToolFailure("http_invalid_response") from None
        except httpx.TransportError:
            raise ToolFailure(
                "http_transport_error",
                recoverable=True,
                transient=True,
                outcome_unknown=not self.spec.retry_safe,
            ) from None
