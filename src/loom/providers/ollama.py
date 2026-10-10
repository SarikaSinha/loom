from __future__ import annotations

import httpx

from loom.config import Config, ProviderConfig
from loom.contracts import (
    JSONValue,
    LoomConfigurationError,
    ToolCall,
    canonical,
    decode_json,
)
from loom.providers.base import (
    ModelAdapter,
    ModelCapabilities,
    ModelFailure,
    ModelRequest,
    ModelResponse,
)
from loom.security.transport import client, loopback, retry_after, validate_url


class OllamaAdapter(ModelAdapter):
    def __init__(self, profile: ProviderConfig, config: Config) -> None:
        if profile.kind != "ollama":
            raise LoomConfigurationError("invalid_provider_kind")
        validate_url(profile.url, ollama=True)
        self.profile = profile
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._verified = False

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            native_tool_calling=self.profile.native_tool_calling,
            structured_output=self.profile.structured_output,
            local=loopback(self.profile.url),
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        body: JSONValue = None,
        seconds: float,
    ) -> dict[str, JSONValue]:
        if self._client is None:
            self._client = client(self.config, timeout=seconds)
        try:
            async with self._client.stream(
                method,
                self.profile.url.rstrip("/") + path,
                json=body,
                timeout=seconds,
                headers={"Accept-Encoding": "identity"},
            ) as response:
                data = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=16384):
                    if len(data) + len(chunk) > 262144:
                        raise ModelFailure("provider_response_too_large")
                    data.extend(chunk)
                if response.status_code >= 300:
                    text = data.decode("utf-8", errors="replace").lower()
                    if response.status_code in {400, 413} and any(
                        term in text for term in ("context", "truncate", "too long", "shift")
                    ):
                        raise ModelFailure("context_limit")
                    raise ModelFailure(
                        "provider_http_error",
                        transient=response.status_code in {408, 429} or response.status_code >= 500,
                        retry_after=retry_after(response.headers.get("retry-after")),
                    )
                value = decode_json(bytes(data))
        except httpx.TimeoutException:
            raise ModelFailure("model_timeout", transient=True) from None
        except httpx.TransportError:
            raise ModelFailure("provider_transport_error", transient=True) from None
        except ValueError:
            raise ModelFailure("provider_protocol_error") from None
        if not isinstance(value, dict):
            raise ModelFailure("provider_protocol_error")
        return value

    async def verify(self, seconds: float) -> None:
        version = await self._request("GET", "/api/version", seconds=seconds)
        if version.get("version") != self.profile.server_version:
            raise LoomConfigurationError("ollama_version_mismatch")
        tags = await self._request("GET", "/api/tags", seconds=seconds)
        models = tags.get("models")
        if not isinstance(models, list):
            raise ModelFailure("provider_protocol_error")
        model = next(
            (
                entry
                for entry in models
                if isinstance(entry, dict) and entry.get("name") == self.profile.model
            ),
            None,
        )
        expected = self.profile.model_digest
        if not model or not expected:
            raise LoomConfigurationError("ollama_model_unavailable")
        digest = model.get("digest")
        if not isinstance(digest, str) or (
            digest.removeprefix("sha256:").lower() != expected.removeprefix("sha256:").lower()
        ):
            raise LoomConfigurationError("ollama_digest_mismatch")
        show = await self._request(
            "POST", "/api/show", body={"model": self.profile.model}, seconds=seconds
        )
        info = show.get("model_info")
        if not isinstance(info, dict):
            raise LoomConfigurationError("ollama_context_unavailable")
        windows = [
            value
            for key, value in info.items()
            if key.endswith(".context_length") and type(value) is int
        ]
        if not windows or self.profile.num_ctx is None or self.profile.num_ctx > max(windows):
            raise LoomConfigurationError("ollama_context_unsupported")
        self._verified = True

    async def generate(self, request: ModelRequest) -> ModelResponse:
        if not self._verified:
            await self.verify(request.timeout_seconds)
        messages: list[JSONValue] = [{"role": "system", "content": request.instructions}]
        for previous_message in request.messages:
            entry: dict[str, JSONValue] = {
                "role": previous_message.role,
                "content": previous_message.content,
            }
            if previous_message.tool_calls:
                entry["tool_calls"] = [
                    {"function": {"name": call.name, "arguments": call.arguments}}
                    for call in previous_message.tool_calls
                ]
            if previous_message.observation:
                entry["content"] = canonical(previous_message.observation).decode("utf-8")
                name = next(
                    (
                        call.name
                        for previous in request.messages
                        for call in previous.tool_calls
                        if call.call_id == previous_message.observation.call_id
                    ),
                    None,
                )
                if name is None:
                    raise ModelFailure("provider_protocol_error")
                entry["tool_name"] = name
            messages.append(entry)
        body: dict[str, JSONValue] = {
            "model": self.profile.model,
            "messages": messages,
            "stream": False,
            "truncate": False,
            "shift": False,
            "options": {
                "num_ctx": self.profile.num_ctx,
                "num_predict": request.max_output_tokens,
                "num_gpu": self.profile.num_gpu,
                "num_thread": self.profile.num_thread,
            },
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.argument_schema,
                    },
                }
                for spec in request.tools
            ]
        if request.output_schema and request.output_mode != "prompt_json":
            body["format"] = (
                request.output_schema if request.output_mode == "json_schema" else "json"
            )
        data = await self._request("POST", "/api/chat", body=body, seconds=request.timeout_seconds)
        finish = data.get("done_reason")
        if data.get("done") is not True or finish not in {"stop", "length"}:
            raise ModelFailure("provider_protocol_error")
        try:
            result = ModelResponse.model_validate(
                {
                    "input_tokens": data.get("prompt_eval_count"),
                    "output_tokens": data.get("eval_count"),
                }
            )
        except ValueError:
            raise ModelFailure("provider_protocol_error") from None
        if finish == "length":
            return result.model_copy(update={"finish_reason": "output_limit"})
        message = data.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content", ""), str):
            raise ModelFailure("provider_protocol_error")
        raw_calls = message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ModelFailure("provider_protocol_error")
        calls: list[ToolCall] = []
        for ordinal, raw in enumerate(raw_calls):
            function = raw.get("function") if isinstance(raw, dict) else None
            if not isinstance(function, dict):
                raise ModelFailure("provider_protocol_error")
            try:
                calls.append(
                    ToolCall.model_validate(
                        {
                            "call_id": f"part_{ordinal}",
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                        }
                    )
                )
            except ValueError:
                raise ModelFailure("provider_protocol_error") from None
        return result.model_copy(
            update={
                "output": message.get("content", ""),
                "tool_calls": tuple(calls),
                "finish_reason": "tool_calls" if calls else "completed",
            }
        )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._verified = False
