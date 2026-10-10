from __future__ import annotations

import os
from typing import ClassVar, cast

import anthropic
import httpx2
import openai
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from google import genai
from google.genai import errors, types
from openai.types.chat.completion_create_params import CompletionCreateParamsNonStreaming

from loom.config import Config, ProviderConfig
from loom.contracts import (
    JSONValue,
    LoomConfigurationError,
    ToolCall,
    canonical,
    decode_json,
    json_value,
)
from loom.providers.base import (
    ModelAdapter,
    ModelCapabilities,
    ModelFailure,
    ModelRequest,
    ModelResponse,
)
from loom.security.transport import retry_after, ssl_context, validate_url


class BoundedTransport(httpx2.AsyncBaseTransport):
    def __init__(self, config: Config) -> None:
        self.transport = httpx2.AsyncHTTPTransport(
            verify=ssl_context(config), retries=0, trust_env=False
        )

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        request.headers["Accept-Encoding"] = "identity"
        response = await self.transport.handle_async_request(request)
        try:
            body = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=16384):
                if len(body) + len(chunk) > 262144:
                    raise ModelFailure("provider_response_too_large")
                body.extend(chunk)
            headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.lower()
                not in {
                    "content-encoding",
                    "content-length",
                }
            ]
            return httpx2.Response(
                response.status_code, headers=headers, content=bytes(body), request=request
            )
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self.transport.aclose()


def calls(request: ModelRequest) -> dict[str, ToolCall]:
    return {call.call_id: call for message in request.messages for call in message.tool_calls}


def openai_messages(request: ModelRequest) -> list[JSONValue]:
    result: list[JSONValue] = [{"role": "system", "content": request.instructions}]
    known = calls(request)
    for message in request.messages:
        item: dict[str, JSONValue] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.provider_call_id or call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": canonical(call.arguments).decode(),
                    },
                }
                for call in message.tool_calls
            ]
        if message.observation:
            call = known.get(message.observation.call_id)
            if call is None:
                raise ModelFailure("provider_protocol_error")
            item["tool_call_id"] = call.provider_call_id or call.call_id
            item["content"] = canonical(message.observation).decode()
        result.append(item)
    return result


def native_calls(items: JSONValue, provider: str) -> tuple[ToolCall, ...]:
    if not isinstance(items, list):
        raise ModelFailure("provider_protocol_error")
    result = []
    for ordinal, item in enumerate(items):
        if not isinstance(item, dict):
            raise ModelFailure("provider_protocol_error")
        function = item.get("function") if provider == "openai" else item
        if not isinstance(function, dict):
            raise ModelFailure("provider_protocol_error")
        arguments = (
            function.get("arguments")
            if provider == "openai"
            else (function.get("input") if provider == "anthropic" else function.get("args"))
        )
        if isinstance(arguments, str):
            arguments = decode_json(arguments)
        native_id = item.get("id")
        if provider in {"openai", "anthropic"} and (
            not isinstance(native_id, str) or not native_id
        ):
            raise ModelFailure("provider_protocol_error")
        try:
            result.append(
                ToolCall.model_validate(
                    {
                        "call_id": f"part_{ordinal}",
                        "provider_call_id": native_id,
                        "name": function.get("name"),
                        "arguments": arguments,
                    }
                )
            )
        except ValueError:
            raise ModelFailure("provider_protocol_error") from None
    return tuple(result)


class HostedAdapter(ModelAdapter):
    provider_kind: ClassVar[str | None] = None

    def __init__(self, profile: ProviderConfig, config: Config) -> None:
        if self.provider_kind is not None and profile.kind != self.provider_kind:
            raise LoomConfigurationError("invalid_provider_kind")
        validate_url(profile.url)
        self.profile, self.config = profile, config
        default_env = {
            "openai": "OPENAI_API_KEY",
            "azure_openai": "AZURE_OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GEMINI_API_KEY",
        }
        if profile.kind not in default_env:
            raise LoomConfigurationError("invalid_provider_kind")
        self.key = os.environ.get(profile.api_key_env or default_env[profile.kind])
        if not self.key:
            raise LoomConfigurationError("provider_credentials_unavailable")
        self.http = httpx2.AsyncClient(
            transport=BoundedTransport(config), trust_env=False, follow_redirects=False
        )
        self.client: openai.AsyncOpenAI | anthropic.AsyncAnthropic | genai.Client
        if profile.kind == "azure_openai":
            assert profile.endpoint is not None
            self.client = openai.AsyncAzureOpenAI(
                api_key=self.key,
                azure_endpoint=profile.endpoint,
                api_version=profile.api_version,
                max_retries=0,
                http_client=self.http,
            )
        elif profile.kind == "openai":
            self.client = openai.AsyncOpenAI(
                api_key=self.key,
                base_url=profile.url,
                max_retries=0,
                http_client=self.http,
            )
        elif profile.kind == "anthropic":
            self.client = anthropic.AsyncAnthropic(
                api_key=self.key,
                base_url=profile.url,
                max_retries=0,
                http_client=self.http,
            )
        elif profile.kind == "gemini":
            self.client = genai.Client(
                api_key=self.key,
                vertexai=False,
                http_options=types.HttpOptions(
                    httpx_async_client=self.http, retry_options=types.HttpRetryOptions(attempts=1)
                ),
            )
        else:
            raise LoomConfigurationError("invalid_provider_kind")

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            native_tool_calling=self.profile.native_tool_calling,
            structured_output=self.profile.structured_output,
            local=False,
        )

    async def generate(self, request: ModelRequest) -> ModelResponse:
        try:
            if isinstance(self.client, openai.AsyncOpenAI):
                return await self._openai(request, self.client)
            if isinstance(self.client, anthropic.AsyncAnthropic):
                return await self._anthropic(request, self.client)
            return await self._gemini(request, self.client)
        except (openai.APIStatusError, anthropic.APIStatusError, errors.APIError) as error:
            status = getattr(error, "status_code", getattr(error, "code", None))
            code = "provider_http_error"
            if status in {401, 403}:
                code = "provider_authentication_failed"
            message = str(error).lower()
            if status in {400, 413} and any(
                word in message for word in ("context", "token limit", "too long")
            ):
                code = "context_limit"
            response = getattr(error, "response", None)
            raise ModelFailure(
                code,
                transient=status in {408, 429} or isinstance(status, int) and status >= 500,
                retry_after=retry_after(response.headers.get("retry-after"))
                if response is not None
                else None,
            ) from None
        except (openai.APITimeoutError, anthropic.APITimeoutError, httpx2.TimeoutException):
            raise ModelFailure("model_timeout", transient=True) from None
        except (
            openai.APIConnectionError,
            anthropic.APIConnectionError,
            httpx2.TransportError,
        ) as error:
            cause: BaseException | None = error
            visited: set[int] = set()
            while cause is not None and id(cause) not in visited:
                if isinstance(cause, ModelFailure):
                    raise cause from None
                visited.add(id(cause))
                cause = cause.__cause__ or cause.__context__
            raise ModelFailure("provider_transport_error", transient=True) from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise ModelFailure("provider_protocol_error") from None

    async def _openai(self, request: ModelRequest, client: openai.AsyncOpenAI) -> ModelResponse:
        body: dict[str, JSONValue] = {
            "model": self.profile.model,
            "messages": openai_messages(request),
            "stream": False,
            "max_completion_tokens": request.max_output_tokens,
        }
        if request.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.argument_schema,
                    },
                }
                for tool in request.tools
            ]
        if request.output_schema is not None and request.output_mode != "prompt_json":
            body["response_format"] = (
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "loom_output",
                        "strict": True,
                        "schema": request.output_schema,
                    },
                }
                if request.output_mode == "json_schema"
                else {"type": "json_object"}
            )
        response = await client.chat.completions.create(
            **cast(CompletionCreateParamsNonStreaming, body), timeout=request.timeout_seconds
        )
        choice = response.choices[0]
        if choice.message.refusal or choice.finish_reason in {"content_filter", "length"}:
            return ModelResponse(
                finish_reason="refused"
                if choice.message.refusal or choice.finish_reason == "content_filter"
                else "output_limit",
                input_tokens=response.usage.prompt_tokens if response.usage else None,
                output_tokens=response.usage.completion_tokens if response.usage else None,
            )
        raw = json_value(choice.message)
        if not isinstance(raw, dict):
            raise ModelFailure("provider_protocol_error")
        tool_calls = native_calls(raw.get("tool_calls") or [], "openai")
        if choice.finish_reason not in {"stop", "tool_calls"} or (
            choice.finish_reason == "tool_calls" and not tool_calls
        ):
            raise ModelFailure("provider_protocol_error")
        return ModelResponse(
            output=choice.message.content,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "completed",
            input_tokens=response.usage.prompt_tokens if response.usage else None,
            output_tokens=response.usage.completion_tokens if response.usage else None,
        )

    async def _anthropic(
        self, request: ModelRequest, client: anthropic.AsyncAnthropic
    ) -> ModelResponse:
        messages: list[JSONValue] = []
        known = calls(request)
        for message in request.messages:
            content: JSONValue = message.content
            role = "assistant" if message.role == "assistant" else "user"
            if message.tool_calls:
                content = message.continuation.get("anthropic_content") or [
                    {
                        "type": "tool_use",
                        "id": call.provider_call_id or call.call_id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                    for call in message.tool_calls
                ]
            if message.observation:
                call = known[message.observation.call_id]
                content = [
                    {
                        "type": "tool_result",
                        "tool_use_id": call.provider_call_id or call.call_id,
                        "content": canonical(message.observation).decode(),
                        "is_error": message.observation.error is not None,
                    }
                ]
            messages.append({"role": role, "content": content})
        body: dict[str, JSONValue] = {
            "model": self.profile.model,
            "system": request.instructions,
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "stream": False,
        }
        if request.tools:
            body["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.argument_schema,
                }
                for tool in request.tools
            ]
        if request.output_schema is not None and request.output_mode == "json_schema":
            body["output_config"] = {
                "format": {"type": "json_schema", "schema": request.output_schema}
            }
        response = await client.messages.create(
            **cast(MessageCreateParamsNonStreaming, body), timeout=request.timeout_seconds
        )
        if response.stop_reason in {"max_tokens", "refusal"}:
            return ModelResponse(
                finish_reason="output_limit" if response.stop_reason == "max_tokens" else "refused",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        raw = json_value(response)
        if not isinstance(raw, dict):
            raise ModelFailure("provider_protocol_error")
        content = raw["content"]
        if not isinstance(content, list):
            raise ModelFailure("provider_protocol_error")
        tool_calls = native_calls(
            [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ],
            "anthropic",
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason not in {"end_turn", "stop_sequence", "tool_use"} or (
            response.stop_reason == "tool_use" and not tool_calls
        ):
            raise ModelFailure("provider_protocol_error")
        return ModelResponse(
            output=text,
            tool_calls=tool_calls,
            finish_reason="tool_calls" if tool_calls else "completed",
            continuation={"anthropic_content": content},
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    async def _gemini(self, request: ModelRequest, client: genai.Client) -> ModelResponse:
        contents: list[types.ContentUnion] = []
        known = calls(request)
        for message in request.messages:
            parts: JSONValue = [{"text": message.content}]
            role = "model" if message.role == "assistant" else "user"
            if message.tool_calls:
                parts = message.continuation.get("gemini_parts") or [
                    {"function_call": {"name": call.name, "args": call.arguments}}
                    for call in message.tool_calls
                ]
            if message.observation:
                call = known[message.observation.call_id]
                parts = [
                    {
                        "function_response": {
                            "name": call.name,
                            "id": call.provider_call_id,
                            "response": json_value(message.observation),
                        }
                    }
                ]
            contents.append(
                types.Content.model_validate_json(canonical({"role": role, "parts": parts}))
            )
        configuration: dict[str, JSONValue] = {
            "system_instruction": request.instructions,
            "max_output_tokens": request.max_output_tokens,
            "http_options": {
                "timeout": max(1, int(request.timeout_seconds * 1000)),
                "retry_options": {"attempts": 1},
            },
            "automatic_function_calling": {"disable": True},
        }
        if request.tools:
            configuration["tools"] = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters_json_schema": tool.argument_schema,
                        }
                        for tool in request.tools
                    ]
                }
            ]
        if request.output_schema is not None and request.output_mode != "prompt_json":
            configuration["response_mime_type"] = "application/json"
            if request.output_mode == "json_schema":
                configuration["response_json_schema"] = request.output_schema
        response = await client.aio.models.generate_content(
            model=self.profile.model,
            contents=[*contents],
            config=types.GenerateContentConfig.model_validate_json(canonical(configuration)),
        )
        if not response.candidates:
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise ModelFailure("model_refusal")
            raise ModelFailure("provider_protocol_error")
        candidate = response.candidates[0]
        if candidate.finish_reason is None:
            raise ModelFailure("provider_protocol_error")
        if candidate.finish_reason != types.FinishReason.STOP:
            return ModelResponse(
                finish_reason="output_limit"
                if candidate.finish_reason == types.FinishReason.MAX_TOKENS
                else "refused",
                input_tokens=response.usage_metadata.prompt_token_count
                if response.usage_metadata
                else None,
                output_tokens=response.usage_metadata.candidates_token_count
                if response.usage_metadata
                else None,
            )
        if candidate.content is None or candidate.content.parts is None:
            raise ModelFailure("provider_protocol_error")
        parts = [json_value(part) for part in candidate.content.parts if not part.thought]
        tool_calls = native_calls(
            [
                part["function_call"]
                for part in parts
                if isinstance(part, dict) and part.get("function_call")
            ],
            "gemini",
        )
        output = "".join(part.text or "" for part in candidate.content.parts if not part.thought)
        return ModelResponse(
            output=output,
            tool_calls=tool_calls,
            continuation={"gemini_parts": parts},
            finish_reason="tool_calls" if tool_calls else "completed",
            input_tokens=response.usage_metadata.prompt_token_count
            if response.usage_metadata
            else None,
            output_tokens=response.usage_metadata.candidates_token_count
            if response.usage_metadata
            else None,
        )

    async def aclose(self) -> None:
        if isinstance(self.client, genai.Client):
            await self.client.aio.aclose()
            self.client.close()
        else:
            await self.client.close()
        await self.http.aclose()
