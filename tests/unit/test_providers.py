import json

import httpx
import httpx2
import pytest

from loom import Agent, Limits, Tool
from loom.config import Config, Defaults, ProviderConfig
from loom.providers.hosted import BoundedTransport


@pytest.fixture
def hosted_agent(monkeypatch):
    monkeypatch.setenv("PROVIDER_TEST_KEY", "test-only")

    def create(kind, handler):
        def transport_init(self, config):
            self.transport = httpx2.MockTransport(handler)

        monkeypatch.setattr(BoundedTransport, "__init__", transport_init)
        fields = dict(
            kind=kind, model="test", api_key_env="PROVIDER_TEST_KEY", native_tool_calling=True
        )
        if kind == "azure_openai":
            fields.update(endpoint="https://azure.example.com", api_version="2025-04-01-preview")
        return Agent(
            name="test",
            instructions="Answer.",
            config=Config(
                defaults=Defaults(provider="test"),
                providers={"test": ProviderConfig(**fields)},
            ),
        )

    return create


@pytest.mark.parametrize("kind", ["openai", "azure_openai", "anthropic", "gemini"])
def test_sdk_wrapped_oversized_responses_are_not_retried(kind, hosted_agent):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(200, content=b"x" * 262145)

    result = hosted_agent(kind, handler).run(input="Hello")
    assert result.status == "failed"
    assert result.error.code == "provider_response_too_large"
    assert result.usage.model_requests == len(requests) == 1


@pytest.mark.parametrize("kind", ["openai", "azure_openai", "anthropic", "gemini"])
def test_truncation_precedes_parsing_partial_tool_arguments(kind, hosted_agent):
    if kind in {"openai", "azure_openai"}:
        payload = {
            "id": "chat-1",
            "object": "chat.completion",
            "created": 1,
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "partial",
                                "type": "function",
                                "function": {"name": "partial", "arguments": '{"a":'},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }
    elif kind == "anthropic":
        payload = {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [{"type": "tool_use", "id": "partial", "name": "partial", "input": "bad"}],
            "stop_reason": "max_tokens",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 4},
        }
    else:
        payload = {
            "candidates": [{"finishReason": "MAX_TOKENS"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4},
        }
    result = hosted_agent(kind, lambda request: httpx2.Response(200, json=payload)).run(input="Hi")
    assert result.status == "failed"
    assert result.error.code == "output_truncated"
    assert result.usage.tool_attempts == 0 and result.usage.model_requests == 1
    assert result.usage.input_tokens == 3 and result.usage.output_tokens == 4


def test_gemini_safety_refusal_may_omit_content(hosted_agent):
    payload = {"candidates": [{"finishReason": "SAFETY"}]}
    result = hosted_agent("gemini", lambda request: httpx2.Response(200, json=payload)).run(
        input="Hi"
    )
    assert result.error.code == "model_refusal"
    assert result.usage.model_requests == 1


@pytest.mark.parametrize("kind", ["openai", "azure_openai", "anthropic", "gemini"])
def test_malformed_sdk_payload_is_a_protocol_failure(kind, hosted_agent):
    result = hosted_agent(kind, lambda request: httpx2.Response(200, json={})).run(input="Hi")
    assert result.status == "failed"
    assert result.error.code == "provider_protocol_error"
    assert result.usage.model_requests == 1


@pytest.mark.parametrize(
    "value,expected",
    [(None, None), ("", None), ("-1", None), ("nan", None), ("inf", None), ("0", 0), ("1.5", 1.5)],
)
def test_retry_after_requires_finite_nonnegative_delay(value, expected):
    from loom.security.transport import retry_after

    assert retry_after(value) == expected


def test_retry_after_supports_http_dates():
    from email.utils import formatdate

    from loom.security.transport import retry_after

    assert retry_after(formatdate(120, usegmt=True), now=100) == 20
    assert retry_after(formatdate(80, usegmt=True), now=100) == 0


@pytest.mark.parametrize("kind", ["openai", "azure_openai", "anthropic", "gemini"])
def test_native_tool_round_trip_uses_verified_transport_without_hidden_retries(kind, monkeypatch):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        first = len(requests) == 1
        if kind in {"openai", "azure_openai"}:
            message = {"role": "assistant", "content": None if first else "3"}
            if first:
                message["tool_calls"] = [
                    {
                        "id": "native-id",
                        "type": "function",
                        "function": {"name": "add", "arguments": '{"a":1,"b":2}'},
                    }
                ]
            payload = {
                "id": "chat-1",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if first else "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        elif kind == "anthropic":
            payload = {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "native-id",
                        "name": "add",
                        "input": {"a": 1, "b": 2},
                    }
                ]
                if first
                else [{"type": "text", "text": "3"}],
                "stop_reason": "tool_use" if first else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        else:
            payload = {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "add",
                                        "args": {"a": 1, "b": 2},
                                        "id": "native-id",
                                    }
                                }
                            ]
                            if first
                            else [{"text": "3"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 5,
                    "totalTokenCount": 15,
                },
            }
        return httpx2.Response(200, json=payload)

    def transport_init(self, config):
        self.transport = httpx2.MockTransport(handler)

    monkeypatch.setattr(BoundedTransport, "__init__", transport_init)
    monkeypatch.setenv("PROVIDER_TEST_KEY", "test-only")
    profile = {
        "kind": kind,
        "model": "test",
        "api_key_env": "PROVIDER_TEST_KEY",
        "native_tool_calling": True,
    }
    if kind == "azure_openai":
        profile.update(endpoint="https://azure.example.com", api_version="2025-04-01-preview")
    config = Config(
        defaults=Defaults(provider="test"), providers={"test": ProviderConfig(**profile)}
    )

    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    result = Agent(
        name="test",
        instructions="Use add.",
        config=config,
        tools=[Tool.from_callable(add, sensitive=False, egress="none")],
    ).run(input="1 + 2")
    assert result.status == "completed", result
    assert result.output == "3" and result.usage.model_requests == len(requests) == 2
    assert result.usage.tool_attempts == 1
    assert result.usage.input_tokens == 20 and result.usage.output_tokens == 10
    if kind in {"openai", "azure_openai"}:
        assert requests[1]["messages"][-1]["tool_call_id"] == "native-id"
        assert requests[0]["max_completion_tokens"] == 2048
    elif kind == "anthropic":
        assert requests[1]["messages"][-1]["content"][0]["tool_use_id"] == "native-id"
        assert requests[0]["max_tokens"] == 2048
    else:
        assert requests[0]["generationConfig"]["maxOutputTokens"] == 2048


def test_ollama_pins_and_explicit_overflow_controls(monkeypatch):
    import loom.providers.ollama as ollama

    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path == "/api/version":
            payload = {"version": "0.40.2"}
        elif request.url.path == "/api/tags":
            payload = {"models": [{"name": "qwen2.5:7b", "digest": "sha256:" + "a" * 64}]}
        elif request.url.path == "/api/show":
            payload = {"model_info": {"qwen2.context_length": 32768}}
        else:
            payload = {
                "done": True,
                "done_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
                "prompt_eval_count": 1,
                "eval_count": 1,
            }
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        ollama,
        "client",
        lambda config, timeout: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    config = Config(
        defaults=Defaults(provider="local"),
        providers={
            "local": ProviderConfig(
                kind="ollama",
                model="qwen2.5:7b",
                model_digest="a" * 64,
                server_version="0.40.2",
                native_tool_calling=True,
                num_ctx=32768,
            )
        },
    )
    result = Agent(
        name="test",
        instructions="Answer.",
        config=config,
        limits=Limits(max_context_bytes=24576),
    ).run(input="Hello")
    assert result.status == "completed", result
    body = json.loads(requests[-1].content)
    assert body["truncate"] is False and body["shift"] is False and body["stream"] is False
    assert body["options"] == {"num_ctx": 32768, "num_predict": 2048, "num_gpu": 0, "num_thread": 4}
    assert len(requests) == 4
