import os

import pytest

from loom import Agent, Config, Tool
from loom.contracts import Message
from loom.providers.base import ModelFailure, ModelRequest
from loom.providers.factory import create_adapter

pytestmark = pytest.mark.live


def configured_provider(kind):
    path = os.environ.get("LOOM_LIVE_CONFIG")
    assert path, "LOOM_LIVE_CONFIG must identify explicitly provisioned provider profiles"
    config = Config.load(path)
    profiles = [name for name, profile in config.providers.items() if profile.kind == kind]
    assert len(profiles) == 1, f"Exactly one {kind} profile is required for live acceptance"
    return config, profiles[0]


@pytest.mark.parametrize("kind", ["openai", "azure_openai", "anthropic", "gemini", "ollama"])
def test_live_native_tool_call_and_structured_answer(kind):
    config, provider = configured_provider(kind)
    calls = []

    def add(a: int, b: int) -> int:
        """Add two integers and return their sum."""
        calls.append((a, b))
        return a + b

    result = Agent(
        name="acceptance",
        instructions="Call the add tool exactly once, then return its result as JSON.",
        model=provider,
        config=config,
        tools=[Tool.from_callable(add, sensitive=False, retry_safe=True, egress="none")],
        output_schema={
            "type": "object",
            "properties": {"sum": {"type": "integer"}},
            "required": ["sum"],
            "additionalProperties": False,
        },
        limits={
            "deadline_seconds": 600,
            "model_timeout_seconds": 180,
            "max_context_bytes": 24576,
            "max_model_turns": 3,
            "max_retries": 0,
        },
    ).run(input='Use add with a=2 and b=3. Return the result as {"sum": 5}.')
    assert result.status == "completed", result.error
    assert result.output == {"sum": 5}
    assert calls == [(2, 3)]


@pytest.mark.parametrize("early_history", [False, True])
async def test_live_ollama_rejects_overflow_without_silent_trimming(early_history):
    config, provider = configured_provider("ollama")
    adapter = create_adapter(config.providers[provider], config)
    oversized = "overflow " * 70000
    messages = (
        (
            Message(role="user", content=oversized),
            Message(role="assistant", content="Retain that full context."),
            Message(role="user", content="Reply with one word."),
        )
        if early_history
        else (Message(role="user", content=oversized),)
    )
    try:
        with pytest.raises(ModelFailure) as failure:
            await adapter.generate(
                ModelRequest(
                    instructions="Never omit earlier context.",
                    messages=messages,
                    tools=(),
                    max_output_tokens=1,
                    timeout_seconds=180,
                )
            )
        assert failure.value.code == "context_limit"
    finally:
        await adapter.aclose()
