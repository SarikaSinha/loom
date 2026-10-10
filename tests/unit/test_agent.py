import asyncio
import threading

import pytest

from loom import Agent, Limits, LoomConfigurationError, Tool
from loom.contracts import ToolCall
from loom.providers.base import ModelResponse
from loom.testing import ScriptedModel


def requested(name, **arguments):
    return ModelResponse(
        tool_calls=(
            ToolCall(call_id="native", provider_call_id="native", name=name, arguments=arguments),
        ),
        finish_reason="tool_calls",
        input_tokens=1,
        output_tokens=1,
    )


def agent(model, **kwargs):
    return Agent(name="test", instructions="Do the requested task.", model=model, **kwargs)


def test_text_completion_and_explicit_continuation():
    model = ScriptedModel(lambda request: ModelResponse(output="ok"))
    runner = agent(model)
    first = runner.run(input="one")
    second = runner.run(input="two")
    continued = runner.run(input="three", state=first.state)
    assert all(result.status == "completed" for result in (first, second, continued))
    assert first.conversation_id != second.conversation_id
    assert first.conversation_id == continued.conversation_id
    assert len(model.requests[0].messages) == len(model.requests[1].messages) == 1
    assert len(model.requests[2].messages) == 3
    assert "state" not in first.to_dict()


def test_function_dispatch_and_strict_validation():
    invocations = []

    def add(a: int, b: int) -> int:
        """Add two integers."""
        invocations.append((a, b))
        return a + b

    model = ScriptedModel(
        [requested("add", a="1", b=2), requested("add", a=1, b=2), ModelResponse(output="3")]
    )
    result = agent(model, tools=[Tool.from_callable(add, sensitive=False, egress="none")]).run(
        input="add"
    )
    assert result.status == "completed"
    assert result.usage.tool_attempts == 1
    assert invocations == [(1, 2)]
    assert model.requests[1].messages[-1].observation.error.code == "invalid_tool_arguments"
    assert model.requests[2].messages[-1].observation.output == 3


def test_headless_sensitive_tool_is_denied(monkeypatch):
    def effect() -> str:
        """A sensitive operation."""
        pytest.fail("must not execute")

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    model = ScriptedModel([requested("effect"), ModelResponse(output="Denied")])
    result = agent(model, tools=[Tool.from_callable(effect)]).run(input="do it")
    assert result.status == "completed"
    assert result.usage.tool_attempts == 0
    assert model.requests[1].messages[-1].observation.error.code == "approval_denied"


def test_schema_correction_is_bounded():
    model = ScriptedModel(
        [ModelResponse(output='{"value":"wrong"}'), ModelResponse(output='{"value":2}')]
    )
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    result = agent(model, output_schema=schema).run(input="produce a value")
    assert result.status == "completed" and result.output == {"value": 2}
    assert result.usage.corrections == 1
    model = ScriptedModel([ModelResponse(output="invalid")])
    result = agent(model, output_schema=schema, limits=Limits(max_corrections=0)).run(input="value")
    assert result.status == "failed" and result.error.code == "output_validation_failed"
    assert len(model.requests) == 1


def test_oversized_success_does_not_repeat_unsafe_effect():
    invoked = []

    def write() -> str:
        """Write and return a large response."""
        invoked.append(1)
        return "x" * 20000

    model = ScriptedModel([requested("write"), requested("write"), ModelResponse(output="done")])
    result = agent(model, tools=[Tool.from_callable(write, sensitive=False, egress="none")]).run(
        input="write"
    )
    assert result.status == "completed" and invoked == [1]
    observation = model.requests[1].messages[-1].observation
    assert observation.execution_status == "succeeded"
    assert observation.error.code == "observation_too_large"
    assert model.requests[2].messages[-1].observation.error.code == "duplicate_effect_prevented"


async def test_sync_facade_rejects_running_loop():
    with pytest.raises(LoomConfigurationError, match="asynchronous"):
        agent(ScriptedModel([ModelResponse(output="ok")])).run(input="x")


async def test_stuck_tool_returns_busy_without_charging_later_run():
    entered = threading.Event()
    release = threading.Event()

    def slow() -> str:
        """Block until released by the test."""
        entered.set()
        release.wait(5)
        return "done"

    tool = Tool.from_callable(slow, sensitive=False, egress="none")
    try:
        first = await agent(
            ScriptedModel([requested("slow")]),
            tools=[tool],
            limits=Limits(tool_timeout_seconds=0.02),
        ).arun(input="start")
        assert entered.is_set() and first.error.code == "unknown_tool_outcome"
        model = ScriptedModel([requested("slow"), ModelResponse(output="busy")])
        second = await agent(model, tools=[tool]).arun(input="again")
        observation = model.requests[1].messages[-1].observation
        assert second.usage.tool_attempts == 0
        assert observation.error.code == "tool_busy"
        assert observation.error.details["blocking_run_id"] == first.run_id
    finally:
        release.set()
        for _ in range(100):
            if tool._claim is None:
                break
            await asyncio.sleep(0.01)
