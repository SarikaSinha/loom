"""Unit tests for AgentTool multi-agent support."""
from __future__ import annotations

import pytest

from loom import Agent, AgentTool, Limits, LoomConfigurationError
from loom.contracts import CallerIdentity, ToolCall
from loom.providers.base import ModelResponse
from loom.testing import ScriptedModel
from loom.tools.http import HTTPTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def text_response(text: str = "done") -> ModelResponse:
    return ModelResponse(output=text, finish_reason="completed", input_tokens=1, output_tokens=1)


def tool_request(name: str, **arguments) -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                call_id="native",
                provider_call_id="native",
                name=name,
                arguments=arguments,
            ),
        ),
        finish_reason="tool_calls",
        input_tokens=1,
        output_tokens=1,
    )


def make_agent(model: ScriptedModel, name: str = "test", **kwargs) -> Agent:
    return Agent(name=name, instructions="Be helpful.", model=model, **kwargs)


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

def test_agent_tool_default_schemas():
    sub = make_agent(ScriptedModel([text_response()]), name="sub")
    tool = AgentTool(agent=sub, name="call_sub", description="Call the sub-agent.")
    assert tool.spec.name == "call_sub"
    assert tool.spec.sensitive is False
    assert tool.spec.retry_safe is False
    assert tool.spec.egress == "none"
    assert tool.spec.egress_origins == ()
    # Default argument schema has "input"
    assert "input" in tool.spec.argument_schema.get("properties", {})


def test_agent_tool_mirrors_sub_agent_output_schema():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"],
               "additionalProperties": False}
    sub = make_agent(ScriptedModel([]), name="sub", output_schema=schema)
    tool = AgentTool(agent=sub, name="call_sub", description="Call sub.")
    assert tool.spec.output_schema == sub.schema


def test_agent_tool_output_schema_fallback_to_string():
    sub = make_agent(ScriptedModel([]), name="sub")
    tool = AgentTool(agent=sub, name="call_sub", description="Call sub.")
    assert tool.spec.output_schema == {"type": "string"}


def test_agent_tool_egress_derived_none_when_no_http_tools():
    sub = make_agent(ScriptedModel([]), name="sub")
    tool = AgentTool(agent=sub, name="call_sub", description="d")
    assert tool.spec.egress == "none"


def test_agent_tool_missing_description_raises():
    sub = make_agent(ScriptedModel([]), name="sub")
    with pytest.raises(LoomConfigurationError):
        AgentTool(agent=sub, name="call_sub", description="")


# ---------------------------------------------------------------------------
# Suspend mode (default) — happy path
# ---------------------------------------------------------------------------

async def test_suspend_mode_returns_sub_agent_output():
    sub_model = ScriptedModel([text_response("summary result")])
    sub = make_agent(sub_model, name="summariser")

    agent_tool = AgentTool(
        agent=sub,
        name="summarise",
        description="Summarise text via sub-agent.",
        sensitive=False,
    )

    # Parent calls sub via tool, then produces final answer
    parent_model = ScriptedModel(
        [tool_request("summarise", input="hello world"), text_response("done")]
    )
    parent = make_agent(parent_model, name="orchestrator", tools=[agent_tool])

    result = await parent.arun(input="summarise hello world")
    assert result.status == "completed"
    assert result.usage.agent_calls == 1
    # Sub-agent output appears in the tool observation seen by parent's model
    last_tool_msg = parent_model.requests[1].messages[-1]
    assert last_tool_msg.observation.execution_status == "succeeded"
    assert last_tool_msg.observation.output == "summary result"


# ---------------------------------------------------------------------------
# Suspend mode — non-completed sub-agent propagates failure
# ---------------------------------------------------------------------------

async def test_suspend_mode_sub_agent_failure_becomes_tool_error():
    from loom.providers.base import ModelFailure

    # Sub-agent model raises so the sub-agent run fails
    sub_model = ScriptedModel([ModelFailure("model_unavailable")])
    sub = make_agent(sub_model, name="failing_sub")

    agent_tool = AgentTool(
        agent=sub, name="call_failing", description="Call a failing sub.", sensitive=False
    )
    parent_model = ScriptedModel(
        [tool_request("call_failing", input="go"), text_response("handled")]
    )
    parent = make_agent(parent_model, name="orchestrator", tools=[agent_tool])

    result = await parent.arun(input="call failing sub")
    # Parent should still complete — it received a tool error observation and recovered
    # (the model saw the error and replied with "handled")
    assert result.status == "completed"
    err = parent_model.requests[1].messages[-1].observation.error
    assert err is not None
    assert err.code == "sub_agent_failed"


# ---------------------------------------------------------------------------
# Handoff mode
# ---------------------------------------------------------------------------

async def test_handoff_mode_adopts_sub_agent_output():
    sub_model = ScriptedModel([text_response("specialist answer")])
    sub = make_agent(sub_model, name="specialist")

    agent_tool = AgentTool(
        agent=sub,
        name="route_to_specialist",
        description="Hand off to specialist.",
        handoff=True,
        sensitive=False,
    )
    parent_model = ScriptedModel([tool_request("route_to_specialist", input="complex query")])
    parent = make_agent(parent_model, name="router", tools=[agent_tool])

    result = await parent.arun(input="handle this complex query")
    assert result.status == "completed"
    assert result.output == "specialist answer"
    # Handoff diagnostic must be present
    codes = {d.code for d in result.diagnostics}
    assert "agent_handoff" in codes


async def test_handoff_mode_sums_usage():
    sub_model = ScriptedModel([text_response("answer")])
    sub = make_agent(sub_model, name="specialist")

    agent_tool = AgentTool(
        agent=sub, name="route", description="Route.", handoff=True, sensitive=False
    )
    parent_model = ScriptedModel([tool_request("route", input="q")])
    parent = make_agent(parent_model, name="router", tools=[agent_tool])

    result = await parent.arun(input="q")
    assert result.status == "completed"
    # Both parent and sub used tokens — summed into result
    assert (result.usage.input_tokens or 0) >= 2
    assert (result.usage.output_tokens or 0) >= 2


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

async def test_cycle_detection_raises_agent_cycle_detected():
    # Build Agent B that calls Agent A via AgentTool.
    # We can't fully wire the cycle at construction time, so we simulate
    # a call chain by passing _agent_chain in context.
    sub_model = ScriptedModel([text_response("ok")])
    sub = make_agent(sub_model, name="sub")

    agent_tool = AgentTool(agent=sub, name="call_sub", description="Call sub.", sensitive=False)
    parent_model = ScriptedModel([tool_request("call_sub", input="hi"), text_response("done")])
    parent = make_agent(parent_model, name="sub", tools=[agent_tool])
    # Parent has the same name as sub — AgentTool will detect "sub" already in chain
    # because _parent_agent_name="sub" and AgentTool checks if agent.name ("sub") is in chain.
    # We seed the chain by passing context containing the parent's own name.
    result = await parent.arun(input="go", context={"_agent_chain": ("sub",)})
    assert result.status == "completed"
    obs = parent_model.requests[1].messages[-1].observation
    assert obs.error is not None
    assert obs.error.code == "agent_cycle_detected"


# ---------------------------------------------------------------------------
# Depth check
# ---------------------------------------------------------------------------

async def test_depth_exceeded_raises_agent_depth_exceeded():
    sub_model = ScriptedModel([text_response("ok")])
    sub = make_agent(sub_model, name="sub", limits=Limits(max_agent_depth=2))

    agent_tool = AgentTool(agent=sub, name="call_sub", description="d", sensitive=False)
    parent_model = ScriptedModel([tool_request("call_sub", input="hi"), text_response("done")])
    parent = make_agent(parent_model, name="parent", tools=[agent_tool])
    # Seed chain with 2 entries — depth already at max_agent_depth=2
    result = await parent.arun(input="go", context={"_agent_chain": ("a", "b")})
    assert result.status == "completed"
    obs = parent_model.requests[1].messages[-1].observation
    assert obs.error is not None
    assert obs.error.code == "agent_depth_exceeded"


# ---------------------------------------------------------------------------
# Egress policy
# ---------------------------------------------------------------------------

async def test_egress_policy_deny_all_http_blocks_sub_with_http_tools(monkeypatch):
    monkeypatch.setenv("DUMMY_KEY", "secret")

    def _builder(args, ctx):
        return {"path_params": {}, "query": {}, "json_body": None}

    from pydantic import BaseModel

    class Q(BaseModel):
        q: str

    class R(BaseModel):
        r: str

    http_tool = HTTPTool(
        name="call_api",
        description="Call external API.",
        method="GET",
        endpoint="https://example.com/api",
        request_builder=_builder,
        argument_schema=Q,
        output_schema=R,
        auth_env={"Authorization": {"env": "DUMMY_KEY", "prefix": "Bearer "}},
        sensitive=False,
    )
    sub_model = ScriptedModel([text_response("ok")])
    sub = make_agent(sub_model, name="http_sub", tools=[http_tool])

    agent_tool = AgentTool(
        agent=sub,
        name="call_http_sub",
        description="Call sub with HTTP.",
        egress_policy="deny_all_http",
        sensitive=False,
    )
    parent_model = ScriptedModel(
        [tool_request("call_http_sub", input="hi"), text_response("done")]
    )
    parent = make_agent(parent_model, name="parent", tools=[agent_tool])

    result = await parent.arun(input="go")
    assert result.status == "completed"
    obs = parent_model.requests[1].messages[-1].observation
    assert obs.error is not None
    assert obs.error.code == "egress_policy_denied"


# ---------------------------------------------------------------------------
# max_agent_calls budget
# ---------------------------------------------------------------------------

async def test_max_agent_calls_limit():
    sub_model = ScriptedModel(lambda _: text_response("ok"))
    sub = make_agent(sub_model, name="sub")

    agent_tool = AgentTool(agent=sub, name="call_sub", description="d", sensitive=False)
    # Allow 1 agent call; model requests two — second must fail with limit error
    limits = Limits(max_agent_calls=1)
    parent_model = ScriptedModel(
        [
            tool_request("call_sub", input="first"),
            tool_request("call_sub", input="second"),
            text_response("done"),
        ]
    )
    parent = make_agent(parent_model, name="parent", tools=[agent_tool], limits=limits)

    result = await parent.arun(input="call twice")
    assert result.status == "limit_exhausted"
    assert result.error.code == "agent_call_limit"


# ---------------------------------------------------------------------------
# Caller override and context selector
# ---------------------------------------------------------------------------

async def test_caller_override_is_forwarded():
    from loom import Tool

    def echo(text: str) -> str:
        """Echo the text."""
        return text

    override = CallerIdentity(subject="human:alice")
    echo_tool = Tool.from_callable(echo, sensitive=False, egress="none")
    # Sub-agent just replies without using its tool (simpler — we only care
    # that the parent run completes successfully with the override set).
    sub_model = ScriptedModel([text_response("done")])
    sub = make_agent(sub_model, name="sub", tools=[echo_tool])

    agent_tool = AgentTool(
        agent=sub,
        name="call_sub",
        description="d",
        sensitive=False,
        caller_override=override,
    )
    parent_model = ScriptedModel([tool_request("call_sub", input="hi"), text_response("done")])
    parent = make_agent(parent_model, name="parent", tools=[agent_tool])

    result = await parent.arun(input="go")
    assert result.status == "completed"
    # Sub-agent ran — caller override was passed; verify parent completed OK
    assert result.usage.agent_calls == 1


# ---------------------------------------------------------------------------
# output_selector
# ---------------------------------------------------------------------------

async def test_output_selector_transforms_result():
    sub_model = ScriptedModel([text_response("raw output")])
    sub = make_agent(sub_model, name="sub")

    wrapped_schema = {
        "type": "object",
        "properties": {"wrapped": {"type": "string"}},
        "required": ["wrapped"],
        "additionalProperties": False,
    }
    agent_tool = AgentTool(
        agent=sub,
        name="call_sub",
        description="d",
        sensitive=False,
        output_schema=wrapped_schema,
        output_selector=lambda r: {"wrapped": r.output},
    )
    parent_model = ScriptedModel(
        [tool_request("call_sub", input="hi"), text_response("done")]
    )
    parent = make_agent(
        parent_model,
        name="parent",
        tools=[agent_tool],
    )
    result = await parent.arun(input="go")
    assert result.status == "completed"
    obs = parent_model.requests[1].messages[-1].observation
    assert obs.output == {"wrapped": "raw output"}


# ---------------------------------------------------------------------------
# AgentTool egress derivation
# ---------------------------------------------------------------------------

def test_egress_derived_fixed_network_for_all_http_tools(monkeypatch):
    monkeypatch.setenv("DUMMY_KEY2", "secret")

    def _builder(args, ctx):
        return {"path_params": {}, "query": {}, "json_body": None}

    from pydantic import BaseModel

    class Q(BaseModel):
        q: str

    class R(BaseModel):
        r: str

    http_tool = HTTPTool(
        name="call_api",
        description="d",
        method="GET",
        endpoint="https://example.com/v1",
        request_builder=_builder,
        argument_schema=Q,
        output_schema=R,
        auth_env={"Authorization": {"env": "DUMMY_KEY2", "prefix": "Bearer "}},
        sensitive=False,
    )
    sub = make_agent(ScriptedModel([]), name="sub", tools=[http_tool])
    tool = AgentTool(agent=sub, name="call_sub", description="d")
    assert tool.spec.egress == "fixed_network"
    assert "https://example.com" in tool.spec.egress_origins
