import asyncio
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import BaseModel

from loom import (
    Agent,
    ApprovalDecision,
    CallerIdentity,
    CancellationToken,
    Config,
    HTTPTool,
    JSONStorage,
    Step,
    Tool,
    ToolContext,
    ToolFailure,
    Workflow,
)
from loom.config import TransportConfig
from loom.contracts import Provenance, ToolCall
from loom.providers.base import ModelAdapter, ModelCapabilities, ModelFailure, ModelResponse
from loom.testing import ScriptedModel
from loom.tools.runner import ToolRunner


class NestedInput(BaseModel):
    value: int


def test_nested_callable_schema_and_immutable_approval():
    def read(item: NestedInput) -> int:
        """Read a typed nested input."""
        assert isinstance(item, NestedInput)
        return item.value

    tool = Tool.from_callable(read, egress="none")
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(call_id="native", name="read", arguments={"item": {"value": 7}}),
                ),
                finish_reason="tool_calls",
            ),
            ModelResponse(output="7"),
        ]
    )

    def approve(request, context):
        with pytest.raises(TypeError):
            request.arguments["item"]["value"] = 100
        with pytest.raises(TypeError):
            context.context["new"] = object()
        return ApprovalDecision(approved=True)

    result = Agent(
        name="test", instructions="Read the value.", model=model, tools=[tool], approval=approve
    ).run(input="Read")
    assert result.status == "completed"
    assert model.requests[1].messages[-1].observation.output == 7
    with pytest.raises(TypeError):
        result.state.snapshots["changed"] = "not a snapshot"


def test_retry_safe_tool_retries_without_reapproving(monkeypatch):
    calls = []
    approvals = []

    def flaky() -> str:
        """Temporarily unavailable operation."""
        calls.append(1)
        if len(calls) < 3:
            raise ToolFailure("temporary_failure", transient=True)
        return "ok"

    def approve(request, context):
        approvals.append(request.call_id)
        return ApprovalDecision(approved=True)

    monkeypatch.setattr("loom.runtime.random.uniform", lambda a, b: 0)
    model = ScriptedModel(
        [
            ModelResponse(
                tool_calls=(ToolCall(call_id="native", name="flaky", arguments={}),),
                finish_reason="tool_calls",
            ),
            ModelResponse(output="done"),
        ]
    )
    result = Agent(
        name="test",
        instructions="Run.",
        model=model,
        approval=approve,
        tools=[Tool.from_callable(flaky, retry_safe=True, egress="none")],
    ).run(input="start")
    assert result.status == "completed", result
    assert result.usage.tool_attempts == 3 and result.usage.retries == 2
    assert len(approvals) == 1


class WaitingModel(ModelAdapter):
    @property
    def capabilities(self):
        return ModelCapabilities(native_tool_calling=True, local=True)

    async def generate(self, request):
        await asyncio.sleep(10)
        return ModelResponse(output="late")

    async def aclose(self):
        return None


async def test_cancellation_interrupts_a_model_wait():
    token = CancellationToken()
    task = asyncio.create_task(
        Agent(
            name="test",
            instructions="Wait.",
            model=WaitingModel(),
        ).arun(input="start", cancellation=token)
    )
    await asyncio.sleep(0.03)
    token.cancel()
    result = await asyncio.wait_for(task, 1)
    assert result.status == "cancelled"


async def test_cancellation_interrupts_long_retry_after(monkeypatch):
    requested = asyncio.Event()
    token = CancellationToken()

    def limited(request):
        requested.set()
        raise ModelFailure("rate_limited", transient=True, retry_after=60)

    monkeypatch.setattr("loom.runtime.random.uniform", lambda a, b: 0)
    model = ScriptedModel(limited)
    task = asyncio.create_task(
        Agent(name="test", instructions="Wait.", model=model).arun(
            input="start", cancellation=token
        )
    )
    await requested.wait()
    await asyncio.sleep(0.1)
    token.cancel()
    result = await asyncio.wait_for(task, 1)
    assert result.status == "cancelled"
    assert len(model.requests) == 1


async def test_retry_after_respects_deadline_without_another_request(monkeypatch):
    from loom import Limits

    def limited(request):
        raise ModelFailure("rate_limited", transient=True, retry_after=60)

    monkeypatch.setattr("loom.runtime.random.uniform", lambda a, b: 0)
    model = ScriptedModel(limited)
    result = await Agent(
        name="test",
        instructions="Wait.",
        model=model,
        limits=Limits(deadline_seconds=0.2),
    ).arun(input="start")
    assert result.status == "limit_exhausted"
    assert result.error.code == "deadline_exhausted"
    assert len(model.requests) == 1


async def test_external_cancellation_is_reraised():
    task = asyncio.create_task(
        Agent(
            name="test",
            instructions="Wait.",
            model=WaitingModel(),
        ).arun(input="start")
    )
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_denied_egress_does_not_invoke_tool():
    from loom.contracts import Provenance
    from loom.security.egress import check_tool_egress

    def send() -> str:
        """Send content."""
        return "not invoked"

    spec = Tool.from_callable(
        send, sensitive=False, egress="fixed_network", egress_origins=("https://example.com",)
    ).spec
    with pytest.raises(ToolFailure, match="document egress denied"):
        check_tool_egress(
            spec,
            (Provenance(store="docs", snapshot_id="saved"),),
            {"docs": "local_only"},
            {"send": ["docs"]},
        )


@pytest.mark.parametrize("revoke_during", ["approval", "reservation"])
async def test_egress_is_rechecked_immediately_before_dispatch(revoke_during):
    effects = []

    def send() -> str:
        """Send document content."""
        effects.append(1)
        return "sent"

    tool = Tool.from_callable(send, egress="fixed_network", egress_origins=("https://example.com",))
    policies = {"docs": "allow_hosted"}
    grants = {"send": ["docs"]}

    def approve(request, context):
        assert request.contains_document_content
        if revoke_during == "approval":
            policies["docs"] = "local_only"
        return ApprovalDecision(approved=True)

    async def reserve():
        if revoke_during == "reservation":
            grants["send"].clear()

    call = ToolCall(call_id="dispatch", name="send", arguments={})
    context = ToolContext(
        root_run_id="root",
        run_id="run",
        conversation_id="conversation",
        call_id=call.call_id,
        attempt=1,
        deadline=datetime.now(UTC) + timedelta(seconds=5),
        cancellation=CancellationToken(),
        caller=CallerIdentity(),
        provenance=(Provenance(store="docs", snapshot_id="saved"),),
        observation_bytes=8192,
    )
    result = await ToolRunner(
        tools={"send": tool}, approval=approve, policies=policies, grants=grants
    ).call(
        call,
        context,
        argument_bytes=8192,
        approval_seconds=1,
        reserve_attempt=reserve,
    )
    assert result.error.code == "document_egress_denied"
    assert result.execution_status == "not_started"
    assert effects == []
    assert not tool.is_active(call.call_id)


@pytest.mark.parametrize("workflow", [False, True])
def test_shared_http_tool_uses_each_owners_configuration(tmp_path, monkeypatch, workflow):
    owners = []

    def client(config, *, timeout):
        owners.append(config)
        return httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"ok": True})),
            trust_env=False,
        )

    monkeypatch.setattr("loom.tools.http.client", client)
    (tmp_path / "loom.toml").write_text("invalid TOML: not the owner's configuration")
    monkeypatch.chdir(tmp_path)
    tool = HTTPTool(
        name="fetch",
        description="Fetch a result.",
        method="GET",
        endpoint="https://example.com/result",
        request_builder=lambda arguments, context: {
            "path_params": {},
            "query": {},
            "json_body": None,
        },
        argument_schema={"type": "object", "additionalProperties": False},
        output_schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        sensitive=False,
        retry_safe=True,
    )
    first = Config(transport=TransportConfig(ca_bundle_env="FIRST_CA"))
    second = Config(transport=TransportConfig(ca_bundle_env="SECOND_CA"))
    for config in (first, second, first):
        if workflow:
            result = Workflow(
                name="fetch_result",
                entry="fetch",
                config=config,
                steps=[Step("fetch", tool, input_selector=lambda view: {})],
                output_selector=lambda view: view.outputs["fetch"],
            ).run(input=None)
        else:
            model = ScriptedModel(
                [
                    ModelResponse(
                        tool_calls=(ToolCall(call_id="native", name="fetch", arguments={}),),
                        finish_reason="tool_calls",
                    ),
                    ModelResponse(output="done"),
                ]
            )
            result = Agent(
                name="fetch_result",
                instructions="Fetch.",
                model=model,
                tools=[tool],
                config=config,
            ).run(input="Fetch")
        assert result.status == "completed", result
    assert len(owners) == 3
    assert owners[0] is owners[2] is first
    assert owners[1] is second


@pytest.mark.parametrize("owner", ["agent", "workflow_agent", "workflow_tool"])
def test_resolved_oversized_effect_is_persistently_protected(tmp_path, encrypted_config, owner):
    effects = []

    def uncertain() -> str:
        """Perform an operation whose acknowledgement was lost."""
        effects.append(1)
        raise ToolFailure("acknowledgement_lost", outcome_unknown=True)

    tool = Tool.from_callable(uncertain, sensitive=False, egress="none")

    def respond(request):
        observations = [message.observation for message in request.messages if message.observation]
        if len(observations) < 2:
            return ModelResponse(
                tool_calls=(ToolCall(call_id="native", name="uncertain", arguments={}),),
                finish_reason="tool_calls",
            )
        return ModelResponse(output="done")

    model = ScriptedModel(respond)
    storage = JSONStorage(path=tmp_path / "state.enc", config=encrypted_config)
    agent = Agent(
        name="effect",
        instructions="Perform the operation.",
        model=model,
        tools=[tool],
        storage=storage if owner == "agent" else None,
    )
    runner = (
        agent
        if owner == "agent"
        else Workflow(
            name="effects",
            entry="effect",
            storage=storage,
            steps=[
                Step(
                    "effect",
                    tool if owner == "workflow_tool" else agent,
                    input_selector=lambda view: {} if owner == "workflow_tool" else "start",
                ),
            ],
            output_selector=lambda view: view.outputs["effect"],
        )
    )
    interrupted = runner.run(input="start")
    assert interrupted.error.code == "unknown_tool_outcome", interrupted
    root = storage.load_run(interrupted.run_id)
    checkpoint = root.child if owner == "workflow_agent" else root
    call_id = checkpoint.pending[0].call.call_id
    runner.resolve_tool_outcome(
        interrupted.run_id,
        call_id,
        resolution="succeeded",
        output="x" * 20000,
        operator_note="Confirmed the operation in the external ledger.",
    )
    root = storage.load_run(interrupted.run_id)
    checkpoint = root.child if owner == "workflow_agent" else root
    assert checkpoint.pending[0].observation.omission is not None
    assert len(checkpoint.omitted_effects) == 1
    result = runner.resume(interrupted.run_id)
    if owner == "workflow_tool":
        assert result.status == "failed"
        assert result.error.code == "observation_too_large"
    else:
        assert result.status == "completed", result
        assert (
            model.requests[-1].messages[-1].observation.error.code == "duplicate_effect_prevented"
        )
    assert result.usage.tool_attempts == 1
    assert effects == [1]
