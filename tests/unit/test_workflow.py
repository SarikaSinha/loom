import uuid

import pytest

from loom import (
    Agent,
    Branch,
    Case,
    JSONStorage,
    LoomConfigurationError,
    SQLiteStorage,
    Step,
    Tool,
    ToolFailure,
    Workflow,
    WorkflowLimits,
)
from loom.contracts import ToolCall
from loom.providers.base import ModelResponse
from loom.testing import ScriptedModel


def plus_one(value: int) -> int:
    """Increment an integer."""
    return value + 1


def function_tool():
    return Tool.from_callable(plus_one, sensitive=False, egress="none")


def test_sequential_tools_and_immutable_view():
    def select(view):
        with pytest.raises(TypeError):
            view.outputs["first"] = 100
        return {"value": view.outputs["first"]}

    workflow = Workflow(
        name="increment",
        entry="first",
        steps=[
            Step("first", function_tool(), input_selector=lambda view: view.input, next="second"),
            Step("second", function_tool(), input_selector=select),
        ],
        output_selector=lambda view: view.outputs[view.last_step],
    )
    result = workflow.run(input={"value": 1})
    assert result.status == "completed", result
    assert result.output == 3
    assert result.usage.tool_attempts == 2
    assert result.usage.logical_model_turns == 0
    with pytest.raises(TypeError):
        result.state.snapshots["changed"] = "not a snapshot"


@pytest.mark.parametrize("backend", [JSONStorage, SQLiteStorage])
def test_agent_steps_charge_root_and_do_not_share_history(tmp_path, encrypted_config, backend):
    model = ScriptedModel(lambda request: ModelResponse(output=request.messages[-1].content + "!"))
    events = []
    first = Agent(name="first", instructions="Append.", model=model, event_sinks=[events.append])
    second = Agent(
        name="second", instructions="Append again.", model=model, event_sinks=[events.append]
    )
    storage = backend(path=tmp_path / "state.enc", config=encrypted_config)
    workflow = Workflow(
        name="sequence",
        entry="first",
        storage=storage,
        steps=[
            Step("first", first, input_selector=lambda view: view.input, next="second"),
            Step("second", second, input_selector=lambda view: view.outputs["first"]),
        ],
        output_selector=lambda view: view.outputs["second"],
    )
    result = workflow.run(input="hello")
    assert result.status == "completed", result.to_json(include_state=True)
    assert result.output == "hello!!"
    assert result.usage.model_requests == result.usage.logical_model_turns == 2
    assert all(len(request.messages) == 1 for request in model.requests)
    assert workflow.resume(result.run_id) == result
    assert len(model.requests) == 2
    assert storage.load_run(result.run_id).activated_steps == ("first", "second")
    assert uuid.UUID(result.run_id).version == 4
    assert {event.run_id for event in events if event.run_id != result.run_id} == {
        str(uuid.uuid5(uuid.UUID(result.run_id), name)) for name in ("first", "second")
    }


def test_aggregate_model_budget_is_reserved_before_dispatch():
    model = ScriptedModel(lambda request: ModelResponse(output="answer"))
    agent = Agent(name="answer", instructions="Answer.", model=model)
    workflow = Workflow(
        name="bounded",
        entry="first",
        limits=WorkflowLimits(max_model_turns=1),
        steps=[
            Step("first", agent, input_selector=lambda view: "first", next="second"),
            Step("second", agent, input_selector=lambda view: "second"),
        ],
        output_selector=lambda view: view.outputs,
    )
    result = workflow.run(input=None)
    assert result.status == "limit_exhausted", result
    assert result.error.code == "workflow_model_turn_limit"
    assert result.usage.logical_model_turns == result.usage.model_requests == 1
    assert len(model.requests) == 1


@pytest.mark.parametrize(
    "predicates,code",
    [
        ((True, True), "branch_ambiguous"),
        ((False, False), "branch_no_match"),
        ((1, False), "branch_evaluation_failed"),
    ],
)
def test_branch_failures_are_explicit(predicates, code):
    workflow = Workflow(
        name="branching",
        entry="first",
        steps=[
            Step(
                "first",
                function_tool(),
                input_selector=lambda view: {"value": 0},
                next=Branch(
                    [
                        Case("left", lambda view: predicates[0], "left"),
                        Case("right", lambda view: predicates[1], "right"),
                    ]
                ),
            ),
            Step("left", function_tool(), input_selector=lambda view: {"value": 10}),
            Step("right", function_tool(), input_selector=lambda view: {"value": 20}),
        ],
        output_selector=lambda view: view.outputs,
    )
    result = workflow.run(input=None)
    assert result.status == "failed", result
    assert result.error.code == code
    assert result.usage.tool_attempts == 1


@pytest.mark.parametrize(
    "edges,entry,code",
    [
        ({"first": "missing"}, "first", "unknown_workflow_target"),
        ({"first": "second", "second": "first"}, "first", "workflow_cycle"),
        ({"first": None, "second": None}, "first", "unreachable_workflow_step"),
    ],
)
def test_graph_validation(edges, entry, code):
    with pytest.raises(LoomConfigurationError) as error:
        Workflow(
            name="invalid",
            entry=entry,
            steps=[
                Step(name, function_tool(), next=following, input_selector=lambda view: {})
                for name, following in edges.items()
            ],
            output_selector=lambda view: None,
        )
    assert error.value.code == code


def test_predicate_toolfailure_is_a_branch_evaluation_error():
    def fail(view):
        raise ToolFailure("arbitrary_application_failure")

    workflow = Workflow(
        name="branch_failure",
        entry="first",
        steps=[
            Step(
                "first",
                function_tool(),
                input_selector=lambda view: {"value": 1},
                next=Branch([Case("check", fail, "second")]),
            ),
            Step("second", function_tool(), input_selector=lambda view: {"value": 2}),
        ],
        output_selector=lambda view: view.outputs,
    )
    result = workflow.run(input=None)
    assert result.status == "failed"
    assert result.error.code == "branch_evaluation_failed"


def test_committed_branch_is_not_reevaluated_after_crash(tmp_path, encrypted_config, monkeypatch):
    class Crash(BaseException):
        pass

    predicates = []
    storage = JSONStorage(path=tmp_path / "state.enc", config=encrypted_config)
    workflow = Workflow(
        name="durable_branch",
        entry="first",
        storage=storage,
        steps=[
            Step(
                "first",
                function_tool(),
                input_selector=lambda view: {"value": 0},
                next=Branch(
                    [
                        Case("choose", lambda view: predicates.append(1) is None, "second"),
                    ]
                ),
            ),
            Step("second", function_tool(), input_selector=lambda view: {"value": 1}),
        ],
        output_selector=lambda view: view.outputs["second"],
    )
    save = storage.save
    root_id = []

    def crash_after_branch(checkpoint):
        result = save(checkpoint)
        if checkpoint.active_step == "second" and checkpoint.activated_steps == ("first",):
            root_id.append(checkpoint.run_id)
            raise Crash()
        return result

    monkeypatch.setattr(storage, "save", crash_after_branch)
    with pytest.raises(Crash):
        workflow.run(input=None)
    monkeypatch.setattr(storage, "save", save)
    result = workflow.resume(root_id[0])
    assert result.status == "completed", result
    assert result.output == 2 and result.usage.tool_attempts == 2
    assert predicates == [1]


@pytest.mark.parametrize("agent_step", [False, True])
def test_uncertain_effect_resolves_under_root_without_replay(
    tmp_path,
    encrypted_config,
    agent_step,
):
    effects = []

    def uncertain() -> int:
        """Make an effect whose remote acknowledgement was lost."""
        effects.append(1)
        raise ToolFailure("acknowledgement_lost", outcome_unknown=True)

    tool = Tool.from_callable(uncertain, sensitive=False, egress="none")

    def respond(request):
        if request.messages[-1].observation:
            return ModelResponse(output="resolved")
        return ModelResponse(
            tool_calls=(ToolCall(call_id="native", name="uncertain", arguments={}),),
            finish_reason="tool_calls",
        )

    target = (
        Agent(
            name="child",
            instructions="Call once.",
            model=ScriptedModel(respond),
            tools=[tool],
        )
        if agent_step
        else tool
    )
    storage = JSONStorage(path=tmp_path / "state.enc", config=encrypted_config)
    workflow = Workflow(
        name="durable",
        entry="effect",
        storage=storage,
        steps=[Step("effect", target, input_selector=lambda view: "start" if agent_step else {})],
        output_selector=lambda view: view.outputs["effect"],
    )
    interrupted = workflow.run(input=None)
    assert interrupted.error.outcome_unknown, interrupted
    root = storage.load_run(interrupted.run_id)
    assert root.phase == "needs_resolution"
    call_id = (root.child if agent_step else root).pending[0].call.call_id
    workflow.resolve_tool_outcome(
        interrupted.run_id,
        call_id,
        resolution="succeeded",
        output=3,
        operator_note="Confirmed the effect in the remote ledger.",
    )
    result = workflow.resume(interrupted.run_id)
    assert result.status == "completed", result
    assert result.output == ("resolved" if agent_step else 3)
    assert result.usage.tool_attempts == 1
    assert effects == [1]
