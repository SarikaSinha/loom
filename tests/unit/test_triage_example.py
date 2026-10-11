import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from loom import Agent, Config, Tool
from loom.providers.base import ModelFailure, ModelResponse
from loom.testing import ScriptedModel

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "sdd" / "triage-agent" / "example"


@pytest.fixture
def example(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "triage_example", EXAMPLE_DIR / "developer-code.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_provider_creation(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Offline example tests must not create a real provider")

    monkeypatch.setattr("loom.providers.factory.create_adapter", fail)


def extraction(category="service_availability", priority="critical"):
    return {
        "summary": "Production API requests return HTTP 503; checkout is blocked.",
        "category": category,
        "priority": priority,
    }


def test_import_does_not_execute_or_read_config(monkeypatch, capsys):
    def fail(*args, **kwargs):
        pytest.fail("Import must not load configuration or execute a workflow")

    monkeypatch.setattr(Config, "load", fail)
    spec = importlib.util.spec_from_file_location(
        "triage_import_only", EXAMPLE_DIR / "developer-code.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "category,team",
    [
        ("service_availability", "API Reliability"),
        ("billing", "Billing Support"),
        ("account_access", "Account Support"),
        ("how_to", "Developer Support"),
        ("other", "General Support"),
    ],
)
@pytest.mark.parametrize(
    "priority,branch,prompt",
    [
        ("critical", "urgent_draft", "urgent internal escalation"),
        ("high", "urgent_draft", "urgent internal escalation"),
        ("medium", "standard_draft", "standard customer-support"),
        ("low", "standard_draft", "standard customer-support"),
    ],
)
def test_real_workflow_routes_and_selects_only_one_draft(
    example, category, team, priority, branch, prompt
):
    triage = extraction(category, priority)
    draft = {"draft_response": "Proposed text for human review."}
    model = ScriptedModel(
        [ModelResponse(output=json.dumps(triage)), ModelResponse(output=json.dumps(draft))]
    )
    workflow = example.build_workflow(model=model)
    ticket = json.loads((EXAMPLE_DIR / "ticket.json").read_text())
    result = workflow.run(input=ticket)

    assert result.status == "completed", result
    assert result.output == {
        **triage,
        "assigned_team": team,
        **draft,
        "next_action": "human_review",
    }
    assert len(model.requests) == result.usage.logical_model_turns == 2
    assert result.usage.tool_attempts == 1
    first, second = model.requests
    assert json.loads(first.messages[-1].content) == ticket
    assert first.output_schema == example.Triage.model_json_schema()
    assert second.output_schema == example.Draft.model_json_schema()
    assert second.instructions.startswith(workflow.steps[branch].target.instructions)
    assert prompt in second.instructions
    assert json.loads(second.messages[-1].content) == {
        "ticket": ticket,
        "triage": triage,
        "assigned_team": team,
    }
    assert len(first.messages) == len(second.messages) == 1
    assert not first.tools and not second.tools
    routing = workflow.steps["route"].target
    assert isinstance(routing, Tool)
    assert routing.spec.egress == "none"
    assert routing.spec.retry_safe and not routing.spec.sensitive
    assert workflow.storage is None
    assert all(
        not step.target.data and not step.target.tools
        for step in workflow.steps.values()
        if isinstance(step.target, Agent)
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {**extraction(), "category": "unknown"},
        {**extraction(), "priority": "urgent"},
        {**extraction(), "summary": 42},
        {**extraction(), "summary": ""},
        {**extraction(), "assigned_team": "Attacker-selected team"},
        {"category": "billing", "priority": "low"},
    ],
)
def test_invalid_extraction_fails_before_routing(example, invalid):
    model = ScriptedModel([ModelResponse(output=invalid)] * 2)
    result = example.build_workflow(model=model).run(input={"message": "Please help."})
    assert result.status == "failed"
    assert result.output is None
    assert result.error.code == "output_validation_failed"
    assert result.usage.corrections == 1
    assert result.usage.tool_attempts == 0
    assert len(model.requests) == 2


def test_corrected_extraction_is_used_for_routing(example):
    model = ScriptedModel(
        [
            ModelResponse(output={**extraction(), "category": "unknown"}),
            ModelResponse(output=extraction("billing", "low")),
            ModelResponse(output={"draft_response": "Please share your invoice number."}),
        ]
    )
    result = example.build_workflow(model=model).run(input={"message": "Billing question."})
    assert result.status == "completed", result
    assert result.usage.corrections == 1
    assert result.usage.tool_attempts == 1
    assert result.output["assigned_team"] == "Billing Support"
    assert "standard customer-support" in model.requests[-1].instructions


def test_draft_cannot_override_routing_or_next_action(example):
    invalid = {
        "draft_response": "Send immediately.",
        "assigned_team": "Different team",
        "next_action": "send",
    }
    model = ScriptedModel(
        [ModelResponse(output=extraction()), *[ModelResponse(output=invalid)] * 2]
    )
    result = example.build_workflow(model=model).run(input={"message": "API unavailable."})
    assert result.status == "failed"
    assert result.output is None
    assert result.error.code == "output_validation_failed"
    assert result.usage.tool_attempts == 1


@pytest.mark.parametrize("message", ["", " \n\t", 123, "x" * 12001])
def test_ticket_rejects_invalid_messages(example, message):
    with pytest.raises(ValidationError):
        example.Ticket.model_validate({"message": message})


def test_bundled_config_matches_reference_pin_and_needs_no_artifacts(example, monkeypatch):
    monkeypatch.delenv("LOOM_ENCRYPTION_KEY", raising=False)
    config = Config.load(EXAMPLE_DIR / "loom.toml")
    reference = Config.load(EXAMPLE_DIR.parents[1] / "query-agent-example" / "loom.toml")
    assert config.providers["primary"] == reference.providers["primary"]
    assert not config.storage.enabled
    assert not config.observability.jsonl_path
    assert not config.stores
    workflow = example.build_workflow()
    assert all(
        step.target.model == "primary"
        for step in workflow.steps.values()
        if isinstance(step.target, Agent)
    )


def test_illustrative_output_matches_workflow_output_shape(example):
    illustrative = json.loads((EXAMPLE_DIR / "output.json").read_text())
    triage = {key: illustrative[key] for key in example.Triage.model_fields}
    model = ScriptedModel(
        [
            ModelResponse(output=triage),
            ModelResponse(output={"draft_response": illustrative["draft_response"]}),
        ]
    )
    result = example.build_workflow(model=model).run(
        input=json.loads((EXAMPLE_DIR / "ticket.json").read_text())
    )
    assert result.status == "completed", result
    assert result.output == illustrative


def use_scripted_workflow(example, monkeypatch, responses):
    model = ScriptedModel(responses)
    build = example.build_workflow
    monkeypatch.setattr(example, "build_workflow", lambda **kwargs: build(model=model, **kwargs))
    return model


def test_cli_uses_script_relative_defaults(example, monkeypatch, tmp_path, capsys):
    model = use_scripted_workflow(
        example,
        monkeypatch,
        [
            ModelResponse(output=extraction()),
            ModelResponse(output={"draft_response": "Proposed escalation note."}),
        ],
    )
    monkeypatch.chdir(tmp_path)
    assert example.main([]) == 0
    captured = capsys.readouterr()
    assert not captured.err
    assert json.loads(captured.out)["next_action"] == "human_review"
    assert len(model.requests) == 2
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("stage", ["triage", "draft"])
def test_cli_reports_unsuccessful_run_without_success_output(example, monkeypatch, capsys, stage):
    responses = [] if stage == "triage" else [ModelResponse(output=extraction())]
    responses.append(ModelFailure("provider_unavailable"))
    use_scripted_workflow(example, monkeypatch, responses)
    assert example.main([]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    error = json.loads(captured.err)
    assert error["status"] == "failed"
    assert error["output"] is None
    assert error["error"]["code"] == "provider_unavailable"


def test_cli_rejects_invalid_input_before_provider_use(example, tmp_path, capsys):
    ticket = tmp_path / "ticket.json"
    ticket.write_text('{"message": " "}')
    with pytest.raises(SystemExit, match="2"):
        example.main(["--ticket", str(ticket)])
    captured = capsys.readouterr()
    assert not captured.out
    assert "Invalid ticket file" in captured.err


def test_cli_reports_invalid_config(example, tmp_path, capsys):
    assert example.main(["--config", str(tmp_path / "missing.toml")]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "Configuration error: invalid_configuration" in captured.err
