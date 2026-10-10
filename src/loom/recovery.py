from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from loom.contracts import (
    CallerIdentity,
    CancellationToken,
    Diagnostic,
    JSONValue,
    LoomConfigurationError,
    RunError,
    RunResult,
    ToolContext,
    ToolFailure,
    ToolObservation,
    canonical,
)
from loom.storage import Checkpoint
from loom.tools.base import Tool
from loom.tools.runner import ToolRunner, effect_key, run_error


def resolve(
    checkpoint: Checkpoint,
    tool: Tool,
    call_id: str,
    *,
    resolution: Literal["succeeded", "failed", "abandoned"],
    output: JSONValue,
    error: ToolFailure | None,
    operator_note: str,
    observation_bytes: int,
) -> Checkpoint:
    if not operator_note.strip():
        raise LoomConfigurationError("operator_note_required")
    if checkpoint.phase != "needs_resolution":
        raise LoomConfigurationError("resolution_not_required")
    pending = list(checkpoint.pending)
    index = next((i for i, op in enumerate(pending) if op.call.call_id == call_id), None)
    if index is None:
        raise LoomConfigurationError("unknown_tool_call")
    operation = pending[index]
    if index < checkpoint.cursor or operation.status not in {"started", "unknown"}:
        raise LoomConfigurationError("resolution_not_required")
    if tool.spec.name != operation.call.name or tool.is_active(call_id):
        raise LoomConfigurationError("tool_unavailable_or_active")
    context = ToolContext(
        root_run_id=checkpoint.root_run_id,
        run_id=checkpoint.run_id,
        conversation_id=checkpoint.state.conversation_id,
        call_id=call_id,
        attempt=max(1, operation.attempts),
        deadline=datetime.now(UTC),
        cancellation=CancellationToken(),
        caller=CallerIdentity(subject=checkpoint.state.caller_subject),
        provenance=checkpoint.state.provenance,
        observation_bytes=observation_bytes,
    )
    if resolution == "succeeded" and error is None:
        observation = ToolRunner.output(
            operation.call, output, tool.spec.output_schema, context, checkpoint.state.provenance
        )
    elif (
        resolution == "failed"
        and output is None
        and error is not None
        and not error.outcome_unknown
    ):
        observation = ToolObservation(
            call_id=call_id,
            execution_status="failed",
            error=run_error(error),
            provenance=checkpoint.state.provenance,
        )
    elif resolution == "abandoned" and output is None and error is None:
        diagnostics = (
            *checkpoint.diagnostics,
            Diagnostic(
                code="tool_outcome_abandoned",
                message="Operator abandoned uncertain work.",
                origin="operator",
                details={"call_id": call_id, "note": operator_note},
            ),
        )
        result = RunResult(
            run_id=checkpoint.run_id,
            conversation_id=checkpoint.state.conversation_id,
            status="failed",
            usage=checkpoint.usage,
            state=checkpoint.state,
            snapshots=checkpoint.state.snapshots,
            diagnostics=diagnostics,
            error=RunError(
                code="tool_outcome_abandoned",
                message="Operator abandoned uncertain work.",
                origin="operator",
            ),
        )
        return checkpoint.model_copy(
            update={
                "phase": "failed",
                "result": result,
                "diagnostics": diagnostics,
            }
        )
    else:
        raise LoomConfigurationError("invalid_resolution")
    pending[index] = operation.model_copy(
        update={
            "status": resolution,
            "observation": observation,
            "operator_note": operator_note,
        }
    )
    omitted_effects = set(checkpoint.omitted_effects)
    if observation.omission is not None and not tool.spec.retry_safe:
        omitted_effects.add(effect_key(tool.spec, canonical(operation.call.arguments)))
    return checkpoint.model_copy(
        update={
            "pending": tuple(pending),
            "phase": "tools_pending",
            "result": None,
            "omitted_effects": tuple(sorted(omitted_effects)),
            "diagnostics": (
                *checkpoint.diagnostics,
                Diagnostic(
                    code="tool_outcome_resolved",
                    message="Operator resolved a tool outcome.",
                    origin="operator",
                    details={"call_id": call_id, "note": operator_note},
                ),
            ),
        }
    )
