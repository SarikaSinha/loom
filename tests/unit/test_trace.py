import uuid

import pytest

from loom import Agent, LoomStorageError
from loom.config import ObservabilityConfig
from loom.events import Event
from loom.providers.base import ModelResponse
from loom.testing import ScriptedModel
from loom.trace import TraceSink


def test_encrypted_trace_rotation_and_identity(tmp_path, encrypted_config, monkeypatch):
    import loom.trace as trace

    monkeypatch.setattr(trace, "SEGMENT_BYTES", 3000)
    config = encrypted_config.model_copy(
        update={
            "observability": ObservabilityConfig(
                jsonl_path=str(tmp_path / "trace" / "events.enc.jsonl")
            )
        }
    )
    diagnostics = []
    sink = TraceSink(config, diagnostics.append)
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    for index in range(30):
        run_id = first if index % 2 else second
        event = Event(
            run_id=run_id,
            root_run_id=run_id,
            sequence=index + 1,
            event_id=f"{run_id}:{index + 1}",
            timestamp="2026-01-01T00:00:00+00:00",
            event_type="run.started",
            payload="private document text",
        )
        sink.append(event)
    manifest = sink.manifest()
    assert len(manifest.segments) == 5
    assert len(list(sink.directory.glob("*.enc.jsonl"))) == 5
    assert any(item.code == "trace_retention_applied" for item in diagnostics)
    events = list(sink.read(first))
    assert events and all(event.run_id == first for event in events)
    for segment in manifest.segments:
        assert b"private document text" not in sink.path(segment).read_bytes()
    path = sink.path(manifest.segments[-1])
    with path.open("ab") as output:
        output.write(b"broken")
    with pytest.raises(LoomStorageError, match="trace corrupt"):
        list(sink.read())


def test_agent_trace_wiring(tmp_path, encrypted_config):
    config = encrypted_config.model_copy(
        update={
            "observability": ObservabilityConfig(
                jsonl_path=str(tmp_path / "traces" / "run.enc.jsonl")
            )
        }
    )
    result = Agent(
        name="test",
        instructions="Answer.",
        model=ScriptedModel([ModelResponse(output="ok")]),
        config=config,
    ).run(input="do not trace this input")
    assert result.status == "completed"
    events = list(TraceSink(config).read(result.run_id))
    assert events and all(event.payload is None for event in events)
