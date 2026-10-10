import pytest

from loom import Agent, JSONStorage, LoomStorageError, SQLiteStorage
from loom.providers.base import ModelResponse
from loom.testing import ScriptedModel


@pytest.mark.parametrize("backend", [JSONStorage, SQLiteStorage])
def test_encrypted_run_continuation_export_and_deletion(tmp_path, encrypted_config, backend):
    storage = backend(path=tmp_path / "private" / "store.enc", config=encrypted_config)
    model = ScriptedModel(lambda request: ModelResponse(output="private answer"))
    agent = Agent(name="test", instructions="Answer.", model=model, storage=storage)
    result = agent.run(input="private input")
    assert result.status == "completed"
    assert storage.load_run(result.run_id).result == result
    assert storage.load_conversation(result.conversation_id) == result.state
    assert b"private input" not in storage.path.read_bytes()
    exported = storage.export_run(result.run_id)
    assert b"private answer" not in exported
    decoded = storage.read_export(exported)
    assert decoded["checkpoint"]["run_id"] == result.run_id
    continued = agent.run(input="continue", state=result.state)
    assert continued.state.version == 2
    with pytest.raises(LoomStorageError):
        storage.delete_run(continued.run_id)
    storage.delete_conversation(result.conversation_id)
    storage.delete_run(continued.run_id)
    storage.delete_run(result.run_id)


def test_completed_resume_does_not_create_a_provider(tmp_path, encrypted_config, monkeypatch):
    storage = JSONStorage(path=tmp_path / "state.enc", config=encrypted_config)
    agent = Agent(
        name="terminal",
        instructions="Answer.",
        model=ScriptedModel([ModelResponse(output="saved")]),
        storage=storage,
    )
    result = agent.run(input="question")

    def unavailable():
        pytest.fail("a terminal resume must not initialize a provider")

    monkeypatch.setattr(agent, "_adapter", unavailable)
    assert agent.resume(result.run_id) == result
