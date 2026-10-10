import base64

import pytest

from loom import Agent, LoomStorageError
from loom.config import Config, EncryptionConfig, ObservabilityConfig, StorageConfig
from loom.providers.base import ModelResponse
from loom.security.admin import rekey, verify
from loom.security.maintenance import Managed
from loom.security.relocation import relocate
from loom.storage import JSONStorage, SQLiteStorage
from loom.testing import ScriptedModel
from loom.trace import TraceSink


def configuration(path, encryption, backend="json"):
    path.mkdir(mode=0o700)
    config = Config(
        encryption=encryption,
        storage=StorageConfig(enabled=True, backend=backend, path=".loom/state.enc"),
        observability=ObservabilityConfig(jsonl_path=".loom/events.enc.jsonl"),
    )
    config._base = path
    return config


def agent(config, instructions="Answer without rewriting paths."):
    return Agent(
        name="test",
        instructions=instructions,
        config=config,
        model=ScriptedModel(lambda request: ModelResponse(output=request.messages[-1].content)),
    )


def destination(tmp_path, encrypted_config, monkeypatch, backend="json"):
    monkeypatch.setenv("LOOM_NEXT_KEY", base64.b64encode(bytes(reversed(range(32)))).decode())
    old = configuration(tmp_path / "old", encrypted_config.encryption, backend)
    new = configuration(
        tmp_path / "new",
        EncryptionConfig(active_key_id="next", keys={"next": "LOOM_NEXT_KEY"}),
        backend,
    )
    return old, new


def test_rekey_preserves_records_ids_and_revisions(tmp_path, encrypted_config, monkeypatch):
    old, new = destination(tmp_path, encrypted_config, monkeypatch)
    original = agent(old).run(input="private input")
    storage = JSONStorage(path=old.path(old.storage.path), config=old)
    before = storage.load_run(original.run_id)
    events = list(TraceSink(old).read())
    assert verify(old)["artifacts"]
    rotated = old.model_copy(
        update={
            "encryption": EncryptionConfig(
                active_key_id="next",
                keys={"test": "LOOM_TEST_KEY", "next": "LOOM_NEXT_KEY"},
            )
        }
    )
    report = rekey(rotated)
    assert report["status"] == "completed"
    assert all(artifact["key_id"] == "next" for artifact in report["artifacts"])
    only_new = rotated.model_copy(update={"encryption": new.encryption})
    monkeypatch.delenv("LOOM_TEST_KEY")
    assert verify(only_new)["status"] == "completed"
    assert agent(only_new).resume(original.run_id) == original
    assert (
        JSONStorage(path=storage.path, config=only_new).load_run(original.run_id).revision
        == before.revision
    )
    assert list(TraceSink(only_new).read()) == events


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_relocation_preserves_state_and_retires_original(
    tmp_path, encrypted_config, monkeypatch, backend
):
    old, new = destination(tmp_path, encrypted_config, monkeypatch, backend)
    text = f"Keep this literal path unchanged: {old.path('documents/manual.md')}"
    original = agent(old, text).run(input=text)
    storage_type = JSONStorage if backend == "json" else SQLiteStorage
    before = storage_type(path=old.path(old.storage.path), config=old).load_run(original.run_id)
    events = list(TraceSink(old).read())
    report = relocate(old, new)
    assert report["state"] == "committed" and report["verified"]
    assert str(old.path(".")) not in str(report) and text not in str(report)
    with pytest.raises(LoomStorageError) as error:
        agent(old, text).resume(original.run_id)
    assert error.value.code == "artifact_relocated"
    assert verify(old)["status"] == "completed"
    assert verify(new)["status"] == "completed"
    assert agent(new, text).resume(original.run_id) == original
    after = storage_type(path=new.path(new.storage.path), config=new).load_run(original.run_id)
    assert after.revision == before.revision
    assert after.usage == before.usage and after.state == before.state
    assert after.configuration["instructions"] == text
    assert after.configuration_fingerprint != before.configuration_fingerprint
    assert list(TraceSink(new).read()) == events
    assert relocate(old, new)["relocation_id"] == report["relocation_id"]


def test_precommit_abort_restores_only_sources(tmp_path, encrypted_config, monkeypatch):
    import loom.security.relocation as module

    class Crash(BaseException):
        pass

    old, new = destination(tmp_path, encrypted_config, monkeypatch)
    original = agent(old).run(input="keep me")
    write = module.atomic_write

    def crash(path, data):
        write(path, data)
        raise Crash()

    monkeypatch.setattr(module, "atomic_write", crash)
    with pytest.raises(Crash):
        relocate(old, new)
    with pytest.raises(LoomStorageError, match="artifact maintenance required"):
        agent(old).resume(original.run_id)
    monkeypatch.setattr(module, "atomic_write", write)
    report = relocate(old, new, abort=True)
    assert report["state"] == "aborted" and not report["verified"]
    assert agent(old).resume(original.run_id) == original
    with pytest.raises(LoomStorageError, match="artifact maintenance required"):
        agent(new).resume(original.run_id)


def test_activation_retry_does_not_overwrite_an_activated_destination(
    tmp_path,
    encrypted_config,
    monkeypatch,
):
    class Crash(BaseException):
        pass

    old, new = destination(tmp_path, encrypted_config, monkeypatch)
    original = agent(old).run(input="keep me")
    write = Managed.write

    def crash_after_storage_activation(self, status, relocation_id=None):
        result = write(self, status, relocation_id)
        if self.config is new and self.kind == "storage_json" and status == "ready":
            raise Crash()
        return result

    monkeypatch.setattr(Managed, "write", crash_after_storage_activation)
    with pytest.raises(Crash):
        relocate(old, new)
    monkeypatch.setattr(Managed, "write", write)
    storage = JSONStorage(path=new.path(new.storage.path), config=new)
    storage.delete_conversation(original.conversation_id)
    assert relocate(old, new)["state"] == "committed"
    with pytest.raises(LoomStorageError, match="conversation not found"):
        storage.load_conversation(original.conversation_id)
    assert storage.load_run(original.run_id).result == original


def test_rekey_refuses_active_resource_ownership(tmp_path, encrypted_config):
    config = configuration(tmp_path / "old", encrypted_config.encryption)
    original = agent(config).run(input="protected")
    storage = JSONStorage(path=config.path(config.storage.path), config=config)
    with storage.ownership(original.run_id):
        with pytest.raises(LoomStorageError, match="resource in use"):
            rekey(config)


@pytest.mark.parametrize("free,succeeds", [(119, False), (120, True)])
def test_relocation_capacity_is_aggregated_per_filesystem(tmp_path, monkeypatch, free, succeeds):
    from types import SimpleNamespace

    import loom.security.relocation as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    allocations = [(tmp_path / "first" / "data.enc", 60), (tmp_path / "second" / "data.enc", 60)]
    if succeeds:
        module.check_capacity(allocations)
    else:
        with pytest.raises(LoomStorageError, match="relocation capacity exceeded"):
            module.check_capacity(allocations)


def test_relocation_detects_source_changes_during_preflight(
    tmp_path, encrypted_config, monkeypatch
):
    import loom.security.relocation as module

    old, new = destination(tmp_path, encrypted_config, monkeypatch)
    original = agent(old).run(input="preserved")
    original_hash = module.ciphertext_hash
    seen = {}

    def changed(artifact):
        seen[artifact.path] = seen.get(artifact.path, 0) + 1
        return original_hash(artifact) if seen[artifact.path] == 1 else "changed"

    monkeypatch.setattr(module, "ciphertext_hash", changed)
    with pytest.raises(LoomStorageError, match="relocation source changed"):
        relocate(old, new)
    assert agent(old).resume(original.run_id) == original
    assert all(not resource.initialized() for resource in module.configured(new).values())
    assert module.journals(old) == []
