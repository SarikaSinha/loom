import hashlib
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from loom import (
    Agent,
    AgentSpec,
    Config,
    DocumentStore,
    JSONStorage,
    LoomConfigurationError,
    Step,
    Workflow,
)
from loom.config import EmbeddingConfig, StorageConfig, StoreConfig
from loom.contracts import validate_output, validate_schema
from loom.documents import embeddings
from loom.providers.base import ModelResponse
from loom.testing import ScriptedModel


def test_agent_spec_is_a_strict_immutable_definition_snapshot():
    agent = Agent(
        name="test", instructions="Answer.", model=ScriptedModel([ModelResponse(output="done")])
    )
    spec = agent.spec
    assert isinstance(spec, AgentSpec)
    assert spec.model_dump(mode="json") == agent.descriptor()
    with pytest.raises(TypeError):
        spec.model["adapter"] = "different"
    with pytest.raises(ValueError):
        AgentSpec.model_validate_json(spec.model_dump_json().replace('"name":', '"unknown":'))
    agent.instructions = "New definition."
    assert spec.instructions == "Answer." and agent.spec.instructions == "New definition."


def test_schema_literals_are_not_treated_as_references():
    literal = {"$ref": "https://example.com/literal", "$id": "ordinary data"}
    schema = {"const": literal, "examples": [literal]}
    validate_schema(schema)
    assert validate_output(literal, schema) == literal


@pytest.mark.parametrize(
    "reference", ["#/missing", "#missing_anchor", "https://example.com/schema"]
)
def test_schema_references_fail_at_construction(reference):
    with pytest.raises(LoomConfigurationError):
        validate_schema({"$defs": {"unused": {"$ref": reference}}})


def test_local_pointers_and_anchors_are_resolved():
    schema = {
        "$defs": {"number": {"$anchor": "number", "type": "integer"}},
        "type": "array",
        "items": {"$ref": "#number"},
    }
    validate_schema(schema)
    assert validate_output([1, 2], schema) == [1, 2]
    validate_schema({"$defs": {"value": {"type": "string"}}, "$ref": "#/$defs/value"})


def test_generated_artifacts_cannot_overlap_sources_or_each_other(tmp_path):
    source = str(tmp_path / "docs")
    index = str(tmp_path / "index")
    stores = {"docs": StoreConfig(path=source, index_path=index)}
    config = Config(
        stores=stores,
        storage=StorageConfig(enabled=True, path=str(tmp_path / "docs" / "state.enc")),
    )
    with pytest.raises(LoomConfigurationError, match="generated path in source"):
        config.validate_paths()
    config = Config(
        stores=stores,
        embeddings=EmbeddingConfig(cache_path=str(tmp_path / "index" / "models")),
    )
    with pytest.raises(LoomConfigurationError, match="artifact paths overlap"):
        config.validate_paths()


def test_workflow_validates_child_bindings_against_root_artifacts(tmp_path, encrypted_config):
    child = Agent(
        name="reader",
        instructions="Read documents.",
        model=ScriptedModel(lambda request: ModelResponse(output="unused")),
        data=[DocumentStore(name="docs", path=tmp_path / "docs", index_path=tmp_path / "index")],
        config=encrypted_config,
    )
    for target, code in (
        (tmp_path / "docs" / "state.enc", "generated_path_in_source"),
        (tmp_path / "index" / "state.enc", "artifact_paths_overlap"),
    ):
        with pytest.raises(LoomConfigurationError) as failure:
            Workflow(
                name="invalid",
                entry="read",
                steps=[Step("read", child, input_selector=lambda view: "read")],
                output_selector=lambda view: view.outputs["read"],
                storage=JSONStorage(path=target, config=encrypted_config),
                config=encrypted_config,
            )
        assert failure.value.code == code
        assert not target.exists()


def test_workflow_rejects_cross_child_index_source_overlap(tmp_path, encrypted_config):
    def child(name, source, index):
        return Agent(
            name=name,
            instructions="Read.",
            model=ScriptedModel(lambda request: ModelResponse(output="unused")),
            data=[DocumentStore(name=name, path=source, index_path=index)],
            config=encrypted_config,
        )

    first = child("first", tmp_path / "first", tmp_path / "second" / "index")
    second = child("second", tmp_path / "second", tmp_path / "second-index")
    with pytest.raises(LoomConfigurationError, match="generated path in source"):
        Workflow(
            name="invalid",
            entry="first",
            steps=[
                Step("first", first, input_selector=lambda view: "read", next="second"),
                Step("second", second, input_selector=lambda view: "read"),
            ],
            output_selector=lambda view: view.outputs,
            config=encrypted_config,
        )


@pytest.fixture
def small_model(tmp_path, monkeypatch):
    payloads = {"first.bin": b"first artifact", "second.bin": b"second artifact"}
    monkeypatch.setattr(
        embeddings,
        "ARTIFACTS",
        {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in payloads.items()},
    )
    config = Config(embeddings=EmbeddingConfig(cache_path=str(tmp_path / "models")))
    return config, payloads


def test_model_setup_never_publishes_a_partial_generation(small_model, monkeypatch):
    config, payloads = small_model
    destination = embeddings.model_path(config)
    original = httpx.Client
    requests = []
    corrupt = [True]

    def respond(request):
        requests.append(request.url)
        assert not destination.exists()
        name = request.url.path.rsplit("/", 1)[-1]
        data = b"wrong" if corrupt[0] and name == "second.bin" else payloads[name]
        return httpx.Response(200, content=data)

    monkeypatch.setattr(
        embeddings.httpx,
        "Client",
        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs),
    )
    with pytest.raises(LoomConfigurationError, match="embedding artifact mismatch"):
        embeddings.setup(config)
    assert not destination.exists()
    assert {path.name for path in destination.parent.iterdir()} == {"setup.lock"}
    corrupt[0] = False
    embeddings.setup(config)
    embeddings.verify(config)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == payloads
    count = len(requests)
    embeddings.setup(config)
    assert len(requests) == count


def test_encoder_reuse_is_single_flight_but_requires_each_cache_path(
    small_model, tmp_path, monkeypatch
):
    from loom.security.files import atomic_write

    config, payloads = small_model
    for name, data in payloads.items():
        atomic_write(embeddings.model_path(config) / name, data)
    initialized = []

    class Encoder:
        def __init__(self, config):
            embeddings.verify(config)
            initialized.append(config)

    monkeypatch.setattr(embeddings, "Encoder", Encoder)
    monkeypatch.setattr(embeddings, "_encoder", None)
    monkeypatch.setattr(embeddings, "_verified_paths", set())
    with ThreadPoolExecutor(max_workers=6) as executor:
        instances = list(executor.map(embeddings.encoder, [config] * 12))
    assert len(initialized) == 1
    assert all(instance is instances[0] for instance in instances)
    missing = Config(embeddings=EmbeddingConfig(cache_path=str(tmp_path / "missing")))
    with pytest.raises(LoomConfigurationError, match="embedding artifact missing"):
        embeddings.encoder(missing)


def test_corrupt_existing_model_is_not_silently_reused(small_model):
    from loom.security.files import atomic_write

    config, payloads = small_model
    for name, data in payloads.items():
        atomic_write(embeddings.model_path(config) / name, data)
    atomic_write(embeddings.model_path(config) / "first.bin", b"wrong")
    with pytest.raises(LoomConfigurationError, match="embedding artifact corrupt"):
        embeddings.setup(config)
