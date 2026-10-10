# Loom

An embedded Python framework for agents, tools, document retrieval, and sequential
workflows. The implementation follows [the v0.1 specification](sdd/basic-plan.md).

**Status: implementation in progress, not an accepted production release.** Local
offline coverage is not a substitute for live-provider, supported-platform,
crash-recovery, security-scanning, or reference-hardware acceptance. Those gates
must pass before release.

## Development installation

Use Python 3.11, 3.12, or 3.13. Python 3.14 is not supported. From the repository root:

```sh
uvx --from 'uv==0.13.0' uv sync --frozen --python 3.13
.venv/bin/loom --version
.venv/bin/pytest -q
```

The distribution is named `loom-agentic-framework`; the import is `loom`.
Dependencies are pinned in `pyproject.toml` and `uv.lock`. On Windows, use
`.venv\Scripts\` instead of `.venv/bin/`.
Provider adapters live in `loom.providers`; ingestion, chunking, embeddings,
index operations and citation checks live in `loom.documents`; persistence
backends and checkpoint records live in `loom.storage`. Workflows live in
`loom.workflows`. Unit, integration and acceptance tests are grouped separately.

## Run the document-question example

The [reference directory](sdd/query-agent-example/) contains the unchanged
`developer-code.py` and a complete `loom.toml`. The source documents are not
bundled: put your PDF, Markdown, and DOCX files in its `product-docs/` folder.

Before running:

- Provision the pinned Ollama server and model explicitly: server `0.40.2`,
  model `qwen2.5:7b`, and the exact model digest in `loom.toml`. Loom checks the
  installed version, digest, and context metadata; it does not install Ollama
  or pull its models.
- Set `LOOM_ENCRYPTION_KEY` to a stable, externally managed, base64-encoded
  32-byte key. Do not commit or log it. Keep the same key available for future
  searches and refreshes; generating a new key does not decrypt old artifacts.
- Allow the explicit embedding setup command to download the pinned BGE
  artifacts over HTTPS. Indexing and querying do not download models.

Then, from the repository root:

```sh
cd sdd/query-agent-example
../../.venv/bin/loom setup embeddings
../../.venv/bin/loom index build product_docs
../../.venv/bin/python developer-code.py
```

After source changes, run `loom index refresh product_docs` from that same
directory. Index commands return exit code **0** for complete publication,
**2** for publication with exclusions or partial extraction, **1** for fatal
failure, and **130** for interruption. Inspect the emitted per-file report;
partial success does not mean every document was indexed.

The script prints a `RunResult`, with the document answer under `output`.
Document answers have `status`, `answer`, and `sources`, not a confidence score.
Answered results contain citations tied to the actual snapshot and returned
chunks. `output.json` illustrates an abstention's output shape; it is not a
recorded provider response or evidence that the reference example has passed
live acceptance.

## Tools and workflows

Tools are explicitly attached in code. Sensitive tools require approval;
headless execution without an approval callback denies them. Declare
`sensitive=False`, `retry_safe=True`, and `egress="none"` only when the callable
actually meets those contracts.

This deterministic workflow needs no model or persisted state:

```python
from loom import Step, Tool, Workflow


def add(a: int, b: int) -> int:
    """Add two integers without external effects."""
    return a + b


addition = Tool.from_callable(
    add, sensitive=False, retry_safe=True, egress="none"
)
workflow = Workflow(
    name="addition",
    entry="add",
    steps=[Step("add", addition, input_selector=lambda view: view.input)],
    output_selector=lambda view: view.outputs["add"],
)
result = workflow.run(input={"a": 2, "b": 3})
assert result.output == 5
```

`Agent` supports the OpenAI, Azure OpenAI, Anthropic, Gemini, and Ollama adapters.
Use `arun`/`aresume` inside an existing event loop; the synchronous facades
reject that context. Continue an agent conversation by explicitly passing
`state=previous_result.state`; separate calls otherwise start separate
conversations. A workflow owns its child agent histories and recovery state.
`agent.spec` exposes a strict, recursively immutable `AgentSpec` snapshot of
the current definition and its resource bindings.

`HTTPTool` has a fixed HTTPS origin and uses its owning agent or workflow's
transport configuration on each call, even when the tool object is shared.
Document-transfer permission is independent of approval and is checked again
immediately before dispatch. Arbitrary Python tools remain trusted application
code, not a sandbox.
Transient retries honor finite numeric or HTTP-date `Retry-After` values.
Backoff remains interruptible by cancellation and never extends the run or
workflow-step deadline.

## Configuration and persistence

Configuration comes from `./loom.toml` or an explicit `config=` value. Loom does
not search parent directories. Paths inside TOML resolve relative to that file.
Generated artifact locations must not overlap one another or sit inside a
source-document root; explicit constructor bindings are checked as well.
See the [reference configuration](sdd/query-agent-example/loom.toml) for the
exact supported fields and the specification for provider-specific settings.

Persistence is disabled in the reference configuration. Enable `[storage]`
or supply `JSONStorage`/`SQLiteStorage` to retain checkpoints. Both backends
persist authenticated encrypted images, not plaintext JSON or SQLite databases.
`JSONStorage` rewrites its entire image and is intended for small stores.

Resume an interrupted durable run with `agent.resume(run_id)` or
`workflow.resume(run_id)`. An unsafe tool with an unknown outcome must be
reconciled by an operator through `resolve_tool_outcome(...)`, including an
operator note, before resume. Do not automatically replay it. A confirmed
operation with an oversized omitted response retains duplicate-effect
protection through recovery.

Source deletion plus refresh removes text from the new index, **not historical
snapshots**. All index generations are retained; similarly sized refreshes grow
disk usage by approximately one full snapshot each. Publication reports both
snapshot bytes and cumulative retained bytes. Queries use a shared admission
ledger, not a hard operating-system RSS limit.

## Encryption, maintenance, and traces

HTTPS uses certificate verification and TLS 1.2 or newer. Plain HTTP is allowed
only for loopback Ollama. Document stores default to `local_only`; hosted-model
access and external-tool access require separate explicit grants.

Encrypted artifacts are bound to their resource identities and canonical paths.
**Do not move an initialized project with a plain directory rename or copy.**
Stop its applications and use the offline relocation command with old and new
configurations:

```sh
loom crypto verify --config old/loom.toml
loom crypto relocate --from-config old/loom.toml --to-config new/loom.toml
loom crypto verify --config new/loom.toml
```

Keep the old location available during relocation. Retry the same command to
recover an interrupted move; `--abort` is available before commit. Successful
relocation retires the source artifacts. Key rotation uses
`loom crypto rekey --config loom.toml`; retain old key references until every
artifact has been resealed and verified.
Relocation preflight aggregates destination and journal space per filesystem
and rejects source changes before placing resources into maintenance mode.

Optional traces use an `.enc.jsonl` path. Payload and source-path capture are
off by default. Traces retain at most five 16 MiB segments, expiring the oldest
closed segments with diagnostics. Read them with
`loom trace read --config loom.toml`; do not rotate encrypted segments using a
generic plaintext-log tool.

## Validation and release

```sh
.venv/bin/ruff check src tests
.venv/bin/mypy src/loom
.venv/bin/pytest -q
uvx --from 'uv==0.13.0' uv run --frozen python .github/scripts/ci.py package
```

Offline tests use scripted providers, mocked HTTP, and synthetic documents.
Real embedding/Ollama acceptance, all five live-provider checks, measured
latency/RSS, supported Python/OS installation coverage, and the full fault
matrix remain separate requirements. Approved Mend inputs, dependency
scanning, CodeQL analysis, and a validated CycloneDX SBOM are release gates,
not implied by a passing unit suite. Windows permission conformance is not
yet established. See specification sections 12-14 for the complete gates.

### CI configuration

`.github/workflows/ci.yml` runs on pull requests, main-branch pushes, daily,
and manual dispatch. It covers Python 3.11-3.13 on Ubuntu x64, macOS arm64 and
Intel, and hosted Windows x64. Each job checks formatting, lint, types, offline
contracts, wheel and source installation, SQLite features, and distinct
runtime/development SBOMs. Evidence contains dependency/model hashes and
platform inventory, not source documents. Hosted Windows Server coverage does
not replace the required Windows 11 acceptance.

The separate CodeQL job pins bundle 2.27.2 and `security-extended`. Its gate
currently blocks **every finding** and missing/failed analysis; there is no
baseline or waiver bypass.

Configure the protected `security-scanning` environment before expecting Mend
to pass. Set the approved `MEND_CLI_VERSION`. For each platform suffix
`LINUX_X64`, `MACOS_ARM64`, `MACOS_X64`, and `WINDOWS_X64`, set
`MEND_CLI_SHA256_<suffix>`, `MEND_CLI_URL_<suffix>` (the approved binary's HTTPS
URL), and `MEND_SCOPE_<suffix>`. Each exact approved scope must use a separate
platform project so concurrent inventory updates do not overwrite one another.
Required secrets are `MEND_URL`, `MEND_EMAIL`, and `MEND_USER_KEY`. Only exported
public dependency requirements are staged for scanning, including development
and build tooling. Each native platform resolves its applicable dependencies;
all platform jobs must pass. Missing inputs, a version/hash mismatch,
resolution failure, or a policy violation fails the job. The enterprise policy
and its synthetic exit-code-9 acceptance must still be verified externally.
Both successful and policy-failing scans retain a sanitized JSON report and
scanner identity. Dependency export includes locked build tooling; local
distribution builds use that installed tooling without a second, floating
build environment.

`.github/workflows/acceptance.yml` is manual-only, uses the protected
`provider-acceptance` environment, and requires a preprovisioned runner labelled
`loom-acceptance`. Supply a nonsecret configuration path with exactly one
profile for each of the five providers; profiles reference provisioned secret
environment variables. Prepare the pinned embeddings and Ollama outside the
test run. Neither workflow publishes a package.

Run the opt-in checks locally only after preparing their prerequisites:

```sh
LOOM_EMBEDDING_CONFIG=sdd/query-agent-example/loom.toml \
  .venv/bin/pytest -q -o addopts="" -m integration
LOOM_LIVE_CONFIG=/path/to/provisioned/loom.toml \
  .venv/bin/pytest -q -o addopts="" -m live
```

Selected acceptance checks fail, rather than skip, when configuration,
credentials, models, or services are missing. Real-model checks are distinct
from the measured reference-hardware benchmarks.

Embedding setup publishes a complete verified directory atomically. An
existing corrupt or incomplete generation fails integrity verification; stop
applications and remove only that model-revision directory before rerunning
explicit setup. Model initialization is shared across runs, while every
newly used cache path must contain the prepared artifact. Idle-fork children
reset process caches and executors; forking during active SDK work is unsupported.

The pinned BGE model is published by BAAI under MIT; the ONNX artifact is
distributed by Qdrant. Weights are not included in this package. See the
[pinned artifact repository](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/tree/aa8f8b060edb00e03bfdd08813a2949946c8ba55)
for model attribution and licensing.

Compatible security patch updates may change pins under the standing security
exception, with corresponding lockfile, SBOM, and evidence updates. Patch
timelines are critical: 24 hours; high: 7 days; medium: 30 days; low: 90 days.

Licensed under [Apache-2.0](LICENSE).
