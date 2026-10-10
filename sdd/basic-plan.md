# Loom v0.1 Implementation Specification and Plan

## 1. Authority and scope

**Status: final design for code generation.** All decisions in this document
are binding for v0.1. There are no alternative architectures, undecided library
selections, or optional implementation branches for the generator to choose.
An implementation that cannot satisfy a contract must report the failure; it
must not silently change the contract, substitute a technology, or reduce scope.

This document specifies work to implement; it does not claim that the framework
exists, dependencies have been installed, or acceptance criteria already pass.
Implementation is now authorized. By the subsequent implementation decision,
coding and offline validation may proceed while external M0 security, provider
and platform/hardware checks are pending. Those checks remain mandatory release
blockers; this does not authorize incompatible dependency substitutions,
weakened contracts, or publication.

Sources, in precedence order:

1. Explicit decisions in the design interview and subsequent review: adopt
   the supplied TLS/encryption/security-patching requirements and use mandatory
   SDK-managed authenticated encryption with externally supplied keys.
2. The contracts in this document, which resolve the remaining implementation
   choices and supersede earlier draft recommendations.
3. [basic-requirement.md](basic-requirement.md).
4. The reference [developer code](query-agent-example/developer-code.py),
   [input](query-agent-example/input.txt),
   [document manifest](query-agent-example/data.txt), and
   [original output](query-agent-example/output.json).

Preserve `developer-code.py` byte-for-byte. Its output fixture is superseded:
remove confidence and use the result/citation schema in section 5. The manifest
does not contain actual product documents; use synthetic source fixtures for
the example rather than treating its rate-limit claim as verified information.
The corrected requirement has no API-review workflow; API review is out of scope.

### 1.1 Included release capabilities

| Area | Final decision |
| --- | --- |
| Product | Embedded Python SDK, plus a preparation/indexing CLI. No hosted execution service. |
| Runtime | CPython 3.11, 3.12, and 3.13; `requires-python = ">=3.11,<3.14"`. |
| Platforms | macOS 13+ on arm64/x86_64; Ubuntu 22.04+ on x86_64; Windows 11 on x86_64. CPU execution, no required GPU or Docker. |
| Installation | One distribution includes dependencies for all five provider adapters and document parsers. Model weights require explicit setup, not an installation extra. |
| Agents | Instructions, model selection, tools, attached document stores, text or schema-validated output. |
| Models | OpenAI, Azure OpenAI, Anthropic, Gemini, and Ollama are all release requirements. |
| Execution | Native async engine, synchronous facade, sequential tool calls, explicit state, bounded execution. No token streaming. |
| Schemas | Python type hints/Pydantic and JSON Schema; native model schema enforcement where compatible, otherwise local validation and bounded correction. |
| Tools | Trusted in-process function/custom tools and typed HTTP tools. Explicit allowlist and sensitive-action approval. |
| Documents | English text-PDF, Markdown, and DOCX; hybrid keyword/semantic retrieval; local embeddings and persistent snapshots. No OCR. |
| Data attachment | A store becomes a model-callable search tool. No automatic pre-retrieval. |
| Output grounding | Search before answering; verified provenance, explicit abstention, conflicts cited on both sides, no confidence score. |
| Privacy | Separate per-agent/per-store consent for hosted document transfer. Local embedding setup does not grant that consent. |
| Transport | TLS 1.2+ with certificate/hostname verification; HTTPS except literal-loopback Ollama. |
| Persistence protection | Mandatory authenticated encryption of Loom-owned sensitive files, owner-only access, and no plaintext SQLite journals or staging files. |
| Memory | Fresh by default; caller explicitly supplies prior state. No implicit Agent-instance memory or automatic summarization. |
| Durability | Opt-in conversations and agent/workflow checkpoints using either `SQLiteStorage` or `JSONStorage`. |
| Workflows | Code-first, typed, sequential acyclic graphs; deterministic branching and bounded retry of eligible operations. |
| Observability | Lifecycle callbacks and encrypted-record JSONL traces; five 16 MiB retained segments with automatic oldest-segment expiry; payload capture disabled by default. |

Python versions/architectures outside this matrix are not claimed in v0.1.
The document target is 100 files and up to 10,000,000 extracted UTF-8 bytes,
not an automatic corpus cap. Section 13 defines the exact benchmark.

## 2. Package, dependencies, and reproducibility

### 2.1 Distribution and modules

Use distribution name `loom-agentic-framework`, version `0.1.0`, import name
`loom`, the existing Apache-2.0 repository license, and console entry point
`loom = loom.cli:main`. Preserve model/dependency license notices.

```text
pyproject.toml
uv.lock
src/loom/
  __init__.py
  py.typed
  contracts.py
  config.py
  agent.py
  runtime.py
  workflows.py
  events.py
  cli.py
  security/
    transport.py
    crypto.py
    files.py
    egress.py
  providers/
    base.py
    openai.py
    azure_openai.py
    anthropic.py
    gemini.py
    ollama.py
  tools/
    base.py
    function.py
    http.py
    approval.py
  documents/
    store.py
    loaders.py
    chunks.py
    embeddings.py
    index.py
    citations.py
    embedding-manifest.json
  storage/
    base.py
    sqlite.py
    json.py
    checkpoint.py
tests/
  unit/
  integration/
  acceptance/
  fixtures/
```

Tests mirror module responsibilities. Add private helpers within these modules;
do not introduce services, plugin discovery, or another orchestration framework.
Imports must not create databases, download weights, index files, or call models.

### 2.2 Exact direct dependencies

Use these direct pins as the reproducible baseline. The standing security-update
exception in section 14 authorizes remediation updates without a new design
interview when the public contracts remain intact. Resolve transitive dependencies with
`uv==0.13.0`, commit `uv.lock`, and use `uv sync --frozen` thereafter.
M0 must prove availability, joint resolution, platform installation, and security
scan results; release-metadata inspection alone does not satisfy this gate.

| Purpose | Pin |
| --- | --- |
| Runtime schemas | `pydantic==2.14.0`, `jsonschema==4.26.0` |
| HTTP | `httpx==0.28.1` |
| Providers | `openai==3.28.0`, `anthropic==1.13.0`, `google-genai==2.29.0`; Ollama uses the versioned public HTTP API through httpx. |
| Parsing | `pypdf==6.20.0`, `python-docx==1.2.0`, `markdown-it-py==4.2.0` |
| Embeddings/search | `fastembed==0.9.0`, `onnxruntime==1.23.2`, `numpy==2.3.5`, `tokenizers==0.23.3` |
| OS file locking | `portalocker==4.4.0` |
| Authenticated encryption | `cryptography==50.0.2` |
| Build backend | `hatchling==1.32.4` |
| Development group | `pytest==9.1.1`, `pytest-asyncio==1.4.0`, `ruff==0.17.0`, `mypy==2.4.0`, `cyclonedx-bom==7.5.0` |

Use standard-library `tomllib`, `sqlite3`, `argparse`, `asyncio`, `hashlib`,
`json`, and `uuid` where applicable. Do not add a CLI framework or vector database.
NumPy 2.3.5 retains Python 3.11 support; ONNX Runtime 1.23.2 has CPU wheels for
the selected macOS architectures and Python versions. These pins were checked
against release metadata, not exercised as an installed dependency set here.

Use SQLite FTS5 for lexical search and NumPy exact cosine search for dense
vectors. Fail setup with `fts5_unavailable` if the Python SQLite build lacks
FTS5; do not silently replace hybrid retrieval with semantic-only search.
SQLite databases are instantiated in memory and serialized into authenticated
encrypted files. Require `Connection.serialize`/`deserialize` in M0. This avoids
an unencrypted FTS index, WAL, journal, or temporary database on disk; it is not
SQLCipher or an assertion that ordinary SQLite files are encrypted.

## 3. Common types and invariants

All public records use Pydantic models with forbidden unknown fields and strict
validation. Mutable execution state stays internal; returned records are frozen.
`JSONValue` means JSON primitives, lists, and string-keyed objects; NaN,
Infinity, arbitrary Python objects, and pickle are prohibited.

IDs are UUID4 strings except child workflow run IDs, which are UUID5 derived
from the root run ID and step name. Timestamps are UTC RFC 3339. JSON persistence
uses UTF-8, sorted keys, compact separators, and `allow_nan=False`.
Set `ensure_ascii=False` for every canonical JSON encoding, including byte
budgets, argument digests, fingerprints, and authenticated headers.

| Record | Required content |
| --- | --- |
| `AgentSpec` | Name, version, instructions, provider/profile or adapter, schema, limits, tool descriptors, store bindings, document access policy. |
| `Message` | Role, text/content, normalized tool calls/results, document provenance, optional opaque provider continuation state. |
| `ToolCall` | Canonical call ID, provider call ID if supplied, tool name, JSON arguments, model turn and ordinal. |
| `ToolObservation` | Call ID, `execution_status` (`not_started`, `succeeded`, `failed`, `unknown`), validated output or null, structured observation error, omission metadata, document provenance. |
| `ConversationState` | Schema version, conversation ID/version, caller subject, messages and cumulative provenance, JSON application state, store/snapshot bindings, continuation eligibility. |
| `RunError` | Stable code, safe message, origin, retryable flag, outcome-unknown flag, and JSON details. |
| `Usage` | Model requests, logical model turns, tool attempts, retries, corrections, nullable input/output token totals, `complete` flag, elapsed milliseconds. |
| `Diagnostic` | Code, safe message, origin, severity, JSON details. |
| `CancellationToken` | Thread-safe `cancel()` and `is_cancelled()` operations usable by sync and async callers. |
| `CallerIdentity` | Application-supplied subject, frozen roles, JSON attributes. Default is anonymous with no roles; never infer identity from prompts. |
| `ApprovalRequest` | Tool/call/version, validated arguments, caller identity, egress destination, document-source IDs, and `contains_document_content`. |
| `ApprovalDecision` | Strict boolean `approved` and string `reason`; denial requires a nonempty reason. |

Canonical tool IDs are `tc_<run_id>_<turn_number>_<ordinal>`. Keep provider IDs
separately for native protocol round trips. Duplicate/malformed native IDs
produce `provider_protocol_error`; never correlate results by list position
alone. For providers without IDs, assign canonical IDs and use the provider's
name/part correlation rules.

Retain required opaque continuation fields, such as provider tool-call
signatures, in checkpoints. Do not interpret or display them as model reasoning.
Do not request hidden chain-of-thought.

### 3.1 Schema contract

Accept `type[BaseModel]`, a JSON Schema object, `"text"`, or `"document_qa"`.
Default to `"text"` only when no schema is supplied by code or configuration.
Raw schemas use Draft 2020-12 with local `$defs`/`$ref`; external references
are rejected at configuration time. Validate schema definitions before running.

Infer tool arguments from fully annotated callable parameters, or accept an
explicit Pydantic/schema definition. Reject missing annotations, `*args`,
`**kwargs`, positional-only parameters, and non-JSON argument types.
An optional keyword-only `context: ToolContext` is injected by Loom and excluded
from model-visible arguments. Output is validated against its declared schema.
When omitted, the tool name is the callable's `__name__`, the description is
its trimmed docstring, and the output schema comes from its return annotation.
A missing description/return annotation without an explicit replacement is a
registration error. Tool names match `[a-z][a-z0-9_]{0,63}`; this accommodates
the `search_` prefix on store names, whose own maximum length is 48.

Use native structured output only when both the model capability profile and
the adapter support the supplied schema. Otherwise issue explicit JSON
instructions and enforce the unchanged schema locally. Report
`native_schema_not_used` as a diagnostic; do not weaken the schema.

## 4. Public API

Export from `loom`: `Agent`, `Tool`, `HTTPTool`, `DocumentStore`, `Workflow`,
`Step`, `Branch`, `Case`, `Storage`, `SQLiteStorage`, `JSONStorage`, `Config`,
`Limits`, `WorkflowLimits`, `RunResult`, `ConversationState`, `CancellationToken`,
`ToolContext`, `ToolFailure`, `CallerIdentity`, `ApprovalRequest`,
`ApprovalDecision`, `ToolSpec`, `LoomConfigurationError`, and `LoomStorageError`.

The following signatures are the required interface. They are specification
signatures, not code already present in the repository.

```text
Agent(
    *, name, instructions, version="1", model=None, tools=(), data=(),
    output_schema=None, limits=None, config=None, storage=None,
    approval=None, event_sinks=(), document_access=None, tool_document_access=None
)
Agent.run(*, input: str, state=None, context=None, caller=None, cancellation=None) -> RunResult
Agent.arun(*, input: str, state=None, context=None, caller=None, cancellation=None) -> RunResult
Agent.resume(run_id: str, *, extend_deadline_seconds=None,
             context=None, caller=None, cancellation=None) -> RunResult
Agent.aresume(run_id: str, *, extend_deadline_seconds=None,
              context=None, caller=None, cancellation=None) -> RunResult
Agent.resolve_tool_outcome(run_id, call_id, *, resolution,
                           output=None, error=None, operator_note: str) -> None

Tool.from_callable(
    fn, *, name=None, description=None, version="1",
    argument_schema=None, output_schema=None,
    sensitive=True, retry_safe=False, timeout_seconds=None,
    egress="unknown", egress_origins=()
) -> Tool

HTTPTool(
    *, name, description, method, endpoint, request_builder,
    argument_schema, output_schema, version="1", headers=None,
    auth_env=None, sensitive=True, retry_safe=False, timeout_seconds=None
)

DocumentStore(*, path, name=None, index_path=None)
DocumentStore.search(query: str, *, snapshot_id=None) -> SearchResult
DocumentStore.asearch(query: str, *, snapshot_id=None) -> SearchResult

SQLiteStorage(*, path, config=None)
JSONStorage(*, path, config=None)
Storage.load_run(run_id) -> Checkpoint
Storage.load_conversation(conversation_id) -> ConversationState
Storage.export_run(run_id) -> bytes
Storage.read_export(encrypted_envelope: bytes) -> JSONValue
Storage.delete_run(run_id) -> None
Storage.delete_conversation(conversation_id) -> None
```

`config` accepts a validated `Config`, a TOML path, or `None` for discovery.
`model` accepts a provider profile name or an application-supplied `ModelAdapter`.
`context` is a runtime-only mapping of application objects; it is not
automatically sent to models, logged, or persisted. Explicit JSON application
state belongs in `ConversationState`.

Every `run` creates a new run ID. Without `state`, it also creates a fresh
conversation. Supplying a completed conversation state is the only automatic
history continuation mechanism; reusing an Agent object does not share history.
Incomplete states must use `resume`, not start a new run as if a pending tool
had completed.

`run`, `resume`, and synchronous search raise configuration errors if invoked
from an active event loop in the same thread. Call their async counterparts.
Sync execution uses the same async engine, not a separate behavioral loop.

Store construction performs no source-folder scan or indexing. Bind a store
by its explicit name or unique normalized absolute path in configuration.
When `index_path` is supplied in code, a name is also required. Duplicate names,
ambiguous path matches, or unresolved bindings are configuration errors.
Query binding uses lexical absolute-path normalization, not live source reads.

Custom tools implement the `Tool` execution contract; they do not require a
different registration or discovery system. Re-creating an Agent on resume
must reattach executable tools and runtime context explicitly.
An explicitly configured Storage uses its own Config for key resolution.
Otherwise it binds the owning Agent/Workflow Config; standalone Storage calls
use normal TOML discovery. Resolve keys at first I/O, not construction.

### 4.1 Custom tools, context, and approval

`Tool` is an abstract base class with an immutable validated `spec: ToolSpec`
and exactly this abstract method:

```text
async execute(
    self, arguments: Mapping[str, JSONValue], context: ToolContext
) -> JSONValue | BaseModel
```

`ToolSpec` contains name/version/description, argument/output schemas,
sensitivity, retry-safety, timeout, `egress` and `egress_origins`.
`egress` is `none`, `fixed_network`, or `unknown`. HTTPTool is
`fixed_network` with its configured HTTPS origin; search is `none`.
Other custom/function tools default to `unknown` unless their trusted author
explicitly declares their behavior. The runner, not `execute`, owns argument
validation, permission checks, approval, counters, serialization, and retries.
Synchronous implementations use `Tool.from_callable`.
Custom subclasses call `Tool.__init__(spec: ToolSpec)`; `ToolSpec` uses the same
field names/defaults as `from_callable`, with required name, description,
argument_schema, and output_schema. Origins are a tuple of canonical
`https://host[:port]` strings without credentials/path/query/fragment; omit
the default port 443 and lowercase hostnames. `fixed_network` requires at least
one origin; `none`/`unknown` require an empty tuple.

`ToolFailure` is an Exception with validated code/message/details and the
existing recoverable/transient/outcome-unknown flags. Returning normally means
the invocation returned; an observation error must not relabel a completed
side effect as an unexecuted operation.

`ToolContext` is a frozen record containing root/run/conversation IDs,
optional step ID, call ID, 1-based attempt, UTC deadline, cancellation token,
trusted `CallerIdentity`, read-only runtime context, cumulative document-source
IDs, and the remaining observation-byte allowance. It contains no mutable
RunState and provides no method to widen permissions or remove provenance.

An approval callback has signature
`(ApprovalRequest, ToolContext) -> ApprovalDecision | Awaitable[ApprovalDecision]`.
Run sync callbacks in the `user_code` pool from section 4.3 and await async
callbacks. Pool saturation denies with `approval_executor_busy`; it never
falls back to a terminal prompt.
Limit them to `approval_timeout_seconds` and the remaining run deadline.
Exceptions/invalid results mean denial plus `approval_callback_failed` or
`approval_invalid_result`; do not silently switch to terminal approval.
Cancellation remains cancellation, never approval. Framework-created
decisions and callback decisions have identical validation.

Callbacks receive validated immutable arguments for authorization; the terminal
uses a separately redacted display. Always show whether document content is
involved. Application-supplied identity is trusted input, not authentication
performed by Loom; reacquire identity and approval on resume.
Caller subject is a nonempty string (default `"anonymous"`), roles are a
tuple of unique strings, and attributes are a JSON object. Resume/continuation
requires the same subject as the saved state; fresh roles/attributes are used
for authorization. A different subject fails with `caller_mismatch`.
Loom does not implement administrator impersonation or identity authentication.

### 4.2 Object-sharing and cache lifetime

Agent, DocumentStore, and Storage instances are shareable across threads and
async tasks. Their configuration is immutable; each run owns state and clients.
Serialize `execute` calls on the same Tool object with a per-instance lock.
Different tool instances may execute in different runs concurrently, never
concurrently within one run.
Claim a Tool instance atomically, without waiting, after validation/egress
checks and before approval. If already claimed, return the recoverable
`tool_busy` observation immediately with `execution_status=not_started`,
`retryable=false`, and `blocking_call_id`, `blocking_run_id`, and
`blocking_phase` (`approval`, `running`, or `timed_out_running`) in details.
Do not charge a tool attempt, wait until the new run's deadline, or schedule an
automatic retry. Direct tool workflow steps fail with the same code.

Keep the claim through approval and execution. Release it on denial or failure
before dispatch; after dispatch release only when the actual invocation finishes,
including a worker that outlives cancellation/timeout. The live-invocation
registry owns release, not the timed-out awaiting task. Late completion cannot
alter finished state or release another call's claim. Even a retry-safe tool is
not automatically retried while its previous invocation is still running.
Do not replace a Tool object or create replacement workers to bypass a stuck
invocation. An explicit application-created instance is still trusted code,
not evidence that the original side effect was resolved.

Use process-local locks as well as OS locks, keyed by canonical artifact path;
do not rely on file locks to serialize threads. Async callers offload blocking
crypto/storage/inference work to their section-4.3 pools. Storage lock waiting consumes
the caller's deadline. Application-injected adapters default to not shareable:
reject simultaneous reuse with `adapter_in_use` unless the adapter explicitly
declares `concurrency_safe=True`.

Keep one process-local FastEmbed instance per
`(artifact revision, CPU provider, thread count)` with single-flight initialization
and serialized inference. Reuse it across repeated synchronous runs; it is not
an event-loop-owned provider client. Decrypted snapshots use the bounded
process-wide cache in section 9.5, not an independent cache per DocumentStore.
Runs retain snapshot IDs, not database objects; a query leases its database only
until the copied SearchResult is ready. Clear inherited caches/locks after fork;
forking with active SDK workers is unsupported.

### 4.3 Executor isolation and bounded admission

Use three lazily created, process-wide ThreadPoolExecutors, independent of the
event loop's default executor. These fixed v0.1 counts are not per-Agent limits:

| Pool | Workers | Maximum queued jobs | Work permitted |
| --- | --- | --- | --- |
| `user_code` | 4 | 0 | Synchronous tools, HTTP request builders, approval/event callbacks, workflow selectors/predicates. |
| `internal_io` | 2 | 8 | Framework-only file I/O, encryption/decryption, checkpoint/index publication, trace appends/rotation. Never application callbacks. |
| `inference` | 1 | 2 | Local embedding initialization/inference and hybrid-search computation. Never application tools or callbacks. |

Bound admission by running plus queued jobs, including reserved submissions.
The standard executor's unbounded queue is not the admission mechanism.
Reserve a slot before submission; fail immediately with `executor_busy` and
pool name/capacity when full. No automatic pool growth or replacement is allowed.
Tools rejected before dispatch produce a recoverable not-started observation
without consuming an attempt. If a later internal stage is rejected after an
attempt began, retain its charge and report the stage failure; do not relabel
earlier execution as not started. Workflow control fails; approval denies; event callbacks disable the
sink with a diagnostic. Unavailable checkpoint I/O stops progress with
`storage_write_failed` and `cause=executor_busy`, not a false successful write.

Approval finishes before reserving the execution slot, so a tool cannot occupy
the last user-code slot while waiting for its own approval callback. Once a
tool execution slot is reserved, checkpoint its start using `internal_io` before
dispatch. Release unused reservations if that checkpoint fails.

Queued waiting counts against the operation deadline. Protect underlying futures
from propagation of asyncio cancellation: mark cancelled/expired queued jobs
to be skipped when dequeued, without invoking user code. Release capacity only
after a job physically leaves the queue and finishes/skips, not merely when an
awaitable is cancelled. This also bounds cancelled queue entries. Running jobs
retain capacity until actual completion. Never let a worker synchronously wait
for another job in its own pool; coordinate cross-pool stages from the async
runtime. Release Tool claims, memory leases, and capacity in actual-completion
handlers even if the originating event loop has closed.

Saturating all four user-code workers must not prevent unrelated checkpoint
writes, decryption, or retrieval. User-code operations then fail/deny visibly
until capacity returns. This is scheduling isolation, not process isolation:
native code holding the GIL, arbitrary allocation, or OS I/O failure remains
outside the guarantee. Python cannot forcibly stop a running thread.
`shutdown(wait=False)` does not guarantee interpreter exit with hung executor
threads; document supervisor-managed process restart for permanently stuck work.

## 5. Result and document-answer schemas

### 5.1 `RunResult`

Fields, with no extra keys:

| Field | Type/meaning |
| --- | --- |
| `schema_version` | Integer literal `1`. |
| `run_id`, `conversation_id` | UUID strings. |
| `status` | `completed`, `failed`, `cancelled`, or `limit_exhausted`. |
| `output` | Validated JSON/text on completion; `null` for other statuses. |
| `error` | `null` on completion; `RunError` otherwise. |
| `usage` | `Usage` record. |
| `snapshots` | Map of store name to immutable snapshot ID. |
| `diagnostics` | Ordered list of diagnostics, including partial index coverage. |
| `state` | Conversation state for explicit continuation; excluded from default display. |

Provide `to_dict(include_state=False)` and `to_json(include_state=False,
indent=2)`. `str(result)` returns `to_json()`, so the unchanged `print(result)`
prints the complete JSON result envelope, without raw conversation state.
Do not implicitly print exception tracebacks, application context, or credentials.

Abstention/conflict are completed domain outcomes. Overall deadline, model-turn,
tool-attempt, context, and workflow limits produce `limit_exhausted` with a
specific reason, including provider-reported `context_limit`. A timed-out
attempt that exhausts its retries before the overall deadline produces
`failed`, not a fabricated success.

### 5.2 Document output

```json
{
  "status": "answered",
  "answer": "The API allows 500 requests/minute.",
  "sources": [
    {
      "store": "product_docs",
      "snapshot_id": "immutable-snapshot-id",
      "chunk_id": "sha256-chunk-id",
      "path": "limits.md",
      "location": {"kind": "markdown", "section": "Rate limits", "line_start": 1},
      "excerpt": "The API allows 500 requests/minute."
    }
  ]
}
```

The IDs above illustrate runtime-generated values, not permitted literal
values in the fixture. Acceptance tests obtain actual IDs from the normal
index builder. Unit tests inject a fixed UUID factory for repeatability;
production snapshot/run IDs still follow the UUID rules above.

`status` is `answered`, `abstained`, or `conflict`; `answer` is a nonempty string.
`sources` is a list of citation records with exactly the keys shown.
Locations are a tagged union:

- PDF: `{"kind":"pdf","page":<1-based integer>}`.
- Markdown: `{"kind":"markdown","section":<string or null>,"line_start":<integer>}`.
- DOCX: `{"kind":"docx","section":<string or null>,"block":<1-based integer>}`.

`answered` requires at least one verified citation. `conflict` requires at
least two citations identifying distinct source locations and an explanation
that does not select an authority. `abstained` uses `sources: []` and a clear
insufficient-evidence explanation. All three require a successful search in
the current run, including an empty-result search for absence-based abstention.
There is no `confidence` field.

Validate citation store, snapshot, chunk, path, and location against retrieved
chunks. The excerpt must be a nonempty contiguous substring of the stored
normalized chunk, no more than 1,200 characters. Deduplicate identical citations.
Paths are source-root-relative POSIX paths, never absolute paths or URLs.
Generation instructions require support for each factual claim; deterministic
validation does not claim to prove semantic entailment or discover every
possible contradiction. There is no second LLM verification call.

## 6. Configuration

### 6.1 Loading and precedence

Read `./loom.toml` from the working directory unless an explicit path/Config
was supplied. Do not search parent directories. If it is absent, purely
programmatic configuration is allowed; any required unresolved value is an
error. Resolve configuration-relative paths against the TOML directory.

For ordinary fields: explicit constructor values override named agent fields,
which override `[defaults]`. Merge nested limits by field. Absence means inherit;
explicit invalid/null values do not mean "silently use a default".

Only attached code tools and store-search tools exist. A configured allowlist
can narrow this set, never register tools. Missing allowlist means precisely
the explicitly attached set. Reject unknown allowlist names.

Document access is a per-store map on an agent, with constructor entry taking
precedence over its named profile; a missing entry is `local_only`. There is no
project-global `allow_hosted` switch. Reject permissions for unattached stores.

Config schema version is `1`; reject unknown top-level sections/fields.
Supported sections: `schema_version`, `defaults`, `providers`, `agents`,
`stores`, `embeddings`, `storage`, `transport`, `encryption`, and `observability`.

Allowed fields for non-provider sections are exhaustive:

| Section | Allowed fields |
| --- | --- |
| `defaults` | `provider`, `output_schema`, `limits`; unset provider means no default provider. |
| `agents.<name>` | `provider`, `output_schema`, `limits`, `allowlist`, `document_access`, `tool_document_access`. Instructions remain explicit constructor input. |
| `stores.<name>` | Required `path` and `index_path`. |
| `embeddings` | `backend`, `model`, `cache_path`; defaults are exactly those in the reference configuration. Other backends/models are rejected in v0.1. |
| `storage` | `enabled`, `backend`, `path`. |
| `transport` | Optional `ca_bundle_env`; no verification-disable, TLS-downgrade, or ambient-proxy option. |
| `encryption` | `active_key_id`, `keys` mapping key IDs to environment-variable names. No plaintext/disabled mode. |
| `observability` | `capture_payloads`, `capture_source_paths`, `jsonl_path`, `redact_keys`. |

Provider/agent/store profile names use the store-name identifier syntax.
Limit fields are exactly the controls listed in section 7.2. When no TOML is
used, relative programmatic paths resolve against the working directory.
Require model-cache and index paths to be outside every configured source root,
so setup and refresh cannot ingest their own generated artifacts.

### 6.2 Complete reference configuration

This is the schema to implement, not a menu of alternatives:

```toml
schema_version = 1

[defaults]
output_schema = "text"

[defaults.limits]
max_model_turns = 10
max_tool_attempts = 20
deadline_seconds = 120
model_timeout_seconds = 30
tool_timeout_seconds = 30
approval_timeout_seconds = 60
max_retries = 2
max_corrections = 1
max_context_bytes = 65536
max_output_tokens = 2048
max_observation_bytes = 8192
max_tool_arguments_bytes = 8192

[providers.primary]
kind = "ollama"
model = "qwen2.5:7b"
model_digest = "sha256:845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"
server_version = "0.40.2"
base_url = "http://127.0.0.1:11434"
native_tool_calling = true
structured_output = "json_schema"
num_ctx = 32768
num_gpu = 0
num_thread = 4

[agents.product_expert]
provider = "primary"
output_schema = "document_qa"

[agents.product_expert.limits]
deadline_seconds = 600
model_timeout_seconds = 180
max_context_bytes = 24576

[agents.product_expert.document_access]
product_docs = "local_only"

[stores.product_docs]
path = "./product-docs"
index_path = ".loom/indexes/product_docs"

[embeddings]
backend = "fastembed"
model = "BAAI/bge-small-en-v1.5"
cache_path = ".loom/models"

[storage]
enabled = false
backend = "sqlite"
path = ".loom/state.sqlite.enc"

[encryption]
active_key_id = "primary"

[encryption.keys]
primary = "LOOM_ENCRYPTION_KEY"

[observability]
capture_payloads = false
capture_source_paths = false
jsonl_path = ""
redact_keys = []
```

The reference server is Ollama 0.40.2, release commit
`b061384d90ff455462bc32745dd0de479de717a3`; the model tag must match the immutable
digest above. Install/pull it explicitly outside Loom. Verify `/api/version`
and the installed `/api/tags` digest before use; reject mismatches rather than
silently updating models. Other Ollama profiles must also declare an expected
version/digest and pass the same overflow compatibility tests.
For digest comparison, strip one optional `sha256:` prefix and validate exactly
64 hexadecimal digits before comparing lowercase values. Read `/api/show` for
the model architecture's declared context length; missing/invalid metadata is
a capability error, not permission to assume an unlimited window.
`num_ctx` is a required positive integer; `num_gpu` defaults to 0 and
`num_thread` to 4. Perform server-dependent checks at run preflight, never at
object construction. Local limit arithmetic uses the agent's fully merged
limits, not the unmodified project defaults.

`providers.<name>.kind` is one of `openai`, `azure_openai`, `anthropic`, `gemini`,
`ollama`. `model` and `native_tool_calling` are required; `structured_output`
is `json_schema`, `json_object`, or `prompt_json`. Adapter-specific unsupported
mode combinations fail configuration validation. Client/model rejections still
fail explicitly even if the operator declared a capability incorrectly.

Provider fields:

| Provider | Required deployment configuration | Credential source |
| --- | --- | --- |
| OpenAI | `model`; optional `base_url` | `api_key_env`, default `OPENAI_API_KEY` |
| Azure OpenAI | `model` containing deployment name, `endpoint`, `api_version` | `api_key_env`, default `AZURE_OPENAI_API_KEY` |
| Anthropic | `model`; optional `base_url` | `api_key_env`, default `ANTHROPIC_API_KEY` |
| Gemini | `model`; Gemini Developer API in v0.1, not Vertex AI | `api_key_env`, default `GEMINI_API_KEY` |
| Ollama | `model`, `model_digest`, `server_version`, `num_ctx`; optional `num_gpu`, `num_thread`; default loopback base URL | Optional environment-key reference, no key by default |

Reject embedded keys and endpoint user-info credentials. Do not persist
resolved environment secrets or silently load `.env` files.

Storage `backend` is `sqlite` or `json`; when enabled, `path` is required and
selects `SQLiteStorage` or `JSONStorage`. An explicit Storage constructor
argument overrides TOML storage. Ordinary runs are ephemeral when disabled.
The search index stays an encrypted SQLite image regardless of the conversation
storage backend. Encryption keys are needed whenever indexes, durable state,
exports, or trace files are opened/written, even when conversation durability
is disabled. Public model setup alone does not require a data-encryption key.

`output_schema` in TOML is `text`, `document_qa`, or a local JSON Schema file
path relative to TOML. Python model classes are supplied only in code, not
imported through configuration strings.

### 6.3 Consent and locality

`document_access.<store>` is `local_only` or `allow_hosted`.
Only Ollama using a literal loopback IP address qualifies as local. All other
provider kinds and non-loopback Ollama endpoints require `allow_hosted` for
each store whose content might enter the request. Do not assume a provider
proxy on loopback makes a hosted provider local.

Preflight an attached document agent before its first provider call, then
check document provenance on every request, including history and resume.
Missing consent is `document_transfer_denied`; transmit no document-containing
request. Provenance tags include store and snapshot IDs for observations,
citations, and derived messages. Retain them across serialization/continuation.
Tools or logs cannot grant new consent.

### 6.4 TLS and verified endpoints

Require HTTPS for provider endpoints, HTTPTool endpoints, model downloads, and
Mend connections. The only HTTP exception is the explicitly configured Ollama
provider on a literal loopback address; it does not apply to HTTPTool.
Reject URL user-info, non-HTTPS redirects, and invalid certificates/hostnames.

Create SSL contexts with `ssl.create_default_context`, minimum TLS version
`TLSv1_2`, `CERT_REQUIRED`, and `check_hostname=True`. A configured
`ca_bundle_env` resolves a trusted CA-file path; it adds deployment trust
configuration without disabling verification. Never expose or pass
`verify=False`, `CERT_NONE`, or an unverified SSL context.
Use httpx with `trust_env=False` to prevent implicit proxy/verification changes.
Pass the verified transport to each provider client. Application-injected
adapters must honor this same contract; arbitrary trusted Python code is not
made safe merely by declaring compliance.

### 6.5 Document egress through tools

Document access permission and sensitive-action approval are separate checks.
Neither `sensitive=False` nor an approving callback may override an egress
denial. Check policy before approval and again immediately before every
dispatch/retry, including workflow steps and resumed work.

Conservatively taint all model-generated tool arguments with the union of
document-source IDs present in that model's context. Propagate taint through
tool outputs, derived messages, selectors, state, and checkpoints. Do not
attempt substring-based proof that an argument is free of document content,
and do not let a tool remove existing provenance.

Rules for tainted arguments:

- `egress=none`: allow only under the trusted tool author's no-network contract.
- `egress=unknown`: deny with `document_egress_denied`.
- `egress=fixed_network`: every tainted store must have `allow_hosted` AND be
  explicitly listed in `agents.<name>.tool_document_access.<tool_name>`.
  Otherwise deny. Configured tool origins must match the dispatched origin.

`tool_document_access` maps attached tool names to lists of attached store
names. Default is an empty map; grants for `local_only` stores are invalid.
For example, permission to use a hosted LLM never by itself grants permission
to post the same documents through an API tool.

These are enforceable checks on Loom-managed dispatch. Custom Python tools,
callbacks, and injected adapters are trusted application code, not a network
sandbox; a developer who lies about a no-network tool can defeat this boundary.
Do not claim protection against malicious application code or direct network
calls outside the framework.

## 7. Model execution, limits, and tools

### 7.1 Provider adapter interface

`ModelAdapter` exposes `capabilities`, `async generate(ModelRequest) ->
ModelResponse`, and `async aclose()`. A request contains instructions,
normalized messages, tool schemas, output schema/mode, maximum output tokens,
and the remaining attempt timeout. A response contains tool calls or final
text/JSON, normalized finish reason, opaque continuation fields, and usage.

Use these client surfaces:

| Adapter | Client surface |
| --- | --- |
| OpenAI | `AsyncOpenAI`, Chat Completions, streaming disabled. |
| Azure OpenAI | `AsyncAzureOpenAI`, Chat Completions, deployment passed as model. |
| Anthropic | `AsyncAnthropic`, Messages API, tool-use/tool-result blocks. |
| Gemini | `google.genai.Client().aio.models.generate_content`, function declarations/responses. |
| Ollama | Verified `httpx.AsyncClient` calling Ollama 0.40.2 `/api/chat`, tools and `format`, `stream=false`. |

Map `max_output_tokens` to OpenAI/Azure `max_completion_tokens`, Anthropic
`max_tokens`, Gemini `config.max_output_tokens`, and Ollama
`options.num_predict`. Do not silently omit a configured output bound.

The earlier Python client 0.6.3 does not expose chat `truncate`/`shift` controls.
Use the public HTTP API directly, not private client methods or an unsupported
keyword. Every Ollama request sends `truncate=false`, `shift=false`,
`options.num_ctx`, and the declared CPU/thread options. Verify the pinned
server/model before use; unsupported controls or overflow behavior block M0.

For Ollama configuration admission require:
`num_ctx >= max_context_bytes + max_output_tokens + 4096`.
This deliberately uses a conservative byte-as-token upper allowance plus
template reserve, not the inaccurate assumption that four bytes always equal
one token. Limit normalized conversation messages to 64 and registered tools
to 16, and validate the pinned template under those bounds in M0.
Reject a context larger than the model's declared supported window.
The server's reject-on-overflow behavior remains authoritative: no truncation
or generation-time context shifting is allowed even if the conservative
admission estimate misses template overhead. Normalize context rejection to
`context_limit`; never retry it as a transient failure.

The reference uses 32,768 context tokens and 24,576 context bytes, leaving
2,048 output tokens plus 4,096 template allowance. Its explicit model timeout
is 180 seconds and run deadline 600 seconds; tools retain the 30-second timeout.
General SDK defaults remain 30 seconds per model/tool attempt and 120 seconds
per run. The end-to-end acceptance gate, not the retrieval-only benchmark,
must prove the reference profile works.

Use native schema mode where the configured adapter supports it; `prompt_json`
is the common fallback for schemas the native mode cannot represent. Do not
fallback to another provider or emulate tool calling by parsing ordinary text.
For tool-enabled runs without native tool support, raise
`model_capability_unsupported` before execution.

Native schema request mapping:

| Adapter | `json_schema` request field | `json_object` request field |
| --- | --- | --- |
| OpenAI/Azure | `response_format={"type":"json_schema","json_schema":{"name":"loom_output","strict":true,"schema":...}}` | `response_format={"type":"json_object"}` |
| Anthropic | `output_config={"format":{"type":"json_schema","schema":...}}` | Unsupported; reject this configured mode. |
| Gemini | `config.response_mime_type="application/json"` and `config.response_json_schema` | `config.response_mime_type="application/json"` |
| Ollama | `format=<schema object>` | `format="json"` |

The common native schema subset is an object root using `type`, `properties`,
`required`, `additionalProperties`, `items`, `enum`, `anyOf`, `description`,
and `title`. Every object must forbid additional properties and list all of
its properties as required; use nullable unions for optional values. Reject
root-level `anyOf` for native dispatch. Schemas containing other keywords use
`prompt_json` with local validation and a diagnostic; do not remove validation
keywords to force native compatibility. `prompt_json` sends the full schema
as an instruction and omits native output-format fields. Native tool calling
remains enabled in every mode; an actual vendor incompatibility fails clearly.

Disable client/transport retries wherever Loom owns retry accounting:
OpenAI/Anthropic retries zero, Gemini automatic HTTP retry attempts one, HTTP
transport retries zero. Assert actual request counts in adapter tests.
Finish reasons normalize to `tool_calls`, `completed`, `refused`, or
`output_limit`. Refusal is `model_refusal`; truncated output is
`output_truncated`. Neither is a valid schema-compliant completion.
Framework-created clients are owned by one run and closed at its end; do not
cache loop-bound async clients across synchronous calls. Application-injected
adapters are caller-owned and are not closed by Loom.

### 7.2 Limit definitions

All overrides must be finite; integer limits are nonnegative, with model
turns, tool attempts, context bytes, and output tokens positive. Deadline and all
timeout controls are positive finite seconds. Observation limits are at least
1,024 bytes and argument limits are positive. No `None`, infinity, or sentinel
for unlimited execution.
`max_retries` is an integer from 0 through 2; `max_corrections` is 0 or 1.
Zero disables that allowance. Check the actual remaining correction allowance
before the first corrective instruction rather than assuming one is available.

| Counter/control | Default | Exact semantics |
| --- | --- | --- |
| Logical model turns | 10 | Increment before the first request for a turn. Retries stay in that turn; observations and corrections start new turns. |
| Tool attempts | 20 | Increment and checkpoint before entering a callable/HTTP dispatch, including retries. Invalid/denied requests do not enter the tool. |
| Deadline | 120 s | Starts when run is admitted, before approvals/model work; includes backoff, events, and waiting. |
| Model timeout | 30 s | `model_timeout_seconds`, capped by remaining run deadline; reference profile explicitly overrides to 180. |
| Tool timeout | 30 s | Tool override or `tool_timeout_seconds`, capped by remaining run deadline. |
| Approval timeout | 60 s | `approval_timeout_seconds`, capped by remaining run deadline; timeout denies. |
| Retries | 2 | Per logical model request or tool invocation: initial attempt plus at most two. |
| Corrections | 1 | Shared by output-schema, citation, and missing-search corrections over the entire run. |
| Context bytes | 65,536 | UTF-8 size of canonical normalized model request excluding transport/auth metadata. Includes instructions, schemas, and conversation. |
| Output tokens | 2,048 | Provider request limit, independent of the local context-byte safeguard. |
| Observation bytes | 8,192 | Maximum canonical model-visible tool observation, including its envelope. Search uses a stricter 4,096-byte cap. |
| Tool argument bytes | 8,192 | Per-call canonical arguments before validation/approval/dispatch. |

The byte budget is a deterministic local context safeguard, not a fabricated
token count. Also surface provider context-window errors as `context_limit`;
do not silently remove history, tool pairs, or instructions. Provider usage
tokens are nullable: aggregate only reported totals and mark `complete=false`
if any attempt's usage is unavailable.
Compute context bytes over canonical JSON containing exactly `instructions`,
`messages`, `tools`, and `output_schema`. Do not include changing timeout
values or credential/transport fields in this measurement.
Aggregate monetary/token-spend budgets are not v0.1 features. Usage accounting
and per-request output bounds satisfy the original requirement; do not report
an inferred currency budget or silently treat unknown token usage as zero.

Persist logical turn/attempt identities before dispatch in durable mode.
A crash cannot reset an attempt. Retrying an interrupted model request
consumes another attempt under the same turn and leaves unknown usage marked.
Resume preserves the original absolute deadline. `extend_deadline_seconds`
is a positive finite duration; it sets the deadline to the later of the
existing deadline and now plus that duration. Record the extension. It never
resets counters or increases step/model/tool ceilings.

### 7.3 Loop and failure policy

1. Validate setup, bind stores/configuration, check capabilities and consent.
2. Allocate isolated state or lock/load the requested durable root run.
3. Check cancellation, deadline, context size, and remaining model turns.
4. Reserve a logical turn and call the adapter under bounded attempt retries.
5. If tools were requested, persist the full ordered list, then validate,
   enforce egress policy, approve, and execute each call sequentially.
   Record every bounded observation and its separate execution status.
6. Return recoverable tool errors as observations; stop on unknown unsafe
   outcomes or nonrecoverable execution failures.
7. For a final answer, validate schema and evidence. On the first violation,
   add one corrective instruction and consume a new model turn. A subsequent
   violation produces `output_validation_failed`. Do not create another loop.
8. Save the terminal result where durable and emit the terminal event.

The document agent must search in this run before producing any final domain
answer. A no-search answer consumes the same correction allowance as an
invalid citation. Exhausted limits take precedence over scheduling a correction.

Use full-jitter exponential backoff: before retry 1, random delay in
`[0, 0.5]` seconds; before retry 2, `[0, 1.0]`. Inject randomness/clock in tests.
If a valid `Retry-After` requires a longer wait, wait at least that long only
when it fits the remaining deadline; otherwise stop at the deadline.

| Failure | Policy |
| --- | --- |
| Invalid setup/schema, missing credentials/artifacts, incompatible checkpoint | Typed configuration/storage exception before new work. |
| Model timeout, transport failure, HTTP 408/429/5xx | Transient bounded retry; exhaustion becomes failed result. |
| Authentication, unsupported capability, invalid request, context overflow | No transient retry. |
| Unknown/denied tool, invalid arguments, known recoverable tool-domain error | Structured observation; no dispatch for invalid/denied calls. |
| Declared retry-safe tool with transient failure | Retry inside both invocation and run attempt limits. |
| Unclassified tool exception or invalid tool output | Failed run, safe error summary, no automatic retry. |
| Unsafe tool with uncertain effect after dispatch | Stop, `unknown_tool_outcome`, require explicit resolution. |
| Durable-write failure | Stop immediately with `storage_write_failed`; never continue on an unrecorded checkpoint. |

`ToolFailure` carries a code, safe message, `recoverable`, `transient`, and
`outcome_unknown` flags. Defaults are false. Validation/schema programming
errors are not converted into success-shaped tool outputs.

### 7.3.1 Bounded observations without repeating side effects

Limit decoded provider responses to 256 KiB and raw serialized tool arguments/
outputs to 1 MiB. Count decoded/serialized bytes incrementally before accumulating
unbounded output. The HTTP wire/body limit and the 8 KiB model-observation
limit are different controls. A trusted callable can still allocate arbitrary
memory internally; this is not a memory sandbox.

Before entering a tool, reserve context space for a 1 KiB minimal observation
per remaining call (or its actual serialized minimal error size, if larger)
plus a 2 KiB continuation/correction allowance. If even
minimal observations will not fit, return `limit_exhausted` before side effects.
The actual observation allowance is the smaller of the configured cap and
remaining context space after reserving the correction allowance and the
minimal observations for all subsequent calls. The current call's minimal
reservation remains available to that call. Recompute using exact canonical
bytes after each observation; argument/observation limits cannot be configured
above the corresponding raw safety caps.

If a returned output exceeds its observation allowance, persist a bounded
observation with `execution_status=succeeded`, `output=null`, and
`error.code=observation_too_large`. Set retryable false and
`details.operation_executed=true`; include byte count and SHA-256 only if the
complete bounded output was available, otherwise mark size/hash unavailable.
The `omission` field is null normally; otherwise it contains exactly
`byte_count: int | null`, `sha256: str | null`, and
`validation: "validated" | "not_performed"`. Do not claim schema validation for
an output omitted before validation. An observed schema violation still fails
the run; the size-error path is not permission to ignore a known violation.
This is a recoverable observation problem, not a failed/unexecuted operation.
Never auto-retry the tool to obtain a smaller result. Prevent a new model call
from repeating the same unsafe tool/version/argument digest after such an
omission in the same run (`duplicate_effect_prevented`).

For HTTP, a received 2xx response whose body exceeds the cap records completed
HTTP execution and omitted output, not an unknown transport outcome. For a
transport failure before an outcome is known, retain the existing unknown-
effect policy. Invalid output schemas still fail explicitly without pretending
the operation did not execute.

Generic outputs are not silently truncated; the agent can request a different
bounded/read-only operation. Retrieval alone can drop complete lowest-ranked
chunks and report clipping. If candidates exist but no chunk fits, return
`observation_too_large`, not a false successful "no evidence" search.
For a standalone workflow tool step, an omitted output fails that step with
`observation_too_large`; it cannot satisfy typed selectors or be replayed as if
the operation had never run.

### 7.4 Execution and approval

All tool calls are sequential within a run. Coroutine tools are awaited;
synchronous tools and HTTP request builders use `user_code`, never directly
block the model event loop or occupy an internal worker. The built-in store
search orchestrates only `internal_io` and `inference` work.
Default `sensitive=True`, `retry_safe=False` for developer/HTTP tools. Store
search is explicitly nonsensitive and retry-safe. Developer declarations are
trusted, not inferred from tool names.

Approve one concrete call: bind tool name, canonical arguments, schema/version,
and call ID. Changed arguments need a new approval. An approved retry of the
same retry-safe invocation retains that approval within the same live run.
After process restart, reacquire approval before a replay; do not persist
blanket authorization.

Use callback first, otherwise a terminal prompt only when stdin and stderr are
interactive TTYs. Serialize terminal prompts. Print tool name and arguments
with credentials removed; only a case-insensitive exact `yes` approves.
Include the trusted caller subject and a "contains document content" indicator
in the prompt. Callback errors deny with diagnostics as specified in section 4.1;
do not interpret an exception or invalid truthy object as consent.
EOF, any other response, missing TTY, or deadline expiry never approves.
Timeout/cancellation must stop accepting input for that request; do not leave
a background terminal reader that consumes input for a subsequent prompt.
Use platform-native nonblocking input handling on POSIX/Windows, not an
uncancellable `input()` worker.

The SDK cancellation token returns a cancelled result. External async
`Task.cancel()` records cancellation and re-raises `CancelledError`.
Synchronous tools cannot be forcibly terminated safely: record uncertain
outcomes for unsafe work and stop scheduling calls. Late completion cannot
mutate a returned result or authorize a replay. Keep a live invocation
registry; explicit resolution is rejected while that local invocation is
still running. Do not claim sandboxing, rollback, or exactly-once effects.

### 7.5 HTTP tool contract

`HTTPTool` requires name, version, description, argument/output schemas,
method, absolute endpoint URL/template, and a code-supplied request builder.
The builder receives validated arguments and ToolContext and returns exactly
`path_params`, `query`, and `json_body`. Path parameters replace named template
segments with percent-encoded scalar values; query values are scalars or lists
of scalars. Reject missing/extra path parameters and GET bodies. The builder
cannot supply headers or change scheme, host, or port. Methods are
GET/POST/PUT/PATCH/DELETE.

`headers` is a static map of nonsecret header strings. `auth_env` maps a header
name to exactly `{"env": <variable name>, "prefix": <string>}`, such as a Bearer
prefix for Authorization. Resolve these variables before dispatch; never take
authentication from LLM arguments or embed secret literals in headers.
Applications needing a different authentication scheme implement a custom
Tool rather than widening this helper's contract.
Disable redirects. Enforce the attempt timeout
and a 1,048,576-byte response limit before JSON decoding, followed by the
universal observation-size policy. Non-JSON responses
are `http_invalid_response`; raw response bodies are not logged automatically.
404 is a recoverable domain error. 401/403 are nonrecoverable authentication/
permission failures. Rate limits/transient failures follow retry policy only
when the tool is explicitly retry-safe. For unsafe requests, an ambiguous
transport failure/5xx after dispatch stops with unknown outcome.

No OpenAPI import, API discovery, or model-selected arbitrary URLs.

## 8. Durable storage and recovery

### 8.1 Records and ownership

Persistent checkpoints are schema version 1. Each root record stores kind
(`agent`/`workflow`), root/run/conversation IDs, revision, phase, configuration
fingerprint, counters/deadline, normalized messages, pending operations,
observations, step cursor/outputs, snapshots, diagnostics, and optional result.
Persist caller subject and cumulative provenance, but not runtime identity
attributes, approval objects, credentials, or encryption keys.

Phases are `ready`, `model_pending`, `tools_pending`, `finalizing`,
`needs_resolution`, and the four terminal statuses. Each pending operation has
`pending`, `approved`, `started`, `succeeded`, `failed`, or `unknown` state.
Checkpoint the reservation/start before invocation and its result afterwards.

The fingerprint includes instructions, model/profile/endpoint, tool and schema
versions/egress origins, limits, document/egress policies, workflow definition/
version, and store bindings. Persist the canonical, nonsecret JSON descriptor
that is hashed, not just the hash, so offline relocation can transform only
typed binding paths. Its `binding_paths` map identifies storage, each named
store's source/index roots, embedding cache, and trace prefix separately from
semantic fields. Explicit programmatic paths must also appear in this map.
Exclude credential values and encryption-key
selection so authenticated key rotation does not invalidate a run.
A mismatched resume fails
`checkpoint_incompatible`; do not silently re-plan or migrate schemas.
Application authors must increment explicit tool/workflow versions when
semantics change; do not infer compatibility from callable memory addresses.

Use `portalocker` OS-backed locks in `<storage-path>.locks/`. Acquire lifecycle
locks for the root run and, for conversation continuation, the conversation
in sorted filename order. A second owner is rejected immediately with
`run_in_use`/`conversation_in_use`. Brief storage-write locks are separate;
never hold them while waiting for a lifecycle lock.

Local filesystems only; network filesystem/distributed lease semantics are not
supported. OS lock release after process exit does not prove tool effects did
not happen. Child workflow runs share root ownership and cannot be resumed
independently from that root.

### 8.2 SQLite and JSON formats

Both backends persist an authenticated encrypted envelope, not a readable
SQLite/JSON file. SQLiteStorage opens `:memory:`, sets `temp_store=MEMORY`,
`journal_mode=MEMORY`, and foreign keys enabled, then deserializes the verified
database image. No disk WAL/journal is created. Tables:

```text
runs(
  run_id TEXT PRIMARY KEY, kind TEXT NOT NULL, revision INTEGER NOT NULL,
  phase TEXT NOT NULL, checkpoint_json TEXT NOT NULL,
  result_json TEXT, updated_at TEXT NOT NULL
)
conversations(
  conversation_id TEXT PRIMARY KEY, version INTEGER NOT NULL,
  head_run_id TEXT NOT NULL, state_json TEXT NOT NULL,
  FOREIGN KEY(head_run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
)
```

Also store one `store_meta(schema_version INTEGER, revision INTEGER)` row;
its revision matches the authenticated envelope revision.
Updating a run and a completed conversation head is one in-memory transaction.
Serialize, encrypt, and atomically publish the complete image before
acknowledging the checkpoint. Use revision compare-and-swap; stale writers fail.

The decrypted payload of `JSONStorage(path=...)` is one JSON document, not a
directory of independently updated records:

```json
{"schema_version":1,"revision":0,"runs":{},"conversations":{}}
```

Its run/conversation values have the same payloads as SQLite. Both backends
encrypt before writing any temporary file, flush/fsync the encrypted sibling,
close its handle, and atomically replace the target. Fsync the directory on
POSIX. Never write plaintext checkpoints or edit an envelope in place.

Readers take the process-local path lock and shared OS lock, read ciphertext,
and close the target handle before releasing the lock. Writers take the
exclusive equivalents and recheck the current revision before replacement.
This protocol also applies between separate Storage instances in one process.
On Windows, retry a sharing-violation replacement at most three attempts with
50/100 ms waits inside the write deadline. Never fall back to an in-place write.
Exhaustion returns `storage_write_failed` and preserves the old valid file.
Test an external reader holding a non-share-delete handle explicitly.

Remove only abandoned, specifically identified encrypted temporary files.
Guarantees cover process-crash recovery on supported local filesystems, not
arbitrary hardware/filesystem failure.

Only completed runs advance conversation heads. A caller-supplied persisted
conversation must match the stored version. Failed/incomplete runs remain in
their run checkpoint and are not silently imported as completed conversations.
Ephemeral runs provide explicit in-memory state with the same validation.

Both encrypted backends rewrite their complete image and have O(total stored
bytes) checkpoint write cost. JSONStorage is a small-state backend capped at
16 MiB decrypted payload; SQLiteStorage is capped at 128 MiB in this release.
Do not advertise SQLiteStorage as an incremental, page-encrypted database.
Check ciphertext size before allocation/decryption and enforce these bounds.

Reserve 4 MiB per admitted root run, plus 256 KiB for its conversation head,
under the store write lock. Admission requires existing data plus all active
reservations to fit. Each root checkpoint is capped at 4 MiB and a conversation
state at 256 KiB. Before dispatch, verify space for bounded arguments/results
and a 4 KiB terminal-error reserve; fail before side effects when unavailable.
Release unused reservations after finalization. Actual retained records still
count toward capacity. Return `storage_capacity_exceeded` rather than silently
purge history. Document explicit deletion/export and migration to a new store.

### 8.3 Resume and uncertain outcomes

On resume, acquire locks, validate the checkpoint, restore budgets and pinned
snapshots, and continue the saved operation/cursor. Completed runs return their
stored result without calls. Explicitly cancelled or deadline-exhausted runs
may resume when their effects are resolved and the deadline permits it;
turn/tool-limit exhaustion cannot be reset by resume. Other terminal failures
are not resumable except through the explicit unknown-outcome resolution path.

An interrupted model/retry-safe tool consumes its reserved attempt and may
retry only if allowance remains. A started unsafe tool with no durable result
enters `needs_resolution`, regardless of whether the crash might have preceded
the external side effect.

`resolve_tool_outcome` requires `resolution` equal to `succeeded`, `failed`,
or `abandoned`, plus a nonempty operator note. `succeeded` validates `output`
against the tool schema, including JSON null when allowed, and forbids error.
`failed` requires a ToolFailure with known outcome and forbids output.
`abandoned` permits neither output nor error and permanently fails the run.
An externally verified success/failure advances the saved cursor. The explicit
resolution tag distinguishes a valid null result from an omitted result.
Record an audit entry and never invoke the tool during resolution. Reject it
while a locally tracked invocation is still active.

Retain state until explicitly deleted. Reject deletion of active runs or
conversations. Reject deleting a run that is a stored conversation head with
`record_referenced`; delete the inactive conversation record first. Conversation
deletion does not cascade into run history. JSON enforces the same reference
rules as SQLite. Exports contain persisted data, not credentials/runtime
objects. `export_run` returns an authenticated encrypted export envelope;
`read_export` explicitly decrypts it into caller memory but never imports it
as an executable checkpoint. An export envelope cannot be substituted for a
store envelope because their authenticated resource types differ.

### 8.4 Mandatory encryption, integrity, and file access

Use AES-256-GCM through cryptography. `[encryption.keys]` maps nonsecret key IDs
to environment-variable names containing strict base64 encodings of 32-byte,
externally generated random master keys. Do not generate, print, persist,
or commit master keys. Empty, invalid, unavailable, or unknown keys fail closed.
There is no encryption-disable flag or plaintext compatibility fallback.

Derive a per-resource 32-byte key with HKDF-SHA256, the resource ID decoded
from hex as salt, and UTF-8 `loom-v1/<resource_type>` as info.
Each encryption uses a fresh
12-byte cryptographically random nonce. The binary envelope is:

```text
8 bytes "LOOMENC1"
4-byte big-endian length of canonical JSON header (maximum 4096)
header: version, algorithm, key_id, resource_type, resource_id, revision, nonce,
        run_id, event_id
AES-GCM ciphertext followed by its 16-byte authentication tag
```

Nonce is base64 in the header. Authenticate magic, header length, and complete
header as additional authenticated data. Validate expected resource identity/
type, size, key ID, and tag before parsing JSON or deserializing SQLite.
For file artifacts, resource ID is SHA-256 of UTF-8
`loom-v1\n<resource_type>\n<canonical_absolute_path>`; canonical paths use
lexical normalization, platform case normalization, and `/` separators without
following symlinks. Readers compute the expected ID from the configured path.
Export IDs instead hash `loom-v1\nexport\n<storage_resource_id>\n<root_run_id>`;
the authenticated export payload includes those source IDs for validation.
Reject symlinks/reparse points in managed artifact paths. A plain filesystem
rename/copy does not change these identities: use the offline relocation
protocol in section 8.5 before abandoning the original artifact locations.
Resource types are `storage_sqlite`, `storage_json`, `index_sqlite`,
`index_manifest`, `index_report`, `index_current`, `trace`, `trace_manifest`, `trace_intent`,
`maintenance`, `relocation_journal`, and `export`.
Header version is integer 1 and algorithm is `"AES-256-GCM"`; forbid extra keys.
Key IDs follow the profile-name syntax; environment-variable names match
`[A-Za-z_][A-Za-z0-9_]*`. `read_export` also verifies that the authenticated
source storage ID matches the receiving Storage instance before returning data.
Mutable storage/CURRENT revisions start at 1 and increment per publication;
immutable snapshot artifacts use revision 1, exports use the exported root
revision, and trace records use their event sequence. Rekey preserves revision.
For trace records, `run_id` and `event_id` are required and must match the
authenticated event payload; `revision` must match its sequence. All other
resource types set those two header fields to null. Trace manifest and
maintenance/journal revisions increment on each committed update.
An invalid tag, swapped artifact, or malformed envelope is
`artifact_authentication_failed`; never load it as an empty store.

Encrypt conversation/checkpoint images, document databases, index manifests,
ingestion reports, CURRENT pointers, trace records, temporary generations,
and exports. Only plaintext-free lock metadata, public model weights/metadata,
and code/dependency/SBOM artifacts are exempt. Model files remain checksum
verified. Original caller-supplied documents and values intentionally returned
in memory/stdout are outside the SDK-owned file perimeter; callers must protect
their sources and captured output under the same deployment policy.

Plaintext database processing occurs only in memory, with SQLite memory
journals/temp storage. Do not spill decrypted databases into temporary files.
The SDK cannot secure OS swap, crash dumps, debugger access, or malicious
application code; the deployment must protect those separately.

Create Loom-owned directories with POSIX 0700 and files with 0600, using explicit
creation modes rather than changing the process-wide umask. Verify ownership
and reject existing overly broad access with `insecure_permissions`.
Do not chmod user source directories. On Windows, use Win32 security APIs via
ctypes to create a protected DACL granting the current user and SYSTEM only,
without inherited Users/Everyone/Authenticated Users access. POSIX chmod alone
is not the Windows implementation. Check effective ACLs where the platform
supports additional grants; inability to establish the required protection is
a setup error, not a warning.

The active key protects new writes. Keep previous key IDs available to read
retained artifacts. `loom crypto rekey` decrypts and atomically reseals configured
artifacts under the active key while applications are stopped and ownership
locks are held. It is restartable; retain all old keys until verification
completes. `loom crypto verify` checks authentication/key IDs in configured
roots without revealing plaintext. Inventory external exports/backups before
retiring a key; the command does not claim to discover caller-owned copies.

AES-GCM supplies integrity as well as confidentiality; a separate unauthenticated
checksum or redundant HMAC is not the protection mechanism. Replay of an entire
older valid store by an actor who controls storage history is outside v0.1's
trust model: an authenticated revision alone is not a trusted freshness anchor.
OS permissions and trusted operators must protect history. Do not claim
anti-rollback or safely resume manually restored historical copies without
operator reconciliation. Key/process compromise is likewise outside the
SDK's protection boundary.

### 8.5 Offline relocation preserving recovery state

**Moving a project requires an explicit offline operation; do not simply rename
its encrypted artifacts or rebuild indexes and discard checkpoints.**
Implement:

```text
loom crypto relocate --from-config OLD_TOML --to-config NEW_TOML [--abort]
```

Keep source artifacts at their original paths until this operation finishes.
The operator prepares the destination code, TOML, source documents and verified
public embedding cache; Loom relocates only configured encrypted artifacts.
All applications using either location must be stopped. Both keyrings must
be available. After a blind filesystem move, restore the original artifact
locations first; no heuristic path guessing or automatic identity bypass ships.

Match resources by storage role/backend and store/profile name, not by scanning
for decryptable files. The two resolved configs may differ only in source/cache/
artifact paths and encryption key references/selection. Other semantic fields
must match, comparing parsed schema contents rather than their file locations.
Destination managed roots must be disjoint from source roots and
absent, except for artifacts belonging to this same interrupted transaction.
Reject overlap, unknown destinations and unconfigured programmatic bindings;
require those bindings to be represented in the supplied configs before moving.
Both original and destination artifact paths must be accessible on the same
supported host for this v0.1 command; cross-host transport is not implemented.
Compute source identities from their verified original canonical paths, never
by accepting arbitrary resource IDs from unauthenticated headers.

Every storage file, index root and logical trace prefix has an encrypted
maintenance marker and a separate OS/process maintenance lock. Use
`<storage-path>.maintenance.enc`, `<index_path>/maintenance.enc`, and
`<jsonl_path>.maintenance.enc`; each corresponding lock replaces `.enc` with
`.lock`. Markers contain schema version, revision, state (`ready`, `pending`,
`retired`, `aborted`), relocation ID/null and canonical source identity.
Create a ready marker on first initialization. Missing markers on initialized
artifacts fail closed; do not recreate one to bypass retirement.

Normal runs hold shared maintenance locks for all bound managed resources
through actual completion of work, including outstanding workers. Direct
storage/search/trace operations hold them for their operation lifetime.
Check markers after acquiring locks: pending/aborted means
`artifact_maintenance_required`; retired means `artifact_relocated`.
These errors never trigger empty-store creation or automatic reindexing.
Acquire multi-resource maintenance locks in sorted canonical order before
lifecycle and data-write locks. Rekey and relocation require exclusive locks;
fail immediately with `resource_in_use` rather than racing active work.

The relocation transaction is:

1. Under exclusive locks, authenticate source markers and the full inventory:
   storage images, every retained snapshot/report/CURRENT pointer, and the
   trace manifest/retained segments. Verify destination permissions, keys,
   capacity and prepared public model files before publication.
2. Write an encrypted journal at
   `<old-config-directory>/.loom/relocations/<relocation_id>/journal.enc`.
   Bound the journal to 16 MiB; reject excess inventory before marking resources
   pending. Record canonical old/new mappings, nonsecret config hashes, source ciphertext
   hashes, and the exact destination inventory. Mark all sources/destinations
   pending before copying any user-data artifact. Once any marker is pending,
   restart requires retrying this command or its precommit abort path.
3. Process one artifact at a time. Authenticate each source against its old identity and reseal it for the
   final destination identity using the destination active key and fresh nonce.
   Write encrypted staging files only. Transform only the descriptor's typed
   `binding_paths` and recompute configuration fingerprints in checkpoints.
   Copy all run/conversation/snapshot/chunk IDs, semantic descriptors, messages,
   counters, deadlines, pending-tool states, observations and trace event IDs
   unchanged. Never rewrite arbitrary strings in prompts, tool arguments or
   application state, and never execute a pending operation.
4. Publish and authenticate every destination artifact while its marker remains
   pending. Record verification in the journal. Increment only storage/container
   revisions affected by descriptor rewrites; preserve root-run revisions,
   conversation versions and snapshot IDs. Verify fingerprint descriptors differ
   solely at authorized bindings. Source artifacts remain intact.
5. Durably mark all sources retired, then atomically commit the journal, then
   mark destinations ready. No destination may be admitted before this global
   commit. A crash during activation leaves remaining pending roots unavailable
   until retry, not accessible with partially transformed state.
6. Retry with the same configs resumes the recorded transaction: verify hashes
   before reusing precommit staging files; after commit, finish activation
   without recopying or overwriting any already activated destination. Abort
   is allowed only before commit: first journal the abort, restore source
   readiness and leave destination remnants marked aborted for explicit cleanup.
   An uncertain/corrupt journal never authorizes rollback or activation.

The final report contains the relocation ID, resource counts and verification
status without plaintext contents/paths. Source copies are retained but retired;
they cannot also resume the same runs through normal APIs. Reattach the same
tool/workflow versions and runtime context at the destination. An expired
deadline or unknown side effect still needs the existing explicit recovery
action; relocation does not replenish budgets or resolve effects.

External exports/backups are not silently discovered or rewritten. Preserve
their original binding information and keys; pure `read_export`/verification
may inspect retired-source archives but never activate them. The old-history
rollback trust limitation remains. Arbitrary application-held paths and callable
code are the operator's responsibility, not strings the relocation tool edits.

## 9. Document pipeline and index format

### 9.1 Fixed embedding artifact

Use `BAAI/bge-small-en-v1.5`, 384-dimensional normalized embeddings, CPU only.
Artifact repository:
`Qdrant/bge-small-en-v1.5-onnx-Q`.
Pinned revision: `aa8f8b060edb00e03bfdd08813a2949946c8ba55`.
The model is MIT-licensed.

`model_optimized.onnx` must be exactly 66,465,124 bytes with SHA-256:

```text
51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431
```

Also retrieve `config.json`, `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `ort_config.json`, and `vocab.txt` from that exact
revision. Record their SHA-256 values in the packaged embedding manifest during
M0 by hashing the revision-addressed bytes; this is deterministic artifact
materialization, not model selection. Setup verifies every file against the
packaged manifest before atomically publishing the model directory.

Only `loom setup embeddings` may download weights. Use HTTPS revision-addressed
artifact URLs; never upload document content. Runtime passes FastEmbed
`specific_model_path`, `local_files_only=True`,
`providers=["CPUExecutionProvider"]`, and `threads=4`; do not use private APIs,
automatic GPU selection, or the library's online fallback. Missing/corrupt
files raise `embedding_artifact_missing`/`embedding_artifact_corrupt`.

Batch size is 32, parallel worker processing disabled. Use the model's
`passage_embed` and `query_embed` surfaces consistently, normalize vectors, and
store little-endian float32. Token boundaries use the pinned `tokenizer.json`;
no remote tokenizer loading.
The process cache in section 4.2 survives synchronous event-loop teardown.
Assert one model initialization across repeated `agent.run` calls and across
concurrent stores sharing the same artifact, with serialized inference.

### 9.2 Parsing and chunking

Only explicit build/refresh reads the source folder. Recursively enumerate
regular files in sorted relative-path order. Accept case-insensitive `.pdf`,
`.md`, `.docx`; report every other regular file as excluded. Do not follow
symlinks. Missing/unreadable root is a fatal indexing error.

Per-file bounds are 25 MiB source bytes, 2,000 pages for PDF, 100 MiB total
uncompressed DOCX ZIP content, and 10 MiB extracted UTF-8 text. Exceeding a
bound excludes that file with a specific diagnostic, not silent truncation.
Guard PDF decoded content streams at 16 MiB per page before text extraction.

Normalize extracted text with Unicode NFC, LF line endings, tabs converted to
spaces, and trailing whitespace removed per line. Retain paragraph boundaries.
Apply exactly the same normalization to citation excerpts before matching.

- PDF: extract text page-by-page with pypdf. Page numbers are 1-based. Reject
  encrypted documents. Report image-only/unextractable pages and retain
  extractable pages; if no text remains, exclude the document as `ocr_required`.
- Markdown: use markdown-it-py's CommonMark parser; retain heading hierarchy,
  text, lists, fenced-code text, and source line numbers. Do not fetch links,
  images, or render/execute embedded HTML.
- DOCX: use python-docx; process body paragraphs and table cells in document
  order. Retain heading text and 1-based block index; do not invent page
  numbers. Images, macros, embedded objects, and headers/footers are not
  searchable; report excluded non-text content in the ingestion diagnostics.

Chunk within one source-location block. Use up to 400 tokenizer tokens with a
50-token overlap for long blocks; short blocks are not overlapped. Count model
special tokens/prefix allowance so no embedded input exceeds 512 tokens.
Retain original normalized character offsets; do not reconstruct citation text
by decoding token IDs. Skip empty normalized chunks.

`document_id` is SHA-256 of store name and relative path separated by LF.
`chunk_id` is SHA-256 of document ID, normalized block hash, location JSON,
and character bounds separated by LF. Stable content and pipeline version
produce stable chunks. `snapshot_id` is a fresh UUID4 for each publication.

### 9.3 Immutable generations

Use this directory format per store:

```text
index_path/
  CURRENT.enc
  writer.lock
  snapshots/<snapshot_id>/
    manifest.json.enc
    index.sqlite.enc
    ingestion-report.json.enc
```

The decrypted `CURRENT.enc` payload contains exactly schema version 1 and
snapshot UUID as a JSON object. Query only published index artifacts, never
the source folder. Authenticate/decrypt the SQLite image into memory, set
`temp_store=MEMORY`, `journal_mode=MEMORY`, then `query_only=ON`.
Never open a decrypted on-disk SQLite file. Tables:

```text
chunks(
  chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
  source_path TEXT NOT NULL, location_json TEXT NOT NULL,
  text TEXT NOT NULL, embedding BLOB NOT NULL
)
chunks_fts USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')
```

Manifest fields: schema version, snapshot ID, store name, creation time,
pipeline fingerprint, embedding artifact revision/dimension, input file hashes,
indexed/excluded/partially-extracted counts, and chunk count. Report every
examined regular file with status and reason.

Build in memory while holding the writer lock; encrypt all components for
their final resource paths before writing a temporary generation. Flush and
rename the complete encrypted generation, then atomically replace `CURRENT.enc`
under the same reader/writer protocol as storage. Never mutate a published
generation. Concurrent queries retain their
selected generation. A failed publication leaves the previous pointer intact.

Refresh reuses vectors only when content/chunk IDs and pipeline fingerprint
match. Exclude newly failed files rather than silently retain their old text.
Remove deleted files from the new generation. If all supported candidates fail,
do not publish. A genuinely empty folder publishes an explicit empty index;
a nonempty folder containing only unsupported/failed files does not.

Keep all published generations in v0.1; no automatic pruning. This eliminates
cross-store/backend ambiguity for resumable snapshots. Source deletion followed
by refresh removes content from the new snapshot, not historical generations.
To erase historical document data, explicitly delete the entire store after
deleting all runs/conversations that reference it. Do not advertise source-file
deletion as an immediate purge.
Store removal is an operator-managed offline filesystem action, not a v0.1
SDK/CLI command: stop applications and remove referencing durable states first.
Do not invent automatic recursive cleanup or an unsafe force-delete option.

Each snapshot stores full text, FTS data, and vectors: vector payload alone is
`chunk_count * 384 * 4` bytes, plus text/FTS/metadata and SQLite page overhead.
Refresh reuses computation, not disk pages across snapshots. After N similarly
sized refreshes disk usage is approximately N times one complete snapshot;
publication additionally needs one full temporary snapshot. Encrypted size
adds a header/tag per artifact; it does not compress historical data.
Report actual snapshot bytes and cumulative retained bytes after publication.
Cap a decrypted index image at 512 MiB and each manifest/report at 16 MiB;
exceeding these fails publication with `index_capacity_exceeded`, preserving
the previous generation. These are memory/file safety limits, not a 100-file
corpus limit. The 2 GiB indexing-RSS acceptance includes serialization and
encryption copies. Query loading additionally requires section-9.5 admission;
an individual file-size limit is not a process-wide memory limit.

### 9.4 Search

Attach one tool named `search_<store_name>`; store names match
`[a-z][a-z0-9_]{0,47}` and must not collide with developer tools.
Arguments are exactly `{"query": <nonempty string up to 4096 characters>}`.
Also validate the query against the embedding model's 512-token bound including
special tokens/prefix; reject oversize queries as recoverable `query_too_long`
rather than silently truncating the semantic query.

Pin the latest snapshot once per fresh run. A continued conversation retains
its existing snapshot bindings; starting a new conversation is how a caller
opts into current content. Resume always uses its saved snapshots and fails
`snapshot_missing` if any are absent, never silently substitutes `CURRENT.enc`.

For each query:

1. Derive lowercase alphanumeric keyword terms, deduplicate, quote them as
   literal FTS terms, and join with OR. Never execute user-supplied FTS syntax.
2. Rank top 20 lexical matches by ascending BM25, then chunk ID.
3. Rank top 20 normalized cosine matches by descending score, then chunk ID.
4. Fuse using `1 / (60 + rank)` for each list containing a chunk, with
   1-based ranks. Sum scores; ties sort by chunk ID.
5. Return at most five chunks. Cap serialized search observations at 4,096
   UTF-8 bytes and the dynamic remaining-context allowance, whichever is less;
   drop complete lowest-ranked chunks until within budget.
   Report `retrieval_context_clipped` when this happens.

If one list is legitimately empty, fuse the other. An index/embedding failure
is an error, not permission to disable half of hybrid search. A zero-chunk
snapshot returns a successful empty search. Do not interpret scores as
probabilities or confidence.
If nonempty candidates cannot fit even one complete chunk, return the explicit
size error from section 7.3.1. Never label a size failure as an empty search.

`SearchResult` contains store, snapshot ID, query, returned chunk records,
clipping flag, and ingestion-coverage diagnostics. Track the exact returned
chunks in run state for citation validation. Search consumes ordinary tool
attempts and timeouts.
Model-visible coverage diagnostics contain aggregate counts/codes, not the
entire per-file ingestion report. Full per-file results remain available in
the encrypted ingestion report and the explicit indexing command's output.

### 9.5 Query memory and snapshot leases

Use one process-wide query-memory admission ledger with a fixed 3 GiB budget
and at most two resident/loading snapshots across all stores and instances.
Cache keys are canonical index path plus snapshot ID. A cache hit must still
check availability/identity of the key that authenticated that entry.
The budget includes active queries, idle cache entries, pending loads and
jobs still running after cancellation; it is not reset at each Agent call.

Let `S` be the encrypted index file's bounded byte length, an upper bound for
its plaintext image. Reject oversize/malformed framing before allocation.
Charge `4*S + 64 MiB` while loading, `2*S + 64 MiB` while resident, and
an additional 64 MiB for each admitted query's scratch/results/metadata.
Loading reservations account for ciphertext, plaintext, SQLite and vector
copies; replace them with resident charges only after temporary buffers are
released. Include query scratch reservations while a job is queued.

Check the byte and snapshot-count limits atomically before file reads or
allocation. Evict least-recently-used entries with zero active leases, closing
their database and releasing vectors, until admission fits. Deduplicate
simultaneous loads of the same cache key. Never evict an in-use snapshot,
silently load another snapshot, or fall back to plaintext/memory-mapped files.
If all eligible entries are leased or the request still cannot fit, return
recoverable `query_memory_exhausted` immediately, with requested/available bytes
and snapshot counts; it is not an empty search or implicit retry.
Standalone search raises ToolFailure with that code.

Use incremental row processing, bounded SQLite page caches (32 MiB per loaded
connection), and bounded NumPy work buffers; do not `fetchall` the corpus into
Python objects. Release ciphertext/plaintext buffers before retaining search
arrays. SQL work on one cached connection is serialized and configured for
the framework's cross-thread use. All returned chunks are copied text/metadata,
not embedding vectors, array views, or database-backed objects.

Acquire a lease only for actual loading/search, and release it after actual
worker completion, including timed-out work. Conversation/checkpoint snapshot
bindings and provenance are IDs only: they never pin a decrypted image for the
whole model conversation. Eviction leaves the encrypted immutable generation
available for a later authenticated reload without changing snapshot IDs.

This is deterministic SDK working-set admission, not a hard OS/process RSS
limit or protection against arbitrary application allocation. Test measured
query RSS separately in section 13.3; do not equate a reservation estimate with
measured RAM consumption.

## 10. Workflow contract

Expose these code-first records:

```text
Case(name, predicate, target)
Branch(cases: Sequence[Case], *, default=None)
Step(name, target: Agent | Tool, *, input_selector, next=None, version="1")
Workflow(
    *, name, steps: Sequence[Step], entry, output_selector,
    version="1", limits=None, storage=None, approval=None, event_sinks=(),
    tool_document_access=None, config=None
)
Workflow.run(*, input: JSONValue, context=None, caller=None, cancellation=None) -> RunResult
Workflow.arun(*, input: JSONValue, context=None, caller=None, cancellation=None) -> RunResult
Workflow.resume / aresume / resolve_tool_outcome
```

The three recovery methods have the same arguments/semantics as Agent.
Selectors/predicates receive an immutable `WorkflowView` containing workflow
input, validated completed step outputs, and last completed step name.
`input_selector` returns the next target's input; `output_selector` returns
the final JSON output. Agent step input must be a string; tool step input must
match its argument schema.
Execute these synchronous control functions in `user_code` with a
30-second timeout capped by the remaining workflow deadline. They consume
elapsed time but no model/tool attempts. Timeout is
`workflow_control_timeout`; do not retry control functions automatically.

`next` is a step-name string, `Branch`, or `None` for termination. Validate
unique names, targets, entry, reachability, and acyclicity at construction.
Reject unused/unreachable steps and nested workflows. Selectors/predicates are
trusted pure Python functions; explicit Step/Workflow versions participate in
the fingerprint. They must not call models/tools or perform side effects.

Evaluate each branch predicate once against the committed output. More than
one true predicate is `branch_ambiguous`; no true predicate without a default
is `branch_no_match`. Non-bool results or exceptions are `branch_evaluation_failed`.
Persist the selected target before starting it; resume does not reevaluate a
committed branch.

Workflow execution is sequential and fail-fast. Save completed outputs, then
advance. A child failure ends the workflow with child/step context in the error.
There is no compensation, whole-agent-step replay, or outer retry multiplier.
Agent steps use their runtime's remaining retry allowances; tool steps use
the same bounded retry-safe tool executor. Recover from the saved checkpoint.

An agent attached to a workflow must not specify its own Storage; the root
workflow owns persistence. Child approvals use the agent callback when set,
otherwise workflow callback, then the normal terminal/headless policy.
Child conversation state is isolated by step; data transfer is explicit through
selectors, not shared mutable history.
Propagate the root caller identity to every step and enforce the same subject
on resume. A Workflow instance follows the same sharing/locking rules as Agent.
Selectors inherit the union of provenance of all inputs visible in WorkflowView;
trusted pure functions cannot clear it. Standalone tool-step grants use
`Workflow.tool_document_access`, keyed by step name and listing store names.
Apply section 6.5's store policy/origin checks regardless of sensitivity.
Agent steps retain their own grants; root grants never widen them. Before any
child model call, check inherited store provenance even if the receiving agent
does not attach the source store; missing store policy/consent denies transfer.

### 10.1 Aggregate limits

For each declared agent step, use its effective `Limits`. A standalone tool
step has deadline 120 seconds, attempt timeout 30 seconds unless overridden
on the Tool, two retries only if retry-safe, and no model turns.

Derive workflow ceilings once at admission:

- Step activations: number of declared steps.
- Model turns: sum of agent-step model-turn ceilings.
- Tool attempts: sum of agent-step tool ceilings plus three for each retry-safe
  standalone tool and one for each other standalone tool.
- Deadline seconds: sum of all step deadlines.

`WorkflowLimits` can set smaller positive values; larger values are rejected.
Zero derived model turns is valid for a tools-only workflow. Every child
reservation atomically charges both its own and root counters. Child deadlines
are capped by the root deadline. Branches may leave some budget unused.
Resume/extensions never replenish counters or permit revisiting a step.

## 11. Events, privacy, and CLI

### 11.1 Events

Event fields are schema version, event ID, root/run/step IDs, monotonic sequence,
timestamp, event type, duration/usage when applicable, metadata, and optional
redacted payload. Sequence starts at 1 within each run; event ID is exactly
`<run_id>:<decimal_sequence>` for deduplication.

Required types:

```text
run.started, run.finished
model.started, model.finished
tool.started, tool.finished
approval.requested, approval.decided
retry.scheduled, output.correction_requested
retrieval.finished, checkpoint.saved, run.resumed
workflow.step_started, workflow.step_finished, workflow.branch_selected
error
```

`run.finished` includes the terminal status. Record events after the associated
state transition is committed. Callback/JSONL delivery is best-effort and not
exactly-once across crashes; durable state is the recovery authority.

Payloads are disabled by default. Metadata is an allowlist of opaque run/call/
document/chunk/snapshot IDs, counters, timing, status, and error codes; it
excludes caller attributes, source names/paths, document text, tool arguments,
and provider response bodies. Source names/paths are also sensitive.
`capture_source_paths=true` explicitly permits structured source-path metadata;
default false omits it. Diagnostic messages use code-specific safe templates,
never raw filesystem/provider exception strings. CLI ingestion reports and
returned citations intentionally disclose sources to their caller, not to
default tracing.

Redact recursively using this exact key-normalization algorithm: strip outer
whitespace; insert `_` between acronym/lowercase-word boundaries
(`([A-Z]+)([A-Z][a-z])`) and lowercase-or-digit/uppercase boundaries
(`([a-z0-9])([A-Z])`); replace whitespace/hyphen runs with `_`, collapse repeated
underscores, and lowercase. Compare the complete normalized key, never a
substring, against:

```text
authorization, proxy_authorization, api_key, apikey, x_api_key, x_goog_api_key,
token, access_token, refresh_token, id_token, session_token, bearer_token,
x_auth_token, password, passwd, secret, client_secret, private_key,
cookie, set_cookie, aws_secret_access_key, aws_session_token
```

Normalize configured `redact_keys` identically and union them with this set;
configuration cannot remove defaults. Replace matching values with
`"[REDACTED]"`, regardless of their type. Thus `accessToken` and `X-API-Key`
are masked, while `max_output_tokens`, `input_tokens`, `output_tokens`,
`token_count`, and other usage fields remain intact.
Additionally scrub exact resolved credential/key strings wherever they occur
inside string values. This supplements key matching, not a promise to identify
arbitrary unknown secrets in free text. Explicit payload capture may expose
document content; `capture_source_paths` controls structured path fields, not
semantic redaction of arbitrary prose.
Opaque provider continuation fields are never trace payloads.

Redact before invoking event callbacks or encrypting trace records. Authorization
callbacks use the separate section-4.1 contract and receive the real validated
arguments; event sinks never do so implicitly. A failed sink is disabled
for that run, adds `event_sink_failed` to diagnostics, and writes one sanitized
stderr warning without recursively logging through the failed sink. Do not
pretend failed durable storage is an ordinary tracing warning.

Each sink invocation is bounded to 500 ms and the remaining run deadline.
Callbacks must not alter run state. No remote telemetry/exporter ships in v0.1.
The requirement's "Agent Thought" is represented by observable tool choices,
branch/correction/retry decisions, and provider-exposed summaries only when
explicit payload capture permits them; never hidden reasoning.

For a nonempty `jsonl_path`, require an `.enc.jsonl` suffix; this is a logical
stream prefix, not a single ever-growing file. Store its files as:

```text
<jsonl_path>.segments/
  manifest.enc
  <segment_number>-<segment_uuid>.enc.jsonl
```

The encrypted `trace_manifest` payload contains exactly schema version,
revision, next segment number, current segment UUID, an ordered list of retained
segments (number, UUID, closed flag), and at most one pending-delete entry
(number, UUID). Bound its plaintext to 64 KiB. Numbers start at 1 and increase;
UUIDs are UUID4. Segment paths never change during rollover. Each gets a new
path-derived resource ID; never rename old ciphertext to a numbered backup.

Each segment line is exactly
`{"envelope":"<base64 LOOMENC1 envelope>"}` plus LF. Its authenticated header
binds segment resource ID, run ID, event ID and event sequence. After decryption,
verify header/payload agreement and the canonical event-ID rule before returning
any event. Attribute events using that verified identity, never adjacent lines
or a caller-supplied label. A run-filtered reader authenticates records before
filtering; it must not relabel a foreign-run event as belonging to the request.
AES-GCM already protects payload run IDs; these checks make contextual binding
explicit, not a claim that ciphertext previously allowed run-ID editing.

Bound plaintext event payloads to 1 MiB and complete encoded segment files to
16 MiB, including headers, base64, JSON framing and LF. Retain at most five
segments (current included), or 80 MiB of event data. Under the stream's shared
maintenance lock and exclusive process/OS append lock, calculate the full
encoded line size before writing. If it would exceed the segment limit,
close the current segment and create a fresh uniquely named segment, then
encrypt the event for that final identity. A line is never split across segments.

Rollover atomically publishes the encrypted manifest. When the sixth segment
is needed, remove the oldest closed segment from the retained list and record
it as pending-delete in that same commit, then delete only that named segment
and clear the pending entry. Report `trace_retention_applied` with the removed
opaque segment ID in the emitting run's diagnostics; do not recursively emit
this notification through the trace sink. This policy expires traces only,
never checkpoints, conversations, or index generations.

Complete recorded deletion/rollover recovery before accepting another append.
, record its assigned ID in an encrypted recovery
intent at `rollover-intent.enc` alongside the manifest (maximum 64 KiB,
resource type `trace_intent`, with its own path identity). Its payload is
exactly schema version, revision, base manifest revision, new segment number,
and new segment UUID. Recovery may finish or remove only that
recorded uncommitted segment. Thus crashes cannot accumulate untracked files
or reset the retention count. A deletion/publication error disables the sink
visibly; do not append indefinitely while retention is broken. Atomic rekey/
recovery may temporarily require one additional 16 MiB segment plus bounded
metadata; this is separate from the 80 MiB retained-data cap.

Trace reads snapshot the manifest/segment bytes under the append lock and close
file handles before releasing it, then decrypt/print outside the lock. Read one
bounded segment at a time; if retention removes an unread segment, report
`trace_segment_expired` rather than silently skipping it. Oversize events fail
the sink visibly. Stop on malformed/incomplete lines with `trace_corrupt`;
do not silently discard a damaged tail.
Per-record authentication does not prevent deletion/replay of entire valid
records or historical streams; tracing is not the recovery authority.

### 11.2 Commands and exit codes

Implement with argparse:

```text
loom setup embeddings [--config PATH]
loom index build STORE [--config PATH]
loom index refresh STORE [--config PATH]
loom crypto verify [--config PATH]
loom crypto rekey [--config PATH]
loom crypto relocate --from-config OLD_TOML --to-config NEW_TOML [--abort]
loom trace read [--config PATH] [--run-id UUID]
```

Setup prepares/verifies the pinned embedding files only. It does not install,
start, or pull an answer-generating LLM. Build refuses to replace an existing
store and instructs the caller to use refresh. Refresh also works for an empty
existing store but requires an existing index; use build for first creation.

Commands print one JSON result/report on stdout and sanitized progress/errors
on stderr. Exit 0 means success without exclusions, 2 means a published index
with exclusions/partial extraction, 1 means fatal failure with no publication.
Argparse usage/config errors exit 1 rather than reusing partial-success code 2.
Ctrl-C exits 130 and must not publish a partial generation. Existing model
setup is idempotent after integrity verification.

Crypto commands inventory configured storage/index/trace paths; they never
print keys or decrypted data. They report each verified/rotated artifact's
opaque ID and key ID, fail on unavailable keys/authentication errors, and
require applications to be stopped for rekey. Unconfigured backups/exports
must be inventoried and re-exported separately before old-key retirement.
Trace inventory includes every retained segment, manifest, recorded recovery
intent and pending-deletion artifact, not only the current segment. Normal
rekey preserves segment names and event IDs. Relocation reseals all retained
segments for their new paths without renumbering or applying extra retention.
Crypto inventory also includes maintenance markers and relocation journals
under the selected config's `.loom/relocations/` directory. Pending relocation
blocks rekey. Retired source copies/journals remain part of the old configuration's
key-retirement inventory; destination verification does not authorize discarding
keys still needed for those originals.
`trace read` requires configured keys and a configured trace path, authenticates
records, then prints the already-redacted events as plaintext JSONL to stdout;
this is explicit disclosure to the caller, not automatic plaintext persistence.
These commands exit 0 on success and 1 on failure; exit 2 is indexing-only.
Relocation requires its explicit old/new configs instead of `--config`, and
returns success only after every destination root is activated or a requested
precommit abort is complete. Document the offline relocation steps prominently
in setup, storage/recovery and project-moving documentation.

The reference flow is: provide TOML, the external encryption-key environment
variable, and source documents; explicitly prepare the pinned local LLM
outside Loom; run setup; build; execute the unchanged script.
Network access is permitted for setup and explicitly configured providers/tools,
not for parsing, ordinary embedding loads, or index opening.

## 12. Implementation milestones

Implement in this order, keeping each milestone's contracts and tests together.
Independent dependencies below describe ordering, not authorization to spawn
parallel agents or implement additional features.

### M0: Package and immutable inputs

Depends on: none.

1. Create the specified package layout, direct dependency pins, uv lock, CLI
   entry point, formatting/type/test configuration, and platform CI matrix.
   Ignore `.loom/`, local secrets and scanner credentials/logs in Git; never
   commit generated user-data artifacts or keys.
2. Materialize the pinned embedding manifest and include attribution. Do not
   download the ONNX weights into the repository or package.
3. Create synthetic three-format fixtures and preserve the reference script.
4. Resolve the exact pins together; install wheel/sdist on every supported
   Python/platform combination. Verify FTS5 and SQLite serialization APIs,
   package metadata, artifact hashes, and licenses. Publish the actual evidence,
   not merely a list of versions found on an index.
5. Run Mend SCA, establish CodeQL SAST, and produce validated CycloneDX SBOMs
   under section 14.1. Missing approved scanner inputs block this gate.
6. Provision the pinned Ollama server/model, verify API version/digest/options,
   and demonstrate that `truncate=false`/`shift=false` reject oversized inputs
   instead of returning a successful shortened conversation. Save redacted
   requests/results and template-bound tests as compatibility evidence.

Exit: installable baseline, immutable input manifest, and recorded compatibility/
security evidence. Incompatible pins are blockers; only the section-14.1
security-update exception permits compatible remediation without an amendment.

### M1: Types, configuration, and test doubles

Depends on: M0.

1. Implement records, schema validation, exact result serialization, and errors.
2. Implement TOML loading/precedence, provider/store binding, consent and egress
   grants, TLS endpoint/context checks, key references, and caller identity.
3. Define the adapter/storage contracts and scripted model, clock, random, tool,
   transport, and event-sink doubles.
4. Implement configuration and schema rejection tests, including no side
   effects at import or object construction.
5. Implement authenticated envelopes, owner-only file creation, and atomic
   encrypted publication/maintenance helpers; prove tamper/key/permission
   failures closed, including trace header/payload identity mismatches.

Exit: all contracts are executable and the supplied script's setup resolves.

### M2: Runtime, tools, and events

Depends on: M1.

1. Implement the async loop and sync facade with precise reservations/budgets,
   the three fixed executor pools, bounded admission and actual-completion
   cleanup. A blocked Tool returns tool_busy rather than consuming a deadline.
2. Implement function/custom/HTTP tools, sequential dispatch, approval, bounded
   observations/errors/retries/correction, cancellation, and uncertain effects.
   Enforce taint-based egress independently of the sensitivity flag.
3. Implement context-byte enforcement, usage completeness, provenance tracking,
   and lifecycle events, exact-key redaction, segmented encrypted trace output/
   readback, five-segment retention and crash-safe rollover.
4. Add checkpoint hooks before/after operations without changing behavior in
   ephemeral mode.

Exit: deterministic boundary/error/approval tests pass with the fake adapter.

### M3: Durable conversations and recovery

Depends on: M2.

1. Implement both encrypted in-memory storage layouts, thread/process locks,
   capacity reservations, revisions, and atomic encrypted writes.
2. Implement conversation heads, fingerprints, safe resume, deadline extension,
   explicit outcome resolution, encrypted export/readback, deletion, rekey,
   and offline relocation of state/index/trace identities and binding descriptors.
3. Inject crashes around starts/results/publication and run the same contract
   suite against SQLite and JSON on each platform, including Windows handle
   contention, permissions, tamper rejection, interrupted key rotation and
   precommit/postcommit relocation recovery without dual active copies.

Exit: neither backend resets budgets, loses acknowledged progress, or
automatically duplicates unsafe effects after restart.

### M4: Provider adapters

Depends on: M2. Does not depend on M3.

1. Implement Ollama first, then OpenAI, Azure OpenAI, Anthropic, and Gemini using the
   specified client surfaces; all five are required, not phased out of v0.1.
2. Normalize messages, native calls, schema modes, signatures, finish reasons,
   usage, error classification, and cancellation.
3. Disable SDK retry multiplication and add mocked HTTP/request-count tests,
   verified TLS tests, Ollama context/identity checks, and overflow rejection.
4. Add opt-in live checks for tool-capable model profiles supplied through
   environment/configuration, with the same contracts as mocks.

Exit: all five pass adapter conformance; unsupported combinations fail clearly.

### M5: Documents and unchanged example

Depends on: M2, M3, and the completed Ollama adapter from M4. Other provider
adapters may still be in progress, but none substitutes for Ollama here.

1. Implement fixed offline embeddings, bounded loaders, normalized locations,
   deterministic chunks, hybrid search, cached model reuse, bounded query-memory
   admission/snapshot leases, and encrypted
   immutable snapshot publication without plaintext temporary databases.
2. Implement setup/build/refresh, all ingestion reports/exit codes, and pinning.
3. Implement search-tool attachment, document schema/citation validation,
   consent checks, abstention/conflict instructions, and one correction budget.
4. Execute the exact reference script with external setup and synthetic files;
   update `output.json` to the final output contract, not the script.
5. Execute the cold/warm end-to-end and finite-conversation acceptance in
   section 13.2 and the multi-store query-memory cases in section 13.3 on
   the specified reference hardware.

Exit: each format works without hidden setup, live-folder reads, invented
citations, or the removed confidence field.

### M6: Workflows

Depends on: M2 and M3.

1. Implement exact Step/Branch/Workflow APIs, validation, selectors, predicates,
   explicit data transfer, and aggregate ceilings.
2. Persist step outputs/branch decisions, inherited provenance and caller
   subject; resume under root ownership without widening egress permissions.
3. Verify no outer retry layer, no completed-step replay, and fail-fast errors.

Exit: sequential/branched workflows recover consistently with agent runs.

### M7: Release acceptance and documentation

Depends on: M3-M6 and all five M4 adapters.

1. Execute section 13 without weakening failing acceptance thresholds.
2. Build wheel/sdist and install in clean environments using the frozen lock.
3. Document setup, exact APIs/configuration, five providers, tool authoring,
   workflows, both backends, recovery, offline project relocation, trace expiry,
   pool saturation/hung-worker recovery, privacy and limitations.
4. Verify package/model notices and that no weights, credentials, or real user
   documents enter distributions/fixtures.
5. Repeat Mend SCA and CodeQL SAST on release inputs; generate/validate final
   runtime and development SBOMs and archive signed-off remediation evidence.

Exit: all required criteria pass. Code generation stops short of publishing;
publication requires a separate user instruction.

## 13. Acceptance contract

| Area | Required cases and assertions |
| --- | --- |
| Public example | Byte-identical `developer-code.py`; config/setup outside it; `print(result)` parses as the specified JSON envelope. |
| Imports | No network, indexing, database creation, or environment-secret resolution at import. |
| Agent state | Reused Agent does not share history; explicit state continues; invalid/incomplete state is rejected. |
| Schemas | Python/Pydantic/raw schemas; forbidden extras; local refs only; native/fallback output validated identically. |
| Model budgets | 10th logical turn allowed, 11th never scheduled; same-turn transient attempts at most three. |
| Tool budgets | 20th actual attempt allowed, 21st never enters callable; retries included; invalid/denied calls never execute. |
| Corrections | Exactly one allowance shared across no-search/schema/citation violations; a second violation fails. |
| Deadlines | Fake-clock tests at 30/120-second boundaries, backoff/approval/sink waiting, explicit resume extension, no counter resets. |
| Approvals | Callback precedence, exact yes, denial/EOF/headless, argument changes, serial TTY prompts, cancellation without orphan input readers. |
| Errors | Recoverable observations, nonrecoverable failures, refusal/truncation, missing usage, SDK retry suppression. |
| Unsafe effects | Timeout/crash after dispatch stops; no model repeat; explicit verified resolution only; live local invocation prevents premature resolution. |
| Storage | Same suite for SQLite/JSON; CAS, exclusive owners, stale conversation heads, atomic replacement, incompatible fingerprints, crash boundaries. |
| Workflow | Cycles/unreachable nodes rejected; zero/one/multiple matching cases; typed selectors; child/root budgets charged; saved branches and completed steps not replayed. |
| Formats | PDF pages, Markdown lines/headings, DOCX blocks/tables, encrypted/image-only/oversize failures, no external content fetching. |
| Indexing | Every file accounted for; partial exit 2; all-failed preserves old CURRENT; empty root publishes empty; crash leaves a valid generation. |
| Snapshots | No source-folder reads during queries; edits/deletes affect new snapshots only; continuation/resume pins old IDs; missing snapshot fails. |
| Retrieval | Exact keywords, paraphrases, deterministic fusion/ties, proper text bounds, empty results, no semantic-only fallback after lexical failure. |
| Grounding | No-search corrected once; valid provenance; fake path/location/excerpt rejected; absent evidence abstains; conflicts cite distinct locations; confidence rejected. |
| Consent | Provider selection alone insufficient; denial before transfer; checks cover retrieved data, history, derived messages, and resumed state. |
| Events | Stable IDs/order within run; no secret payloads by default; redaction before all sinks; failed sinks visible; durable failures stop progress. |
| Providers | Conformance for all five; real tool/schema smoke cases on explicitly provisioned profiles; no cross-provider fallback. |
| Installation | Python/platform matrix, CLI behavior, frozen dependencies, local embedding integrity, no runtime downloads with outbound networking disabled. |
| TLS | Reject non-loopback HTTP, including remote Ollama and every HTTPTool; accept only literal-loopback Ollama HTTP; fail bad certificates/hostnames and TLS below 1.2; no insecure SDK transport overrides or downgrade redirects. |
| Encryption/integrity | Missing/wrong/unknown keys fail; ciphertext cannot be parsed as JSON/SQLite; distinctive fixture plaintext absent from all managed files, temp copies and journals; bit flips, swapped resources and malformed headers rejected before loading state; authenticated historical replay limitation documented. |
| File permissions | POSIX 0700 directories/0600 files, restrictive inherited ACL behavior, rejection of preexisting broad permissions; protected Windows DACL; no permission changes to caller sources. |
| Key lifecycle | Interrupted rekey remains readable with retained keys; new writes use active key; verify inventories all configured artifacts; old-key removal fails visibly while old artifacts remain; exports remain encrypted. |
| Document egress | A model copies, paraphrases, encodes, or omits local document text in arguments: cumulative taint still denies network/unknown-egress dispatch; test `sensitive=False`, approving callback, retries, workflow selectors and resume; hosted LLM consent alone does not authorize tool egress. |
| Redaction/privacy | Exact aliases including `accessToken`, `refresh_token`, `X-API-Key`, nested lists/objects, extra configured aliases, known-secret values; preserve usage/budget fields; default metadata/errors contain no source filenames/paths or caller attributes. |
| Custom tools/approval | Concrete Tool subclass and sync/async callable; immutable ToolContext; trusted caller propagated; exact callback return validation; callback exception, timeout and invalid return deny visibly without terminal fallback. |
| Observation limits | Exactly-at/one-byte-over 8 KiB observation and 1 MiB raw caps; 4 KiB retrieval cap and dynamic headroom; oversized successful unsafe tool is recorded once, never auto-retried or replayed, and reports omission rather than fake execution failure. |
| Concurrency | Shared Agent/Store/Storage across threads/tasks and sync loops; same Tool serialization; same-run contention; adapters reject unsafe reuse; model initialization single-flight and reused across repeated synchronous calls. |
| Capacity/Windows | Exact JSON16MiB/SQLite128MiB and checkpoint/reservation boundaries; refusal before unsafe work; failed Windows replacement leaves old authenticated file; reader/writer crash and lock cases use both backends. |
| Ollama | Required num_ctx/version/digest; invalid byte/output/template allowance rejected; emitted `truncate=false`, `shift=false`, explicit num_ctx/num_predict; live oversized input refused without silent message loss. |
| Supply chain | Actual pin resolution/install matrix; Mend policy violation and missing-scan failures block release; CodeQL security findings handled; validated runtime/development SBOMs match released artifacts; critical patch path does not require a new design interview. |
| Stuck tools | Timeout a controlled synchronous tool; a later run using the same Tool gets tool_busy with the original call/run ID, not deadline exhaustion; no new attempt, approval or automatic retry; claim is released only after actual completion. |
| Executor isolation | Fill all four user-code slots with timed-out workers; internal checkpoint/encryption work and unrelated retrieval still run. Assert worker/queue limits, immediate saturation errors, cancelled-queue accounting, actual-completion release and safe sync-loop teardown. |
| Query memory | Shared ledger across multiple agents/stores; two-snapshot limit includes pending loads; exact-at/over byte reservations; atomic admission, single-flight loading, zero-lease LRU eviction, no eviction during active work, release after late completion, identical snapshot IDs after reload. |
| Relocation | Move fixtures to new directory/home/mount paths with both storage backends; queries and continuation/resume preserve IDs, counters, deadlines and unresolved effects. Test every copy/retirement/commit/activation crash boundary, precommit abort, occupied targets, tampering, missing keys, live-owner rejection and semantic-config changes. Old sources remain intact but cannot execute after retirement; activated destinations are never recopied on retry. |
| Trace retention | Count encoded bytes, not plaintext size; exact segment boundary, rollover-before-overflow, five retained segments, oldest-closed expiry diagnostic, simultaneous writers, interrupted manifest/deletion/rekey recovery, bounded staging, and visible reader expiry. No deletion of run state or index generations. |
| Trace identity | Interleave runs with identical sequence numbers; verify header/payload run ID, event ID and sequence agreement; foreign-run filtering never relabels an event. Modified headers, copied records under another segment identity, and malformed IDs fail; whole-record replay/deletion remains explicitly outside the guarantee. |

Unit tests use fake models/embeddings/clocks and must run offline. Mark prepared
local integration tests `integration`; mark paid/provider tests `live`. Default
`pytest -m "not integration and not live"` must not contact services. A release
requires recorded live smoke evidence for all five; missing credentials skip
ordinary development runs but do not waive the release gate.

Run Ruff check/format check and mypy in strict mode over `src/loom`. Adapter
boundary conversions use proper types/guards, not blanket `Any` casts.
Do not assert exact prose in live tests. Assert statuses, schema, provenance,
counter bounds, and source-supported behavior on synthetic facts.
Concurrency tests use controllable blocking workers and unblock them in
`finally`; do not leave permanently hung threads or helper processes behind.

### 13.1 Fixed performance acceptance

Benchmark on Apple M1, 8 CPU cores, 16 GiB RAM, local SSD, macOS 13+, CPython
3.11, four ONNX threads, no GPU. Record OS/dependency versions.

- Corpus: 100 Markdown files, exactly 100,000 extracted ASCII bytes each,
  10,000,000 bytes total. Mixed-format correctness is covered separately.
- Warm model and index with 20 queries; measure 100 subsequent fixed queries.
- Warm retrieval p95 must be at most 1,000 ms, including query embedding and
  fusion, excluding any answer-generating LLM call.
- p95 is the sorted sample at `ceil(0.95 * count) - 1`.
- Peak process RSS during full indexing must be at most 2 GiB.
- Record cold setup/index duration and installed/download size; no other
  retrieval/indexing latency or installation-size SLA is claimed. The separate
  end-to-end reference gate below includes model latency.

These are fixed release criteria, not measurements already achieved. A failing
implementation must be optimized; changing thresholds requires user approval.

### 13.2 End-to-end reference acceptance

Use the section-13.1 machine, pinned Ollama 0.40.2/model digest, four CPU threads,
`num_gpu=0`, and the exact reference TOML/script. Prepare embeddings, encryption
keys, and synthetic PDF/Markdown/DOCX fixtures explicitly. Place the 500-requests-
per-minute fact in `limits.md`; the other files contain distinct supporting
facts and no conflicting rate limit.

Run three fresh-process cold trials with both the server model and embedding
cache unloaded, then ten warm trials with one reused Agent/DocumentStore.
Measure from `agent.run` entry through returned result, including model load,
prompt evaluation, retrieval, validation, and final generation; do not exclude
LLM latency. Every trial must complete within 600 seconds and every model
attempt within 180 seconds, with supported citations, no context/limit error,
and the required JSON envelope. Record every duration, peak RSS, model digest,
effective num_ctx, and attempt/correction counts; do not report unmeasured
performance as a guarantee.

In a separate continuation test, ask the initial question plus exactly two
follow-up questions, passing returned state each time. Require one successful
retrieval per turn on this fixed fixture, grounded completion on all three,
and at most 24,576 canonical context bytes before each model request.
This proves the documented minimum three-question conversation for the
reference fixture, not an unlimited chat capacity. Larger histories still
fail explicitly when full; no hidden trimming/summarization is introduced.

### 13.3 Query-memory and resource-pressure acceptance

On the section-13.1 machine, prepare three independent stores using that same
100-file/10,000,000-byte corpus per store, plus one retained historical snapshot.
Use direct searches or the scripted model so this measures Loom rather than
the Ollama server's memory. Exercise repeated synchronous runs and four
concurrent callers across current/historical snapshots for 100 queries.

Assert at most two resident/loading snapshots and at most 3 GiB of combined
reservations at every admission/transition. Saturation may return the specified
busy/memory error, never wrong/empty results; retry explicitly after pressure is
released and verify all queries succeed against their original snapshot IDs.
Peak whole-process RSS for this fixed query workload must be at most 2 GiB,
including embedding initialization, decryption, SQLite and NumPy copies.
Record load/eviction counts and prove reuse across synchronous event loops.

Also test real maximum-size index loads and exact reservation boundaries using
controlled scheduling. Measure their RSS and retain the evidence; the fixed
2 GiB workload threshold is not a claim that arbitrary applications or every
allowed 512 MiB snapshot combination fits within 2 GiB.

## 14. Explicit exclusions and change control

Do not implement TypeScript, Python 3.14+, OCR, multilingual embeddings, token
streaming, hosted execution, OpenAPI import/discovery, remote telemetry,
automatic provider fallback, prompt-emulated tools, automatic history trimming/
summarization, untrusted-code sandboxing, parallel/nested/cyclic workflows,
compensation/sagas, distributed workers, automatic snapshot pruning, aggregate
token/currency-spend budgets, Azure Entra ID/managed-identity authentication,
or trusted external anti-rollback/freshness services. Azure supports API keys
only; usage accounting does not imply cost accounting.

There is no automatic schema migration in v0.1. Unsupported versions fail
explicitly. Deployment inputs such as model names, credentials, filesystem
paths, and approved finite limit overrides are intentionally configurable
within the exact schemas above; they are not unresolved design questions.

Mechanical work remaining for code generation is implementing these contracts,
resolving/persisting transitive dependencies under the fixed direct pins,
materializing the pinned artifact manifest, and passing acceptance. If the
specification proves internally inconsistent or a dependency cannot implement
a required contract, report that precise blocker and request an amendment.
Do not ask the generator to choose a different API, model, database, algorithm,
recovery policy, or reduced release scope.

### 14.1 Security maintenance and release gates

The critical-remediation deadline is 24 hours from the first project notification
of an applicable vulnerability. Project deadlines for other severities are
7 days high, 30 days medium, and 90 days low. These are this plan's adopted
controls, not a claim that an IBM policy document was independently inspected.
Record notification time, applicability, owner, fix/mitigation, and evidence.
If a fix is unavailable, isolate the affected feature/deployment, block affected
releases, and escalate before the deadline; never silently waive the finding.

**Standing security exception:** compatible security updates to direct/transitive
dependencies, uv, scanner tools, Ollama, or the embedding artifact do not require
a new design interview. Update exact pins/digests, lock, notices, SBOMs and
conformance evidence together. Preserve public API, platforms, privacy,
retrieval behavior and recovery contracts. A breaking change still requires an
amendment; do not leave a vulnerable release unmitigated while waiting for it.

Run checks at M0, on pull requests, daily, and before M7 release acceptance:

1. **Mend SCA:** use the enterprise-approved CLI with protected CI inputs
   `MEND_CLI_VERSION` and per-runner `MEND_CLI_SHA256` (64 lowercase hex digits),
   plus service credentials and an exact organization/application/project
   `MEND_SCOPE`. Verify binary version/hash before execution. Missing inputs,
   resolution errors, and policy failures are failing checks, not skipped passes.
   Stage only the frozen public dependency manifest/lock-derived requirements;
   never scan or upload application source, `.loom`, real documents, credentials,
   or user state. Do not enable extended/source/reachability scanning.
   Execute `mend dep --dir <dependency-stage> --dev --strict --fail-policy
   --format json --non-interactive --scope <approved-scope>`.
   Configure the organization policy to block critical/high findings and overdue
   medium/low findings; verify exit code 9 on a synthetic policy violation.
   Scan the complete runtime/development dependency inventory, including
   platform-specific packages, not only packages installed on the CI host.
2. **SAST:** use CodeQL Python `security-extended`, bundle 2.27.2 and
   `github/codeql-action` pinned to
   `24c54180a607b1449ed407dd24f251e4e9147c8d`. Analyze locally in CI and retain
   SARIF in the repository's authorized security/artifact storage. Fail critical/
   high security findings and overdue lower-severity findings; scanner failure
   is never success. Do not send source to Mend as part of this separate check.
3. **SBOM:** use cyclonedx-bom 7.5.0 against clean runtime and development
   environments. Run `cyclonedx-py environment <environment-python>
   --spec-version 1.6 --output-format JSON --output-file <sbom-path> --validate`.
   Retain distinct validated runtime/development SBOMs, dependency hashes,
   lockfile hash, platform inventory, and model artifact/license manifest with
   release evidence. Exclude credentials, source documents and private paths.

M0 records actual installed versions, all direct/transitive resolution results,
scanner identity/policy checks, and platform failures. Metadata existence is
not an installation or vulnerability result. M7 reruns against the exact built
release artifacts. A scanner outage or missing approval cannot produce a
"clean" release. No package installation, scan or hardware acceptance result
is claimed by this planning document.

## 15. References

- [Pinned FastEmbed implementation](https://github.com/qdrant/fastembed/tree/v0.9.0):
  public local model-path loading and CPU execution.
- [Pinned embedding artifact](https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q/tree/aa8f8b060edb00e03bfdd08813a2949946c8ba55):
  model identity, files, revision, and license.
- [ONNX Runtime 1.23.2](https://pypi.org/project/onnxruntime/1.23.2/):
  CPU wheel/platform baseline.
- [pypdf extraction](https://pypdf.readthedocs.io/en/stable/user/extract-text.html):
  text extraction, image-only documents, and decoded-stream memory limits.
- [python-docx](https://python-docx.readthedocs.io/en/latest/):
  paragraphs, tables, and document structure.
- [SQLite FTS5](https://www.sqlite.org/fts5.html):
  embedded lexical retrieval.
- [Anthropic structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs):
  the native `output_config.format` contract.
- [Ollama 0.40.2 chat implementation](https://github.com/ollama/ollama/blob/v0.40.2/server/routes.go):
  explicit truncation/context-shift controls; live refusal remains an M0 gate.
- [Ollama 0.40.2 prompt construction](https://github.com/ollama/ollama/blob/v0.40.2/server/prompt.go):
  message selection and overflow behavior.
- [AES-GCM authenticated encryption](https://cryptography.io/en/latest/hazmat/primitives/aead/):
  nonce handling and authenticated-data contract.
- [Mend SCA configuration](https://docs.mend.io/legacy-sca/latest/configure-the-mend-cli-for-sca):
  strict resolution, policy-failure, dependency-only scan and output flags.
- [CycloneDX Python environment command](https://cyclonedx-bom-tool.readthedocs.io/en/latest/usage.html):
  environment inventory and validated SBOM output.
