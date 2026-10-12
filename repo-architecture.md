# Loom Repo Architecture

Loom is an embedded Python framework for agents, tools, document retrieval, and sequential workflows. It is distributed as the `loom-agentic-framework` package and imported as `loom`. The entry-point binary is `loom` (mapped to [`loom.cli:main`](src/loom/cli.py)).

---

## Repository layout

```
loom/
├── src/loom/            # Framework source (all importable modules)
│   ├── agent.py         # Agent and _Run execution engine
│   ├── workflows.py     # Workflow, Step, Branch, Case orchestration
│   ├── contracts.py     # Shared data types and Pydantic records
│   ├── config.py        # Config, ProviderConfig, AgentProfile, and related records
│   ├── events.py        # Event, EventSink, Events dispatcher
│   ├── trace.py         # Encrypted rotating trace sink (TraceSink)
│   ├── recovery.py      # Unsafe-tool outcome reconciliation
│   ├── runtime.py       # Internal async utilities
│   ├── testing.py       # Scripted provider helpers for offline tests
│   ├── cli.py           # `loom` CLI entry point
│   ├── tools/           # Tool abstraction layer
│   ├── providers/       # LLM provider adapters
│   ├── documents/       # Document ingestion, embedding, and retrieval
│   ├── storage/         # Persistence backends
│   └── security/        # Crypto, egress, transport, and maintenance
├── tests/
│   ├── unit/            # Offline unit tests (default pytest run)
│   ├── integration/     # Opt-in: requires local embeddings or Ollama
│   └── acceptance/      # Opt-in: requires live provider credentials
├── sdd/
│   ├── basic-plan.md         # v0.1 specification
│   ├── basic-requirement.md  # Requirements document
│   └── query-agent-example/  # Reference example with loom.toml and developer-code.py
├── pyproject.toml       # Build metadata, dependencies, and tool config
└── uv.lock              # Pinned dependency lockfile
```

---

## Core concepts

### Config ([`src/loom/config.py`](src/loom/config.py))

The [`Config`](src/loom/config.py) record is the single source of truth for every runtime setting. It is loaded from `./loom.toml` (or an explicit path) at construction time and never re-read during a run. Key sub-records:

| Record | Purpose |
|---|---|
| `ProviderConfig` | LLM provider kind, model, capabilities, and credentials (via env vars) |
| `AgentProfile` | Per-agent provider selection, output schema, limits, and document access |
| `StoreConfig` | Source path and index path for a document store |
| `EmbeddingConfig` | BGE model cache location |
| `StorageConfig` | Persistence backend selection and path |
| `EncryptionConfig` | Active key ID and key map (base64-encoded 32-byte keys) |
| `TransportConfig` | Custom CA bundle for HTTPS |
| `ObservabilityConfig` | Trace payload/source-path capture and redaction |

### Contracts ([`src/loom/contracts.py`](src/loom/contracts.py))

All shared data structures are strict, frozen Pydantic `Record` instances. Notable types:

- **`RunResult`** — the value returned by every `agent.run` / `workflow.run` call; carries `status`, `output`, `usage`, `diagnostics`, and a `ConversationState` for continuation.
- **`ConversationState`** — the full message history and snapshot references; pass as `state=` to continue a conversation.
- **`ToolSpec` / `AgentSpec`** — immutable snapshots of a tool's or agent's definition at bind time.
- **`ToolContext`** — injected into tool callables at execution time; carries run/call identity, deadline, cancellation token, and caller identity.
- **`Limits` / `WorkflowLimits`** — configurable ceilings for turns, tool attempts, deadlines, token budgets, and context window size.
- **`DocumentAnswer` / `Citation` / `SearchResult`** — document retrieval output types.

---

## Module walkthrough

### Agent ([`src/loom/agent.py`](src/loom/agent.py))

`Agent` is the top-level object application code instantiates. It holds a resolved `ModelAdapter`, bound `DocumentStore` instances, attached `Tool` objects, an optional `Storage` backend, and event sinks.

Execution flow (`arun` / `aresume` → `_execute` → `_Run.execute`):

```
Agent.run(input)
  └─ _Run.execute(resume=False)
        ├─ model_turn()            # send ModelRequest → ModelAdapter → ModelResponse
        ├─ execute_tools()         # fan-out concurrent tool calls
        │     ├─ approval check    # sensitive tools require callback
        │     ├─ egress check      # allowlist enforcement
        │     └─ Tool.execute()    # actual callable or HTTP dispatch
        ├─ validate_answer()       # JSON schema validation of output
        └─ finish()                # emit RunResult, persist checkpoint
```

- `agent.run` / `agent.resume` are synchronous facades; `arun` / `aresume` are the native async methods.
- Conversation continuation requires `state=previous_result.state`; omitting it starts a fresh conversation.
- `agent.spec` returns a recursively immutable `AgentSpec` snapshot.

### Workflows ([`src/loom/workflows.py`](src/loom/workflows.py))

A `Workflow` is a directed graph of `Step` nodes. Each step targets either an `Agent` or a `Tool`. Control flow is declared with `next: str | Branch | None`; a `Branch` contains ordered `Case` predicates and an optional default target.

```
Workflow
  ├─ steps: {name → Step}
  │     └─ Step.target: Agent | Tool
  │     └─ Step.input_selector: (WorkflowView) → JSONValue
  │     └─ Step.next: str | Branch | None
  └─ output_selector: (WorkflowView) → JSONValue
```

The `_WorkflowRun` internal class drives execution: it advances the graph step by step, persists intermediate checkpoints, enforces `WorkflowLimits`, and owns all child agent histories. Workflows expose the same `run` / `arun` / `resume` / `aresume` API as `Agent`.

### Tools ([`src/loom/tools/`](src/loom/tools/))

| File | Class | Purpose |
|---|---|---|
| `base.py` | `Tool` (ABC) | Claim/phase/release lifecycle; `Tool.from_callable` factory |
| `function.py` | `FunctionTool` | Wraps a Python callable |
| `http.py` | `HTTPTool` | Fixed-origin HTTPS calls; per-call auth from env vars |
| `agent.py` | `AgentTool` | Wraps an `Agent` as a callable tool; supports suspend and handoff modes |
| `approval.py` | — | Approval-request helpers |
| `runner.py` | `ToolRunner` | Concurrent tool execution with retry, timeout, and egress enforcement |

Tools declare `sensitive`, `retry_safe`, and `egress` in their `ToolSpec`. Sensitive tools are denied when no approval callback is supplied (headless execution).

### Providers ([`src/loom/providers/`](src/loom/providers/))

All providers implement the `ModelAdapter` ABC ([`base.py`](src/loom/providers/base.py)) with `complete(request) → ModelResponse`.

| Adapter | Provider |
|---|---|
| `openai.py` | OpenAI |
| `azure_openai.py` | Azure OpenAI |
| `anthropic.py` | Anthropic Claude |
| `gemini.py` | Google Gemini |
| `ollama.py` | Ollama (local) |

`factory.py` selects the adapter from `ProviderConfig.kind`. `hosted.py` contains shared helpers for hosted providers. Plain HTTP is only permitted for Ollama loopback; all other providers require TLS 1.2+.

### Documents ([`src/loom/documents/`](src/loom/documents/))

End-to-end RAG pipeline:

```
source files (PDF / DOCX / Markdown)
  └─ loaders.py       extract text
  └─ chunks.py        split into overlapping chunks
  └─ embeddings.py    BGE ONNX embeddings via fastembed
  └─ index.py         build / refresh encrypted snapshot
        └─ store.py   DocumentStore — pin snapshot, search, expose SearchTool
              └─ citations.py   chunk → Citation mapping
```

- Snapshots are AES-256-GCM encrypted blobs stored at `index_path`.
- All index generations are retained; queries use a shared admission ledger for memory budgeting.
- `DocumentStore.tool(snapshot_id)` returns a `SearchTool` that can be attached to an agent.
- `loom index build` / `loom index refresh` drive ingestion from the CLI.

### Storage ([`src/loom/storage/`](src/loom/storage/))

Persistence backends store encrypted checkpoint images (not plaintext).

| Class | File | Notes |
|---|---|---|
| `Storage` (base) | `base.py` | Locking, encryption, checkpoint record lifecycle |
| `JSONStorage` | `json.py` | Rewrites full image; for small stores |
| `SQLiteStorage` | `sqlite.py` | Row-level writes; for larger stores |

`checkpoint.py` defines the `Checkpoint` record shape. Both backends support `save`, `load_run`, `load_conversation`, `export_run`, and `delete_*` operations. Resume an interrupted run with `agent.resume(run_id)`.

### Security ([`src/loom/security/`](src/loom/security/))

| File | Responsibility |
|---|---|
| `crypto.py` | `Crypto` class — AES-256-GCM seal/open, key derivation, `LOOMENC1` magic |
| `egress.py` | Allowlist enforcement for tool network access |
| `transport.py` | HTTPS client with CA-bundle override and TLS 1.2 minimum |
| `admin.py` | Key rotation (`loom crypto rekey`) |
| `relocation.py` | Offline relocation (`loom crypto relocate`) |
| `maintenance.py` | Maintenance-mode markers for safe relocation |
| `files.py` | Path canonicalisation and overlap checks |
| `windows.py` | Windows permission conformance helpers |

Encrypted artifacts are bound to their resource identity and canonical path. Moving a project requires `loom crypto relocate`, not a plain directory rename.

### Events and Traces ([`src/loom/events.py`](src/loom/events.py), [`src/loom/trace.py`](src/loom/trace.py))

`Events` is the internal event bus. Each run creates an `Events` instance that emits `Event` records to one or more `EventSink` callables (sync or async). Secret values and configurable keys are redacted before emission.

`TraceSink` is a built-in `EventSink` that writes events to rotating AES-256-GCM encrypted `.enc.jsonl` segments (up to five 16 MiB segments). Traces are read with `loom trace read`.

---

## Data flow diagram

```
Application code
      │
      ▼
  Agent / Workflow  ──── Config (loom.toml)
      │
      ├──► ModelAdapter (providers/)  ──── LLM API (OpenAI / Anthropic / Gemini / Azure / Ollama)
      │
      ├──► Tool execution (tools/)
      │         ├── FunctionTool  (Python callable)
      │         ├── HTTPTool      (HTTPS fixed-origin)
      │         └── AgentTool     (sub-agent delegation)
      │
      ├──► DocumentStore (documents/)
      │         ├── loaders → chunks → embeddings
      │         └── encrypted index snapshot
      │
      ├──► Storage (storage/)
      │         └── encrypted checkpoint (JSON or SQLite backend)
      │
      └──► Events → EventSink(s)
                    └── TraceSink (encrypted rotating .enc.jsonl)
```

---

## CLI commands

The `loom` binary (entry point `loom.cli:main`) provides:

| Command | Action |
|---|---|
| `loom setup embeddings` | Download and verify pinned BGE ONNX artifacts |
| `loom index build <store>` | Build a new encrypted document index snapshot |
| `loom index refresh <store>` | Incrementally refresh an existing snapshot |
| `loom trace read` | Decrypt and stream trace events |
| `loom crypto verify` | Verify encrypted artifact integrity |
| `loom crypto relocate` | Safely relocate an initialized project |
| `loom crypto rekey` | Rotate the active encryption key |

Exit codes for index commands: **0** = complete, **2** = partial/exclusions, **1** = fatal failure, **130** = interrupted.

---

## Key design constraints

- **No parent-directory config search.** `loom.toml` must be in the working directory or passed explicitly.
- **No plaintext persistence.** All storage and trace artifacts are AES-256-GCM encrypted.
- **No secrets in code.** API keys are read from environment variables; `LOOM_ENCRYPTION_KEY` provides the storage key.
- **TLS 1.2+ everywhere** except Ollama loopback.
- **Sensitive tools denied headless.** Without an approval callback, sensitive tools are automatically denied.
- **Immutable records.** All `contracts.py` types are strict, frozen Pydantic models; `AgentSpec` and `ToolSpec` are recursively immutable.
- **Python 3.11–3.13 only.** Python 3.14 is explicitly unsupported.
- **Sub-agent depth bounded.** `max_agent_depth` (default 5) and cycle detection via `_agent_chain` in `ToolContext.context` prevent unbounded recursive agent invocation.
