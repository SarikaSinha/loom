# Loom Multi-Agent Support — Implementation Specification

## 1. Authority and scope

This document specifies the addition of **`AgentTool`** to Loom: a `Tool` subclass whose callable is another `Agent.arun`. It enables one agent to dynamically invoke another at runtime without requiring a statically-declared `Workflow`.

**What is in scope:**
- New `src/loom/tools/agent.py` containing the `AgentTool` class
- Changes to `src/loom/contracts.py` (`Limits`, `ToolContext`)
- Changes to `src/loom/tools/base.py` (optional `Tool.from_agent` is explicitly excluded — see §2)
- Changes to `src/loom/__init__.py` (export `AgentTool`)
- Changes to `architecture.md` (document the new primitive)

**What is explicitly out of scope:**
- Workflow changes
- New storage backends
- New provider adapters
- Any changes to existing `Agent`, `Workflow`, `Tool`, `FunctionTool`, or `HTTPTool` behaviour beyond what is required to thread `ToolContext` depth-tracking

---

## 2. Design decisions

| # | Topic | Decision |
|---|-------|----------|
| 1 | Execution mode | Hybrid: `handoff: bool = False`. Default (`False`) = suspend Agent A, run Agent B, return result as tool observation. `True` = Agent A terminates, B's output becomes A's output with a handoff diagnostic. |
| 2 | Deadline / limits | Intersection: Agent B runs under `min(its own configured limits, Agent A's remaining budget)`. |
| 3 | Agent call budget | New `max_agent_calls: int = Field(default=20, gt=0)` added to `Limits`. Decremented independently from `max_tool_attempts`. |
| 4 | `CallerIdentity` | Configurable: `AgentTool(caller_override=...)`. If omitted, a derived machine identity `CallerIdentity(subject="agent:<parent_agent_name>")` is used. |
| 5 | Context propagation | Configurable: `AgentTool(context_selector=...)`. If omitted, Agent B receives an empty context dict. |
| 6 | Conversation state | Configurable: `AgentTool(state_selector=...)`. Receives `ToolContext`, returns `ConversationState | None`. If omitted, Agent B always starts a fresh `ConversationState`. |
| 7 | Persistence | Agent B uses its own configured `Storage`. Sub-runs are independently recoverable. |
| 8 | Tool observation | Configurable: `AgentTool(output_selector=...)`. Defaults to returning `RunResult.output` only. |
| 9 | `sensitive` | `False` by default. Developer opts in explicitly. |
| 10 | Events | Separate streams. Agent B's events go to its own `event_sinks`. Both share `root_run_id`. |
| 11 | Egress policy | `AgentTool` accepts an optional `egress_policy` that can only restrict (not expand) Agent B's own egress. Deny-only. |
| 12 | Recursion guard | Depth limit (`max_agent_depth`) + cycle detection via agent-name chain in `ToolContext`. `ToolFailure("agent_cycle_detected")` raised on re-entry. |
| 13 | `handoff=True` result | Agent A returns `status="completed"`, Agent B's output as its own output (via `output_selector`), and a `Diagnostic(code="agent_handoff", ...)` entry. Usage is summed. |
| 14 | Public API | `AgentTool` class in `src/loom/tools/agent.py`, exported from `loom` root. No `Tool.from_agent()` factory. |
| 15 | Argument schema | Optional `argument_schema` parameter. Defaults to `{"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}`. |
| 16 | Output schema | Optional `output_schema` parameter. Defaults to mirroring Agent B's `output_schema`; falls back to `{"type": "string"}` if Agent B has none. |
| 17 | `retry_safe` | Developer-controlled: `AgentTool(retry_safe=False)` (default `False`). |
| 18 | Testing | No special mock support. Developers configure Agent B with a scripted provider from `testing.py` for offline tests. |
| 19 | Configuration fingerprint | `ToolSpec` fields only (name, version, description, schemas). Agent B's full `AgentSpec` is not embedded. The `version` string on `AgentTool` is the compatibility signal. |
| 20 | `max_agent_depth` default | `5`, added to `Limits`. |
| 21 | `egress` field on `ToolSpec` | Derived at construction: `"none"` if Agent B has no `HTTPTool`s; `"fixed_network"` (aggregating all origins) if all HTTP tools are `"fixed_network"`; `"unknown"` otherwise. |

---

## 3. Contracts changes (`src/loom/contracts.py`)

### 3.1 `Limits` — two new fields

```python
class Limits(Record):
    # ... existing fields unchanged ...
    max_agent_calls: int = Field(default=20, gt=0)
    max_agent_depth: int = Field(default=5, gt=0)
```

`max_agent_calls` is the per-run budget for `AgentTool` invocations. It is decremented in `_Run.execute_tools` when an `AgentTool` observation is successfully completed (same accounting point as `tool_attempts`).

`max_agent_depth` is the maximum number of nested agent frames permitted in a single call chain. It is enforced by reading the `_agent_chain` list from `ToolContext.context` at `AgentTool.execute` time.

### 3.2 `ToolContext` — no schema change required

`ToolContext.context` is already `Mapping[str, object]`. The depth-tracking chain is passed as a reserved key `"_agent_chain"` (a `tuple[str, ...]` of agent names from outermost to innermost). The `AgentTool` reads and extends this key when building the context for Agent B.

The `"_agent_chain"` key is a private runtime convention — it is never exposed in events or persisted. The key name starts with `_` to signal it is framework-internal and should not be set by application code.

---

## 4. `AgentTool` — full specification (`src/loom/tools/agent.py`)

### 4.1 Constructor signature

```python
class AgentTool(Tool):
    def __init__(
        self,
        agent: Agent,
        *,
        name: str,
        description: str,
        version: str = "1",
        handoff: bool = False,
        argument_schema: Schema | type[BaseModel] | None = None,
        output_schema: Schema | type[BaseModel] | None = None,
        caller_override: CallerIdentity | None = None,
        context_selector: Callable[[ToolContext], Mapping[str, object]] | None = None,
        state_selector: Callable[[ToolContext], ConversationState | None] | None = None,
        output_selector: Callable[[RunResult], JSONValue] | None = None,
        egress_policy: Literal["deny_all_http"] | None = None,
        sensitive: bool = False,
        retry_safe: bool = False,
        timeout_seconds: float | None = None,
    ) -> None
```

**Parameter rules:**

| Parameter | Rule |
|-----------|------|
| `name` | Required. Must match `TOOL_NAME` regex (`^[a-z][a-z0-9_]{0,63}$`). Must not collide with any other tool on the parent agent. |
| `description` | Required, non-empty. |
| `version` | Compatibility signal for checkpoint fingerprinting. Bump on breaking changes. |
| `handoff` | `False` (default) = suspend-and-return. `True` = terminal hand-off. |
| `argument_schema` | Optional. Defaults to `{"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]}`. |
| `output_schema` | Optional. Defaults to Agent B's `output_schema` if set, otherwise `{"type": "string"}`. |
| `caller_override` | Optional `CallerIdentity`. If omitted, defaults to `CallerIdentity(subject="agent:<parent_agent_name>")` constructed at execute-time from `ToolContext`. |
| `context_selector` | Optional callable `(ToolContext) -> Mapping[str, object]`. If omitted, Agent B receives `{}`. |
| `state_selector` | Optional callable `(ToolContext) -> ConversationState | None`. If omitted, Agent B receives `None` (fresh conversation). |
| `output_selector` | Optional callable `(RunResult) -> JSONValue`. If omitted, defaults to `lambda r: r.output`. |
| `egress_policy` | `None` (no extra restriction) or `"deny_all_http"` (Agent B's HTTP tools are blocked at the `AgentTool.execute` boundary before Agent B is invoked). |
| `sensitive` | `False` by default (unlike `HTTPTool`). Opt in explicitly if approval is required. |
| `retry_safe` | `False` by default. Developer sets `True` if Agent B's operation is idempotent. |
| `timeout_seconds` | Optional per-call timeout ceiling. Applied on top of the deadline intersection rule. |

### 4.2 `ToolSpec` construction

At `__init__` time, `AgentTool` constructs its `ToolSpec` as follows:

- `name`, `description`, `version`: from constructor parameters.
- `argument_schema`: resolved schema (default or provided).
- `output_schema`: resolved schema (default mirrors Agent B's `output_schema`).
- `sensitive`: from constructor parameter.
- `retry_safe`: from constructor parameter.
- `timeout_seconds`: from constructor parameter.
- `egress`: derived by inspecting Agent B's tools at construction time:
  - If Agent B has no `HTTPTool` instances → `"none"`.
  - If all `HTTPTool` instances on Agent B have `egress="fixed_network"` → `"fixed_network"`, with `egress_origins` set to the union of all their `egress_origins` tuples.
  - Otherwise → `"unknown"`.
- `egress_origins`: aggregated as described above; empty tuple if `egress != "fixed_network"`.

### 4.3 `execute` method

```
AgentTool.execute(arguments, context) → JSONValue
```

**Full execution sequence:**

1. **Cycle detection.** Read `_agent_chain` from `context.context` (default `()`). If `agent.name` is already in the chain → raise `ToolFailure("agent_cycle_detected", recoverable=True)`.

2. **Depth check.** If `len(_agent_chain) >= agent.limits.max_agent_depth` → raise `ToolFailure("agent_depth_exceeded", recoverable=True)`.

3. **Agent call budget check.** The `AgentTool` itself does not decrement this — it is decremented by the parent `_Run` after the observation is recorded (see §5). The check at this point is informational only (no action).

4. **Egress policy.** If `egress_policy="deny_all_http"` → verify Agent B has no `HTTPTool` in its `tools` tuple. If it does → raise `ToolFailure("egress_policy_denied", recoverable=True)`. *(Note: this is a construction-time guarantee if the policy is known statically, but checked at runtime to handle future dynamic configurations.)*

5. **Deadline intersection.** Compute effective deadline: `min(context.deadline, now + agent.limits.deadline_seconds)`. Compute effective `timeout_seconds` ceiling: if `self.spec.timeout_seconds` is set, further cap by `now + timeout_seconds`.

6. **Caller identity.** Resolved as: `caller_override` if provided, otherwise `CallerIdentity(subject=f"agent:{parent_name}")` where `parent_name` is derived from the `run_id` prefix embedded in `context.run_id` (the parent agent's run). *(The parent agent name is not directly available in `ToolContext`; it is passed via a reserved `"_parent_agent_name"` key in `context.context`, set by the parent `_Run` when building the `ToolContext` — see §5.)*

7. **Context for Agent B.** Build:
   ```python
   child_context = {
       **(context_selector(context) if context_selector else {}),
       "_agent_chain": (*_agent_chain, agent.name),
       "_parent_agent_name": agent.name,  # overwritten by AgentTool for next level
   }
   ```
   The `_agent_chain` always includes Agent B's name so Agent B cannot call itself.

8. **State for Agent B.** `child_state = state_selector(context) if state_selector else None`.

9. **Cancellation.** A new `CancellationToken` is created for Agent B. Agent A's cancellation token is polled in a task that cancels Agent B's token if Agent A is cancelled, ensuring cascading cancellation.

10. **Run Agent B.** Call `await agent.arun(input=arguments["input"], state=child_state, context=child_context, caller=caller, cancellation=child_cancellation)`. If `argument_schema` is non-default and the schema has fields beyond `"input"`, the full `arguments` dict is passed as a JSON-serialised string in `input` — the developer's `context_selector` / Agent B's instructions are responsible for interpreting it. *(This keeps `agent.arun`'s `input: str` contract unchanged.)*

11. **Handle result.**
    - If `result.status != "completed"` and `handoff=False`: return `ToolFailure` with code `"sub_agent_failed"`, `recoverable=True`, `outcome_unknown=result.error.outcome_unknown if result.error else False`.
    - If `result.status != "completed"` and `handoff=True`: same failure propagation.
    - If `result.status == "completed"`:
      - Compute `output_value = output_selector(result) if output_selector else result.output`.
      - If `handoff=False`: return `output_value` as the tool observation (the parent `_Run.execute_tools` path handles it as a normal `ToolObservation`).
      - If `handoff=True`: raise a special internal sentinel `_HandoffSignal(result)` that is caught in the parent `_Run.execute` loop (see §5.2).

### 4.4 Error codes

| Code | Meaning | Recoverable |
|------|---------|-------------|
| `agent_cycle_detected` | Agent name already in the call chain | `True` |
| `agent_depth_exceeded` | `max_agent_depth` reached | `True` |
| `agent_call_limit` | `max_agent_calls` exhausted on parent | `True` |
| `egress_policy_denied` | `egress_policy="deny_all_http"` blocks HTTP tool | `True` |
| `sub_agent_failed` | Agent B returned non-`completed` status | `True` |
| `sub_agent_cancelled` | Agent B was cancelled | `True` |

---

## 5. Parent agent changes (`src/loom/agent.py`)

### 5.1 `ToolContext` enrichment

In `_Run.__init__`, when building `ToolContext` for tool calls, two reserved keys are added to `context`:

```python
"_agent_chain": context.get("_agent_chain", ()),
"_parent_agent_name": agent.name,
```

These are passed through unmodified from whatever `context` the parent run received, extended by `AgentTool.execute` for each level.

### 5.2 `max_agent_calls` accounting

In `_Run.execute_tools`, after a tool observation is recorded as `execution_status="succeeded"` and the tool is an `AgentTool`:

```python
if isinstance(tool, AgentTool):
    self.usage(agent_calls=self.cp.usage.agent_calls + 1)
    if self.cp.usage.agent_calls > self.limits.max_agent_calls:
        raise ToolFailure("agent_call_limit")
```

This requires adding `agent_calls: int = 0` to the `Usage` record in `contracts.py`.

### 5.3 `handoff=True` signal handling

`AgentTool.execute` raises `_HandoffSignal(result: RunResult)` (a private exception defined in `tools/agent.py`) when `handoff=True` and Agent B completes successfully.

In `_Run.execute`, the tool-execution loop catches `_HandoffSignal`:

```python
except _HandoffSignal as signal:
    output_value = (
        signal.result.output
        if self.agent._agent_tool_output_selector(tool_name) is None
        else self.agent._agent_tool_output_selector(tool_name)(signal.result)
    )
    handoff_diagnostic = Diagnostic(
        code="agent_handoff",
        message=f"Run handed off to sub-agent; terminal result adopted.",
        origin="agent",
        details={"sub_agent_run_id": signal.result.run_id},
        severity="info",
    )
    return await self.finish("completed", output=output_value, extra_diagnostics=(handoff_diagnostic,))
```

Usage for the `handoff=True` case: the parent `_Run`'s usage is updated to include the sub-agent's `usage` token counts summed into its own.

**Implementation note:** `_HandoffSignal` must only be catchable inside `_Run.execute` and must not escape to `ToolRunner.call`. `ToolRunner.call` wraps tool execution in a broad `except Exception` clause — `_HandoffSignal` must be re-raised there so it propagates up to `_Run.execute`. This is achieved by catching it explicitly in `ToolRunner.call` and re-raising before the generic handler.

---

## 6. `Usage` record change (`src/loom/contracts.py`)

```python
class Usage(Record):
    # ... existing fields unchanged ...
    agent_calls: int = 0
```

`agent_calls` is included in `RunResult.usage` and propagated normally through all existing usage accounting paths (charge callbacks, event payloads, etc.).

---

## 7. Security constraints

### 7.1 Egress policy enforcement

When `egress_policy="deny_all_http"` is set on an `AgentTool`, the check in `AgentTool.execute` (step 4 of §4.3) prevents invocation entirely if Agent B has `HTTPTool` instances. This is a coarse deny — it does not inspect individual tool origins. The rationale is that Agent A's operator cannot grant Agent B new permissions, only restrict.

### 7.2 Document provenance isolation

Agent B's document access policies are entirely self-contained. Provenance accumulated by Agent B's tool calls is not merged into Agent A's `ConversationState`. Agent B's `RunResult.state` carries its own provenance; if the `state_selector` passes Agent B's state back on subsequent calls, provenance accumulates within Agent B's conversation. Agent A's provenance is never polluted by Agent B's document access.

### 7.3 Secrets isolation

Agent A's `ToolContext.context` may contain values (passed via `context_selector`) but the `AgentTool` does not forward Agent A's `ToolContext` directly. Only the values explicitly returned by `context_selector` reach Agent B. Reserved keys (`_agent_chain`, `_parent_agent_name`) are injected by the framework, not by user code.

### 7.4 Cancellation cascading

If Agent A is cancelled while Agent B is running, Agent B's `CancellationToken` is also cancelled. This prevents orphaned sub-agent runs consuming model/tool budget after the parent is abandoned.

### 7.5 Cycle detection prevents unbounded recursion

The `_agent_chain` check (§4.3 step 1) catches both direct self-recursion (`A → A`) and indirect cycles (`A → B → A`). The `max_agent_depth` check (§4.3 step 2) caps linear chains regardless of cycle detection. Together they bound the call tree.

---

## 8. `AgentTool` `ToolSpec.egress` derivation (detail)

At `AgentTool.__init__` time, inspect `agent.tools`:

```python
http_tools = [t for t in agent.tools if isinstance(t, HTTPTool)]
if not http_tools:
    egress = "none"
    egress_origins = ()
elif all(t.spec.egress == "fixed_network" for t in http_tools):
    egress = "fixed_network"
    egress_origins = tuple(sorted({o for t in http_tools for o in t.spec.egress_origins}))
else:
    egress = "unknown"
    egress_origins = ()
```

This is computed once at construction. If Agent B's tool list changes after `AgentTool` is constructed (which is impossible — `Agent.tools` is a `tuple` set at `Agent.__init__`), the derived value remains valid.

---

## 9. Public API

### 9.1 Export

`src/loom/__init__.py` additions:

```python
from loom.tools.agent import AgentTool

__all__ = [
    # ... existing exports ...
    "AgentTool",
]
```

### 9.2 Developer usage example — suspend mode (default)

```python
from loom import Agent, AgentTool

summariser = Agent(
    name="summariser",
    instructions="Summarise the text passed as input. Return a plain string summary.",
    tools=[],
)

summariser_tool = AgentTool(
    agent=summariser,
    name="summarise_text",
    description="Delegate text summarisation to the summariser sub-agent.",
    sensitive=False,
)

orchestrator = Agent(
    name="orchestrator",
    instructions="You are a research orchestrator. Use summarise_text to summarise passages.",
    tools=[summariser_tool],
)

result = orchestrator.run(input="Summarise the following: ...")
```

### 9.3 Developer usage example — handoff mode

```python
router_tool = AgentTool(
    agent=specialist_agent,
    name="route_to_specialist",
    description="Hand off to the specialist agent when the query requires deep domain expertise.",
    handoff=True,
    output_selector=lambda r: r.output,
)
```

### 9.4 Developer usage example — configurable caller and context

```python
AgentTool(
    agent=child_agent,
    name="child_call",
    description="Run child agent with caller forwarding.",
    caller_override=CallerIdentity(subject="human:alice", roles=("admin",)),
    context_selector=lambda ctx: {"tenant_id": ctx.context.get("tenant_id")},
    state_selector=lambda ctx: ctx.context.get("child_state"),  # type: ignore[return-value]
    output_selector=lambda r: {"status": r.status, "output": r.output},
)
```

---

## 10. `architecture.md` update

The following additions are required in `architecture.md`:

1. In the **Tools** table (§ "Tools" module walkthrough), add a row:

   | `agent.py` | `AgentTool` | Wraps an `Agent` as a callable tool; supports suspend and handoff modes |

2. In the **Data flow diagram**, add `AgentTool` under the `Tool execution` branch:

   ```
   ├──► Tool execution (tools/)
   │         ├── FunctionTool  (Python callable)
   │         ├── HTTPTool      (HTTPS fixed-origin)
   │         └── AgentTool     (sub-agent delegation)
   ```

3. In the **Key design constraints** list, add:

   > **Sub-agent depth bounded.** `max_agent_depth` (default 5) and cycle detection via `_agent_chain` in `ToolContext.context` prevent unbounded recursive agent invocation.

---

## 11. Implementation milestones

### M1 — Contracts and limits (no behaviour change)

**Files:** `src/loom/contracts.py`

- Add `max_agent_calls: int = Field(default=20, gt=0)` to `Limits`.
- Add `max_agent_depth: int = Field(default=5, gt=0)` to `Limits`.
- Add `agent_calls: int = 0` to `Usage`.

**Validation:** All existing unit tests pass. `Limits` and `Usage` model validation tests cover new fields.

---

### M2 — `AgentTool` core implementation

**Files:** `src/loom/tools/agent.py` (new file)

Implement:
- `_HandoffSignal` private sentinel exception.
- `AgentTool.__init__` with full parameter validation and `ToolSpec` construction (including egress derivation per §8).
- `AgentTool.execute` with the full sequence from §4.3:
  - Cycle detection
  - Depth check
  - Egress policy check
  - Deadline intersection
  - Caller resolution
  - Context and state assembly
  - Cascading cancellation setup
  - `agent.arun(...)` invocation
  - Result handling (suspend path only; handoff path in M3)

**Validation:** Unit tests covering:
- Cycle detection raises `ToolFailure("agent_cycle_detected")`.
- Depth exceeded raises `ToolFailure("agent_depth_exceeded")`.
- Egress policy `"deny_all_http"` raises `ToolFailure("egress_policy_denied")` when Agent B has `HTTPTool`.
- `ToolSpec.egress` is derived correctly for all three cases (`none`, `fixed_network`, `unknown`).
- Default `output_selector` returns `result.output`.
- Non-`completed` sub-agent result raises `ToolFailure("sub_agent_failed")`.
- Cancellation of parent cancels child token.

---

### M3 — Handoff mode

**Files:** `src/loom/tools/agent.py`, `src/loom/agent.py`

- Complete `AgentTool.execute` handoff path (raises `_HandoffSignal`).
- In `_Run.execute` (in `agent.py`), catch `_HandoffSignal` from tool execution loop.
- In `ToolRunner.call` (in `tools/runner.py`), explicitly re-raise `_HandoffSignal` before the generic `except Exception` handler.
- Produce `Diagnostic(code="agent_handoff", ...)` in the parent `RunResult`.
- Sum sub-agent `usage` into parent `usage` for the handoff case.

**Validation:** Unit tests covering:
- `handoff=True` returns `status="completed"` with `agent_handoff` diagnostic.
- Parent `RunResult.usage.agent_calls` is incremented.
- Summed usage tokens are correct.

---

### M4 — Parent agent integration

**Files:** `src/loom/agent.py`

- Inject `_agent_chain` and `_parent_agent_name` into `ToolContext.context` in `_Run.__init__` or at the point `ToolContext` is constructed per call in `execute_tools`.
- Add `max_agent_calls` budget check after a successful `AgentTool` observation (see §5.2).

**Validation:** Unit tests covering:
- `max_agent_calls` exhaustion raises `ToolFailure("agent_call_limit")`.
- `_agent_chain` is correctly threaded through two levels of nesting.
- `agent_calls` counter in `Usage` increments per successful sub-agent run.

---

### M5 — Export and documentation

**Files:** `src/loom/__init__.py`, `architecture.md`

- Add `AgentTool` to `__all__` and the import in `__init__.py`.
- Update `architecture.md` per §10.

**Validation:** `from loom import AgentTool` works. Import smoke test passes.

---

### M6 — Acceptance tests

**Files:** `tests/unit/` and/or `tests/acceptance/`

End-to-end acceptance tests (using the scripted provider from `testing.py`):

1. **Two-level suspend chain:** Orchestrator agent calls a summariser sub-agent via `AgentTool`. Verify orchestrator `RunResult` contains the summariser's output as the tool observation and `usage.agent_calls == 1`.

2. **Handoff:** Router agent invokes specialist via `AgentTool(handoff=True)`. Verify `RunResult.diagnostics` contains `agent_handoff`, output matches specialist output, and `usage.agent_calls == 1`.

3. **Cycle detection:** Agent A has an `AgentTool` wrapping Agent B; Agent B has an `AgentTool` wrapping Agent A. Verify `ToolFailure("agent_cycle_detected")` is raised and neither agent loops.

4. **Depth exceeded:** Five-level linear chain with `max_agent_depth=3`. Verify `ToolFailure("agent_depth_exceeded")` at the fourth level.

5. **Cancellation propagation:** Parent agent is cancelled mid sub-agent run. Verify sub-agent's `CancellationToken` is cancelled and the parent returns `status="cancelled"`.

6. **Egress policy:** Agent B has an `HTTPTool`. `AgentTool(egress_policy="deny_all_http")`. Verify `ToolFailure("egress_policy_denied")`.

---

## 12. Explicit exclusions

The following are **not** part of this specification and must not be added:

- `Tool.from_agent()` factory method — not consistent with `AgentTool`'s rich constructor.
- Workflow-level `AgentTool` registration — workflows already support `Agent` as step targets.
- Shared `ConversationState` between parent and child — each agent owns its own state.
- Parent-agent egress grants to sub-agent — the boundary is deny-only, never grant.
- Automatic secret forwarding from parent context to child — only explicit `context_selector` values reach Agent B.
- `AgentTool` inside a `Workflow` `Step.target` — steps target `Agent` or `Tool`; an `AgentTool` is a `Tool` and may be used by an Agent that is itself a step target, but the `Step` type does not need to change.
- CLI commands for multi-agent inspection — out of scope for this release.

---

## 13. References

- [`src/loom/agent.py`](../src/loom/agent.py) — `Agent`, `_Run`
- [`src/loom/contracts.py`](../src/loom/contracts.py) — `Limits`, `Usage`, `ToolContext`, `ToolSpec`, `RunResult`
- [`src/loom/tools/base.py`](../src/loom/tools/base.py) — `Tool` ABC, claim/phase/release lifecycle
- [`src/loom/tools/runner.py`](../src/loom/tools/runner.py) — `ToolRunner.call`
- [`src/loom/tools/http.py`](../src/loom/tools/http.py) — `HTTPTool` (structural reference for `AgentTool`)
- [`src/loom/tools/function.py`](../src/loom/tools/function.py) — `FunctionTool`
- [`src/loom/workflows.py`](../src/loom/workflows.py) — `Workflow`, `_WorkflowRun` (reference for child run handling)
- [`src/loom/storage/checkpoint.py`](../src/loom/storage/checkpoint.py) — `Checkpoint`, `Usage`
- [`architecture.md`](../architecture.md) — current architecture overview
