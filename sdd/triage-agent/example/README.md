# Support-ticket triage example

This example uses Loom's `Agent`, `Tool`, and `Workflow` APIs to classify a
ticket, select a support team, and draft text **for human review only**.
It does not send responses, notify teams, create incidents, or connect to a
ticketing system. It requires the framework from [PR #2](https://github.com/SarikaSinha/loom/pull/2).

## Run

From the repository root, install the locked development environment with
Python 3.11, 3.12, or 3.13 (not 3.14):

```sh
uvx --from 'uv==0.13.0' uv sync --frozen --python 3.13
.venv/bin/python sdd/triage-agent/example/developer-code.py
```

Before the second command, explicitly provision the reference Ollama server
`0.40.2`, model `qwen2.5:7b`, exact model digest and context settings from
[`loom.toml`](loom.toml). Loom verifies the server/model contract; it does not
install or start Ollama or pull a model. A different client version alone does
not satisfy that contract.

Alternatively, replace the entire `[providers.primary]` table in a copy of
the configuration with a supported hosted provider, for example:

```toml
[providers.primary]
kind = "openai"
model = "YOUR_PROVISIONED_MODEL_ID"
api_key_env = "OPENAI_API_KEY"
native_tool_calling = true
structured_output = "json_schema"
```

Use a real model available to your account and export its API key outside the
repository. Do not retain Ollama-only fields in a hosted-provider table.
Hosted execution sends the ticket and generated triage to that provider, so
use only data you are authorized to share. See the [framework README](../../../README.md)
for other supported adapters.

The script defaults to the ticket and config beside itself, independent of the
working directory. Override either path explicitly:

```sh
.venv/bin/python sdd/triage-agent/example/developer-code.py \
  --ticket /path/to/ticket.json --config /path/to/loom.toml
```

Ticket input is a JSON object with only a nonblank `message` string, at most
12,000 characters. Success prints the output object as JSON to stdout and
exits 0. An unsuccessful run emits the `RunResult` error to stderr and exits
nonzero (130 for cancellation); malformed input exits 2. Configuration errors
also fail explicitly. Nothing overwrites the bundled sample files. Importing
`developer-code.py` does not read a ticket, build a workflow, or call a provider.

No document store, index, embedding download, external ticketing account, or
encryption key is needed. Keep storage and traces disabled as provided; all
state is in memory. The framework package includes embedding dependencies,
but this example never loads embedding models.

## Behavior

The `ticket_triage` Agent extracts a schema-validated `summary`, `category`,
and `priority`. The deterministic `assign_team` Python Tool uses this mapping:

| Category | Assigned team |
| --- | --- |
| `service_availability` | API Reliability |
| `billing` | Billing Support |
| `account_access` | Account Support |
| `how_to` | Developer Support |
| `other` | General Support |

A workflow `Branch` sends `critical` and `high` priorities to `urgent_draft`
for an internal escalation-note draft. `medium` and `low` use `standard_draft`
for a customer-response draft. Only the selected drafting Agent runs.
Production unavailability or blocked core business is critical; major
degradation/time-sensitive issues are high; other functional issues are medium;
routine questions are low. These are example policies, not a production SLA.

The output combines the validated extraction, deterministic team, and draft;
`next_action` is always the Python-assigned value `human_review`. Drafting
agents cannot override routing fields. No externally acting tool is attached,
and the routing tool declares `egress="none"`. Prompts treat ticket text as
untrusted and prohibit claims that an action was taken; human review is still
essential because model classification and prose can be wrong.

[`output.json`](output.json) is **illustrative**, not recorded live-provider
output or an acceptance result. It shows the intended result for the bundled
production-503 ticket. Actual wording and classification can vary. Invalid
structured output gets at most one correction before failing.

## Offline checks

From the repository root:

```sh
.venv/bin/pytest -q tests/unit/test_triage_example.py
.venv/bin/ruff check sdd/triage-agent/example/developer-code.py tests/unit/test_triage_example.py
.venv/bin/mypy sdd/triage-agent/example/developer-code.py
```

These tests import and execute the real example with `ScriptedModel` responses,
checking schema validation, every routing category, both branches, and failure
handling without model credentials, a running server, or embeddings. They do
not establish a live Ollama or hosted-provider pass.
