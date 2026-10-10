# Basic Requirement

## Context 
We are creating a lightweight, modern agentic framework with following capabilities and information-

### Agent Definition - 
System prompt
Model selection
Tool attachment
Structured outputs

### Tool Calling
Function tools
API tools
Custom tools
Tool validation

### Conversation Memory:
Messages
Tool results
Agent state

### Workflow Execution
Sequential execution
Branching
Retry

### LLM Provider support
OpenAI
Azure OpenAI
Anthropic
Gemini
Ollama

### Observability
Tracing for 
User Input
 ↓
Agent Thought
 ↓
Tool Calls
 ↓
Results
 ↓
Output

## Feature guidance
1. Agent definition - Instructions, model configuration, available tools, optional output schema
2. Model adapter interface
3. Agent execution loop - Call model → validate requested tools → execute → return observations → repeat or finish
4. Typed tool interface - Name, description, argument schema, callable, validation, and structured errors
5. Explicit run state - Conversation history, tool-call IDs/results, run ID, and application context
6. Output contract - Support text or schema-validated output; distinguish completion, failure, cancellation, and limit exhaustion
7. Bounded execution - Maximum steps/tool calls, timeouts, cancellation, and usage accounting
8. Error handling and retries - Classify failures; bounded retries for transient errors; surface exhausted failures
9. Permission boundary - Explicit tool allowlist and an approval callback for sensitive actions
10. Events and debugging - Model/tool start/end events, errors, latency, and usage; configurable payload redaction
11. Testability - Fake model adapter, replaceable tools, and deterministic loop tests

## Goal
Create a plan document named loom/sdd/basic-plan.md which will detail the steps to be carried out to create the agentic framework

## Workflow
Solution is to provide a template, so that user provides the list of API changes(add/modify/delete) which is in the scope of current review.
This API review tool should now take this file from user as an input and review the APIs only which was provided in the scope.


## Stop Condition
Interview me relentlessly about every aspect of this plan until we reach a shared understanding.
