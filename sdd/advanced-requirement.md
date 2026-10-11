# Advanced feature requirements

## 1. Multi-agent / handoff support
The framework has single-agent and workflow orchestration, but no first-class way for one agent to hand off to another agent dynamically (as opposed to statically in a Workflow). An AgentTool — a tool whose callable is another Agent.run — would enable this pattern without a full framework redesign.

## 2. Streaming output
src/loom/providers/base.py appears to use a request/response model. Streaming (stream_complete) is now expected for chat-style UIs. This would require a streaming ModelAdapter variant and a corresponding Agent.stream() facade.

## 3. Structured tool-call retry with LLM feedback
When the LLM produces a malformed tool call, the current loop classifies it as a failure. A smarter loop would feed the validation error back to the model as an observation, letting the model self-correct (a pattern popularized by ReAct/Reflexion variants).

## 4. Memory / long-term context management
ConversationState carries the full message history. For long conversations this will breach the context window. A summarisation or sliding-window strategy (compress old turns before sending to the model) would handle production-length sessions gracefully.

## 5. Tool result schema validation
Tool input arguments are schema-validated via ToolSpec. Tool outputs are not. Adding an optional output_schema to ToolSpec would let the framework validate what comes back before the model sees it, catching bugs in tool implementations early.

## 6. A --dry-run / sandbox mode
A flag that lets developers run the full agentic loop with all tool calls intercepted and echoed (but not executed) would make local development and demos much faster — especially for sensitive or egress-restricted tools.




