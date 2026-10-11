from loom import Agent, AgentTool, Tool

# ---------------------------------------------------------------------------
# Specialist sub-agent: summarises any text passed to it
# ---------------------------------------------------------------------------

summariser = Agent(
    name="summariser",
    instructions="""
    You are a concise summarisation assistant.
    Summarise the text given to you in at most three sentences.
    Return only the summary — no preamble, no commentary.
    """,
)

# Wrap the summariser as a tool the triage agent can call dynamically.
# handoff=False (default): the triage agent suspends, waits for the summary,
# then continues reasoning with the result as a normal tool observation.
summarise_tool = AgentTool(
    agent=summariser,
    name="summarise_text",
    description=(
        "Delegate text summarisation to the summariser sub-agent. "
        "Pass the full text as 'input'. Returns a short plain-text summary."
    ),
    sensitive=False,
    retry_safe=True,
)


# ---------------------------------------------------------------------------
# Utility tool attached to the triage agent
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


word_count_tool = Tool.from_callable(
    word_count,
    sensitive=False,
    retry_safe=True,
    egress="none",
)


# ---------------------------------------------------------------------------
# Orchestrator (triage) agent: decides what to do and delegates when needed
# ---------------------------------------------------------------------------

triage = Agent(
    name="triage",
    instructions="""
    You are a triage assistant. For every request:
    1. If the input is longer than 50 words, use summarise_text to produce
       a short summary and report both the word count and the summary.
    2. If the input is 50 words or fewer, just report the word count directly.
    Always use word_count to measure the length first.
    """,
    tools=[word_count_tool, summarise_tool],
)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

short_input = "The quick brown fox jumps over the lazy dog."

long_input = (
    "Artificial intelligence is transforming the way organisations operate. "
    "From automating repetitive tasks to generating novel insights from large "
    "datasets, AI systems are being deployed across industries ranging from "
    "healthcare to finance. As these systems grow more capable, questions around "
    "safety, transparency, and accountability are becoming increasingly important "
    "for practitioners and policy-makers alike."
)

print("=== Short input ===")
result = triage.run(input=short_input)
print(result.output)
print(f"agent_calls={result.usage.agent_calls}, tool_attempts={result.usage.tool_attempts}")

print()
print("=== Long input (triggers sub-agent delegation) ===")
result = triage.run(input=long_input)
print(result.output)
print(f"agent_calls={result.usage.agent_calls}, tool_attempts={result.usage.tool_attempts}")
