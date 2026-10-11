from loom import Agent, Tool


def word_count(text: str) -> int:
    """Count the number of words in a text string."""
    return len(text.split())


def char_count(text: str, include_spaces: bool = True) -> int:
    """Count the number of characters in a text string.

    Set include_spaces to False to exclude whitespace characters.
    """
    if include_spaces:
        return len(text)
    return len(text.replace(" ", ""))


def reverse_text(text: str) -> str:
    """Reverse the characters in a text string."""
    return text[::-1]


word_count_tool = Tool.from_callable(
    word_count,
    sensitive=False,
    retry_safe=True,
    egress="none",
)

char_count_tool = Tool.from_callable(
    char_count,
    sensitive=False,
    retry_safe=True,
    egress="none",
)

reverse_tool = Tool.from_callable(
    reverse_text,
    sensitive=False,
    retry_safe=True,
    egress="none",
)

agent = Agent(
    name="text_analyst",
    instructions="""
    You are a text analysis assistant. Use the available tools to answer
    questions about text. Always show exact numbers from the tools.
    """,
    tools=[word_count_tool, char_count_tool, reverse_tool],
)

result = agent.run(
    input="How many words and characters are in 'The quick brown fox'? Also reverse it."
)

print(result)
