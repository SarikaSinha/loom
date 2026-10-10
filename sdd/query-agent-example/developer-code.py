from loom import Agent, Tool, DocumentStore

kb = DocumentStore(
    path="./product-docs"
)

agent = Agent(
    name="product_expert",
    instructions="""
    Answer questions using the provided documents.
    Cite sources for every answer.
    """,
    data=[kb]
)

result = agent.run(
    input="What are the rate limits for the API?"
)

print(result)
