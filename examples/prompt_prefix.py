"""The cacheable prefix, which on CPU is the difference between usable and unusable.

Four scenarios: the layers a prompt is assembled from; a second turn that leaves the
prefix untouched; tool declarations reordered without cost, and a toolset that genuinely
changed; and counting the prefix with the server's own tokenizer instead of the estimate.

Run it with `python examples/prompt_prefix.py`. Nothing here reaches the network.
"""

from __future__ import annotations

from tesserix_adk.core import Agent, Message, NoOutput, TextPart
from tesserix_adk.runtime import (
    PROMPT_LAYERS,
    ToolDeclaration,
    approximate_tokens,
    assemble_prompt,
)

AGENT: Agent[NoOutput] = Agent(
    name="clerk",
    instructions="Answer from the file. Cite the page.",
    model="llama-3.1-8b-instruct",
    free_text=True,
)

SEARCH = ToolDeclaration(
    name="search",
    description="Search the file.",
    parameters={"type": "object", "properties": {"q": {"type": "string"}}},
)
BOOK = SEARCH.model_copy(update={"name": "book", "description": "Book a slot."})

FILE = ["the file runs to 400 pages and does not change between questions"]


def the_layers() -> None:
    """One documented order, and a label on every message saying which layer it came from."""
    prompt = assemble_prompt(
        AGENT,
        "what does page 12 say?",
        pinned=FILE,
        retrieved=["page 12: the lease was signed on a Tuesday"],
        history=[Message(role="user", content=[TextPart(text="who signed it?")])],
        tools=[SEARCH],
    )
    print("documented:", " > ".join(layer.value for layer in PROMPT_LAYERS))  # noqa: T201
    print("assembled: ", " > ".join(layer.value for layer in prompt.layers))  # noqa: T201
    print(f"prefix is {len(prompt.prefix)} of {len(prompt.messages)} messages")  # noqa: T201


def a_second_turn() -> None:
    """Prefill is skipped on every turn after the first, or it is paid for again."""
    first = assemble_prompt(AGENT, "who signed it?", pinned=FILE, tools=[SEARCH])
    second = assemble_prompt(
        AGENT,
        "and on what date?",
        pinned=FILE,
        tools=[SEARCH],
        history=[Message(role="user", content=[TextPart(text="who signed it?")])],
        retrieved=["page 12: the lease was signed on a Tuesday"],
    )
    print(  # noqa: T201
        f"turn 1 {first.fingerprint}, turn 2 {second.fingerprint}:",
        "cache hits" if first.fingerprint == second.fingerprint else "REFILL",
    )


def tools_in_whatever_order() -> None:
    """A registry iterating a dict differently must not cost a refill; a new tool must."""
    one_way = assemble_prompt(AGENT, "hi", tools=[SEARCH, BOOK])
    other_way = assemble_prompt(AGENT, "hi", tools=[BOOK, SEARCH])
    fewer = assemble_prompt(AGENT, "hi", tools=[SEARCH])
    print(  # noqa: T201
        f"sorted to {[tool.name for tool in one_way.tools]};",
        f"reordered: {'same' if one_way.fingerprint == other_way.fingerprint else 'REFILL'};",
        f"tool removed: {'same' if one_way.fingerprint == fewer.fingerprint else 'refill'}",
    )


def counting_it() -> None:
    """The estimate is for a log line. A context-window check wants the real tokenizer."""
    estimated = assemble_prompt(AGENT, "hi", pinned=FILE, tools=[SEARCH])
    counted = assemble_prompt(
        AGENT, "hi", pinned=FILE, tools=[SEARCH], tokenizer=lambda text: len(text.split())
    )
    print(  # noqa: T201
        f"four-characters-a-token says {estimated.prefix_tokens},",
        f"a word count says {counted.prefix_tokens},",
        f"and 'a' * 40 is {approximate_tokens('a' * 40)} either way",
    )


def main() -> None:
    """Run every scenario in order."""
    the_layers()
    a_second_turn()
    tools_in_whatever_order()
    counting_it()


if __name__ == "__main__":
    main()
