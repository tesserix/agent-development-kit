"""What the model is holding, and what leaves when there is no more room.

Four scenarios: a chunk retrieved twice and sent once; eviction order under pressure; the
prefix refusing to be trimmed; and handing what survived to `assemble_prompt`.

Run it with `python examples/context_window.py`. Nothing here reaches the network.
"""

from __future__ import annotations

from tesserix_adk.core import Agent, NoOutput
from tesserix_adk.core.errors import ContextWindowExceededError
from tesserix_adk.runtime import ContextWindow, PromptLayer, Segment, assemble_prompt

AGENT: Agent[NoOutput] = Agent(
    name="clerk",
    instructions="Answer from the file. Cite the page.",
    model="llama-3.1-8b-instruct",
    free_text=True,
)

PAGE_12 = "page 12: the lease was signed on a Tuesday"


def words(text: str) -> int:
    """A word count stands in for the server's tokenizer, so the numbers below are readable."""
    return len(text.split())


def retrieved_twice() -> None:
    """Re-injecting a chunk the model already has is prefill spent on nothing."""
    window = ContextWindow(limit_tokens=100, tokenizer=words)
    first = window.admit(Segment(text=PAGE_12, layer=PromptLayer.RETRIEVED, key="p12"))
    again = window.admit(Segment(text=PAGE_12, layer=PromptLayer.RETRIEVED, key="p12"))
    print(f"first {first}, again {again}, held {len(window.segments)}")  # noqa: T201


def under_pressure() -> None:
    """Conversation goes oldest-first, then retrieval goes lowest-scored-first."""
    window = ContextWindow(limit_tokens=3, tokenizer=words)
    window.admit(Segment(text="relevant chunk", layer=PromptLayer.RETRIEVED, key="a", score=0.9))
    window.admit(Segment(text="marginal chunk", layer=PromptLayer.RETRIEVED, key="b", score=0.1))
    window.admit(Segment(text="two turns", layer=PromptLayer.CONVERSATION))
    window.admit(Segment(text="ago now", layer=PromptLayer.CONVERSATION))
    evicted = window.fit()
    print(  # noqa: T201
        f"evicted {[segment.text for segment in evicted]},",
        f"kept {[segment.text for segment in window.segments]},",
        f"and 'b' readmissible: {not window.holds('b')}",
    )


def the_prefix_is_not_negotiable() -> None:
    """Trimming the prefix refills every cache downstream; refusing says so out loud."""
    window = ContextWindow(limit_tokens=3, tokenizer=words)
    window.admit(Segment(text="the case file runs to four hundred pages", layer=PromptLayer.PINNED))
    try:
        window.fit()
    except ContextWindowExceededError as refused:
        print(f"refused: {refused.counted} tokens against {refused.limit}")  # noqa: T201


def into_the_prompt() -> None:
    """What survived is the shape `assemble_prompt` takes."""
    window = ContextWindow(limit_tokens=100, tokenizer=words)
    window.admit(Segment(text=PAGE_12, layer=PromptLayer.RETRIEVED, key="p12"))
    prompt = assemble_prompt(
        AGENT,
        "what does page 12 say?",
        retrieved=window.texts(PromptLayer.RETRIEVED),
        tokenizer=words,
    )
    layers = [layer.value for layer in prompt.layers]
    print(f"{window.tokens} tokens held, prompt layers {layers}")  # noqa: T201


def main() -> None:
    """Run every scenario in order."""
    retrieved_twice()
    under_pressure()
    the_prefix_is_not_negotiable()
    into_the_prompt()


if __name__ == "__main__":
    main()
