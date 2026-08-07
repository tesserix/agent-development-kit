"""Deterministic prompt assembly.

The order is fixed and stated here, not left to whichever call site got there first:
system, tools, pinned context, retrieved context, conversation. Tool declarations are
sorted by name, because they are part of the cacheable prefix and a registry that
iterates a dict in a different order must not cost a refill.

The prefix — everything up to and including the pinned layer — is byte-stable across the
turns of a conversation, and `Prompt.fingerprint` is its identity. On CPU inference that
is not a micro-optimisation: prefill dominates, and a prefix that shifts by one byte is
tens of seconds spent re-evaluating what the server already had.

Content the agent did not author — recalled memory, retrieved documents, tool results —
is wrapped as data. A model cannot be relied on to ignore an instruction handed to it as
prose, so it is never handed one.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field

from tesserix_adk.core import Message, TextPart
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.provider import ToolDeclaration

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tesserix_adk.core import Agent
    from tesserix_adk.runtime.structured import OutputContract

__all__ = [
    "PROMPT_LAYERS",
    "Prompt",
    "PromptLayer",
    "Tokenizer",
    "ToolDeclaration",
    "approximate_tokens",
    "assemble_prompt",
    "wrap_untrusted",
]


_BEGIN = "<untrusted-data"
_END = "</untrusted-data>"
_ESCAPED = {_BEGIN: "&lt;untrusted-data", _END: "&lt;/untrusted-data&gt;"}
_SOURCE = re.compile(r"^[a-z0-9_-]+$")
_VERSION_LENGTH = 12

# Roughly four characters to a token across English prose and JSON alike. Wrong by up to a
# fifth on either, which is why it is an estimate with a name that says so.
_CHARACTERS_PER_TOKEN = 4


class PromptLayer(StrEnum):
    """A band of the prompt, in the order the kit assembles them.

    `TOOLS` labels the declarations rather than a message: they travel in their own field
    and each provider renders them where its wire format puts them. They sit in the prefix
    all the same, which is why they are sorted and fingerprinted with it.
    """

    SYSTEM = "system"
    TOOLS = "tools"
    PINNED = "pinned"
    RETRIEVED = "retrieved"
    CONVERSATION = "conversation"


PROMPT_LAYERS = (
    PromptLayer.SYSTEM,
    PromptLayer.TOOLS,
    PromptLayer.PINNED,
    PromptLayer.RETRIEVED,
    PromptLayer.CONVERSATION,
)

# Everything from here up is assembled fresh each turn, so the prefix ends before it.
_FIRST_VARIABLE_LAYER = PromptLayer.RETRIEVED


class Tokenizer(Protocol):
    """Counts the tokens in a string, by whatever rule the model behind it uses."""

    def __call__(self, text: str) -> int:
        """Return how many tokens `text` occupies."""
        ...


def approximate_tokens(text: str) -> int:
    """Estimate the tokens in `text` at four characters each.

    The default, because it needs no model and no download. It is an estimate: good enough
    for a log line or a cache-hit ratio, and wrong for a context-window check, where the
    server's own tokenizer should be passed to `assemble_prompt` instead.

    Example:
        >>> approximate_tokens("a" * 40)
        10
    """
    return len(text) // _CHARACTERS_PER_TOKEN


def wrap_untrusted(content: str, *, source: str) -> str:
    """Return `content` marked as data the model must not take instruction from.

    Args:
        content: The untrusted text.
        source: Where it came from, e.g. `memory` or `tool_result`. "Untrusted" alone is
            not actionable; naming the origin is.

    Raises:
        ValueError: If `source` is not lowercase alphanumeric with `_` or `-`, which
            would let it break out of the marker.

    Example:
        >>> wrap_untrusted("3 rows", source="tool_result").splitlines()[0]
        '<untrusted-data source="tool_result">'
    """
    if not _SOURCE.match(source):
        raise ValueError(f"source must match {_SOURCE.pattern}, got {source!r}")
    safe = content
    for marker, escaped in _ESCAPED.items():
        safe = safe.replace(marker, escaped)
    return f'{_BEGIN} source="{source}">\n{safe}\n{_END}'


class Prompt(AdkModel):
    """What is actually sent for one turn, plus the identity of its cacheable prefix.

    Args:
        messages: The conversation in assembly order.
        tools: The tools the model is told about, sorted by name.
        layers: Which layer each message belongs to, parallel to `messages`.
        version: A short digest of the prompt's design — instructions, tool declarations
            and the output contract. Two runs sharing a version were shaped by the same
            prompt, whatever was asked of it, which is what makes a regression
            attributable.
        fingerprint: A short digest of the prefix as it goes on the wire, pinned context
            included. Equal fingerprints on two turns mean the server can reuse its
            evaluated prefix; unequal ones mean prefill runs again.
        prefix_tokens: How much of the prompt the fingerprint covers, as counted by the
            tokenizer assembly was given.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolDeclaration, ...] = ()
    layers: tuple[PromptLayer, ...] = ()
    version: str = Field(min_length=1)
    fingerprint: str = Field(min_length=1)
    prefix_tokens: int = Field(ge=0)

    @property
    def prefix(self) -> tuple[Message, ...]:
        """The messages the fingerprint covers — the part a cache can hold between turns."""
        return tuple(
            message
            for message, layer in zip(self.messages, self.layers, strict=True)
            if PROMPT_LAYERS.index(layer) < PROMPT_LAYERS.index(_FIRST_VARIABLE_LAYER)
        )


def _digest(
    agent: Agent[Any], tools: Sequence[ToolDeclaration], output: OutputContract | None
) -> str:
    prefix = json.dumps(
        {
            "instructions": agent.instructions,
            "tools": [tool.model_dump(mode="json") for tool in tools],
            "output": None if output is None else [output.hash, output.native],
        },
        sort_keys=True,
    )
    return hashlib.sha256(prefix.encode()).hexdigest()[:_VERSION_LENGTH]


def _fingerprint(prefix: Sequence[Message], tools: Sequence[ToolDeclaration]) -> str:
    """Digest the prefix as bytes, because bytes are what the server's cache compares."""
    rendered = json.dumps(
        {
            "messages": [message.model_dump(mode="json") for message in prefix],
            "tools": [tool.model_dump(mode="json") for tool in tools],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode()).hexdigest()[:_VERSION_LENGTH]


def _declared(tools: Iterable[ToolDeclaration]) -> tuple[ToolDeclaration, ...]:
    sorted_tools = tuple(sorted(tools, key=lambda tool: tool.name))
    names = [tool.name for tool in sorted_tools]
    duplicated = {name for name in names if names.count(name) > 1}
    if duplicated:
        raise ValueError(
            f"tool names must be unique; got {', '.join(sorted(duplicated))} more than once"
        )
    return sorted_tools


def assemble_prompt(
    agent: Agent[Any],
    user_input: str,
    *,
    history: Iterable[Message] = (),
    pinned: Iterable[str] = (),
    retrieved: Iterable[str] = (),
    tools: Iterable[ToolDeclaration] = (),
    output: OutputContract | None = None,
    tokenizer: Tokenizer | None = None,
) -> Prompt:
    """Compose one turn's prompt in the documented order.

    Args:
        agent: The declaration whose instructions open the prompt.
        user_input: What is being asked this turn.
        history: The conversation so far, in order.
        pinned: Context that holds for the whole conversation — a case file, a style
            guide, a schema. Part of the cacheable prefix, and wrapped as untrusted data.
        retrieved: Context fetched for this turn — recalled memory, retrieved documents.
            Outside the prefix, because content that changes per turn would invalidate it
            per turn. Wrapped as untrusted data.
        tools: Tool declarations. Sorted by name here, so the order a registry happens to
            iterate in cannot cost a prefix refill.
        output: The answer's declared shape. Where the provider does not enforce a schema
            itself, it is stated in the prompt instead; either way it is part of the
            version, because a changed schema is a changed prompt.
        tokenizer: How to count `prefix_tokens`. `None` uses `approximate_tokens`; pass
            the server's own where the count has to be right.

    Returns:
        The assembled `Prompt`.

    Raises:
        ValueError: If `user_input` is blank — a run with nothing asked of it has no
            terminal state that means anything — or if two tools share a name, which
            sorting would hide and the model could not disambiguate.

    Example:
        >>> from tesserix_adk.core import Agent
        >>> agent = Agent(
        ...     name="planner",
        ...     instructions="Plan trips.",
        ...     model="claude-sonnet-5",
        ...     free_text=True,
        ... )
        >>> [m.role for m in assemble_prompt(agent, "plan a trip").messages]
        ['system', 'user']
    """
    if not user_input.strip():
        raise ValueError("user_input is empty; a run with nothing asked of it cannot finish")

    declared = _declared(tools)
    layered: list[tuple[PromptLayer, Message]] = [
        (
            PromptLayer.SYSTEM,
            Message(role="system", content=[TextPart(text=agent.instructions)]),
        )
    ]
    if output is not None and not output.native:
        layered.append(
            (
                PromptLayer.SYSTEM,
                Message(role="system", content=[TextPart(text=output.instruction)]),
            )
        )
    for layer, content in ((PromptLayer.PINNED, pinned), (PromptLayer.RETRIEVED, retrieved)):
        block = tuple(content)
        if block:
            wrapped = wrap_untrusted("\n".join(block), source=layer.value)
            layered.append((layer, Message(role="system", content=[TextPart(text=wrapped)])))
    layered.extend((PromptLayer.CONVERSATION, message) for message in history)
    layered.append(
        (
            PromptLayer.CONVERSATION,
            Message(role="user", content=[TextPart(text=user_input)]),
        )
    )

    messages = tuple(message for _, message in layered)
    layers = tuple(layer for layer, _ in layered)
    variable = PROMPT_LAYERS.index(_FIRST_VARIABLE_LAYER)
    prefix = tuple(message for (layer, message) in layered if PROMPT_LAYERS.index(layer) < variable)

    return Prompt(
        messages=messages,
        tools=declared,
        layers=layers,
        version=_digest(agent, declared, output),
        fingerprint=_fingerprint(prefix, declared),
        prefix_tokens=_prefix_tokens(prefix, declared, tokenizer or approximate_tokens),
    )


def _prefix_tokens(
    prefix: Sequence[Message], tools: Sequence[ToolDeclaration], tokenizer: Tokenizer
) -> int:
    text = "".join(
        part.text for message in prefix for part in message.content if isinstance(part, TextPart)
    )
    declarations = "".join(
        json.dumps(tool.model_dump(mode="json"), sort_keys=True) for tool in tools
    )
    return tokenizer(text + declarations)
