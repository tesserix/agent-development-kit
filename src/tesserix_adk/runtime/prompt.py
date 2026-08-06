"""Deterministic prompt assembly.

The order is fixed and stated here, not left to whichever call site got there first:
instructions, then memory, then history, then the new input. Tool declarations keep the
order the registry gave them, because they are part of the cacheable prefix and
reordering them refills it.

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
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tesserix_adk.core import Message, TextPart
from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from tesserix_adk.core import Agent
    from tesserix_adk.runtime.structured import OutputContract

__all__ = ["Prompt", "ToolDeclaration", "assemble_prompt", "wrap_untrusted"]


_BEGIN = "<untrusted-data"
_END = "</untrusted-data>"
_ESCAPED = {_BEGIN: "&lt;untrusted-data", _END: "&lt;/untrusted-data&gt;"}
_SOURCE = re.compile(r"^[a-z0-9_-]+$")
_VERSION_LENGTH = 12


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


class ToolDeclaration(AdkModel):
    """A tool as the model is told about it.

    Provisional and owned by the runtime until the Tools epic lands a registry type
    (#49). It carries the JSON Schema as data so that a declaration can be hashed into
    the prompt version and diffed in review.
    """

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class Prompt(AdkModel):
    """What is actually sent for one turn, plus the identity of its cacheable prefix.

    Args:
        messages: The conversation in assembly order.
        tools: The tools the model is told about, in registry order.
        version: A short digest of the prefix — instructions and tool declarations. Two
            runs sharing a version were shaped by the same prompt, whatever was asked of
            it, which is what makes a regression attributable.
    """

    messages: tuple[Message, ...]
    tools: tuple[ToolDeclaration, ...] = ()
    version: str = Field(min_length=1)


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


def assemble_prompt(
    agent: Agent[Any],
    user_input: str,
    *,
    history: Iterable[Message] = (),
    memory: Iterable[str] = (),
    tools: Iterable[ToolDeclaration] = (),
    output: OutputContract | None = None,
) -> Prompt:
    """Compose one turn's prompt in the documented order.

    Args:
        agent: The declaration whose instructions open the prompt.
        user_input: What is being asked this turn.
        history: The conversation so far, in order.
        memory: Recalled text, wrapped as untrusted data.
        tools: Tool declarations, in the registry's order.
        output: The answer's declared shape. Where the provider does not enforce a schema
            itself, it is stated in the prompt instead; either way it is part of the
            version, because a changed schema is a changed prompt.

    Returns:
        The assembled `Prompt`.

    Raises:
        ValueError: If `user_input` is blank — a run with nothing asked of it has no
            terminal state that means anything.

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

    declared = tuple(tools)
    messages: list[Message] = [Message(role="system", content=[TextPart(text=agent.instructions)])]
    if output is not None and not output.native:
        messages.append(Message(role="system", content=[TextPart(text=output.instruction)]))
    recalled = tuple(memory)
    if recalled:
        messages.append(
            Message(
                role="system",
                content=[
                    TextPart(text=wrap_untrusted("\n".join(recalled), source="memory")),
                ],
            )
        )
    messages.extend(history)
    messages.append(Message(role="user", content=[TextPart(text=user_input)]))

    return Prompt(
        messages=tuple(messages), tools=declared, version=_digest(agent, declared, output)
    )
