"""What makes two runs the same run, expressed as something a test can compare.

A run is reproducible only if the thing it asked is reproducible. The fingerprint below
canonicalises everything that shapes a provider call — the prompt, the tool schemas the
model was told about, the model itself, its output schema and the hook chain that could
rewrite any of them — so that "the same inputs" is a claim with a digest behind it rather
than a hope.

It names *which* field diverged, because a replay that fails with "cassette miss" and
nothing else sends the reader to diff two blobs by eye.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import Field

from tesserix_adk.core.models import AdkModel

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tesserix_adk.core import Message
    from tesserix_adk.runtime.loop import ModelRequest
    from tesserix_adk.runtime.prompt import ToolDeclaration

__all__ = ["RunFingerprint", "canonical_digest", "fingerprint_of"]


_FIELDS = ("hooks", "messages", "model", "output_schema", "tools")


class RunFingerprint(AdkModel):
    """A canonical summary of one provider call, field by field.

    Each field is a digest rather than the content it summarises: a fingerprint travels
    with a cassette, and a cassette is a file people commit and read.

    Args:
        model: The model identifier, plain — it is not a secret and naming it is the point.
        messages: Digest of the assembled prompt.
        tools: Digest of the tool declarations the model was told about.
        output_schema: Digest of the declared output schema, or `"none"`.
        hooks: Digest of the hook chain's names, in order.

    Example:
        >>> RunFingerprint(model="m", messages="a", tools="b", output_schema="c", hooks="d").diff(
        ...     RunFingerprint(model="m", messages="a", tools="b", output_schema="c", hooks="e")
        ... )
        ('hooks',)
    """

    model: str = Field(min_length=1)
    messages: str = Field(min_length=1)
    tools: str = Field(min_length=1)
    output_schema: str = Field(min_length=1)
    hooks: str = Field(min_length=1)

    @property
    def digest(self) -> str:
        """One digest over every field, for keying a recorded interaction."""
        return canonical_digest({field: getattr(self, field) for field in _FIELDS})

    def diff(self, other: RunFingerprint) -> tuple[str, ...]:
        """Name every field on which this fingerprint and `other` disagree, in order."""
        return tuple(field for field in _FIELDS if getattr(self, field) != getattr(other, field))


def fingerprint_of(request: ModelRequest, *, hooks: Iterable[str] = ()) -> RunFingerprint:
    """Fingerprint one provider call, plus the hook chain that could have shaped it.

    Args:
        request: The call about to go out.
        hooks: The hook names in the chain, in declaration order. Order matters: rewrites
            chain, so the same hooks in another order can assemble another prompt.
    """
    return RunFingerprint(
        model=request.model,
        messages=canonical_digest([_message(message) for message in request.messages]),
        tools=canonical_digest([_tool(tool) for tool in request.tools]),
        output_schema=canonical_digest(request.output_schema) if request.output_schema else "none",
        hooks=canonical_digest(list(hooks)),
    )


def canonical_digest(value: Any) -> str:
    """SHA-256 over `value` serialised key-order independently.

    Two dicts that differ only in iteration order are one value; anything JSON cannot
    represent is rendered by `repr` rather than raising, because a digest that refuses to
    be taken is a run that cannot be recorded at all.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=repr, separators=(",", ":")).encode()
    ).hexdigest()


def _message(message: Message) -> dict[str, Any]:
    return {
        "role": message.role,
        "tool_call_id": message.tool_call_id,
        "content": [part.model_dump(mode="json") for part in message.content],
    }


def _tool(tool: ToolDeclaration) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": _ordered(tool.parameters),
    }


def _ordered(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _ordered(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_ordered(item) for item in value]
    return value
