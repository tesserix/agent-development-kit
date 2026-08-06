"""The contract between a declared answer type and what a provider actually returned.

An agent that declares `output_type` is asking for an object, not prose about one. This
module holds the schema the model is told about, the hash that identifies it, and the
parse that either produces a validated instance or raises. Nothing here repairs, coerces
or partially accepts: a half-parsed answer handed on as if it were whole is the bug that
structured output exists to remove.

Unwrapping is explicit and narrow. A model that fences its JSON gets the fence stripped
and the strip recorded; a model that wraps its JSON in prose does not get scraped, because
a regex that finds an object inside a sentence also finds one inside a refusal.
"""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from tesserix_adk.core import STRICT_SUBSET, SchemaViolationError, schema_for, schema_hash

if TYPE_CHECKING:
    from pydantic import BaseModel

__all__ = ["OutputContract", "unwrap_fenced"]

_FENCE = "```"


def unwrap_fenced(content: str) -> tuple[str, bool]:
    r"""Return `content` with an enclosing code fence removed, and whether one was removed.

    The whole answer must be the fence. Prose either side of it means the model answered
    with something other than the object it was asked for, and that is a violation to
    report rather than a payload to go looking for.

    Args:
        content: What the provider returned.

    Returns:
        The unwrapped text and whether unwrapping happened.

    Example:
        >>> unwrap_fenced('```json\n{"a": 1}\n```')
        ('{"a": 1}', True)
    """
    text = content.strip()
    if not text.startswith(_FENCE) or not text.endswith(_FENCE) or len(text) <= 2 * len(_FENCE):
        return content, False
    body = text[len(_FENCE) : -len(_FENCE)]
    opening, _, remainder = body.partition("\n")
    if opening.strip() and not opening.strip().isalnum():
        return content, False
    return remainder.strip(), True


@dataclasses.dataclass(frozen=True)
class OutputContract:
    """What the model was asked for, and what will be accepted back.

    Args:
        output_type: The type the answer must validate against.
        schema: The schema derived from it, in the closed dialect every provider accepts.
        hash: Identity of that schema. It moves when the type moves, so a cached prompt,
            a replayed cassette and a recorded violation all name the same shape or
            visibly disagree.
        native: Whether the provider enforces the schema itself. When it does not, the
            schema goes into the prompt instead and the answer is validated the same way.
    """

    output_type: type[BaseModel]
    schema: dict[str, Any]
    hash: str
    native: bool

    @classmethod
    def of(cls, output_type: type[BaseModel], *, native: bool = True) -> OutputContract:
        """Build the contract for `output_type`.

        Args:
            output_type: The declared answer type.
            native: Whether the provider declared it enforces schemas itself.

        Returns:
            The contract.

        Raises:
            SchemaGenerationError: If the type cannot be described faithfully, which is a
                configuration failure raised where the agent is built.
        """
        schema = schema_for(output_type, dialect=STRICT_SUBSET)
        return cls(output_type=output_type, schema=schema, hash=schema_hash(schema), native=native)

    @property
    def instruction(self) -> str:
        """The prompt text used where the provider does not enforce the schema itself."""
        return (
            "Answer with one JSON object and nothing else: no prose, no code fence, no "
            "explanation. It must validate against this JSON Schema:\n"
            f"{json.dumps(self.schema, sort_keys=True, indent=2)}"
        )

    def parse(self, content: str) -> BaseModel:
        """Return `content` as a validated instance of the declared type.

        Args:
            content: The answer, already unwrapped.

        Returns:
            The validated instance.

        Raises:
            SchemaViolationError: If the content is not JSON, or is JSON that the type
                refuses. Carries the raw output, every failing path and the schema hash.
        """
        try:
            payload = json.loads(content)
        except ValueError as broken:
            raise self._violation(f"output is not valid JSON: {broken}", content, ()) from broken
        try:
            return self.output_type.model_validate(payload)
        except ValidationError as refused:
            problems = {_path(error["loc"]): str(error["msg"]) for error in refused.errors()}
            raise self._violation(
                f"output did not satisfy {self.output_type.__name__}",
                content,
                tuple(sorted(problems)),
                problems=problems,
            ) from refused

    def _violation(
        self,
        message: str,
        payload: str,
        paths: tuple[str, ...],
        *,
        problems: dict[str, str] | None = None,
    ) -> SchemaViolationError:
        return SchemaViolationError(
            message,
            model=self.output_type.__name__,
            paths=paths,
            problems=problems,
            payload=payload,
            details={"schema_hash": self.hash},
        )


def _path(location: tuple[int | str, ...]) -> str:
    return ".".join(str(part) for part in location)
