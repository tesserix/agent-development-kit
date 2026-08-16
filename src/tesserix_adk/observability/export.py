"""Redaction in the export path, so attaching an attribute is never a data-leak decision.

The most useful debugging attributes — prompt text, tool arguments, retrieved snippets —
are exactly the ones most likely to hold a card number or a bearer token, and a trace store
is a queryable place with wide read access and long retention. So nothing is trusted to
have been careful upstream: every attribute, every span event and every exception message
is rewritten here, on the way out, whatever attached it.

The processor covers what the kit exports. Third-party instrumentation emitting into the
same process is outside it — that path needs its own processor or its own denylist.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import Field, SecretStr

from tesserix_adk.core import MASK, AdkModel, scrub
from tesserix_adk.observability.attribution import ATTRIBUTE_PREFIX
from tesserix_adk.observability.redaction import Redaction

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "DENIED_KEYS",
    "PAYLOAD_ATTRIBUTES",
    "REFERENCE_SUFFIX",
    "ExportedEvent",
    "ExportedSpan",
    "PendingSpan",
    "RedactingSpanProcessor",
    "RedactionPolicy",
    "RedactionStats",
    "SpanEvent",
]

DENIED_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "private_key",
        "secret",
        "session_id",
        "set_cookie",
        "token",
    }
)
"""Key names a credential travels under. Dropped whatever the value happens to look like."""

PAYLOAD_ATTRIBUTES: frozenset[str] = frozenset(
    {
        f"{ATTRIBUTE_PREFIX}completion",
        f"{ATTRIBUTE_PREFIX}memory.content",
        f"{ATTRIBUTE_PREFIX}prompt",
        f"{ATTRIBUTE_PREFIX}retrieved.text",
        f"{ATTRIBUTE_PREFIX}tool.arguments",
        f"{ATTRIBUTE_PREFIX}tool.result",
    }
)
"""Model-payload attributes. Off unless a deployment allowlists them, and redacted even so."""

REFERENCE_SUFFIX = ".ref"
"""Suffix of the content reference left where a payload was dropped."""


class RedactionPolicy(AdkModel):
    """What a deployment is willing to export.

    Args:
        payload_attributes: Which of `PAYLOAD_ATTRIBUTES` may travel at all. Empty by
            default, because a payload captured for one investigation outlives it.
        denied_keys: Key names dropped outright, matched on the last dotted segment.
        extra_patterns: Shapes this deployment knows about, such as a case reference.
        max_length: Longest value scanned and exported. Applied before scanning, so a
            large payload cannot cost more than a bounded scan.
        content_references: Whether a dropped payload leaves a digest to correlate on.
    """

    payload_attributes: frozenset[str] = frozenset()
    denied_keys: frozenset[str] = DENIED_KEYS
    extra_patterns: tuple[str, ...] = ()
    max_length: int = Field(default=4096, ge=16)
    content_references: bool = True


class SpanEvent(AdkModel):
    """An event recorded on a span before redaction.

    Args:
        name: What happened.
        attributes: Whatever was attached to it.
    """

    name: str
    attributes: Mapping[str, object] = {}


class ExportedEvent(AdkModel):
    """A span event as an exporter receives it.

    Args:
        name: What happened.
        attributes: The rewritten attributes.
    """

    name: str
    attributes: Mapping[str, str] = {}


class PendingSpan(AdkModel):
    """A span about to be exported, with whatever a caller attached to it.

    Args:
        name: The span name.
        attributes: Attributes of any type, including `SecretStr`.
        events: Events recorded on the span.
        exception: The exception text, whose frames carry argument values.
    """

    name: str
    attributes: Mapping[str, object] = {}
    events: tuple[SpanEvent, ...] = ()
    exception: str | None = None


class ExportedSpan(AdkModel):
    """A span that has been through redaction and may leave the process.

    Args:
        name: The span name, unchanged.
        attributes: What survived, rewritten.
        events: The events, rewritten the same way.
        exception: The exception text, scrubbed.
        redaction: Which attribute names were dropped, so a gap reads as a decision.
    """

    name: str
    attributes: Mapping[str, str] = {}
    events: tuple[ExportedEvent, ...] = ()
    exception: str | None = None
    redaction: Redaction = Redaction()


@dataclass(slots=True)
class RedactionStats:
    """What the processor did, for the counter that should sit at zero in steady state.

    Args:
        redacted: Values rewritten because they matched a shape.
        dropped: Attributes removed entirely.
        failures: Attributes dropped because redaction could not complete.
    """

    redacted: int = 0
    dropped: int = 0
    failures: int = 0


class RedactingSpanProcessor:
    """Rewrites a span so nothing sensitive can leave the process.

    Enforcement is layered, because each layer catches what the one before it misses: a
    `SecretStr` never renders, a denylisted key is dropped whatever its value looks like,
    a payload attribute travels only if allowlisted, and every remaining value is scanned
    for sensitive shapes — recursively, where it is structured.

    Failure is closed. A detector that raises or a value that cannot be scanned costs that
    attribute, not the span: dropping the whole span would lose the causality that explains
    the failure, while exporting it unredacted is the incident this exists to prevent.

    Args:
        policy: What this deployment is willing to export.
        detector: An extra predicate over an already-scrubbed value, such as the PII
            classifier. A value it accepts is masked; an exception it raises drops the
            attribute and increments `stats.failures`.

    Example:
        >>> RedactingSpanProcessor().process(
        ...     PendingSpan(name="adk.tool", attributes={"who": "ada@example.com"})
        ... ).attributes["who"]
        '[redacted]'
    """

    def __init__(
        self,
        policy: RedactionPolicy | None = None,
        *,
        detector: Callable[[str], bool] | None = None,
    ) -> None:
        self.policy = policy or RedactionPolicy()
        self.stats = RedactionStats()
        self._detector = detector

    def process(self, span: PendingSpan) -> ExportedSpan:
        """Return `span` with everything sensitive rewritten or removed.

        Args:
            span: The span about to be exported.

        Returns:
            The span an exporter may have, and what was dropped from it.
        """
        attributes, dropped = self._attributes(span.attributes)
        events = tuple(
            ExportedEvent(name=event.name, attributes=self._attributes(event.attributes)[0])
            for event in span.events
        )
        return ExportedSpan(
            name=span.name,
            attributes=attributes,
            events=events,
            exception=self._exception(span.exception),
            redaction=Redaction(dropped=tuple(sorted(dropped))),
        )

    def _exception(self, exception: str | None) -> str | None:
        """The exception text, or a mask where scrubbing it did not complete."""
        if exception is None:
            return None
        try:
            return self._text(exception)
        except Exception:
            self.stats.failures += 1
            return MASK

    def _attributes(self, attributes: Mapping[str, object]) -> tuple[dict[str, str], list[str]]:
        """Rewrite one attribute mapping, returning what survived and what was dropped."""
        kept: dict[str, str] = {}
        dropped: list[str] = []
        for key, value in attributes.items():
            if self._denied(key):
                dropped.append(key)
                self.stats.dropped += 1
                continue
            if key in PAYLOAD_ATTRIBUTES and key not in self.policy.payload_attributes:
                dropped.append(key)
                self.stats.dropped += 1
                if self.policy.content_references:
                    kept[f"{key}{REFERENCE_SUFFIX}"] = _reference(value)
                continue
            try:
                rendered = self._value(key, value)
            except Exception:
                dropped.append(key)
                self.stats.failures += 1
                continue
            kept[key] = rendered
        return kept, dropped

    def _value(self, key: str, value: object) -> str:
        """One attribute value, masked, capped and scanned in that order."""
        if isinstance(value, SecretStr):
            self.stats.redacted += 1
            return MASK
        text = str(value)[: self.policy.max_length]
        if key.startswith(ATTRIBUTE_PREFIX) and key not in PAYLOAD_ATTRIBUTES:
            return text
        rewritten = self._text(text)
        if rewritten != text:
            self.stats.redacted += 1
        return rewritten

    def _text(self, text: str) -> str:
        """Scrub free text, or the leaves of a structured value where there are any."""
        capped = text[: self.policy.max_length]
        try:
            document = json.loads(capped)
        except ValueError:
            return self._scrubbed(capped)
        if not isinstance(document, dict | list):
            return self._scrubbed(capped)
        return json.dumps(self._nested(document))

    def _nested(self, value: object) -> object:
        """Walk a structured value, because a top-level match misses what tools send."""
        if isinstance(value, dict):
            return {
                key: MASK if self._denied(str(key)) else self._nested(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._nested(item) for item in value]
        if isinstance(value, str):
            return self._scrubbed(value)
        return value

    def _scrubbed(self, text: str) -> str:
        """Pattern-scrub, then hand what is left to the detector."""
        rewritten = scrub(text, self.policy.extra_patterns)
        if self._detector is not None and self._detector(rewritten):
            return MASK
        return rewritten

    def _denied(self, key: str) -> bool:
        """Whether a key names a credential, matched on its last dotted segment."""
        return key.rsplit(".", 1)[-1].replace("-", "_").lower() in self.policy.denied_keys


def _reference(value: object) -> str:
    """A digest of a payload that was not exported, so it can still be correlated."""
    digest = hashlib.sha256(str(value).encode()).hexdigest()
    return f"sha256:{digest[:16]}"
