"""Scrubbing applied before anything is exported, not after it is stored.

Cost data is queried by finance, by platform owners and by whoever is investigating a
spike — people who were never cleared to read prompts. An attribute a consumer attached
for their own convenience is how a customer email address ends up in a queryable store,
so consumer attributes are pattern-scrubbed on the way out and the drop is recorded rather
than made silently.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from tesserix_adk.core import MASK, SENSITIVE_SHAPES, AdkModel
from tesserix_adk.observability.attribution import ATTRIBUTE_PREFIX

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = ["MASK", "Redaction", "Redactor"]


class Redaction(AdkModel):
    """What was removed on the way out.

    Args:
        dropped: The attribute names whose values were masked, sorted. Recorded so a
            missing attribute reads as a decision rather than as a bug in the exporter.
    """

    dropped: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        """Whether anything was removed."""
        return bool(self.dropped)


class Redactor:
    """Masks attribute values that match a sensitive shape.

    The kit's own `adk.` attributes are structural identity — a tenant id, an agent name, a
    token count — and are passed through unscrubbed. Redacting them would leave spend
    attributed to nobody, which is the failure this whole surface exists to prevent; case
    content never reaches them in the first place.

    Args:
        extra_patterns: Additional regular expressions a deployment knows about, such as a
            local case or account reference. Applied alongside the built-in shapes.

    Example:
        >>> Redactor().scrub({"who": "ada@example.com"})[0]["who"]
        '[redacted]'
    """

    def __init__(self, extra_patterns: Sequence[str] = ()) -> None:
        self._patterns = tuple(re.compile(p) for p in (*SENSITIVE_SHAPES, *extra_patterns))

    def scrub(self, attributes: Mapping[str, str]) -> tuple[dict[str, str], Redaction]:
        """Return `attributes` with sensitive values masked, and what was masked.

        Args:
            attributes: The attributes about to be exported.

        Returns:
            The attributes to export, and a `Redaction` naming what was dropped.
        """
        kept: dict[str, str] = {}
        dropped: list[str] = []
        for key, value in attributes.items():
            if key.startswith(ATTRIBUTE_PREFIX) or not self._matches(value):
                kept[key] = value
                continue
            kept[key] = MASK
            dropped.append(key)
        return kept, Redaction(dropped=tuple(sorted(dropped)))

    def _matches(self, value: str) -> bool:
        return any(pattern.search(value) for pattern in self._patterns)
