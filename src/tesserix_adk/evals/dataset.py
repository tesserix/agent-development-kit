"""The golden dataset an agent is judged against, and the file it is committed as.

An eval dataset is source code: it is reviewed, it is diffed, and a case added today has to
mean the same thing to the run that reads it in a year. Two things follow. The file carries
its own format number, so a reader that meets a newer dataset says so instead of silently
dropping the fields it does not know about. And a case identifies itself, because a result
is only a regression if it can be compared to the same case's result yesterday.

Cases are redacted on the way to disk. A dataset is committed, and a customer email
committed once outlives every run that used it — see `docs/eval-datasets.md`.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.redaction import looks_sensitive, scrub

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["DATASET_FORMAT", "EvalCase", "EvalSuite"]

DATASET_FORMAT = 1
"""The on-disk format this build writes and is willing to read."""

_IDENTITY_FIELDS = ("id", "tenant", "user")


class EvalCase(AdkModel):
    """One example: what the agent is asked, as whom, and what a good answer looks like.

    Args:
        id: Stable across edits to the rest of the case — it is the key a result is
            compared under, so renaming it starts the history over.
        input: What the user says.
        tenant: Who the case runs as. There is no default: a case that does not say whose
            data it touches cannot be replayed safely.
        user: The acting principal, where the case is about one.
        expected: The answer a human would accept, for metrics that need a reference. A
            case without one is still useful to metrics that judge cost or a schema.
        tags: Labels a run can select on, such as the release the case was added for.
        metadata: Anything a project's own metrics read.

    Example:
        >>> EvalCase(id="refund-1", input="refund my order", tenant="acme").tenant
        'acme'
    """

    id: str = Field(min_length=1)
    input: str
    tenant: str = Field(min_length=1)
    user: str | None = None
    expected: str | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _identities_are_not_blank(self) -> EvalCase:
        """Refuse whitespace where an id was meant, which compares against nothing."""
        for field in ("id", "tenant"):
            if not str(getattr(self, field)).strip():
                raise ConfigurationError(f"a case's {field} cannot be blank")
        return self

    def redacted(self) -> EvalCase:
        """Return this case with sensitive-looking content masked.

        Identity fields are refused rather than masked: masking an id would break the
        comparison the id exists for, so a credential-shaped one is the author's to fix.

        Raises:
            ConfigurationError: An identity field looks sensitive.

        Example:
            >>> EvalCase(id="c1", input="mail ada@example.com", tenant="acme").redacted().input
            'mail [redacted]'
        """
        for field in _IDENTITY_FIELDS:
            value = getattr(self, field)
            if value is not None and looks_sensitive(str(value)):
                raise ConfigurationError(
                    f"a case's {field} cannot be redacted without breaking the comparison "
                    f"it exists for; edit the dataset instead"
                )
        return self.model_copy(
            update={
                "input": scrub(self.input),
                "expected": None if self.expected is None else scrub(self.expected),
                "metadata": {key: scrub(value) for key, value in self.metadata.items()},
            }
        )


class EvalSuite(AdkModel):
    """A named, versioned set of cases, read and written as one file.

    Args:
        name: What the suite measures. It names the artefact directory too.
        version: The dataset's own version, bumped when a case changes meaning. Two runs
            are only comparable across the same version.
        cases: In file order, which is the order they are reported in.

    Example:
        >>> suite = EvalSuite(
        ...     name="refunds",
        ...     version="1",
        ...     cases=(EvalCase(id="c1", input="hi", tenant="acme"),),
        ... )
        >>> len(suite.cases)
        1
    """

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    cases: tuple[EvalCase, ...]

    @model_validator(mode="after")
    def _one_case_per_id(self) -> EvalSuite:
        """Refuse a repeated id, which would report two results under one history."""
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ConfigurationError(f"case id {case.id!r} appears more than once")
            seen.add(case.id)
        return self

    def tagged(self, tag: str) -> tuple[EvalCase, ...]:
        """Return the cases carrying `tag`, in file order."""
        return tuple(case for case in self.cases if tag in case.tags)

    def to_jsonl(self, path: Path) -> None:
        """Write the suite as JSONL: a header line, then one redacted case per line.

        Nothing is written until every case has been rendered, so a case the redaction
        refuses leaves no half-file behind for the next reader to trust.

        Args:
            path: The file to write, created or replaced.

        Raises:
            ConfigurationError: A case carries a sensitive-looking identity field.
        """
        header = json.dumps({"format": DATASET_FORMAT, "name": self.name, "version": self.version})
        rendered = (json.dumps(case.redacted().model_dump(mode="json")) for case in self.cases)
        lines = [header, *rendered]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def from_jsonl(cls, path: Path) -> EvalSuite:
        """Read a suite written by `to_jsonl`.

        Args:
            path: The dataset file.

        Returns:
            The suite, with its cases in file order.

        Raises:
            ConfigurationError: The header is missing, or the format is one this build
                does not read.
        """
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            raise ConfigurationError(f"{path} is empty; a dataset starts with a header line")
        header = cls._header(path, lines[0])
        return cls(
            name=str(header["name"]),
            version=str(header["version"]),
            cases=tuple(EvalCase.model_validate_json(line) for line in lines[1:]),
        )

    @staticmethod
    def _header(path: Path, line: str) -> Mapping[str, object]:
        """Parse and check the first line, which says what the rest of the file is."""
        try:
            header = json.loads(line)
        except json.JSONDecodeError as unreadable:
            raise ConfigurationError(f"{path} has no readable header line") from unreadable
        if not isinstance(header, dict) or "format" not in header:
            raise ConfigurationError(
                f"{path} has no header line declaring its format, name and version"
            )
        found = header["format"]
        if found != DATASET_FORMAT:
            raise ConfigurationError(
                f"{path} is dataset format {found}, and this build reads {DATASET_FORMAT}; "
                f"migrate it with `adk eval migrate`"
            )
        return header
