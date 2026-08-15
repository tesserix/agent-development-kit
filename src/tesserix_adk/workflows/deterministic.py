"""The workflow-safe replacements for the things that break a replay.

A workflow function is re-executed from the start every time it is resumed, and every
decision it makes has to come out the same way. `uuid4`, `time.time()`, `datetime.now()` and
an unordered iteration do not, and none of them fails in development — the divergence arrives
on the first replay in production, where it either wedges the run or silently re-decides it.

Everything here derives its answer from the run's own recorded state, so the second execution
computes what the first one did.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type

from tesserix_adk.core.models import AdkModel

__all__ = ["DeterministicIds", "Patches", "WorkflowClock", "stable"]

_ID_BYTES = 16


class WorkflowClock(AdkModel):
    """Time as the run recorded it, not as the worker's clock reads it now.

    A replay a week later must see the same instants the original execution saw. The clock
    is therefore state: it starts where the run started and only moves when the workflow
    says it moved, from a value that came back through an activity.

    Args:
        started_at: When the run began, in Unix seconds, taken once by the caller that
            started it.
        elapsed: How far the run has advanced since. Never read from a wall clock.

    Example:
        >>> clock = WorkflowClock(started_at=1_700_000_000.0)
        >>> clock.advanced(90.0).now()
        1700000090.0
    """

    started_at: float
    elapsed: float = 0.0

    def now(self) -> float:
        """The current instant, in Unix seconds, as the run understands it."""
        return self.started_at + self.elapsed

    def advanced(self, seconds: float) -> WorkflowClock:
        """Return the clock moved forward, which is the only way it moves.

        Raises:
            ValueError: If `seconds` is negative. A clock that can go backwards makes a
                deadline that has passed pass again.
        """
        if seconds < 0:
            raise ValueError(f"a workflow clock only moves forward, got {seconds}")
        return self.model_copy(update={"elapsed": self.elapsed + seconds})


class DeterministicIds(AdkModel):
    """Ids derived from the run and the call's position, never from randomness.

    An idempotency key from `uuid4` is a different key on every replay, which is how a
    payment is made twice by a run that was only trying to finish. These ids depend on the
    run id and how many have been asked for, so the replay produces the same sequence.

    Args:
        run_id: The run they belong to. Two runs never share an id.
        issued: How many have been handed out. A replay reaches the same count at the same
            point, which is what makes the ids match.

    Example:
        >>> ids = DeterministicIds(run_id="r1")
        >>> ids.next("payment")[0] == DeterministicIds(run_id="r1").next("payment")[0]
        True
    """

    run_id: str
    issued: int = 0

    def next(self, kind: str = "") -> tuple[str, DeterministicIds]:
        """Return the next id and the generator that has issued it."""
        seed = f"{self.run_id}:{kind}:{self.issued}".encode()
        return (
            hashlib.blake2b(seed, digest_size=_ID_BYTES).hexdigest(),
            self.model_copy(update={"issued": self.issued + 1}),
        )


class Patches(AdkModel):
    """Which changes in the workflow's logic this history already knows about.

    Agent logic does legitimately change. A run started before the change must keep taking
    the old path, or its replay diverges from the history it is being replayed against; a
    run started after it takes the new one. `applied` is what that decision reads.

    Args:
        known: The patch names recorded in this run's history. A history recorded before a
            patch existed does not name it, which is exactly how the old path is chosen.

    Example:
        >>> Patches(known=("prompt-v2",)).applied("prompt-v2")
        True
        >>> Patches().applied("prompt-v2")
        False
    """

    known: tuple[str, ...] = ()

    def applied(self, name: str) -> bool:
        """Whether this run takes the patched path."""
        return name in self.known

    def applying(self, name: str) -> Patches:
        """Return these patches with `name` recorded, for a run reaching it the first time."""
        if name in self.known:
            return self
        return self.model_copy(update={"known": (*self.known, name)})


def stable[ValueT](registry: Mapping[str, ValueT]) -> tuple[tuple[str, ValueT], ...]:
    """Return a registry's entries in an order that cannot change between executions.

    A tool registry iterated in load order feeds the prompt in load order, and the order
    changes when an import does. Sorting by name is not cosmetic here: it is what stops a
    deploy from re-deciding a run that was already in flight.

    Example:
        >>> stable({"search": 2, "book": 1})
        (('book', 1), ('search', 2))
    """
    return tuple(sorted(registry.items()))
