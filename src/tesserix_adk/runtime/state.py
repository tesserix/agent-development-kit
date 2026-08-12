"""Session and run state in a dict, for a single replica and for tests.

Nothing here outlives the process, which is what surviving a restart is not. It exists so
the concurrency rules a real adapter has to keep — versioned writes, commuting patches,
listings ordered by insertion rather than by clock — can be exercised without a database.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.errors import StateConflictError, StateInUseError, StateNotFoundError
from tesserix_adk.core.state import (
    RunRecord,
    SessionRecord,
    StateDelta,
    StateKey,
    StatePage,
    StateQuery,
)

if TYPE_CHECKING:
    from tesserix_adk.core.protocols import Clock

__all__ = ["MemoryStateStore"]


class MemoryStateStore:
    """A `StateStore` in two dicts, keyed by tenant and id.

    A record takes a store-wide sequence number when it is first written and keeps it, and
    listings are ordered by that. A store that paged by timestamp would skip records
    whenever two writers' clocks disagreed, and disagreeing clocks are what a distributed
    store has instead of a shared one.
    """

    def __init__(self, clock: Clock | None = None) -> None:
        self._sessions: dict[tuple[str, str], SessionRecord] = {}
        self._runs: dict[tuple[str, str], tuple[int, RunRecord]] = {}
        self._sequence = 0
        self._clock = clock

    async def get_session(self, key: StateKey) -> SessionRecord | None:
        """Return the session, or `None` where there is not one."""
        return self._sessions.get(_at(key))

    async def put_session(self, record: SessionRecord) -> SessionRecord:
        """Store the session at its version plus one."""
        key = record.key
        stored = self._sessions.get(_at(key))
        self._refuse_a_moved_version(0 if stored is None else stored.version, record.version, key)
        written = record.model_copy(
            update={"version": record.version + 1, "updated_at": self._now()}
        )
        self._sessions[_at(key)] = written
        return written

    async def delete_session(self, key: StateKey, *, cascade: bool = False) -> None:
        """Remove the session, and its runs where `cascade` was asked for."""
        if _at(key) not in self._sessions:
            return
        owned = [run for _, run in self._runs.values() if run.session_id == key.id]
        live = tuple(run.run_id for run in owned if not run.state.is_terminal)
        if live and not cascade:
            raise StateInUseError(
                f"session {key.id!r} still has {len(live)} unfinished run(s)",
                key=_named(key),
                live_runs=live,
            )
        if cascade:
            for run in owned:
                self._runs.pop((run.tenant, run.run_id), None)
        del self._sessions[_at(key)]

    async def get_run(self, key: StateKey) -> RunRecord | None:
        """Return the run, or `None` where there is not one."""
        held = self._runs.get(_at(key))
        return None if held is None else held[1]

    async def put_run(self, record: RunRecord) -> RunRecord:
        """Store the run at its version plus one, with anything sensitive masked."""
        key = record.key
        stored = self._runs.get(_at(key))
        self._refuse_a_moved_version(
            0 if stored is None else stored[1].version, record.version, key
        )
        written = record.scrubbed().model_copy(
            update={"version": record.version + 1, "updated_at": self._now()}
        )
        self._runs[_at(key)] = (self._placed(stored), written)
        return written

    async def patch_run(self, key: StateKey, delta: StateDelta) -> RunRecord:
        """Add `delta` to the run. Takes no version: additions commute."""
        held = self._runs.get(_at(key))
        if held is None:
            raise StateNotFoundError(f"no run {key.id!r} to add to", key=_named(key), kind="run")
        sequence, record = held
        if delta.empty:
            return record
        patched = delta.applied_to(record).model_copy(update={"updated_at": self._now()})
        self._runs[_at(key)] = (sequence, patched)
        return patched

    async def delete_run(self, key: StateKey) -> None:
        """Remove the run. Deleting an absent run is not an error."""
        self._runs.pop(_at(key), None)

    async def list_runs(self, query: StateQuery) -> StatePage:
        """Return one page of matching runs, oldest first."""
        after = -1 if query.cursor is None else int(query.cursor)
        matching = sorted(
            (
                (sequence, record)
                for sequence, record in self._runs.values()
                if sequence > after and _matches(record, query)
            ),
            key=lambda held: held[0],
        )
        page = matching[: query.limit]
        more = len(matching) > query.limit
        return StatePage(
            records=tuple(record for _, record in page),
            cursor=str(page[-1][0]) if more else None,
        )

    def _refuse_a_moved_version(self, actual: int, version: int, key: StateKey) -> None:
        """A write states the version it read, or it is not a write anyone can trust."""
        if actual != version:
            raise StateConflictError(
                f"{key.id!r} moved from version {version} to {actual}",
                key=_named(key),
                expected_version=version,
                actual_version=actual,
            )

    def _placed(self, stored: tuple[int, RunRecord] | None) -> int:
        """Where a record sits in the listing: assigned once, so paging cannot re-see it."""
        if stored is not None:
            return stored[0]
        self._sequence += 1
        return self._sequence

    def _now(self) -> float:
        return 0.0 if self._clock is None else self._clock.now()


def _at(key: StateKey) -> tuple[str, str]:
    return (key.tenant, key.id)


def _named(key: StateKey) -> str:
    return f"{key.tenant}/{key.id}"


def _matches(record: RunRecord, query: StateQuery) -> bool:
    if record.tenant != query.tenant:
        return False
    if query.state is not None and record.state != query.state:
        return False
    return not (query.updated_before is not None and record.updated_at >= query.updated_before)
