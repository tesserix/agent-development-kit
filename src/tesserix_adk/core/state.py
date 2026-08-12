"""Where a session and a run live between processes, and what keeps two writers honest.

Session state is a Redis blob in one product and a JSON column in another, each with its
own key layout and no concurrency control. Two workers holding the same run write it back
and the second silently wins; nobody can tell afterwards that the first ever happened.

So state has one shape here, the tenant is part of the key rather than an argument beside
it, and every write states the version it read. A write against a version that moved is a
typed refusal rather than a lost update. Amounts that accumulate — tokens, cost, iterations
— go through `patch_run`, which adds rather than sets, because two workers adding commute
and two workers setting do not.

The decisions behind these types are in `docs/state.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.primitives import ToolCall, Usage
from tesserix_adk.core.redaction import scrub
from tesserix_adk.core.run import RunState

if TYPE_CHECKING:
    from typing import Self

__all__ = [
    "RunRecord",
    "SessionRecord",
    "StateDelta",
    "StateKey",
    "StatePage",
    "StateQuery",
    "StateStore",
]


class StateKey(AdkModel):
    """Whose record, and which one.

    The tenant is part of the key rather than a parameter beside it: a store that takes
    them separately can be called with one of them missing, and a cross-tenant read is
    the kind of bug that is found by an auditor rather than by a test.

    Args:
        tenant: The isolation boundary.
        id: The session or run this names, within that tenant.

    Example:
        >>> StateKey(tenant="acme", id="run_1").id
        'run_1'
    """

    tenant: str = Field(min_length=1)
    id: str = Field(min_length=1)


class SessionRecord(AdkModel):
    """A conversation across runs, as a store holds it.

    Args:
        session_id: What the session is called.
        tenant: The isolation boundary.
        user: The acting principal, where there is one.
        version: The version this was read at. Zero means it has never been stored, so a
            write of it is a create and fails if something is there already.
        runs: The runs that belong to it, in the order they started.
        updated_at: Unix seconds of the last write, for age queries. Never for ordering:
            two workers' clocks disagree and the version counter does not.
        metadata: Consumer-owned annotations. The kit reads nothing from it.
    """

    session_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    user: str | None = None
    version: int = Field(default=0, ge=0)
    runs: tuple[str, ...] = ()
    updated_at: float = 0.0
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def key(self) -> StateKey:
        """Where this record is stored."""
        return StateKey(tenant=self.tenant, id=self.session_id)


class RunRecord(AdkModel):
    """A run as a store holds it: far enough along to be resumed, and no further.

    Args:
        run_id: What the run is called.
        tenant: The isolation boundary.
        agent_name: Which agent is running.
        session_id: The conversation it belongs to, where it belongs to one.
        state: Where the run is. See `RunState`.
        version: The version this was read at. Zero means it has never been stored.
        message_cursor: How much of the conversation has been handed to the model, so a
            resumed run neither repeats nor skips a turn.
        pending_tool_calls: Calls the model asked for that have not come back. A run
            abandoned mid tool call is this, and it is a state rather than a corruption.
        usage: What the run has spent.
        cost_micros: What that cost, in millionths of the ledger's currency. An integer,
            because money that accumulates in a float stops adding up.
        iterations: How many times round the loop it has been.
        awaiting_approval: The approval record it is held on, where a human has to clear
            something before it continues.
        updated_at: Unix seconds of the last write, for age queries.

    Example:
        >>> RunRecord(run_id="run_1", tenant="acme", agent_name="planner").mid_tool_call
        False
    """

    run_id: str = Field(min_length=1)
    tenant: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    session_id: str | None = None
    state: RunState = RunState.PENDING
    version: int = Field(default=0, ge=0)
    message_cursor: int = Field(default=0, ge=0)
    pending_tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    cost_micros: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    awaiting_approval: str | None = None
    updated_at: float = 0.0

    @property
    def key(self) -> StateKey:
        """Where this record is stored."""
        return StateKey(tenant=self.tenant, id=self.run_id)

    @property
    def mid_tool_call(self) -> bool:
        """Whether the run stopped with a call it never got an answer to."""
        return bool(self.pending_tool_calls)

    def scrubbed(self) -> Self:
        """Return the record as it may be written down.

        Persisted state is a queryable store: a token that reached a tool argument is a
        token an operator can later grep for. Every implementation writes this rather
        than what it was handed, and the conformance suite fails one that does not.

        Example:
            >>> call = ToolCall(id="c1", name="pay", arguments={"who": "ada@example.com"})
            >>> record = RunRecord(
            ...     run_id="run_1", tenant="acme", agent_name="planner",
            ...     pending_tool_calls=(call,),
            ... )
            >>> record.scrubbed().pending_tool_calls[0].arguments["who"]
            '[redacted]'
        """
        return self.model_copy(
            update={
                "pending_tool_calls": tuple(_scrubbed(call) for call in self.pending_tool_calls)
            }
        )


class StateDelta(AdkModel):
    """Amounts to add to a run, never values to set.

    Two workers each adding what they spent commute; two workers each writing a total do
    not, and the one that arrives second is the only one that happened.

    Args:
        usage: Tokens to add.
        cost_micros: Cost to add.
        iterations: Loop passes to add.
        messages_read: How far to advance the message cursor.

    Example:
        >>> StateDelta(iterations=1).empty
        False
    """

    usage: Usage | None = None
    cost_micros: int = Field(default=0, ge=0)
    iterations: int = Field(default=0, ge=0)
    messages_read: int = Field(default=0, ge=0)

    @property
    def empty(self) -> bool:
        """Whether this would change nothing, which a store may skip writing."""
        return self.usage is None and not (
            self.cost_micros or self.iterations or self.messages_read
        )

    def applied_to(self, record: RunRecord) -> RunRecord:
        """Return `record` with these amounts added. The one place the arithmetic lives."""
        usage = record.usage if self.usage is None else record.usage + self.usage
        return record.model_copy(
            update={
                "usage": usage,
                "cost_micros": record.cost_micros + self.cost_micros,
                "iterations": record.iterations + self.iterations,
                "message_cursor": record.message_cursor + self.messages_read,
            }
        )


class StateQuery(AdkModel):
    """Which runs to list, for operator tooling and for whatever reaps abandoned work.

    Args:
        tenant: The isolation boundary. Required; there is no cross-tenant listing.
        state: Only runs in this state, where one is named.
        updated_before: Only runs last written before this Unix second, which is how
            abandoned work is found.
        limit: How many at most. A page, not the whole table.
        cursor: Where to carry on from, as the previous page returned it. Opaque: it
            names a position in the store's own write order, not a timestamp.

    Example:
        >>> StateQuery(tenant="acme", limit=10).limit
        10
    """

    tenant: str = Field(min_length=1)
    state: RunState | None = None
    updated_before: float | None = None
    limit: int = Field(default=50, ge=1, le=1000)
    cursor: str | None = None


class StatePage(AdkModel):
    """One page of runs, and where the next one starts.

    Args:
        records: The runs, in the order the store first saw them. Never in clock order:
            two workers' clocks disagree and a listing that depends on them skips records.
        cursor: What to pass as the next query's `cursor`, or `None` at the end.
    """

    records: tuple[RunRecord, ...] = ()
    cursor: str | None = None

    @model_validator(mode="after")
    def _refuse_a_cursor_that_leads_nowhere(self) -> Self:
        """An empty page with a cursor would loop a reaper forever."""
        if self.cursor is not None and not self.records:
            raise ValueError("an empty page cannot carry a cursor to carry on from")
        return self


@runtime_checkable
class StateStore(Protocol):
    """Where session and run state lives, for agents running in more than one process.

    Every write states the version it read: a store commits at that version plus one, or
    refuses with `StateConflictError` carrying both numbers. A lost update is impossible
    rather than unlikely. Version zero means "this is a create", so two workers racing to
    start the same run resolve to one.

    Implementations write `record.scrubbed()` rather than the record they were handed,
    order listings by their own insertion counter rather than by a clock, and raise
    `StatePersistenceError` rather than partially applying a write. A store that cannot be
    reached is not a store that accepted the write.
    """

    async def get_session(self, key: StateKey) -> SessionRecord | None:
        """Return the session, or `None` where there is not one."""
        ...

    async def put_session(self, record: SessionRecord) -> SessionRecord:
        """Store the session and return it at its new version.

        Raises:
            StateConflictError: If the stored version is not the one the record was read
                at, including where a create found something already there.
            StatePersistenceError: If the store could not take the write.
        """
        ...

    async def delete_session(self, key: StateKey, *, cascade: bool = False) -> None:
        """Remove the session. Deleting an absent session is not an error.

        Args:
            key: Which session.
            cascade: Also remove its runs. Without it, a session that still has runs
                which have not reached a terminal state is refused.

        Raises:
            StateInUseError: If runs are still live and `cascade` was not asked for.
        """
        ...

    async def get_run(self, key: StateKey) -> RunRecord | None:
        """Return the run, or `None` where there is not one."""
        ...

    async def put_run(self, record: RunRecord) -> RunRecord:
        """Store the run and return it at its new version.

        Raises:
            StateConflictError: If the stored version is not the one the record was read
                at, including where a create found something already there.
            StatePersistenceError: If the store could not take the write.
        """
        ...

    async def patch_run(self, key: StateKey, delta: StateDelta) -> RunRecord:
        """Add `delta` to the run and return it, without reading it first.

        Concurrent patches accumulate rather than overwrite, so this takes no version.

        Raises:
            StateNotFoundError: If there is no such run. There is nothing to add to.
            StatePersistenceError: If the store could not take the write.
        """
        ...

    async def delete_run(self, key: StateKey) -> None:
        """Remove the run. Deleting an absent run is not an error."""
        ...

    async def list_runs(self, query: StateQuery) -> StatePage:
        """Return one page of runs matching `query`, oldest first."""
        ...


def _scrubbed(call: ToolCall) -> ToolCall:
    """A call with anything sensitive-looking in its arguments masked."""
    return call.model_copy(
        update={
            "arguments": {
                name: scrub(value) if isinstance(value, str) else value
                for name, value in call.arguments.items()
            }
        }
    )
