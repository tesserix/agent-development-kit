"""Typed boundaries between kit layers.

Every seam the kit crosses is a `Protocol` declared here. Nothing in this module
performs I/O, imports a vendor SDK or names a vendor type: a protocol that accepts
or returns a vendor type is not substitutable, which defeats the reason for having
it. Concrete implementations live in `models`, `memory`, `adapters` and friends;
fakes live in `testing`.

Conformance is checked in three places, each catching what the others cannot.
`mypy --strict` verifies signatures at authoring time. `verify_conformance` verifies
member presence at construction time, so a missing member fails before the run
starts. The conformance suite in `tesserix_adk.testing` verifies behaviour.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from tesserix_adk.core.errors import ProtocolConformanceError

__all__ = [
    "BudgetPolicy",
    "Clock",
    "Guardrail",
    "IdFactory",
    "MemoryStore",
    "ModelProvider",
    "ToolRegistry",
    "Tracer",
    "members_of",
    "verify_conformance",
]


@runtime_checkable
class Clock(Protocol):
    """Source of time, injected so that behaviour under time is testable.

    Nothing in the kit calls `time` or `datetime.now` directly. A test that must
    wait for a timeout should advance a fake clock, not sleep.
    """

    def now(self) -> float:
        """Return the current time as a monotonic seconds value."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend for `seconds`.

        Raises:
            asyncio.CancelledError: If the surrounding task is cancelled.
        """
        ...


@runtime_checkable
class IdFactory(Protocol):
    """Source of identifiers, injected so that a run can be compared with another.

    A `uuid4` in the loop is a field no assertion can name: two runs of the same agent on
    the same inputs differ, and the difference means nothing. Production passes nothing and
    gets random ids; a test passes a factory and gets ids it can write down.
    """

    def __call__(self) -> str:
        """Return the next identifier."""
        ...


@runtime_checkable
class ModelProvider(Protocol):
    """A source of model completions.

    Implementations are responsible for transport and for translating vendor errors
    into the kit's error hierarchy. They are not responsible for policy, budget or
    tracing — those are crossed by the bus around this protocol, so that a new
    provider cannot accidentally skip them.
    """

    @property
    def name(self) -> str:
        """Stable identifier for this provider, used in records and routing."""
        ...

    async def complete(self, request: Any) -> Any:
        """Return a completion for `request`.

        Raises:
            ProviderError: On any transport or upstream failure, after translation.
        """
        ...

    async def stream(self, request: Any) -> Any:
        """Return an async iterator of incremental completion events.

        Raises:
            ProviderError: On any transport or upstream failure, after translation.
        """
        ...


@runtime_checkable
class ToolRegistry(Protocol):
    """The set of tools an agent may call, and their declared schemas."""

    def declarations(self) -> Any:
        """Return tool declarations in a stable order.

        Order is part of the cacheable prompt prefix, so it must not vary between
        calls or every turn pays a full prefill.
        """
        ...

    async def invoke(self, name: str, arguments: Any) -> Any:
        """Invoke tool `name` with `arguments`.

        Raises:
            ToolNotFoundError: If no tool is registered under `name`.
            ToolInputError: If `arguments` do not satisfy the declared schema.
        """
        ...


@runtime_checkable
class MemoryStore(Protocol):
    """Durable key-scoped storage for memory records.

    Every operation is tenant-scoped by the caller. An implementation that ignores
    the scope is a cross-tenant read waiting to happen.
    """

    async def get(self, key: str) -> Any:
        """Return the record stored under `key`, or None if absent."""
        ...

    async def put(self, key: str, value: Any) -> None:
        """Store `value` under `key`, replacing any existing record."""
        ...

    async def delete(self, key: str) -> None:
        """Remove `key`. Deleting an absent key is not an error."""
        ...


@runtime_checkable
class Guardrail(Protocol):
    """An inline check on the model or tool call path.

    Guardrails fail closed: if a check cannot be evaluated, the call does not happen.
    This is the opposite of `Tracer`, and the distinction is deliberate.
    """

    @property
    def name(self) -> str:
        """Stable identifier, recorded when the guardrail trips."""
        ...

    async def check(self, subject: Any) -> Any:
        """Evaluate `subject` and return the decision.

        Raises:
            GuardrailEvaluationError: If the check cannot be evaluated, which the
                caller must treat as a denial rather than a pass.
        """
        ...


@runtime_checkable
class BudgetPolicy(Protocol):
    """Spend and usage limits applied before and after a metered call."""

    async def reserve(self, estimate: int) -> None:
        """Reserve `estimate` units before a call.

        Raises:
            BudgetExceededError: If the reservation would breach the limit.
        """
        ...

    async def record(self, actual: int) -> None:
        """Record the `actual` units consumed, releasing any over-reservation."""
        ...


@runtime_checkable
class Tracer(Protocol):
    """Sideband emission of spans and events.

    Tracing fails open. A collector outage degrades observability; it must never
    stop a run. Implementations therefore swallow their own transport failures
    rather than propagating them.
    """

    def span(self, name: str, **attributes: Any) -> Any:
        """Return a context manager recording a span named `name`."""
        ...

    def event(self, name: str, **attributes: Any) -> None:
        """Record a point-in-time event. Never raises."""
        ...


_PROTOCOL_INTERNALS = frozenset(
    {"__init__", "__subclasshook__", "__class_getitem__", "__protocol_attrs__"}
)


def members_of(protocol: type) -> tuple[str, ...]:
    """Return the member names a `protocol` requires, sorted.

    Args:
        protocol: A `Protocol` subclass.

    Returns:
        Sorted member names, excluding dunders and protocol machinery.
    """
    declared: set[str] = set(getattr(protocol, "__protocol_attrs__", set()))
    return tuple(sorted(declared - _PROTOCOL_INTERNALS))


def verify_conformance(obj: object, protocol: type) -> None:
    """Raise unless `obj` provides every member `protocol` requires.

    `runtime_checkable` `isinstance` checks report presence as a single boolean and
    say nothing about which member is absent, which turns a one-line configuration
    mistake into a search. This reports the whole set at once, at construction time.

    Signature conformance is not checked here and cannot be — that is `mypy --strict`
    at authoring time and the conformance suite at run time.

    Args:
        obj: The candidate implementation.
        protocol: The protocol it claims to satisfy.

    Raises:
        ProtocolConformanceError: If any required member is absent or, for a method,
            present but not callable.
    """
    missing: list[str] = []
    for name in members_of(protocol):
        candidate = getattr(obj, name, None)
        if candidate is None:
            missing.append(name)
            continue
        if callable(getattr(protocol, name, None)) and not callable(candidate):
            missing.append(name)
    if missing:
        raise ProtocolConformanceError(
            protocol=protocol.__name__,
            missing=tuple(sorted(missing)),
            obj_type=type(obj).__name__,
        )
