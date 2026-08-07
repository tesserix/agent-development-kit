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

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from tesserix_adk.core.errors import ProtocolConformanceError

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from tesserix_adk.core.budget import BudgetDecision, BudgetLimits, ResolvedBudget
    from tesserix_adk.core.capabilities import ModelCapabilities
    from tesserix_adk.core.primitives import Message, Usage
    from tesserix_adk.core.provider import ModelRequest, ModelResponse
    from tesserix_adk.core.streaming import StreamEvent

__all__ = [
    "BudgetPolicy",
    "Clock",
    "DeclaresEmulation",
    "Guardrail",
    "IdFactory",
    "MemoryStore",
    "ModelProvider",
    "SecretProvider",
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

    @property
    def capabilities(self) -> ModelCapabilities:
        """What this provider's model can do, as a record the kit checks before calling.

        Read at wiring time and again before each request. A capability discovered from
        an upstream error is a capability discovered after paying for it.
        """
        ...

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return a completion for `request`.

        Raises:
            ProviderError: On any transport or upstream failure, after translation.
            ModelResponseError: If the upstream body cannot be read as a response.
        """
        ...

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Return an async iterator of incremental completion events.

        The events are the kit's own — see `tesserix_adk.core.streaming` — so a consumer
        reads one vocabulary whichever vendor produced the stream. The last event is a
        `StreamEnd` carrying the assembled response.

        Raises:
            CapabilityError: If this provider does not declare `streaming`. Buffering one
                chunk and calling it a stream is worse than refusing.
            ProviderError: On any transport or upstream failure, after translation.
        """
        ...

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Return how many tokens `messages` occupy, by this provider's own count.

        Every vendor counts differently, so the answer belongs to the provider. It is
        what the declared context window is checked against.
        """
        ...


@runtime_checkable
class DeclaresEmulation(Protocol):
    """A provider with an opinion about being stood in for.

    Where a model cannot enforce a schema itself, the kit can ask for JSON in the prompt
    and validate what comes back. That is worth having against a vendor model that is
    good at following the instruction, and it is a trap against a small self-hosted one:
    the schema is enforced by nothing, and the failure is an answer in the wrong shape
    rather than an error. A provider that knows which it is says so here; one that says
    nothing is emulated, which is what every provider did before this existed.
    """

    @property
    def emulates(self) -> bool:
        """Whether the kit may stand in for a capability this provider has not declared."""
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
class SecretProvider(Protocol):
    """Where a credential is looked up, asked once per use rather than once per process.

    Implemented by the environment by default, and by a secret manager where one is
    injected. Nothing in the kit reads a key from a config file or a database: one is
    found by whoever clones the repository, the other cannot be rotated without a restart.
    """

    def secret(self, name: str) -> str | None:
        """Return the current value for `name`, or None where there is not one."""
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
    """Spend and usage limits applied before and after a metered call.

    Ceilings are checked before the spend, never reported after it, and an implementation
    may keep its ledger in the process or in a shared store — the shape is the same either
    way. See `tesserix_adk.core.budget` for the limits vocabulary and the default policy.
    """

    @property
    def resolved(self) -> ResolvedBudget:
        """The effective ceiling and the scope each dimension of it came from."""
        ...

    def limits(self) -> BudgetLimits:
        """What is left, as limits — the allowance a child run starts from."""
        ...

    def child(self) -> BudgetPolicy:
        """A budget for a run this one called, spending what this one has left.

        A sub-agent handed a fresh allowance is a way to spend one ceiling twice.
        """
        ...

    async def reserve(self, estimate: int) -> None:
        """Reserve `estimate` input tokens before a call.

        Raises:
            BudgetExceededError: If the reservation would breach a ceiling.
            BudgetUnavailableError: If a shared ledger holding a ceiling is unreachable.
        """
        ...

    async def record(
        self,
        usage: Usage,
        *,
        model_calls: int = 0,
        tool_calls: int = 0,
        iterations: int = 0,
        peer_invocations: int = 0,
    ) -> None:
        """Record what was consumed, releasing any over-reservation.

        Raises:
            BudgetExceededError: If the run has now passed a ceiling.
            BudgetUnavailableError: If a shared ledger holding a ceiling is unreachable.
        """
        ...

    def check(self) -> BudgetDecision:
        """Whether there is room left, and where there is not. Never raises."""
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
