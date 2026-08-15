"""Declaring a reversal next to the work it reverses, and a scope that drives it.

The forward step and the step that takes it back are written together, because the pair
drifts apart the moment they live in different files. What the scope adds over the runtime
engine is the ordering and the refusal: every applied step is written down *before* it is
attempted, so a call whose result never came back is still on the list, and a step nothing
can take back is refused before it runs rather than apologised for afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tesserix_adk.core.compensation import Applied, arguments_digest
from tesserix_adk.core.errors import IrreversibleActionError
from tesserix_adk.runtime.compensation import Compensator

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping
    from types import TracebackType

    from pydantic import JsonValue

    from tesserix_adk.core.compensation import (
        CompensatedFailure,
        CompensationIncomplete,
        CompensationLedger,
    )
    from tesserix_adk.core.idempotency import IdempotencyStore
    from tesserix_adk.core.protocols import Clock

__all__ = ["CompensatingActivity", "Compensations", "Saga", "compensating_activity"]


class CompensatingActivity:
    """One reversal, bound to the name of the forward tool it takes back.

    Example:
        >>> @compensating_activity("book_hotel")
        ... async def cancel_hotel(action: Applied) -> str:
        ...     return f"cancelled {action.idempotency_key}"
        >>> cancel_hotel.forward
        'book_hotel'
    """

    def __init__(
        self,
        forward: str,
        reverse: Callable[[Applied], Awaitable[str]],
        *,
        irreversible: bool,
    ) -> None:
        self.forward = forward
        self.irreversible = irreversible
        self._reverse = reverse
        self.__doc__ = reverse.__doc__
        self.__name__ = getattr(reverse, "__name__", forward)

    async def compensate(self, action: Applied) -> str:
        """Take `action` back, and say what the world did about it."""
        return await self._reverse(action)


def compensating_activity(
    forward: str, *, irreversible: bool = False
) -> Callable[[Callable[[Applied], Awaitable[str]]], CompensatingActivity]:
    """Pair a reversal with the forward tool it undoes.

    Args:
        forward: The name of the tool this reverses.
        irreversible: Declare that the forward action cannot be taken back at all. The
            reversal is then never called automatically — it exists for a person who
            reconciles by hand, and the run says so rather than pretending it unwound.

    Returns:
        A decorator producing a `CompensatingActivity`.

    Example:
        >>> @compensating_activity("charge_card", irreversible=True)
        ... async def unwind_charge(action: Applied) -> str:
        ...     return "reconcile by hand"
        >>> unwind_charge.irreversible
        True
    """

    def decorate(reverse: Callable[[Applied], Awaitable[str]]) -> CompensatingActivity:
        return CompensatingActivity(forward, reverse, irreversible=irreversible)

    return decorate


class Compensations:
    """Which reversal belongs to which forward tool.

    Two reversals claiming one tool is refused at construction: whichever ran would
    otherwise depend on registration order, and that is not something to find out
    during a rollback.

    Example:
        >>> @compensating_activity("book_hotel")
        ... async def cancel_hotel(action: Applied) -> str:
        ...     return "cancelled"
        >>> Compensations(cancel_hotel).of("book_hotel").forward
        'book_hotel'
    """

    def __init__(self, *reversals: CompensatingActivity) -> None:
        self._by_tool: dict[str, CompensatingActivity] = {}
        for reversal in reversals:
            if reversal.forward in self._by_tool:
                raise ValueError(f"two reversals claim {reversal.forward}")
            self._by_tool[reversal.forward] = reversal

    def of(self, tool: str) -> CompensatingActivity | None:
        """The reversal paired with `tool`, or `None` where nothing takes it back."""
        return self._by_tool.get(tool)

    def irreversible(self, tool: str) -> bool:
        """Whether `tool` is declared as something no reversal can undo."""
        reversal = self._by_tool.get(tool)
        return reversal is not None and reversal.irreversible

    @property
    def tools(self) -> Iterable[str]:
        """Every forward tool something is paired with."""
        return tuple(self._by_tool)

    @property
    def mapping(self) -> Mapping[str, CompensatingActivity]:
        """The pairs, as the runtime `Compensator` wants them."""
        return dict(self._by_tool)


class Saga:
    """A scope whose applied work comes back off when something inside it fails.

    Args:
        run_id: The run the applied work belongs to.
        tenant: Its isolation boundary. Nothing crosses it, in either direction.
        ledger: Where applied actions are written down.
        compensations: The reversal paired with each forward tool.
        idempotency: Where a call whose result never came back is queried before the
            unwind decides anything.
        clock: What stamps each applied action.
        attempts: How many times one reversal is tried before it is called failed.

    Example:
        >>> from tesserix_adk.runtime import MemoryCompensationLedger
        >>> saga = Saga("r1", tenant="acme", ledger=MemoryCompensationLedger())
        >>> saga.outcome is None
        True
    """

    def __init__(
        self,
        run_id: str,
        *,
        tenant: str,
        ledger: CompensationLedger,
        compensations: Compensations | None = None,
        idempotency: IdempotencyStore | None = None,
        clock: Clock | None = None,
        attempts: int = 3,
    ) -> None:
        self.run_id = run_id
        self.tenant = tenant
        self.ledger = ledger
        self.compensations = compensations or Compensations()
        self.outcome: CompensatedFailure | CompensationIncomplete | None = None
        self._clock = clock
        self._step = 0
        self._compensator = Compensator(
            ledger,
            reversals=self.compensations.mapping,
            idempotency=idempotency,
            clock=clock,
            attempts=attempts,
        )

    async def __aenter__(self) -> Saga:
        """Enter the scope. Nothing is applied yet, so there is nothing to take back."""
        return self

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Unwind on the way out, then let the original failure carry on to the caller."""
        if error is not None:
            await self.unwind(str(error) or type(error).__name__)
        return False

    async def apply(
        self,
        tool: str,
        *,
        key: str,
        call: Callable[[], Awaitable[str]],
        arguments: Mapping[str, JsonValue] | None = None,
        branch: str = "",
        approved: bool = False,
    ) -> str:
        """Write the step down, run it, and record what came back.

        The record goes in first. A crash between the write and the result leaves an
        action whose outcome is unknown, which is a thing the unwind can query; leaving
        no record at all leaves a hotel room nobody knows is booked.

        Args:
            tool: The forward tool being applied.
            key: The idempotency key the call is made under.
            call: The forward call itself.
            arguments: What it was called with. Only a digest is kept.
            branch: The fan-out branch this belongs to.
            approved: That a person or an autonomy grant allowed an irreversible step.

        Returns:
            Whatever `call` returned.

        Raises:
            IrreversibleActionError: Where `tool` is declared irreversible and nothing
                approved it. Raised before the call, so the money has not moved.
        """
        irreversible = self.compensations.irreversible(tool)
        if irreversible and not approved:
            raise IrreversibleActionError(
                f"{tool} cannot be taken back and was not approved",
                run_id=self.run_id,
                tool=tool,
                idempotency_key=key,
            )
        self._step += 1
        action = Applied(
            run_id=self.run_id,
            tenant=self.tenant,
            branch=branch,
            step=self._step,
            tool=tool,
            idempotency_key=key,
            arguments_digest=arguments_digest(arguments or {}),
            irreversible=irreversible,
            applied_at=self._clock.now() if self._clock else 0.0,
        )
        await self.ledger.record(action)
        result = await call()
        await self.ledger.record(action.model_copy(update={"result_ref": result}))
        return result

    async def unwind(
        self, cause: str, *, branch: str | None = None
    ) -> CompensatedFailure | CompensationIncomplete:
        """Take back what this run applied, and keep the outcome where a caller can read it."""
        self.outcome = await self._compensator.unwind(
            self.run_id, tenant=self.tenant, cause=cause, branch=branch
        )
        return self.outcome
