# Compensation

An agent booked the hotel and the flight leg failed. There is no transaction to roll back:
the booking happened, in someone else's system, and the only way it comes off is a second
call that says so. This is what the kit gives you to make that second call happen, in the
right order, exactly once, and to say plainly when it did not.

## The three rules

1. **The record goes in before the call.** A step is written to the ledger, then attempted.
   A crash between the two leaves an action whose outcome is unknown — which is something
   the unwind can query. Recording afterwards leaves a hotel room nobody knows is booked.
2. **The unwind never asks a model anything.** It reads the ledger, walks it newest first,
   and calls the reversal paired with each tool. A rollback that depends on a sampled token
   is not a rollback.
3. **What cannot come back is refused, not apologised for.** A tool declared irreversible is
   refused *before* it runs unless something approved it, and is never auto-compensated
   afterwards. The run ends `CompensationIncomplete` and a person reconciles.

## Pairing the reversal with the forward action

Write them together. Two files drift.

```python
from tesserix_adk.core import Applied
from tesserix_adk.workflows import Compensations, compensating_activity

@compensating_activity("book_hotel")
async def cancel_hotel(action: Applied) -> str:
    """Give the room back."""
    return await hotels.cancel(action.idempotency_key)

@compensating_activity("charge_card", irreversible=True)
async def unwind_charge(action: Applied) -> str:
    """Never called automatically. Money does not come back on its own."""
    return "reconcile by hand"

compensations = Compensations(cancel_hotel, unwind_charge)
```

Two reversals claiming one tool is refused at construction. Which of them ran would
otherwise depend on registration order, and that is not a thing to discover mid-rollback.

## The scope

```python
from tesserix_adk.workflows import Saga

saga = Saga("run-91", tenant="acme", ledger=ledger, compensations=compensations)
async with saga:
    pnr = await saga.apply("book_hotel", key="run-91:hotel", call=book, branch="hotel")
    await saga.apply("book_flight", key="run-91:flight", call=fly, branch="flight")
```

If anything inside raises, the applied work comes back off and the original error carries on
to the caller — compensating is not handling. Afterwards `saga.outcome` holds the terminal
state:

| Outcome | Means |
|---|---|
| `None` | The scope finished. Nothing was unwound. |
| `CompensatedFailure` | It failed, and the world holds nothing of this run any more. |
| `CompensationIncomplete` | It failed and the world still holds something. **Alert-grade.** |

`CompensationIncomplete.outstanding` is the list a person works from. It is never empty —
the model refuses an incomplete state that names nothing outstanding, because that is
indistinguishable from success to everything downstream.

## What the ledger holds

`Applied` is a record, not a payload. It carries the run, the tenant, the branch, the step,
the tool, the idempotency key, a **digest** of the arguments, and a reference to the result.
Arguments themselves are never stored: the digest is order-independent and a card number
that went into it cannot be read back out of it.

```python
class CompensationLedger(Protocol):
    async def record(self, action: Applied) -> None: ...
    async def mark(self, reversal: Reversal) -> None: ...
    async def outstanding(self, run_id, *, tenant, branch=None) -> Sequence[Applied]: ...
    async def forget(self, *, tenant: str) -> int: ...
```

`record` is keyed on `(tenant, run_id, idempotency_key)`, so a write retried after a crash
records one action. `forget` is the erasure path and returns how many records went.
`MemoryCompensationLedger` implements this for tests and single-process consumers; anything
whose worker can be replaced mid-run needs a durable one, since what is not written down
survives nothing.

## The action nobody can decide

A call whose result never came back is neither applied nor not applied. The engine does not
guess in either direction:

- Give the `Compensator` (or the `Saga`) an `IdempotencyStore` and the unknown outcome is
  **queried** — where the store recorded a result, the action is reversed like any other.
- With no store, or where the store has nothing, the action is left `UNKNOWN` and stays
  outstanding. The run ends incomplete.

## Cancellation

The whole walk runs under `asyncio.shield`. A cancellation arriving mid-unwind does not
abandon it, because half a rollback is worse than none: what is left applied is no longer on
any list a person is looking at.

## Fan-out

Pass `branch=` to `apply` and each leg keeps its own sequence. `saga.unwind(cause,
branch="flight")` takes back one branch and leaves its sibling standing. With no branch,
everything the run applied comes off.

## Retries

One reversal is tried `attempts` times (three by default) before it is called `FAILED`. A
supplier being down is never a reason to stop the walk — the other three bookings still come
off, and the failed one is named in `outstanding`.

See [`examples/compensation.py`](../examples/compensation.py) for a runnable version of all
of this.
