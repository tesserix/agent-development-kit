# The spend ledger

A budget kept in a process is a budget per process. Run the same agent on eight replicas
with a 10.00 USD hourly tenant ceiling and the tenant can spend 80.00, because each pod
believes it has the whole allowance. Autoscaling makes that a certainty rather than a risk.

The ledger is where the window actually lives, so the ceiling means the same thing however
many pods are serving.

```python
from decimal import Decimal

from tesserix_adk.adapters import InMemoryLedger
from tesserix_adk.core import LedgerKey, Window, WindowKind

ledger = InMemoryLedger(clock=clock)
key = LedgerKey(tenant="acme", agent=None, window=Window(kind=WindowKind.ROLLING, seconds=3_600))

held = await ledger.reserve(key, Decimal("0.40"), ceiling=Decimal("10.00"))
answer = await agent.run(...)
await ledger.settle(held, answer.usage.cost.total)
```

Reserve before the spend, settle after it. Checking a ceiling after the money is gone is
reporting, not enforcement.

## Reserve, settle, release

| Call | What it does |
|---|---|
| `reserve(key, amount, *, ceiling, lease_seconds=300)` | Holds `amount`, atomically with the ceiling check. Raises `BudgetExceededError` if the hold would pass `ceiling`. |
| `settle(reservation, actual)` | Turns the hold into spend of `actual` and returns whatever was over-held. |
| `release(reservation)` | Gives the hold back unspent. |
| `record_progress(reservation, spent)` | Tells the ledger what a run has spent so far. |
| `read_window(key)` | Settled, reserved, and when the allowance returns. |
| `reconcile()` | Closes out holds whose lease has lapsed, and says how many. |
| `forget(tenant)` | Erases a tenant's records, returning the aggregate that was dropped. |

A reservation counts against the ceiling before it settles. Without that, every replica
reserves against the same empty window at the same moment and all of them are told yes.

Settling twice is refused. A retried settlement that double-counts is a ceiling that
quietly halves.

## Windows

A rolling window is the last `seconds`, moving continuously — no cliff, no thundering herd
on the hour. A calendar window is fixed buckets aligned to the epoch, which is what an
invoice period looks like.

```python
Window(kind=WindowKind.ROLLING, seconds=3_600)    # the last hour, always
Window(kind=WindowKind.CALENDAR, seconds=86_400)  # today, resetting at midnight UTC
```

Two things the window guarantees:

- **A stepped clock cannot open a second allowance.** Time inside the ledger is monotonic:
  it never reads earlier than it already has. An NTP correction backwards does not hand a
  tenant a fresh window.
- **A run crossing a boundary is not granted a fresh allowance mid-run.** The reservation
  was taken from the old window and is held until it settles.

## Keys carry identifiers, nothing else

`LedgerKey` is tenant, optional agent, and window. Ledger contents get read by operators
who were never cleared to read prompts, so a key carrying a user's question would put it
in front of them by accident.

A tenant or agent name containing the key separator (`:`) is rejected at construction —
one that contains it could name another tenant's window.

An agent-scoped window is a separate window from the tenant's. Spending 2.00 under
`("acme", "researcher")` does not show up when reading `("acme", None)`; enforce both if
you want both.

## Leases and reconciliation

A replica that dies holding a reservation must not hold a tenant's allowance until the
window rolls. Every hold carries a lease (`lease_seconds`, five minutes by default).

`reconcile()` sweeps lapsed leases. A hold that recorded progress settles against what it
admitted spending; one that recorded none is released. The window is then neither
permanently reduced by a dead replica nor credited with spend that did happen.

Call it from a small periodic task, not from the request path.

## Failing closed

Every method raises `BudgetUnavailableError` when the store cannot be reached. Carrying on
without the ledger is how one outage becomes an unbounded bill.

Degraded mode exists and is off:

```python
RedisLedger(client, clock=clock, degraded_allowed=True)
```

It is never inferred from a failure — a deployment decides in advance that an unreachable
ledger should not stop the service. Every hold it waves through is marked
`Reservation.degraded`, the ledger counts them in `degradations`, and a settlement against
one does not pretend to have been recorded. A bill can then be explained afterwards.

## Stores

| Store | For |
|---|---|
| `InMemoryLedger` | Local development, tests, and single-replica deployments that are honest about being single-replica. |
| `RedisLedger` | Shared ceiling with a Redis. One Lua script per operation. |
| `PostgresLedger` | Shared ceiling with the database you already run. One statement per operation. |
| `CoalescingLedger` | Wraps another ledger and buys allowance in blocks. |

`RedisLedger` takes anything with `eval`; `PostgresLedger` takes anything with `fetch`.
Neither `redis` nor a Postgres driver is a hard dependency of the kit.

The atomicity is structural. Each operation is one server-side script or one CTE
statement, because the ceiling check and the write it authorises cannot be two round trips
without a race between them.

`PostgresLedger.ensure_schema()` creates the table. Call it from a deployment step, never
on a request — a ledger that runs DDL while serving fails at the worst moment.

### Sharding

A busy tenant's writes contend on one counter. `shards=16` spreads them; reads sum every
shard, so the ceiling is unaffected by the choice. Pick it for write rate, not for
correctness.

### Coalescing

A shared ledger consulted on every model call adds its latency to every model call.

```python
ledger = CoalescingLedger(inner, block=Decimal("5.00"))
```

The block is *held* in the underlying ledger for its whole life, so no other replica can
spend it and the ceiling still holds. What the block does not use goes back on `flush()`.
Larger blocks mean fewer round trips and more allowance parked on one replica — which
matters when that replica dies, since the block waits for its lease.

Call `flush()` when a run finishes and on shutdown.

## Erasure

`forget(tenant)` deletes the tenant's records and returns the aggregate that went. What
remains after erasure is a number, not a history: enough to answer what the window held,
carrying nothing that names anybody.

## Deploying it on GKE

The library takes a client; it does not read your infrastructure. The pattern that works:

- **Credentials from Secret Manager, never from source.** An `ExternalSecret` syncs the
  Redis or Postgres URL into a `Secret`, which is mounted as an env var. The pod's
  Kubernetes service account is bound to a GCP service account by Workload Identity, so
  nothing carries a static key.

  ```yaml
  apiVersion: external-secrets.io/v1beta1
  kind: ExternalSecret
  metadata:
    name: adk-ledger
  spec:
    secretStoreRef: {name: gcpsm, kind: ClusterSecretStore}
    target: {name: adk-ledger}
    data:
      - secretKey: REDIS_URL
        remoteRef: {key: adk-ledger-redis-url}
  ```

- **Memory-only resource requests for the ledger client.** It holds connections and small
  counters, not case data or caches. Request memory and a modest CPU; do not give it
  ephemeral storage it will not use.
- **One reconciler, not one per replica.** Run `reconcile()` from a single `CronJob` or a
  leader-elected sidecar. Every replica sweeping is wasted round trips against the same
  keys.
- **Redis needs persistence or it needs to be small.** A Redis that loses its dataset
  loses the window, which reads as a fresh allowance. Either enable AOF, or keep windows
  short enough that the loss is bounded and accept it deliberately.

## Known limitations

- The Lua and the SQL run server-side and cannot be unit-tested without a server. The unit
  tests cover the translation — script chosen, keys, arguments, replies, refusals — and
  `SpendLedgerConformance` proves the semantics against a real store.
- `CoalescingLedger` wraps a local `InMemoryLedger`; the block sizing is a static choice,
  not adaptive to observed spend rate.
- `reconcile()` settles a lapsed hold against recorded progress. A run that spent money and
  recorded nothing before dying is under-counted by whatever it did not admit.
- Sharding uses the built-in hash, which is per-process salted for strings. Shard choice is
  therefore not stable across processes — reads sum all shards, so this is a distribution
  question rather than a correctness one.

## Conformance

A store you write yourself is held to the same suite:

```python
from tesserix_adk.testing import SpendLedgerConformance


class TestMyLedger(SpendLedgerConformance):
    def make_ledger(self) -> MyLedger:
        return MyLedger(...)
```

It covers the ceiling across concurrent writers, tenant isolation, reservation accounting,
double-settlement, and erasure.
