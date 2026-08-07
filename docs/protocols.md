# Protocols

Every seam in the kit is a `typing.Protocol` in `tesserix_adk.core.protocols`. A
consumer can replace any row below without touching kit code, and can unit-test agent
logic with no network by using the fake.

## The table

| Protocol | Remit | Shipped implementations | Fake | Conformance suite |
|---|---|---|---|---|
| `Clock` | Time, injected so timeouts are testable | *(pending — `adapters`)* | `FakeClock` | `ClockConformance` |
| `ModelProvider` | Completions and streaming from one provider | `models` | `ScriptedProvider` | `ModelProviderConformance` |
| `ToolRegistry` | Declared tools and their invocation | *(pending — `tools`)* | *(pending)* | *(pending)* |
| `MemoryStore` | Durable key-scoped record storage | *(pending — `adapters`)* | `FakeMemoryStore` | `MemoryStoreConformance` |
| `Guardrail` | An inline check on the call path | *(pending — `guardrails`)* | *(pending)* | *(pending)* |
| `BudgetPolicy` | Spend and usage limits around metered calls | `RunBudget`, `UnlimitedBudget` | `FakeBudgetPolicy` | `BudgetPolicyConformance` |
| `Tracer` | Sideband spans and events | *(pending — `observability`)* | `FakeTracer` | `TracerConformance` |

Rows marked pending are owned by the epic named in the cell. A story that adds an
implementation adds its row in the same change.

## Three checks, three different things

Substitutability is not one property and cannot be verified in one place.

| Check | When | Catches |
|---|---|---|
| `mypy --strict` | Authoring | A signature that does not match the protocol |
| `verify_conformance` | Construction | A member that is absent or not callable |
| Conformance suite | Test run | Behaviour the type system cannot express |

The third is the one people skip, and it is the one that matters. Structural typing
cannot state that deleting an absent key is not an error, that `put` replaces rather
than merges, or that a tracer must not swallow the exception raised inside its span.
Those live in `tesserix_adk.testing` and are inherited, not re-derived:

```python
from tesserix_adk.testing import MemoryStoreConformance


class TestRedisStore(MemoryStoreConformance):
    def make_store(self):
        return RedisMemoryStore(url="redis://localhost")
```

## Rules

**Construction fails, not the first call.** `verify_conformance` runs when an agent
is assembled. An implementation missing a member added in a later minor release fails
immediately, naming every missing member at once, rather than degrading into an
ambiguous failure halfway through a run.

**`isinstance` is not enough.** `runtime_checkable` reports presence as a single
boolean and says nothing about which member is absent, so a one-line configuration
mistake becomes a search. It also does not check signatures at all.

**Sync implementations are adapted explicitly.** The kit never hides a blocking call
inside an async protocol method — one synchronous call stalls every coroutine in the
process. Wrap it in `asyncio.to_thread` at the adapter, where the cost is visible.

**No vendor types.** A protocol that accepts or returns a vendor type is not
substitutable, whatever its shape. Enforced by the `no-vendor-types-inward` contract
in `.importlinter`.

**Adding a member is a breaking change.** Every existing implementation stops
conforming. It follows the deprecation policy, and the conformance suite gains the
new case in the same change.
