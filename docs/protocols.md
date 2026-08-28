# Protocols

Every seam in the kit is a `typing.Protocol` in `tesserix_adk.core.protocols`. A
consumer can replace any row below without touching kit code, and can unit-test agent
logic with no network by using the fake.

## The table

| Protocol | Remit | Shipped implementations | Fake | Conformance suite |
|---|---|---|---|---|
| `Clock` | Time, injected so timeouts are testable | `SystemClock` | `FakeClock` | `ClockConformance` |
| `ModelProvider` | Completions and streaming from one provider | Native and compatible providers in `models.providers` | `FakeModelProvider` | `ModelProviderConformance` |
| `ToolRegistry` | Declared tools and their invocation | `tesserix_adk.tools.ToolRegistry` | `FakeToolRegistry` | Runtime and registry tests |
| `KeyValueStore` | Durable key-scoped value storage | Consumer adapters | `FakeKeyValueStore` | `KeyValueStoreConformance` |
| `MemoryStore` | Working, profile, episodic and semantic memory under a scope — see [memory.md](memory.md) | `InMemoryMemoryStore` | `InMemoryMemoryStore` | `MemoryStoreConformance` |
| `ContradictionPolicy` | What an incoming profile record does to the live one — see [beliefs.md](beliefs.md) | `SupersedeMatching` | `SupersedeMatching` | *(covered by `MemoryStoreConformance`)* |
| `DecayPolicy` | How much of a record survives its age or its own uncertainty | `HalfLife`, `ConfidenceFloor` | `HalfLife` | *(covered by `MemoryStoreConformance`)* |
| `Guardrail` | An inline check on the call path | `Guard`, PII, injection, content, and output guards | `FakeGuardrail` | `GuardrailConformance` |
| `BudgetPolicy` | Spend and usage limits around metered calls | `RunBudget`, `UnlimitedBudget` | `FakeBudgetPolicy` | `BudgetPolicyConformance` |
| `Tracer` | Sideband spans and events | Consumer/OpenTelemetry bridge | `FakeTracer` | `TracerConformance` |

The wider package also defines focused protocols for secrets, idempotency, memory, state,
checkpoints, leases, work queues, events, peer transports, registries, and gateway
sessions. Use the protocol closest to the boundary rather than a generic plugin object.

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
from tesserix_adk.testing import KeyValueStoreConformance


class TestRedisStore(KeyValueStoreConformance):
    def make_store(self):
        return RedisKeyValueStore(url="redis://localhost")
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

## Known limitations

- Runtime conformance checks verify member presence, not signatures. Keep
  `mypy --strict` and the behavioral suite in the adapter project.
- Protocol implementations are wired explicitly; the base package does not auto-import
  third-party plugins. This keeps startup and the dependency graph deterministic.
- A Python `Agent.output_type` is not serialized as executable code.
  `AgentDefinition` stores its JSON Schema so the reviewed artifact keeps the contract.
- Adding a required member affects every external implementation and must follow the
  versioning policy.
