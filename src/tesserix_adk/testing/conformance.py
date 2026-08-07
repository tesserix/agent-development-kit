"""Shared conformance suites for the core protocols.

An implementation is substitutable only if it behaves the way the runtime assumes,
and structural typing cannot express "deleting an absent key is not an error". These
suites carry those assumptions. First- and third-party implementations subclass one,
supply the implementation under test, and inherit the whole suite:

```python
from tesserix_adk.testing import MemoryStoreConformance


class TestRedisStore(MemoryStoreConformance):
    def make_store(self):
        return RedisMemoryStore(url="redis://localhost")
```

Adding a member to a protocol means adding its case here in the same change, so
every implementation learns about it by failing rather than by drifting.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from decimal import Decimal

import pytest

from tesserix_adk.core.capabilities import Capability, ModelCapabilities
from tesserix_adk.core.errors import (
    BudgetExceededError,
    BudgetUnavailableError,
    CapabilityError,
)
from tesserix_adk.core.ledger import LedgerKey, SpendLedger, Window, WindowKind
from tesserix_adk.core.primitives import Message, TextPart, Usage
from tesserix_adk.core.protocols import (
    BudgetPolicy,
    Clock,
    MemoryStore,
    ModelProvider,
    Tracer,
    verify_conformance,
)
from tesserix_adk.core.provider import ModelRequest, ModelResponse

__all__ = [
    "BudgetPolicyConformance",
    "ClockConformance",
    "MemoryStoreConformance",
    "ModelProviderConformance",
    "TracerConformance",
]


def _request(*texts: str) -> ModelRequest:
    """The smallest request a provider can be asked to answer."""
    return ModelRequest(
        model="conformance",
        messages=tuple(Message(role="user", content=[TextPart(text=t)]) for t in texts),
    )


class ModelProviderConformance(ABC):
    """Behaviour every `ModelProvider` implementation must exhibit.

    A provider is substitutable only if the kit can read its limits before it calls, so
    most of this suite is about the capability record rather than the completion.
    """

    @abstractmethod
    def make_provider(self) -> ModelProvider:
        """Return a provider under test, able to answer several requests."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_provider(), ModelProvider)

    def test_it_is_named(self) -> None:
        name = self.make_provider().name
        assert isinstance(name, str)
        assert name

    def test_it_declares_what_it_can_do(self) -> None:
        assert isinstance(self.make_provider().capabilities, ModelCapabilities)

    def test_the_declaration_does_not_change_between_reads(self) -> None:
        """A record that varies is a record nothing can be checked against."""
        provider = self.make_provider()
        assert provider.capabilities == provider.capabilities

    async def test_it_answers_with_a_response(self) -> None:
        assert isinstance(await self.make_provider().complete(_request("hello")), ModelResponse)

    def test_it_counts_tokens_without_going_negative(self) -> None:
        assert self.make_provider().count_tokens(_request("hello").messages) >= 0

    def test_a_longer_prompt_does_not_count_for_less(self) -> None:
        provider = self.make_provider()
        short = provider.count_tokens(_request("hello").messages)
        long = provider.count_tokens(_request("hello", "hello again at some length").messages)
        assert long >= short

    async def test_streaming_it_never_declared_is_refused(self) -> None:
        """A provider that buffers one chunk and calls it a stream is worse than a refusal."""
        provider = self.make_provider()
        if provider.capabilities.supports(Capability.STREAMING):
            return
        with pytest.raises(CapabilityError):
            await provider.stream(_request("hello"))


class MemoryStoreConformance(ABC):
    """Behaviour every `MemoryStore` implementation must exhibit."""

    @abstractmethod
    def make_store(self) -> MemoryStore:
        """Return a fresh, empty store under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_store(), MemoryStore)

    async def test_get_returns_none_for_an_absent_key(self) -> None:
        assert await self.make_store().get("absent") is None

    async def test_put_then_get_round_trips(self) -> None:
        store = self.make_store()
        await store.put("k", {"v": 1})
        assert await store.get("k") == {"v": 1}

    async def test_put_replaces_rather_than_merges(self) -> None:
        store = self.make_store()
        await store.put("k", {"a": 1})
        await store.put("k", {"b": 2})
        assert await store.get("k") == {"b": 2}

    async def test_delete_removes_the_key(self) -> None:
        store = self.make_store()
        await store.put("k", "v")
        await store.delete("k")
        assert await store.get("k") is None

    async def test_deleting_an_absent_key_is_not_an_error(self) -> None:
        await self.make_store().delete("never-existed")

    async def test_keys_do_not_collide_across_distinct_values(self) -> None:
        store = self.make_store()
        await store.put("a", 1)
        await store.put("b", 2)
        assert (await store.get("a"), await store.get("b")) == (1, 2)


class ClockConformance(ABC):
    """Behaviour every `Clock` implementation must exhibit."""

    @abstractmethod
    def make_clock(self) -> Clock:
        """Return a fresh clock under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_clock(), Clock)

    def test_now_does_not_go_backwards(self) -> None:
        clock = self.make_clock()
        assert clock.now() <= clock.now()

    async def test_sleep_does_not_move_time_backwards(self) -> None:
        clock = self.make_clock()
        before = clock.now()
        await clock.sleep(0)
        assert clock.now() >= before


class BudgetPolicyConformance(ABC):
    """Behaviour every `BudgetPolicy` implementation must exhibit."""

    @abstractmethod
    def make_policy(self) -> BudgetPolicy:
        """Return a fresh policy under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_policy(), BudgetPolicy)

    async def test_a_reservation_within_budget_is_permitted(self) -> None:
        await self.make_policy().reserve(1)

    async def test_recording_after_reserving_does_not_raise(self) -> None:
        policy = self.make_policy()
        await policy.reserve(10)
        await policy.record(Usage(input_tokens=8, output_tokens=0))

    def test_a_ceiling_is_readable_off_the_policy(self) -> None:
        """A limit nobody can read afterwards is a limit nobody can audit."""
        assert self.make_policy().resolved.limits is not None

    def test_a_child_spends_what_the_parent_has_left(self) -> None:
        """A sub-agent handed a fresh allowance is a way to spend one ceiling twice."""
        policy = self.make_policy()
        assert policy.child().limits() == policy.limits()


class TracerConformance(ABC):
    """Behaviour every `Tracer` implementation must exhibit.

    The defining requirement is that tracing fails open: a collector outage must
    degrade observability, never stop a run.
    """

    @abstractmethod
    def make_tracer(self) -> Tracer:
        """Return a fresh tracer under test."""

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_tracer(), Tracer)

    def test_event_never_raises(self) -> None:
        self.make_tracer().event("anything", attribute=object())

    def test_span_yields_and_closes(self) -> None:
        with self.make_tracer().span("work", attribute=1):
            pass

    def test_span_does_not_swallow_the_bodys_exception(self) -> None:
        tracer = self.make_tracer()
        raised = False
        try:
            with tracer.span("work"):
                raise ValueError("from the body")
        except ValueError:
            raised = True
        assert raised, "a span must not swallow the exception its body raised"


class SpendLedgerConformance(ABC):
    """Behaviour every `SpendLedger` implementation must exhibit.

    A store that gets any of these wrong lets a tenant past its ceiling, which is the one
    thing the ledger exists to prevent. Structural typing cannot express "the ceiling check
    and the hold happen together", so it is asserted here and every implementation —
    in-process, Redis, PostgreSQL, or a deployment's own — inherits the suite.
    """

    ceiling = Decimal("10.00")

    @abstractmethod
    def make_ledger(self) -> SpendLedger:
        """Return a fresh, empty ledger under test."""

    def a_key(self, tenant: str = "conformance") -> LedgerKey:
        """A window to spend against. Rolling by default, because that is the harder one."""
        return LedgerKey(
            tenant=tenant, agent=None, window=Window(kind=WindowKind.ROLLING, seconds=3_600)
        )

    def test_satisfies_the_protocol(self) -> None:
        verify_conformance(self.make_ledger(), SpendLedger)

    async def test_settled_spend_is_visible_to_the_next_reader(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.settle(
            await ledger.reserve(key, Decimal("1.00"), ceiling=self.ceiling), Decimal("1.00")
        )
        assert (await ledger.read_window(key)).settled == Decimal("1.00")

    async def test_a_hold_counts_before_it_settles(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.reserve(key, Decimal("4.00"), ceiling=self.ceiling)
        assert (await ledger.read_window(key)).committed == Decimal("4.00")

    async def test_the_ceiling_refuses_rather_than_overshoots(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.reserve(key, Decimal("9.50"), ceiling=self.ceiling)
        with pytest.raises(BudgetExceededError):
            await ledger.reserve(key, Decimal("1.00"), ceiling=self.ceiling)

    async def test_concurrent_holds_do_not_pass_the_ceiling(self) -> None:
        """The reserve-and-check has to be one operation, whatever the store."""
        ledger, key = self.make_ledger(), self.a_key()
        granted = await asyncio.gather(
            *(self._try(ledger, key, Decimal("1.00")) for _ in range(40))
        )
        assert sum(granted) == 10

    async def test_a_release_gives_the_allowance_back(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.release(await ledger.reserve(key, Decimal("9.00"), ceiling=self.ceiling))
        assert (await ledger.read_window(key)).reserved == Decimal(0)

    async def test_settling_twice_is_refused(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        held = await ledger.reserve(key, Decimal("1.00"), ceiling=self.ceiling)
        await ledger.settle(held, Decimal("1.00"))
        with pytest.raises(BudgetUnavailableError):
            await ledger.settle(held, Decimal("1.00"))

    async def test_one_tenant_cannot_read_another_s_window(self) -> None:
        ledger = self.make_ledger()
        await ledger.settle(
            await ledger.reserve(self.a_key("one"), Decimal("2.00"), ceiling=self.ceiling),
            Decimal("2.00"),
        )
        assert (await ledger.read_window(self.a_key("two"))).settled == Decimal(0)

    async def test_erasure_leaves_nothing_of_the_tenant_behind(self) -> None:
        ledger, key = self.make_ledger(), self.a_key()
        await ledger.settle(
            await ledger.reserve(key, Decimal("2.00"), ceiling=self.ceiling), Decimal("2.00")
        )
        assert (await ledger.forget(key.tenant)).settled == Decimal("2.00")
        assert (await ledger.read_window(key)).settled == Decimal(0)

    async def _try(self, ledger: SpendLedger, key: LedgerKey, amount: Decimal) -> bool:
        try:
            held = await ledger.reserve(key, amount, ceiling=self.ceiling)
        except BudgetExceededError:
            return False
        await ledger.settle(held, amount)
        return True
