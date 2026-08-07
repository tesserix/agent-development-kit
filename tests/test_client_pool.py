"""Connections are opened once and reused, and somebody owns closing them.

A client built per run pays a TLS handshake and a DNS lookup for every agent turn, and
under load the sockets outlive the runs that opened them. So clients are keyed, shared and
retired on purpose: two runs against one endpoint with one credential share a pool, two
tenants with different credentials never do, and a rotated key retires the old pool without
cutting off what is still in flight.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from tesserix_adk.core import Message, ModelRequest, TextPart
from tesserix_adk.core.errors import PoolExhaustedError, ProviderTimeoutError
from tesserix_adk.models.pool import ClientPool, PoolConfig
from tesserix_adk.models.providers import OpenAIProvider
from tesserix_adk.testing import FakeClock, HttpCassette, HttpReplay
from tesserix_adk.testing.http_cassette import HttpExchange

OPENAI = "https://api.openai.com/v1"


def transport(handler: Any = None) -> httpx.MockTransport:
    """A transport that answers everything, so nothing reaches the network."""
    return httpx.MockTransport(handler or (lambda _: httpx.Response(200, json={"ok": True})))


def pool(**overrides: Any) -> ClientPool:
    fields: dict[str, Any] = {"config": PoolConfig(), "transport": transport()}
    return ClientPool(**{**fields, **overrides})


class TestReuseAcrossRuns:
    """A hundred runs against one endpoint open one pool, not a hundred."""

    async def test_the_same_key_returns_the_same_client(self) -> None:
        async with pool() as clients:
            first = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            again = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

            assert first is again
            assert clients.metrics.opened == 1

    async def test_a_hundred_runs_open_one_pool(self) -> None:
        async with pool() as clients:
            for _ in range(100):
                clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

            assert clients.metrics.opened == 1
            assert clients.metrics.reused == 99

    async def test_a_different_endpoint_is_a_different_pool(self) -> None:
        async with pool() as clients:
            here = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            there = clients._client(
                provider="openai", base_url="https://eu.openai.example/v1", credential="sk-1"
            )

            assert here is not there
            assert clients.metrics.opened == 2

    async def test_different_transport_settings_are_different_pools(self) -> None:
        async with pool() as clients:
            quick = clients._client(
                provider="openai", base_url=OPENAI, credential="sk-1", timeout_seconds=5.0
            )
            patient = clients._client(
                provider="openai", base_url=OPENAI, credential="sk-1", timeout_seconds=600.0
            )

            assert quick is not patient


class TestTenantIsolation:
    """Two credentials against one endpoint are two pools, and the key never records one."""

    async def test_two_credentials_never_share_a_client(self) -> None:
        async with pool() as clients:
            theirs = clients._client(provider="openai", base_url=OPENAI, credential="sk-acme")
            ours = clients._client(provider="openai", base_url=OPENAI, credential="sk-globex")

            assert theirs is not ours

    async def test_the_key_holds_a_digest_rather_than_the_secret(self) -> None:
        async with pool() as clients:
            clients._client(provider="openai", base_url=OPENAI, credential="sk-super-secret")

            assert all("sk-super-secret" not in repr(key) for key in clients.keys)

    async def test_a_pooled_client_carries_no_per_run_state(self) -> None:
        async with pool() as clients:
            client = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

            assert "authorization" not in {name.lower() for name in client.headers}
            assert "x-tenant" not in {name.lower() for name in client.headers}


class TestCredentialRotation:
    """A rotated key is a new client, and the old one finishes what it started."""

    async def test_a_rotated_credential_opens_a_new_client(self) -> None:
        async with pool() as clients:
            before = clients._client(provider="openai", base_url=OPENAI, credential="sk-first")
            after = clients._client(provider="openai", base_url=OPENAI, credential="sk-second")

            assert before is not after
            assert clients.metrics.opened == 2

    async def test_the_retiring_pool_is_not_closed_under_an_in_flight_request(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def slow(_: httpx.Request) -> httpx.Response:
            started.set()
            await release.wait()
            return httpx.Response(200, json={"ok": True})

        async with pool(transport=httpx.MockTransport(slow)) as clients:
            old = clients._client(provider="openai", base_url=OPENAI, credential="sk-first")
            async with clients._borrowed_client(
                provider="openai", base_url=OPENAI, credential="sk-first"
            ) as borrowed:
                flight = asyncio.ensure_future(borrowed.get("/models"))
                await asyncio.wait_for(started.wait(), 2.0)
                await clients.retire(provider="openai", base_url=OPENAI, credential="sk-first")

                assert not old.is_closed
                release.set()
                assert (await asyncio.wait_for(flight, 2.0)).status_code == 200

            assert clients.metrics.retired == 1

    async def test_a_retired_key_is_reopened_rather_than_handed_back(self) -> None:
        async with pool() as clients:
            before = clients._client(provider="openai", base_url=OPENAI, credential="sk-first")
            await clients.retire(provider="openai", base_url=OPENAI, credential="sk-first")
            after = clients._client(provider="openai", base_url=OPENAI, credential="sk-first")

            assert after is not before
            assert before.is_closed


class TestExhaustion:
    """A full pool waits a bounded time and then says so, rather than growing."""

    async def test_a_pool_timeout_becomes_a_typed_retryable_error(self) -> None:
        def refuse(_: httpx.Request) -> httpx.Response:
            raise httpx.PoolTimeout("no free connection")

        async with pool(transport=httpx.MockTransport(refuse)) as clients:
            client = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            with pytest.raises(PoolExhaustedError) as raised:
                await clients._request(client, "GET", "/models", provider="openai")

            assert raised.value.retryable
            assert "openai" in str(raised.value)

    async def test_the_wait_is_bounded_by_the_declared_acquisition_timeout(self) -> None:
        async with pool(config=PoolConfig(acquire_seconds=2.5)) as clients:
            client = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

            assert client.timeout.pool == pytest.approx(2.5)

    async def test_the_pool_never_grows_past_its_declared_ceiling(self) -> None:
        async with pool(config=PoolConfig(max_connections=4, max_keepalive=2)) as clients:
            clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

            assert clients.config_for("openai").max_connections == 4
            assert clients.config_for("openai").max_keepalive == 2

    async def test_exhaustion_is_counted_before_it_becomes_latency(self) -> None:
        def refuse(_: httpx.Request) -> httpx.Response:
            raise httpx.PoolTimeout("no free connection")

        async with pool(transport=httpx.MockTransport(refuse)) as clients:
            client = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            with pytest.raises(PoolExhaustedError):
                await clients._request(client, "GET", "/models", provider="openai")

            assert clients.metrics.exhaustions == 1


class TestPerProviderOverrides:
    """One vendor's limits are not every vendor's."""

    async def test_a_provider_override_replaces_the_default(self) -> None:
        config = PoolConfig(
            max_connections=100,
            per_provider={"anthropic": PoolConfig(max_connections=8, max_keepalive=4)},
        )
        async with pool(config=config) as clients:
            clients._client(
                provider="anthropic", base_url="https://api.anthropic.com", credential="k"
            )

            assert clients.config_for("anthropic").max_connections == 8
            assert clients.config_for("openai").max_connections == 100

    async def test_keep_alive_is_declared_rather_than_left_to_the_library(self) -> None:
        assert PoolConfig().keepalive_seconds > 0
        assert PoolConfig().max_keepalive <= PoolConfig().max_connections

    async def test_a_keepalive_longer_than_the_pool_allows_is_refused(self) -> None:
        with pytest.raises(ValueError, match="keepalive"):
            PoolConfig(max_connections=4, max_keepalive=8)


class TestLifecycle:
    """Somebody owns closing the pool, and a leak check says whether they did."""

    async def test_closing_the_registry_closes_every_client(self) -> None:
        clients = pool()
        opened = [
            clients._client(provider="openai", base_url=OPENAI, credential=f"sk-{n}")
            for n in range(3)
        ]
        await clients.aclose()

        assert all(client.is_closed for client in opened)
        assert clients.metrics.open_now == 0

    async def test_the_block_closes_what_it_opened(self) -> None:
        async with pool() as clients:
            client = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

        assert client.is_closed

    async def test_nothing_is_left_running_after_close(self) -> None:
        before = len(asyncio.all_tasks())
        async with pool() as clients:
            clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

        assert len(asyncio.all_tasks()) == before

    async def test_a_closed_registry_refuses_rather_than_handing_out_a_dead_client(self) -> None:
        clients = pool()
        await clients.aclose()
        with pytest.raises(RuntimeError, match="closed"):
            clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

    async def test_closing_twice_is_not_an_error(self) -> None:
        clients = pool()
        clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
        await clients.aclose()
        await clients.aclose()

        assert clients.metrics.open_now == 0


class TestInheritedPools:
    """A pool that came from another process is half-open, whatever its bookkeeping says."""

    async def test_a_client_from_another_process_is_never_handed_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        clients = pool()
        inherited = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
        monkeypatch.setattr("os.getpid", lambda: 999999)

        after_fork = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

        assert after_fork is not inherited
        assert clients.metrics.inherited == 1
        await clients.aclose()


class TestMetrics:
    """Exhaustion is visible before it is latency."""

    async def test_the_counters_read_as_a_snapshot(self) -> None:
        async with pool() as clients:
            clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            snapshot = clients.metrics

            assert (snapshot.opened, snapshot.reused, snapshot.open_now) == (1, 1, 1)
            assert snapshot.waited_seconds == pytest.approx(0.0)

    async def test_time_spent_waiting_for_a_connection_is_recorded(self) -> None:
        clock = FakeClock()
        slow = _slow_transport(clock, seconds=0.75)
        async with pool(transport=slow, clock=clock) as clients:
            client = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            await clients._request(client, "GET", "/models", provider="openai")

            assert clients.metrics.waited_seconds > 0

    async def test_a_snapshot_does_not_change_under_the_reader(self) -> None:
        async with pool() as clients:
            snapshot = clients.metrics
            clients._client(provider="openai", base_url=OPENAI, credential="sk-1")

            assert snapshot.opened == 0


def _slow_transport(clock: FakeClock, *, seconds: float) -> httpx.MockTransport:
    """A transport whose response costs `seconds` of the clock's time."""

    async def handler(_: httpx.Request) -> httpx.Response:
        clock.advance(seconds)
        return httpx.Response(200, json={"ok": True})

    return httpx.MockTransport(handler)


class Rotating:
    """A secret provider whose value a test rotates between calls."""

    def __init__(self, key: str = "sk-first") -> None:
        self.key = key

    def secret(self, name: str) -> str | None:
        return self.key if name == "OPENAI_API_KEY" else None


def answered() -> HttpCassette:
    return HttpCassette(
        provider="openai",
        exchanges=(
            HttpExchange(
                path="/v1/chat/completions",
                body={
                    "id": "chatcmpl_1",
                    "object": "chat.completion",
                    "model": "gpt-4o",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "it rained"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            ),
        )
        * 3,
    )


def saturated() -> httpx.MockTransport:
    """A transport standing in for a pool with nothing free."""

    def refuse(_: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("no connection came free")

    return httpx.MockTransport(refuse)


def asked() -> ModelRequest:
    return ModelRequest(
        model="gpt-4o",
        messages=(Message(role="user", content=[TextPart(text="did it rain")]),),
    )


class TestRefusedConfigurations:
    """A shape the transport would accept and behave strangely under is refused early."""

    def test_a_negative_ceiling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            PoolConfig(max_connections=-1, max_keepalive=-2)

    def test_a_wait_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must both be positive"):
            PoolConfig(acquire_seconds=0)


class TestNestedBorrows:
    """A borrow inside a borrow keeps the client until the outer one is done with it."""

    async def test_the_inner_borrow_does_not_release_the_client(self) -> None:
        async with pool() as clients:
            held = clients._client(provider="openai", base_url=OPENAI, credential="sk-1")
            async with clients._borrowed_client(
                provider="openai", base_url=OPENAI, credential="sk-1"
            ):
                async with clients._borrowed_client(
                    provider="openai", base_url=OPENAI, credential="sk-1"
                ):
                    pass
                await clients.retire(provider="openai", base_url=OPENAI, credential="sk-1")
                assert not held.is_closed

            assert held.is_closed

    async def test_a_borrow_of_a_client_nobody_retired_leaves_it_open(self) -> None:
        async with pool() as clients:
            async with clients._borrowed_client(
                provider="openai", base_url=OPENAI, credential="sk-1"
            ) as client:
                pass

            assert not client.is_closed


class TestRetiringWhatIsNotHeld:
    """Retiring a key the registry never opened is a no-op, not an error."""

    async def test_an_unknown_key_retires_quietly(self) -> None:
        async with pool() as clients:
            await clients.retire(provider="openai", base_url=OPENAI, credential="never-used")

            assert clients.metrics.retired == 0


class TestProvidersOnAPool:
    """A provider handed a pool borrows from it rather than opening its own."""

    async def test_two_providers_on_one_pool_share_a_client(self) -> None:
        replay = HttpReplay(answered(), expect_provider="openai")
        async with ClientPool(transport=replay.transport) as clients:
            first = OpenAIProvider("gpt-4o", secrets=Rotating(), pool=clients)
            second = OpenAIProvider("gpt-4o", secrets=Rotating(), pool=clients)
            await first.complete(asked())
            await second.complete(asked())

            assert clients.metrics.opened == 1
            assert clients.metrics.reused >= 1

    async def test_a_provider_on_a_pool_does_not_close_what_it_does_not_own(self) -> None:
        replay = HttpReplay(answered(), expect_provider="openai")
        async with ClientPool(transport=replay.transport) as clients:
            model = OpenAIProvider("gpt-4o", secrets=Rotating(), pool=clients)
            await model.complete(asked())
            await model.aclose()

            assert clients.metrics.open_now == 1
            assert (await model.complete(asked())).content == "it rained"

    async def test_a_rotated_key_retires_the_old_client(self) -> None:
        replay = HttpReplay(answered(), expect_provider="openai")
        secrets = Rotating()
        async with ClientPool(transport=replay.transport) as clients:
            model = OpenAIProvider("gpt-4o", secrets=secrets, pool=clients)
            await model.complete(asked())
            secrets.key = "sk-rotated"
            await model.complete(asked())

            assert clients.metrics.opened == 2
            assert clients.metrics.retired == 1
            assert clients.metrics.open_now == 1

    async def test_a_saturated_pool_reaches_the_caller_as_pool_exhaustion(self) -> None:
        async with ClientPool(transport=saturated()) as clients:
            model = OpenAIProvider("gpt-4o", secrets=Rotating(), pool=clients)
            with pytest.raises(PoolExhaustedError, match="every connection to openai"):
                await model.complete(asked())

            assert clients.metrics.exhaustions == 1

    async def test_a_saturated_pool_stops_a_stream_the_same_way(self) -> None:
        async with ClientPool(transport=saturated()) as clients:
            model = OpenAIProvider("gpt-4o", secrets=Rotating(), pool=clients)
            with pytest.raises(PoolExhaustedError, match="every connection to openai"):
                async for _ in await model.stream(asked()):
                    pass

            assert clients.metrics.exhaustions == 1

    async def test_a_pool_of_its_own_running_out_is_still_a_timeout(self) -> None:
        model = OpenAIProvider("gpt-4o", secrets=Rotating(), transport=saturated())
        with pytest.raises(ProviderTimeoutError) as refused:
            async for _ in await model.stream(asked()):
                pass
        await model.aclose()

        assert refused.value.details["phase"] == "pool"

    async def test_a_provider_without_a_pool_still_owns_its_own_client(self) -> None:
        replay = HttpReplay(answered(), expect_provider="openai")
        model = OpenAIProvider("gpt-4o", secrets=Rotating(), transport=replay.transport)
        await model.complete(asked())
        await model.aclose()

        assert model.client.is_closed
