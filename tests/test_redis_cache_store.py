"""What the shared cache store puts on the wire, and what it refuses to put there.

A cache entry is a model's answer to a customer's question, so what a replica writes into
Redis is personal data sitting in somebody else's process. These assert the shape of the
key — which is what makes an erasure request a pattern rather than a scan of every value —
and that the model's private reasoning never leaves the process at all.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tesserix_adk.adapters.cache import RedisCacheStore
from tesserix_adk.core.primitives import Usage
from tesserix_adk.core.provider import ModelResponse
from tesserix_adk.models.cache import CacheEntry, CacheKey


class FakeRedis:
    """Records what was evaluated and answers with whatever the test says the server said."""

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, int, tuple[str, ...]]] = []
        self.fail: Exception | None = None

    async def eval(self, script: str, numkeys: int, *args: str) -> Any:
        if self.fail is not None:
            raise self.fail
        self.calls.append((script, numkeys, args))
        return self.replies.pop(0) if self.replies else None

    @property
    def keys(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[:numkeys]

    @property
    def argv(self) -> tuple[str, ...]:
        _, numkeys, args = self.calls[-1]
        return args[numkeys:]


def key(**overrides: str) -> CacheKey:
    """A key with every determinant filled, varied by whatever the test varies."""
    parts = {
        "tenant": "acme",
        "model": "gpt-4o",
        "prompt": "p" * 8,
        "tools": "t" * 8,
        "output_schema": "",
        "parameters": "x" * 8,
        "prompt_version": "v1",
        "model_version": "2026-05-01",
    }
    return CacheKey(**{**parts, **overrides})


def entry(**overrides: object) -> CacheEntry:
    """One stored answer, at a fixed time, with a one-minute life."""
    fields: dict[str, object] = {
        "key": key(),
        "response": ModelResponse(
            content="it rained",
            reasoning="thinking out loud",
            usage=Usage(input_tokens=10, output_tokens=4),
        ),
        "stored_at": 100.0,
        "expires_at": 160.0,
    }
    return CacheEntry(**{**fields, **overrides})  # type: ignore[arg-type]


class TestWriting:
    """What a put sends, and what it deliberately leaves behind."""

    async def test_the_key_carries_the_determinants_erasure_needs(self) -> None:
        client = FakeRedis()
        await RedisCacheStore(client).put(entry())

        assert client.keys == (f"adk:cache:acme:v1:2026-05-01:{key().digest}",)

    async def test_the_namespace_is_configurable(self) -> None:
        client = FakeRedis()
        await RedisCacheStore(client, namespace="courts").put(entry())

        assert client.keys[0].startswith("courts:acme:")

    async def test_the_ttl_is_what_the_entry_has_left(self) -> None:
        client = FakeRedis()
        await RedisCacheStore(client).put(entry())

        assert client.argv[1] == "60000"

    async def test_an_already_expired_entry_is_not_written(self) -> None:
        client = FakeRedis()
        await RedisCacheStore(client).put(entry(expires_at=100.0))

        assert client.calls == []

    async def test_the_models_reasoning_is_not_persisted(self) -> None:
        client = FakeRedis()
        await RedisCacheStore(client).put(entry())

        written = json.loads(client.argv[0])
        assert written["response"]["reasoning"] == ""

    async def test_reasoning_can_be_kept_where_a_deployment_has_decided_to(self) -> None:
        client = FakeRedis()
        await RedisCacheStore(client, redact_reasoning=False).put(entry())

        written = json.loads(client.argv[0])
        assert written["response"]["reasoning"] == "thinking out loud"


class TestReading:
    """What comes back, and what a missing or unreadable value does."""

    async def test_a_stored_entry_round_trips(self) -> None:
        client = FakeRedis()
        store = RedisCacheStore(client)
        await store.put(entry())
        client.replies.append(client.argv[0])

        read = await store.get(key().digest)

        assert read is not None
        assert read.response.content == "it rained"
        assert read.key == key()

    async def test_a_missing_key_is_none(self) -> None:
        store = RedisCacheStore(FakeRedis(None))

        assert await store.get(key().digest) is None

    async def test_a_lookup_matches_on_the_digest_alone(self) -> None:
        client = FakeRedis(None)
        await RedisCacheStore(client).get(key().digest)

        assert client.keys == ()
        assert client.argv == (f"adk:cache:*:*:*:{key().digest}",)

    async def test_an_unreadable_value_is_a_miss_rather_than_a_crash(self) -> None:
        store = RedisCacheStore(FakeRedis("{not json"))

        assert await store.get(key().digest) is None

    async def test_bytes_from_the_server_are_read(self) -> None:
        client = FakeRedis()
        store = RedisCacheStore(client)
        await store.put(entry())
        client.replies.append(client.argv[0].encode())

        read = await store.get(key().digest)

        assert read is not None
        assert read.response.content == "it rained"


class TestPurging:
    """Erasure is a key pattern, because scanning every value to find a tenant is not one."""

    async def test_a_tenant_purge_matches_only_that_tenant(self) -> None:
        client = FakeRedis(3)
        removed = await RedisCacheStore(client).purge(tenant="acme")

        assert client.argv == ("adk:cache:acme:*:*:*",)
        assert removed == 3

    async def test_a_prompt_version_purge_narrows_to_that_version(self) -> None:
        client = FakeRedis(1)
        await RedisCacheStore(client).purge(tenant="acme", prompt_version="v1")

        assert client.argv == ("adk:cache:acme:v1:*:*",)

    async def test_a_model_version_purge_narrows_to_that_build(self) -> None:
        client = FakeRedis(0)
        await RedisCacheStore(client).purge(model_version="2026-05-01")

        assert client.argv == ("adk:cache:*:*:2026-05-01:*",)

    async def test_a_digest_purge_names_one_entry(self) -> None:
        client = FakeRedis(1)
        await RedisCacheStore(client).purge(digest=key().digest)

        assert client.argv == (f"adk:cache:*:*:*:{key().digest}",)

    async def test_purging_nothing_in_particular_removes_everything(self) -> None:
        client = FakeRedis(9)
        removed = await RedisCacheStore(client).purge()

        assert client.argv == ("adk:cache:*:*:*:*",)
        assert removed == 9

    async def test_a_none_criterion_is_a_wildcard_not_a_literal(self) -> None:
        client = FakeRedis(0)
        await RedisCacheStore(client).purge(tenant="acme", prompt_version=None)

        assert client.argv == ("adk:cache:acme:*:*:*",)


class TestFailures:
    """A store outage is the caller's to degrade past, so it is raised rather than hidden."""

    async def test_a_failed_lookup_is_raised(self) -> None:
        client = FakeRedis()
        client.fail = ConnectionError("no route to host")

        with pytest.raises(ConnectionError):
            await RedisCacheStore(client).get(key().digest)

    async def test_a_failed_purge_is_raised(self) -> None:
        client = FakeRedis()
        client.fail = ConnectionError("no route to host")

        with pytest.raises(ConnectionError):
            await RedisCacheStore(client).purge(tenant="acme")


class TestRefusedConfigurations:
    """A key part that could collide with the separator is refused rather than encoded."""

    async def test_a_tenant_containing_the_separator_is_refused(self) -> None:
        with pytest.raises(ValueError, match="tenant"):
            await RedisCacheStore(FakeRedis()).put(entry(key=key(tenant="acme:globex")))

    async def test_a_version_containing_a_wildcard_is_refused(self) -> None:
        with pytest.raises(ValueError, match="prompt_version"):
            await RedisCacheStore(FakeRedis()).put(entry(key=key(prompt_version="v*")))
