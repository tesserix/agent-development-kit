"""Where a provider's connections live, and what may share one with what.

Two providers on one pool against the same endpoint share a warm client; a second
credential, a second endpoint or a second set of timeouts does not. A rotated key opens a
new pool while the requests already on the old one finish on it.

A mock transport stands in for the vendors, so nothing here reaches the network and no key
is needed. Run it with `python examples/connection_pooling.py`.
"""

from __future__ import annotations

import asyncio

import httpx

from tesserix_adk.core import Message, ModelRequest, TextPart
from tesserix_adk.core.errors import PoolExhaustedError
from tesserix_adk.models import ClientPool, PoolConfig
from tesserix_adk.models.providers import OpenAIProvider

ANSWERED = {
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
}


class Keyring:
    """A secret source an operator rotates under a running process."""

    def __init__(self, key: str) -> None:
        self.key = key

    def secret(self, name: str) -> str | None:
        """The current value of the vendor's key variable."""
        return self.key if name == "OPENAI_API_KEY" else None


def transport() -> httpx.MockTransport:
    """A transport that answers every request the same way."""
    return httpx.MockTransport(lambda _: httpx.Response(200, json=ANSWERED))


def asked() -> ModelRequest:
    """One trivial request, reused throughout."""
    return ModelRequest(
        model="gpt-4o", messages=(Message(role="user", content=[TextPart(text="did it rain")]),)
    )


async def two_providers_one_connection() -> None:
    """Same provider, endpoint, credential and timeouts: one client between them."""
    async with ClientPool(transport=transport()) as clients:
        first = OpenAIProvider("gpt-4o", secrets=Keyring("sk-acme"), pool=clients)
        second = OpenAIProvider("gpt-4o-mini", secrets=Keyring("sk-acme"), pool=clients)
        await first.complete(asked())
        await second.complete(asked())

        print("\ntwo providers against one endpoint")  # noqa: T201
        print(f"  opened: {clients.metrics.opened}, reused: {clients.metrics.reused}")  # noqa: T201


async def two_tenants_never_share() -> None:
    """A second credential is a second key, and a second key is a second pool."""
    async with ClientPool(transport=transport()) as clients:
        theirs = OpenAIProvider("gpt-4o", secrets=Keyring("sk-acme"), pool=clients)
        ours = OpenAIProvider("gpt-4o", secrets=Keyring("sk-globex"), pool=clients)
        await theirs.complete(asked())
        await ours.complete(asked())

        print("\ntwo tenants against one endpoint")  # noqa: T201
        print(f"  clients held: {len(clients.keys)}")  # noqa: T201
        print(f"  digests:      {[key.credential_digest for key in clients.keys]}")  # noqa: T201


async def a_rotation_retires_rather_than_closes() -> None:
    """The old client stops being handed out; what is already on it finishes on it."""
    keyring = Keyring("sk-first")
    async with ClientPool(transport=transport()) as clients:
        model = OpenAIProvider("gpt-4o", secrets=keyring, pool=clients)
        await model.complete(asked())
        keyring.key = "sk-rotated"
        await model.complete(asked())

        metrics = clients.metrics
        print("\nthe operator rotated the key mid-process")  # noqa: T201
        print(f"  opened: {metrics.opened}, retired: {metrics.retired}, held: {metrics.open_now}")  # noqa: T201


async def saturation_is_reported_not_queued() -> None:
    """A pool with nothing free fails inside its wait rather than past the deadline.

    The transport reports what a full pool reports, since a mock one has no sockets to run
    out of.
    """

    def nothing_free(_: httpx.Request) -> httpx.Response:
        raise httpx.PoolTimeout("no connection came free")

    config = PoolConfig(max_connections=1, max_keepalive=1, acquire_seconds=0.25)
    async with ClientPool(config, transport=httpx.MockTransport(nothing_free)) as clients:
        model = OpenAIProvider("gpt-4o", secrets=Keyring("sk-acme"), pool=clients)
        results = await asyncio.gather(
            *(model.complete(asked()) for _ in range(4)), return_exceptions=True
        )
        refused = [one for one in results if isinstance(one, PoolExhaustedError)]

        print("\nfour calls through a pool with nothing free")  # noqa: T201
        print(f"  refused:     {len(refused)} of {len(results)}")  # noqa: T201
        print(f"  retryable:   {refused[0].retryable}")  # noqa: T201
        print(f"  exhaustions: {clients.metrics.exhaustions}")  # noqa: T201


async def main() -> None:
    """Run every pattern."""
    await two_providers_one_connection()
    await two_tenants_never_share()
    await a_rotation_retires_rather_than_closes()
    await saturation_is_reported_not_queued()


if __name__ == "__main__":
    asyncio.run(main())
