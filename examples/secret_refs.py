"""Configuration holding references, resolved at the point of use and cached.

Run it with `uv run python examples/secret_refs.py`.
"""

from __future__ import annotations

import asyncio

from pydantic import SecretStr

from tesserix_adk.core import (
    AdkModel,
    CachingSecrets,
    ChainedSecrets,
    EnvironmentSecrets,
    SecretRef,
    SecretResolutionError,
    literal_credentials,
)
from tesserix_adk.testing import FakeClock

ENVIRONMENT = {"ACME_OPENAI_KEY": "sk-test-acme", "GLOBEX_OPENAI_KEY": "sk-test-globex"}


class ProviderSettings(AdkModel):
    """What a deployment writes down: an endpoint, and where the key lives."""

    endpoint: str
    api_key_ref: SecretRef


class Leaky(AdkModel):
    """What a deployment writes down by mistake."""

    endpoint: str
    api_key: str = ""


async def main() -> None:
    """Show what config holds, what a refusal says, and what the ttl does."""
    settings = ProviderSettings(
        endpoint="https://api.example.invalid",
        api_key_ref=SecretRef(name="{tenant}-openai-key"),
    )
    print(f"config, safe to log: {settings.model_dump_json()}")  # noqa: T201

    clock = FakeClock()
    secrets = CachingSecrets(
        ChainedSecrets((EnvironmentSecrets(environ=ENVIRONMENT),)),
        clock=clock,
        ttl_seconds=300.0,
    )

    for tenant in ("acme", "globex"):
        held = await secrets.resolve(settings.api_key_ref.for_tenant(tenant))
        print(f"{tenant} resolves to: {held}")  # noqa: T201
        expected = ENVIRONMENT[f"{tenant.upper()}_OPENAI_KEY"]
        delivered = held.get_secret_value() == expected
        print(f"  delivered only at provider boundary: {delivered}")  # noqa: T201

    print(f"cached references: {secrets.cached}")  # noqa: T201

    try:
        await secrets.resolve(settings.api_key_ref.for_tenant("initech"))
    except SecretResolutionError as refused:
        print(f"a tenant with no key: {refused}")  # noqa: T201

    acme = settings.api_key_ref.for_tenant("acme")
    try:
        acme.for_tenant("globex")
    except SecretResolutionError as refused:
        print(f"rebinding a bound reference: {refused}")  # noqa: T201

    leaky = Leaky(endpoint="https://api.example.invalid", api_key="sk-test-oops")
    if literal_credentials(leaky) != ("api_key",):
        raise RuntimeError("literal credential detection contract failed")
    print("the linter finds the literal credential field: api_key")  # noqa: T201

    masked = SecretStr("sk-test-oops")
    if str(masked) == masked.get_secret_value():
        raise RuntimeError("SecretStr rendering contract failed")
    print("a SecretStr masks its value when rendered")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
