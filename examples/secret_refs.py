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
        print(f"  revealed only where asked: {held.get_secret_value()}")  # noqa: T201

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
    print(f"the linter finds: {literal_credentials(leaky)}")  # noqa: T201
    print(f"a SecretStr renders as: {SecretStr('sk-test-oops')}")  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
