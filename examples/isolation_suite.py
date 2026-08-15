"""Two tenants, one leaky cache, and the suite that catches it.

Run it with `uv run python examples/isolation_suite.py`.
"""

from __future__ import annotations

from tesserix_adk.core import current_tenant
from tesserix_adk.testing import (
    IsolationScenario,
    LeakReport,
    Observed,
    Step,
    Surface,
    assert_no_leak,
    interleaved,
)

SCENARIO = IsolationScenario.confusable("acme", "rival")


class SharedCache:
    """A cache keyed on the prompt alone, which is the bug."""

    def __init__(self, *, keyed_by_tenant: bool) -> None:
        self.keyed_by_tenant = keyed_by_tenant
        self.entries: dict[str, str] = {}

    def answer(self, prompt: str, body: str) -> str:
        """Serve a cached summary, or store this tenant's."""
        key = f"{current_tenant().tenant}:{prompt}" if self.keyed_by_tenant else prompt
        return self.entries.setdefault(key, body)


async def run_both(cache: SharedCache) -> dict[str, tuple[LeakReport, ...]]:
    """Drive both tenants through the cache, one step each, in turn."""

    async def run(tenant: str, step: Step) -> list[LeakReport]:
        seeded = SCENARIO.fixture(tenant)
        await step()
        answer = cache.answer(seeded.cache_input, seeded.documents[0].body)
        return [
            LeakReport.over(
                SCENARIO,
                tenant=tenant,
                observed=[Observed(Surface.OUTPUT, answer)],
                absent=[surface for surface in Surface if surface is not Surface.OUTPUT],
            )
        ]

    return await interleaved(SCENARIO, run)


async def main() -> None:
    """The same run against a cache that forgets the tenant, and one that does not."""
    for keyed_by_tenant in (False, True):
        reports = await run_both(SharedCache(keyed_by_tenant=keyed_by_tenant))
        verdicts = {
            tenant: "clean" if report.clean else report.describe()
            for tenant, (report, *_) in reports.items()
        }
        print(f"\ncache keyed by tenant={keyed_by_tenant}:")  # noqa: T201
        for tenant, verdict in verdicts.items():
            print(f"  {tenant}: {verdict}")  # noqa: T201

    try:
        assert_no_leak(SCENARIO, tenant="acme", observed=[Observed(Surface.OUTPUT, "clean")])
    except AssertionError as unread:
        print(f"\nnot a pass: {unread}")  # noqa: T201


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
