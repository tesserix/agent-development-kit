"""Proving isolation rather than assuming it.

Cross-tenant leakage is the failure nobody notices: the code looks right, the tests
pass because they use one tenant, and the defect only appears once two tenants hold
similar data. So the scenario here always has two tenants, their data is deliberately
confusable, and each tenant's data carries a marker that cannot be confused for the
other's — a sentinel found where it does not belong is a leak, not a heuristic.

Every payload is invented. The fixtures ship with the kit and get copied into consumer
repositories, so nothing here may be an extract of anything real.
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from tesserix_adk.core import tenant_scope

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

__all__ = [
    "CONFUSABLE_FIXTURES",
    "DEFAULT_TENANTS",
    "SENTINEL_KINDS",
    "IsolationScenario",
    "Leak",
    "LeakReport",
    "Observed",
    "SeededDocument",
    "Step",
    "Surface",
    "TenantFixture",
    "assert_no_leak",
    "interleaved",
    "sentinel_for",
]

DEFAULT_TENANTS = ("tenant-alpha", "tenant-beta")
"""The two tenants the shipped fixtures are seeded for."""

SENTINEL_KINDS = ("document", "profile", "cache")
"""What a sentinel can mark. Every kind is checked on every surface."""

type Step = Callable[[], Awaitable[None]]
"""Hand control to the other run. Awaiting it is where an interleaving happens."""


class Surface(StrEnum):
    """Somewhere a leak could show.

    A run is only proven isolated over the surfaces that were actually inspected, so
    these are enumerated rather than left to whatever the caller remembered to pass.
    """

    OUTPUT = "output"
    MEMORY = "memory"
    SEARCH = "search"
    CACHE = "cache"
    SPANS = "spans"
    EVENTS = "events"


def sentinel_for(tenant: str, kind: str) -> str:
    """The marker embedded in one tenant's data of one kind.

    Alphanumeric and lowercase, so it survives the mangling a summariser or a tokeniser
    does to it: a marker split on a hyphen is a marker the leak check no longer finds.
    """
    digest = hashlib.sha256(f"{tenant}/{kind}".encode()).hexdigest()
    return f"zq{digest[:12]}"


@dataclass(frozen=True, slots=True)
class SeededDocument:
    """One document a tenant holds, near-identical to the other tenant's.

    Args:
        title: The same across tenants on purpose, so a title match proves nothing.
        body: Differs only where the sentinel is, so a similarity search that forgets
            its filter ranks the neighbour's copy just as highly.
    """

    title: str
    body: str


@dataclass(frozen=True, slots=True)
class TenantFixture:
    """What one tenant has been seeded with.

    Args:
        tenant: Whose data this is.
        documents: The confusable documents.
        profile: Profile keys that collide with the other tenant's, holding values that
            do not.
        cache_input: The prompt both tenants send, so a cache keyed without the tenant
            produces a hit across them.
    """

    tenant: str
    documents: tuple[SeededDocument, ...]
    profile: Mapping[str, str]
    cache_input: str

    @property
    def sentinels(self) -> frozenset[str]:
        """Every marker belonging to this tenant."""
        return frozenset(sentinel_for(self.tenant, kind) for kind in SENTINEL_KINDS)


def _fixture(tenant: str) -> TenantFixture:
    """Seed one tenant with data built to be mistaken for the other tenant's."""
    document = sentinel_for(tenant, "document")
    profile = sentinel_for(tenant, "profile")
    return TenantFixture(
        tenant=tenant,
        documents=(
            SeededDocument(
                title="Q3 renewal terms",
                body=f"The Q3 renewal terms run to 24 months at the standard rate. {document}",
            ),
            SeededDocument(
                title="Escalation contacts",
                body=f"Escalations go to the duty rota, then to support@example.com. {document}",
            ),
        ),
        profile={"display_name": f"Renewals desk {profile}", "timezone": "Europe/London"},
        cache_input="summarise the Q3 renewal terms",
    )


CONFUSABLE_FIXTURES = tuple(_fixture(tenant) for tenant in DEFAULT_TENANTS)
"""The shipped pair, seeded for `DEFAULT_TENANTS`. Synthetic throughout."""


@dataclass(frozen=True, slots=True)
class IsolationScenario:
    """Two tenants, seeded with data each could be mistaken for the other's.

    Args:
        fixtures: One per tenant, keyed by tenant.
    """

    fixtures: Mapping[str, TenantFixture]

    @classmethod
    def confusable(cls, *tenants: str) -> IsolationScenario:
        """A scenario over the named tenants, or over `DEFAULT_TENANTS`.

        Raises:
            ValueError: Given fewer than two tenants. One tenant proves nothing, which
                is exactly why single-tenant suites miss this class of defect.
        """
        named = tenants or DEFAULT_TENANTS
        if len(named) < 2:
            raise ValueError("an isolation scenario needs at least two tenants to prove anything")
        return cls(fixtures={tenant: _fixture(tenant) for tenant in named})

    @property
    def tenants(self) -> tuple[str, ...]:
        """Who the scenario covers, in the order they were named."""
        return tuple(self.fixtures)

    def fixture(self, tenant: str) -> TenantFixture:
        """What one tenant was seeded with.

        Raises:
            KeyError: For a tenant the scenario does not cover. An empty fixture would
                let a suite report green over data nobody seeded.
        """
        if tenant not in self.fixtures:
            raise KeyError(f"{tenant!r} is not in this scenario; it covers {self.tenants}")
        return self.fixtures[tenant]

    def foreign_sentinels(self, tenant: str) -> Mapping[str, str]:
        """Every other tenant's markers, keyed by marker."""
        self.fixture(tenant)
        return {
            sentinel: other
            for other, fixture in self.fixtures.items()
            if other != tenant
            for sentinel in fixture.sentinels
        }


@dataclass(frozen=True, slots=True)
class Observed:
    """What one surface actually held after the run.

    Args:
        surface: Which surface this is.
        content: Everything the surface held, rendered as text. A structure is stringified
            by the caller, because a leak in a key is as much a leak as one in a value.
    """

    surface: Surface
    content: str


@dataclass(frozen=True, slots=True)
class Leak:
    """One tenant's marker found on another tenant's surface.

    Args:
        surface: Where it showed.
        sentinel: The marker found.
        owner: Whose data it is.
    """

    surface: Surface
    sentinel: str
    owner: str

    def describe(self) -> str:
        """The failure as a reader needs it: the surface, the marker and its owner."""
        return f"{self.surface} held {self.sentinel} which belongs to {self.owner}"


@dataclass(frozen=True, slots=True)
class LeakReport:
    """What the suite found, and what it was unable to look at.

    Args:
        tenant: Whose run was inspected.
        leaks: Every marker found where it did not belong, on every surface.
        inspected: The surfaces that were read.
        declared_absent: The surfaces the caller stated this deployment does not have.
    """

    tenant: str
    leaks: tuple[Leak, ...]
    inspected: tuple[Surface, ...]
    declared_absent: tuple[Surface, ...] = ()

    @classmethod
    def over(
        cls,
        scenario: IsolationScenario,
        *,
        tenant: str,
        observed: Iterable[Observed],
        absent: Iterable[Surface] = (),
    ) -> LeakReport:
        """Scan one run's surfaces for markers belonging to anyone else."""
        foreign = scenario.foreign_sentinels(tenant)
        seen = tuple(observed)
        leaks = tuple(
            Leak(surface=entry.surface, sentinel=sentinel, owner=owner)
            for entry in seen
            for sentinel, owner in foreign.items()
            if sentinel in entry.content
        )
        return cls(
            tenant=tenant,
            leaks=leaks,
            inspected=tuple(sorted({entry.surface for entry in seen})),
            declared_absent=tuple(sorted(set(absent))),
        )

    @property
    def missing(self) -> tuple[Surface, ...]:
        """Surfaces neither inspected nor declared absent."""
        accounted = set(self.inspected) | set(self.declared_absent)
        return tuple(surface for surface in Surface if surface not in accounted)

    @property
    def clean(self) -> bool:
        """Whether the run leaked nothing, over surfaces that were all accounted for.

        A report over surfaces nobody read is not clean. A suite that reports green
        because it never looked is worse than no suite.
        """
        return not self.leaks and not self.missing

    def describe(self) -> str:
        """Why this report is not clean, in the order a reader needs it."""
        parts = [leak.describe() for leak in self.leaks]
        if self.missing:
            missed = ", ".join(self.missing)
            parts.append(
                f"these surfaces were not inspected and not declared absent: {missed}; "
                f"a pass over them would be a pass nobody earned"
            )
        return "; ".join(parts)


def assert_no_leak(
    scenario: IsolationScenario,
    *,
    tenant: str,
    observed: Iterable[Observed],
    absent: Iterable[Surface] = (),
) -> LeakReport:
    """Assert one tenant's run shows nothing of anybody else's.

    Args:
        scenario: The seeded tenants.
        tenant: Whose run this was.
        observed: What each surface held.
        absent: Surfaces this deployment genuinely does not have. Declared rather than
            skipped, so the report says which guarantee it actually verified.

    Returns:
        The report, when it is clean.

    Raises:
        AssertionError: Naming every leaking surface and every surface nobody inspected.
    """
    report = LeakReport.over(scenario, tenant=tenant, observed=observed, absent=absent)
    if not report.clean:
        raise AssertionError(f"isolation not proven for {tenant}: {report.describe()}")
    return report


async def interleaved[ResultT](
    scenario: IsolationScenario,
    run: Callable[[str, Step], Awaitable[Sequence[ResultT] | None]],
) -> dict[str, tuple[ResultT, ...]]:
    """Run the scenario for every tenant at once, handing control over at each step.

    The interleaving is deterministic — each run proceeds exactly one step, then the next
    tenant does — so a context-bleed defect fails every time rather than occasionally.

    Args:
        scenario: The seeded tenants.
        run: The scenario body. It is called inside `tenant_scope`, and awaiting the
            `Step` it is handed passes control to the next tenant's run.

    Returns:
        Whatever each run returned, keyed by tenant.
    """
    order = scenario.tenants
    batons = {tenant: asyncio.Event() for tenant in order}
    results: dict[str, tuple[ResultT, ...]] = {}

    async def drive(index: int) -> None:
        tenant = order[index]
        following = order[(index + 1) % len(order)]

        async def step() -> None:
            batons[following].set()
            await batons[tenant].wait()
            batons[tenant].clear()

        await batons[tenant].wait()
        batons[tenant].clear()
        try:
            with tenant_scope(tenant):
                results[tenant] = tuple(await run(tenant, step) or ())
        finally:
            batons[following].set()

    tasks = [asyncio.ensure_future(drive(index)) for index in range(len(order))]
    batons[order[0]].set()
    await asyncio.gather(*tasks)
    return results
