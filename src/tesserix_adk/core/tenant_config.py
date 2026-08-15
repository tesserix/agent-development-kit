"""What a tenant is permitted, as configuration rather than as a branch in agent code.

A plan tier written as an `if` inside an agent is in one code path and not the next. The
one it misses is a background task or a second entry point, and the symptom is a run on
the cheap plan reaching the reasoning model with nobody able to say afterwards what that
tenant was actually allowed.

So the entitlement is data. It is resolved once, at the boundary that already knows whose
work this is, and bound for the run — `TenantPolicy` is what everything below reads. It
sits above the deployment's own configuration, `adk.toml` and `TESSERIX_ADK_*` included,
because it is the narrowest thing that knows about this run.

Budget is the exception to "the narrowest wins", and deliberately: a tenant ceiling
arrives as one more `ScopedLimits` and `most_restrictive` decides, so a tenant
configuration cannot widen a ceiling the deployment already set. Everything else — the
model allowlist, the tool allowlist, the region, the retention window — the tenant layer
states outright.

Every path that could end in a permissive default ends in a refusal instead: an unknown
tenant, an unreachable store, an entry too old to trust. A limits system that keeps
running when it cannot read the limits is not a limits system.
"""

from __future__ import annotations

import contextvars
import time
import tomllib
from collections import OrderedDict
from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from pydantic import Field, model_validator

from tesserix_adk.core.budget import BudgetLimits, BudgetScope, ScopedLimits
from tesserix_adk.core.errors import (
    ConfigurationError,
    SecretResolutionError,
    TenantLimitError,
    TenantUnconfiguredError,
    ToolNotPermittedError,
    UnknownTenantError,
)
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.tenancy import current_tenant

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tesserix_adk.core.protocols import Clock, SecretProvider

__all__ = [
    "CachingTenantConfig",
    "FileTenantConfig",
    "SecretRef",
    "TenantConfig",
    "TenantConfigProvider",
    "TenantLimits",
    "TenantPolicy",
    "current_policy",
    "policy_here",
    "resolve_tenant_policy",
    "tenant_policy",
]

_CURRENT: contextvars.ContextVar[TenantPolicy | None] = contextvars.ContextVar(
    "tesserix_adk_tenant_policy", default=None
)

# Long enough that a plan change is picked up within a support call, short enough that a
# store outage is not survived by a stale ceiling for the rest of the day.
DEFAULT_TTL = timedelta(minutes=5)

# A cache bounded by tenant count is a memory leak shaped like a customer list.
DEFAULT_KEEP = 1024


class SecretRef(AdkModel):
    """The name of a secret, never the secret.

    Tenant configuration is a file somebody commits and a row somebody dumps, so a
    credential in one is a credential in a backup. The reference names it and the
    `SecretProvider` in force where the run happens supplies the value.

    Args:
        name: What to ask the secret provider for. A `{tenant}` placeholder is filled by
            `for_tenant`, so one line of configuration serves every tenant.
        version: Which version to read, where the backend has versions. Absent means the
            backend's own notion of current, which is what rotation updates.
        tenant: Which tenant this reference was bound to by `for_tenant`. A reference
            already bound to one tenant cannot be rebound to another.

    Example:
        >>> SecretRef(name="ACME_PROVIDER_KEY").name
        'ACME_PROVIDER_KEY'
    """

    name: str = Field(min_length=1)
    version: str | None = None
    tenant: str | None = None

    def describe(self) -> str:
        """The reference as one string for a log line or a refusal. Never a value.

        Example:
            >>> SecretRef(name="openai-key").describe()
            'openai-key@latest'
        """
        return f"{self.name}@{self.version or 'latest'}"

    def for_tenant(self, tenant: str) -> Self:
        """The same reference bound to one tenant, filling a `{tenant}` placeholder.

        Args:
            tenant: Whose secret to read.

        Returns:
            A new reference. Nothing is mutated.

        Raises:
            SecretResolutionError: Where this reference is already bound to another
                tenant. A templated reference that can be rebound is one tenant's
                configuration reading another tenant's secret.

        Example:
            >>> SecretRef(name="{tenant}-openai-key").for_tenant("acme").name
            'acme-openai-key'
        """
        if self.tenant is not None and self.tenant != tenant:
            raise SecretResolutionError(
                f"{self.describe()} is bound to {self.tenant!r} and cannot be read for {tenant!r}",
                ref=self.describe(),
            )
        return self.model_copy(update={"name": self.name.format(tenant=tenant), "tenant": tenant})

    def resolve(self, secrets: SecretProvider) -> str:
        """The value behind this name.

        Args:
            secrets: Where the value actually lives.

        Returns:
            The secret.

        Raises:
            ConfigurationError: Where nothing is configured under the name. An unset
                secret reading as an empty string is an unauthenticated call to a
                provider that answers with something other than the obvious error.
        """
        value = secrets.secret(self.name)
        if value is None:
            raise ConfigurationError(
                f"tenant configuration names the secret {self.name!r} and nothing supplies it"
            )
        return value


class TenantLimits(AdkModel):
    """What one tenant may spend, reach and keep.

    Every field is optional, and an absent one means this layer states nothing rather
    than that it permits everything: an absent allowlist leaves the decision to whatever
    else is in force, while an empty one is a stated refusal of the lot.

    Args:
        budget: The tenant's ceiling. Applied as a `TENANT` scope, so it can narrow a
            run's limits but never widen them.
        models: The models this tenant may reach at all.
        task_class_models: Which model answers each task class for this tenant, so a
            plan tier is a routing table rather than a conditional.
        tools: The tools this tenant may call.
        memory_retention: How long this tenant's memory writes are kept.
        region: Where this tenant's data may be processed. A provider outside it is
            refused rather than preferred against.
    """

    budget: BudgetLimits | None = None
    models: frozenset[str] | None = None
    task_class_models: Mapping[str, str] = Field(default_factory=dict)
    tools: frozenset[str] | None = None
    memory_retention: timedelta | None = None
    region: str | None = None

    @model_validator(mode="after")
    def _the_routing_table_stays_inside_the_allowlist(self) -> Self:
        """Two settings that contradict each other are caught where they are written."""
        if self.models is None:
            return self
        outside = sorted(set(self.task_class_models.values()) - self.models)
        if outside:
            raise ConfigurationError(
                f"task classes route to {', '.join(outside)}, which the model allowlist excludes"
            )
        return self


class TenantConfig(AdkModel):
    """One tenant's entitlement, as it is stored and as it arrives.

    Args:
        tenant: Who this is for.
        limits: What they are permitted.
        secrets: Tenant-specific credentials, by name only. The type is the enforcement:
            a literal where a reference belongs does not validate.
    """

    tenant: str = Field(min_length=1)
    limits: TenantLimits
    secrets: Mapping[str, SecretRef] = Field(default_factory=dict)


@runtime_checkable
class TenantConfigProvider(Protocol):
    """Where tenant configuration is read from.

    One tenant per call rather than a bulk load, because a deployment with a hundred
    thousand tenants cannot hold them all and should not have to.
    """

    async def config_for(self, tenant: str) -> TenantConfig:
        """The configuration for one tenant.

        Args:
            tenant: Who to answer for.

        Returns:
            Their configuration.

        Raises:
            UnknownTenantError: Where the tenant is not configured. Never a default.
            TenantUnconfiguredError: Where the store could not be read.
        """
        ...


class TenantPolicy(AdkModel):
    """The resolved entitlement, bound for the work it applies to.

    Resolved once at the boundary and read below, so a limit lowered while a run is in
    flight applies to the next run rather than retroactively failing work that has
    already happened under the ceiling that was in force.

    Args:
        tenant: Who this is for.
        limits: What they are permitted.
    """

    tenant: str
    limits: TenantLimits

    @classmethod
    def of(cls, config: TenantConfig) -> Self:
        """The policy a configuration describes.

        Args:
            config: What was read from the store.

        Returns:
            The policy to bind.

        Example:
            >>> TenantPolicy.of(TenantConfig(tenant="acme", limits=TenantLimits())).tenant
            'acme'
        """
        return cls(tenant=config.tenant, limits=config.limits)

    @property
    def retention(self) -> timedelta | None:
        """How long this tenant's memory writes are kept, where they said."""
        return self.limits.memory_retention

    def budget_scope(self) -> ScopedLimits | None:
        """This tenant's ceiling, ready to resolve against the run's own.

        Returns:
            The ceiling attributed to `TENANT`, so a breach says whose it was, or None
            where the tenant states no ceiling — which is not the same as an unlimited
            one, and is why this is not a `BudgetLimits` with everything unset.
        """
        if self.limits.budget is None:
            return None
        return ScopedLimits(scope=BudgetScope.TENANT, limits=self.limits.budget)

    def model_for(self, task_class: str) -> str | None:
        """The model this tenant's plan puts behind a task class, where it names one."""
        return self.limits.task_class_models.get(task_class)

    def check_model(self, model: str) -> None:
        """Refuse a model this tenant may not reach.

        A ceiling catches the spend after the call; the allowlist is what stops the call.

        Args:
            model: What the router chose.

        Raises:
            TenantLimitError: Where the tenant states an allowlist and this is not on it.
        """
        if self.limits.models is not None and model not in self.limits.models:
            raise TenantLimitError(
                f"{self.tenant!r} is not permitted the model {model!r}",
                tenant=self.tenant,
                limit="models",
            )

    def check_tool(self, name: str) -> None:
        """Refuse a tool this tenant may not call.

        Args:
            name: The tool the model asked for.

        Raises:
            ToolNotPermittedError: Where the tenant states an allowlist and this is not
                on it. The kit's own tool refusal, so a caller already handling one does
                not need a second branch for the tenant case.
        """
        if self.limits.tools is not None and name not in self.limits.tools:
            raise ToolNotPermittedError(
                name, agent=self.tenant, permitted=sorted(self.limits.tools)
            )

    def check_region(self, region: str) -> None:
        """Refuse a provider outside where this tenant's data may be processed.

        Args:
            region: Where the work would happen.

        Raises:
            TenantLimitError: Where the tenant is pinned to a different region. A
                residency setting nothing enforces is a residency setting nobody has.
        """
        if self.limits.region is not None and region != self.limits.region:
            raise TenantLimitError(
                f"{self.tenant!r} is pinned to {self.limits.region!r} and this would run "
                f"in {region!r}",
                tenant=self.tenant,
                limit="region",
            )


class FileTenantConfig:
    """Tenant configuration as one TOML file per tenant in a directory.

    The simplest deployment there is: somebody edits a file, the next resolution reads
    it. One file per tenant rather than one file of tenants, so answering for a tenant
    costs one read no matter how many there are.

    Args:
        directory: Where the files live, named `<tenant>.toml`.
    """

    def __init__(self, directory: Path | str) -> None:
        self._directory = Path(directory)
        self.files_read = 0

    async def config_for(self, tenant: str) -> TenantConfig:
        """Read one tenant's file.

        Args:
            tenant: Who to answer for, which on somebody's deployment arrives from a
                request header — so it is checked against the directory rather than
                joined onto it.

        Returns:
            Their configuration.

        Raises:
            UnknownTenantError: Where there is no file for them.
            TenantUnconfiguredError: Where the file does not parse or does not validate.
        """
        path = (self._directory / f"{tenant}.toml").resolve()
        if not path.is_file() or path.parent != self._directory.resolve():
            raise UnknownTenantError(f"no configuration for the tenant {tenant!r}", tenant=tenant)
        self.files_read += 1
        try:
            payload = tomllib.loads(path.read_text(encoding="utf-8"))
            # TOML has no frozenset and no decimal, so strictness gives way at this edge.
            return TenantConfig.model_validate({**payload, "tenant": tenant}, strict=False)
        except (tomllib.TOMLDecodeError, ValueError) as err:
            raise TenantUnconfiguredError(
                f"the configuration for {tenant!r} could not be read: {err}", tenant=tenant
            ) from err


class CachingTenantConfig:
    """A provider in front of a provider, so a limit lookup is not a round trip per run.

    Bounded and per-tenant: one tenant's plan change invalidates one entry, and the
    number of entries is capped rather than following the customer list.

    An entry past its window is not used, even where the store behind is unreachable. A
    ceiling nobody can confirm is a ceiling that may have been lowered since, and serving
    it is the permissive path this module exists to close.

    Args:
        source: Where configuration actually comes from.
        ttl: How long an entry may be served for.
        clock: The source of time, injected so expiry is testable without sleeping.
        keep: How many tenants to hold. The least recently used goes first.
    """

    def __init__(
        self,
        source: TenantConfigProvider,
        *,
        ttl: timedelta = DEFAULT_TTL,
        clock: Clock | None = None,
        keep: int = DEFAULT_KEEP,
    ) -> None:
        self._source = source
        self._ttl = ttl.total_seconds()
        self._clock = clock
        self._keep = keep
        self._entries: OrderedDict[str, tuple[float, TenantConfig]] = OrderedDict()

    def __len__(self) -> int:
        """How many tenants are held, so a bound can be asserted rather than assumed."""
        return len(self._entries)

    async def config_for(self, tenant: str) -> TenantConfig:
        """The cached configuration, or a fresh read.

        Args:
            tenant: Who to answer for.

        Returns:
            Their configuration.

        Raises:
            UnknownTenantError: Where the tenant is not configured. Refusals are not
                cached: a newly onboarded tenant would be unknown for the whole window.
            TenantUnconfiguredError: Where the store could not be read and nothing
                unexpired is held.
        """
        now = self._clock.now() if self._clock is not None else time.monotonic()
        held = self._entries.get(tenant)
        if held is not None and now - held[0] < self._ttl:
            self._entries.move_to_end(tenant)
            return held[1]
        config = await self._source.config_for(tenant)
        self._entries[tenant] = (now, config)
        self._entries.move_to_end(tenant)
        while len(self._entries) > self._keep:
            self._entries.popitem(last=False)
        return config

    def invalidate(self, tenant: str) -> None:
        """Drop one tenant's entry, so a plan change applies without a redeploy.

        Args:
            tenant: Whose entry to drop. Only theirs — flushing the cache to apply one
                tenant's change is an outage for everybody else's.
        """
        self._entries.pop(tenant, None)


async def resolve_tenant_policy(source: TenantConfigProvider) -> TenantPolicy:
    """The policy for the tenant bound here.

    Args:
        source: Where configuration comes from.

    Returns:
        The resolved policy, bound to nothing — binding is `tenant_policy`, so the
        boundary that resolves decides how far the answer reaches.

    Raises:
        MissingTenantContextError: Where no tenant is bound. Resolving limits for nobody
            in particular is resolving nobody's limits.
        UnknownTenantError: Where the tenant is not configured.
        TenantUnconfiguredError: Where configuration could not be read.

    Example:
        >>> import asyncio
        >>> class OneTenant:
        ...     async def config_for(self, tenant: str) -> TenantConfig:
        ...         return TenantConfig(tenant=tenant, limits=TenantLimits())
        >>> from tesserix_adk.core.tenancy import tenant_scope
        >>> with tenant_scope("acme"):
        ...     asyncio.run(resolve_tenant_policy(OneTenant())).tenant
        'acme'
    """
    context = current_tenant(where="resolve_tenant_policy")
    return TenantPolicy.of(await source.config_for(context.tenant))


@contextmanager
def tenant_policy(policy: TenantPolicy) -> Iterator[TenantPolicy]:
    """Bind a resolved policy for the duration of the block.

    Args:
        policy: What everything below reads.

    Yields:
        The bound policy.

    Example:
        >>> config = TenantConfig(tenant="acme", limits=TenantLimits())
        >>> with tenant_policy(TenantPolicy.of(config)):
        ...     current_policy().tenant
        'acme'
    """
    token = _CURRENT.set(policy)
    try:
        yield policy
    finally:
        _CURRENT.reset(token)


def policy_here() -> TenantPolicy | None:
    """The policy bound here, or None outside one.

    For code that legitimately runs with no tenant entitlement in force — a health check,
    a corpus job — and has to branch on that rather than fail.

    Example:
        >>> policy_here() is None
        True
    """
    return _CURRENT.get()


def current_policy(*, where: str = "current_policy") -> TenantPolicy:
    """The policy bound here, or a refusal.

    Args:
        where: What was about to happen, so the refusal names the enforcement point.

    Returns:
        The policy established by the nearest enclosing `tenant_policy`.

    Raises:
        ConfigurationError: Where nothing is bound. A missing policy is not an unlimited
            one, so it does not read as absence.
    """
    here = _CURRENT.get()
    if here is None:
        raise ConfigurationError(
            f"{where} needs a resolved tenant policy and none is bound; resolve one at the "
            f"boundary that knows whose work this is"
        )
    return here
