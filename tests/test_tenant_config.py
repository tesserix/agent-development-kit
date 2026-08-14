"""What a tenant is allowed, resolved from configuration rather than from conditionals.

The failure these are about is a plan tier expressed as an `if` in agent code: it is in
one code path and not the next, nobody can read off what a tenant is actually permitted,
and the run that skips the branch spends against a ceiling that was never applied. So the
limits are data, they are resolved once at the boundary, and every path that could widen
them — an unknown tenant, an unreachable store, a stale cache — refuses instead.
"""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    Agent,
    CachingTenantConfig,
    ConfigurationError,
    FileTenantConfig,
    ModelCapabilities,
    Run,
    RunState,
    SecretRef,
    TenantConfig,
    TenantLimits,
    TenantPolicy,
    TenantUnconfiguredError,
    ToolCall,
    UnknownTenantError,
    Usage,
    current_policy,
    policy_here,
    resolve_tenant_policy,
    tenant_policy,
    tenant_scope,
)
from tesserix_adk.core.budget import BudgetLimits, BudgetScope
from tesserix_adk.core.errors import TenantLimitError, ToolNotPermittedError
from tesserix_adk.core.routing import CHEAP, REASONING
from tesserix_adk.runtime import AgentRunner, ModelResponse
from tesserix_adk.testing import FakeClock, FakeSecrets, FakeToolRegistry, ScriptedProvider

if TYPE_CHECKING:
    from pathlib import Path

_CALLING = ModelResponse(
    content="",
    tool_calls=(ToolCall(id="call_1", name="lookup", arguments={}),),
    usage=Usage(input_tokens=10, output_tokens=5),
)

CHEAP_TENANT = TenantConfig(
    tenant="acme",
    limits=TenantLimits(
        budget=BudgetLimits(max_cost=Decimal("1.00"), max_model_calls=5),
        models=frozenset({"gpt-4o-mini"}),
        task_class_models={"cheap": "gpt-4o-mini"},
        tools=frozenset({"search"}),
        memory_retention=timedelta(days=30),
        region="eu-west-1",
    ),
)
RICH_TENANT = TenantConfig(
    tenant="globex",
    limits=TenantLimits(
        budget=BudgetLimits(max_cost=Decimal("100.00")),
        models=frozenset({"gpt-4o-mini", "o3"}),
        task_class_models={"cheap": "gpt-4o-mini", "reasoning": "o3"},
    ),
)


class Configured:
    """A store holding the two tenants, which can be made to fail on demand."""

    def __init__(self, *configs: TenantConfig) -> None:
        self.known = {config.tenant: config for config in configs}
        self.unreachable = False
        self.reads = 0

    async def config_for(self, tenant: str) -> TenantConfig:
        self.reads += 1
        if self.unreachable:
            raise TenantUnconfiguredError("the configuration store is unreachable")
        try:
            return self.known[tenant]
        except KeyError:
            raise UnknownTenantError(f"{tenant!r} is not a configured tenant") from None


@pytest.fixture
def store() -> Configured:
    return Configured(CHEAP_TENANT, RICH_TENANT)


class TestTheSameCodeRunsDifferentlyPerTenant:
    """The primary scenario: one agent, two entitlements, no branch in between."""

    async def test_the_task_class_resolves_to_the_model_the_tenant_pays_for(
        self, store: Configured
    ) -> None:
        with tenant_scope("acme"):
            cheap = await resolve_tenant_policy(store)
        with tenant_scope("globex"):
            rich = await resolve_tenant_policy(store)
        assert cheap.model_for(REASONING) is None
        assert rich.model_for(REASONING) == "o3"

    async def test_a_model_outside_the_allowlist_is_refused_rather_than_charged(
        self, store: Configured
    ) -> None:
        """A ceiling catches the spend afterwards; the allowlist is what stops it."""
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
        with pytest.raises(TenantLimitError, match="o3"):
            policy.check_model("o3")

    async def test_a_tool_the_tenant_does_not_have_is_refused(self, store: Configured) -> None:
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
        policy.check_tool("search")
        with pytest.raises(ToolNotPermittedError, match="wire_transfer"):
            policy.check_tool("wire_transfer")

    async def test_the_tenant_ceiling_reaches_the_run_as_a_tenant_scoped_limit(
        self, store: Configured
    ) -> None:
        """Attributed to the tenant, so a breach says whose ceiling was reached."""
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
        scoped = policy.budget_scope()
        assert scoped is not None
        assert scoped.scope is BudgetScope.TENANT
        assert scoped.limits.max_cost == Decimal("1.00")

    async def test_a_tenant_with_no_ceiling_stated_contributes_none(self) -> None:
        """No ceiling from this layer is not the same as an unlimited one."""
        store = Configured(TenantConfig(tenant="acme", limits=TenantLimits()))
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
        assert policy.budget_scope() is None


class TestFailingClosed:
    """Every path that could end in a permissive default instead ends in a refusal."""

    async def test_an_unknown_tenant_is_rejected_rather_than_given_the_defaults(
        self, store: Configured
    ) -> None:
        with tenant_scope("nobody"), pytest.raises(UnknownTenantError, match="nobody"):
            await resolve_tenant_policy(store)

    async def test_an_unreachable_store_with_nothing_cached_refuses(
        self, store: Configured
    ) -> None:
        store.unreachable = True
        with tenant_scope("acme"), pytest.raises(TenantUnconfiguredError):
            await resolve_tenant_policy(CachingTenantConfig(store, clock=FakeClock()))

    async def test_an_unreachable_store_does_not_fall_back_to_an_expired_entry(
        self, store: Configured
    ) -> None:
        """A ceiling nobody can confirm is a ceiling that may have been lowered since."""
        clock = FakeClock()
        cached = CachingTenantConfig(store, ttl=timedelta(seconds=60), clock=clock)
        with tenant_scope("acme"):
            await resolve_tenant_policy(cached)
            clock.advance(61)
            store.unreachable = True
            with pytest.raises(TenantUnconfiguredError):
                await resolve_tenant_policy(cached)

    async def test_resolving_outside_a_tenant_context_refuses(self, store: Configured) -> None:
        """Resolving limits for nobody in particular is resolving nobody's limits."""
        from tesserix_adk.core import MissingTenantContextError

        with pytest.raises(MissingTenantContextError):
            await resolve_tenant_policy(store)

    async def test_a_region_outside_the_tenants_own_is_refused(self, store: Configured) -> None:
        """A residency setting nothing enforces is a residency setting nobody has."""
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
        policy.check_region("eu-west-1")
        with pytest.raises(TenantLimitError, match="us-east-1"):
            policy.check_region("us-east-1")

    async def test_a_tenant_that_pins_no_region_accepts_the_provider_it_is_given(
        self, store: Configured
    ) -> None:
        with tenant_scope("globex"):
            policy = await resolve_tenant_policy(store)
        policy.check_region("us-east-1")

    async def test_an_allowlist_nobody_stated_permits_rather_than_refuses_everything(
        self,
    ) -> None:
        """An unstated allowlist is an absent limit; an empty one is a stated refusal."""
        store = Configured(TenantConfig(tenant="acme", limits=TenantLimits()))
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
        policy.check_model("o3")
        policy.check_tool("wire_transfer")

    async def test_an_empty_allowlist_refuses_everything_it_covers(self) -> None:
        store = Configured(
            TenantConfig(tenant="acme", limits=TenantLimits(models=frozenset(), tools=frozenset()))
        )
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
        with pytest.raises(TenantLimitError):
            policy.check_model("gpt-4o-mini")
        with pytest.raises(ToolNotPermittedError):
            policy.check_tool("search")


class TestTheCache:
    """A limit change has to apply without a redeploy, and without flushing everyone."""

    async def test_a_second_resolution_within_the_window_does_not_reach_the_store(
        self, store: Configured
    ) -> None:
        cached = CachingTenantConfig(store, ttl=timedelta(seconds=60), clock=FakeClock())
        with tenant_scope("acme"):
            await resolve_tenant_policy(cached)
            await resolve_tenant_policy(cached)
        assert store.reads == 1

    async def test_the_entry_is_read_again_once_it_has_expired(self, store: Configured) -> None:
        clock = FakeClock()
        cached = CachingTenantConfig(store, ttl=timedelta(seconds=60), clock=clock)
        with tenant_scope("acme"):
            await resolve_tenant_policy(cached)
            clock.advance(61)
            await resolve_tenant_policy(cached)
        assert store.reads == 2

    async def test_invalidating_one_tenant_leaves_the_others_cached(
        self, store: Configured
    ) -> None:
        """One tenant's plan change is not an outage for everybody else's cache."""
        cached = CachingTenantConfig(store, ttl=timedelta(hours=1), clock=FakeClock())
        with tenant_scope("acme"):
            await resolve_tenant_policy(cached)
        with tenant_scope("globex"):
            await resolve_tenant_policy(cached)
        cached.invalidate("acme")
        with tenant_scope("globex"):
            await resolve_tenant_policy(cached)
        with tenant_scope("acme"):
            await resolve_tenant_policy(cached)
        assert store.reads == 3

    async def test_the_cache_does_not_grow_without_a_bound(self) -> None:
        """A tenant per key is a memory leak shaped like a customer list."""
        many = Configured(*(TenantConfig(tenant=f"t{n}", limits=TenantLimits()) for n in range(10)))
        cached = CachingTenantConfig(many, ttl=timedelta(hours=1), clock=FakeClock(), keep=3)
        for n in range(10):
            with tenant_scope(f"t{n}"):
                await resolve_tenant_policy(cached)
        assert len(cached) == 3

    async def test_a_refusal_is_not_cached_as_though_it_were_an_answer(
        self, store: Configured
    ) -> None:
        """Caching "unknown" makes a newly onboarded tenant unknown for the whole window."""
        cached = CachingTenantConfig(store, ttl=timedelta(hours=1), clock=FakeClock())
        with tenant_scope("nobody"):
            with pytest.raises(UnknownTenantError):
                await resolve_tenant_policy(cached)
            store.known["nobody"] = TenantConfig(tenant="nobody", limits=TenantLimits())
            await resolve_tenant_policy(cached)


class TestBindingThePolicyForTheRun:
    """Resolved once at the boundary, read everywhere below, unchanged underneath a run."""

    async def test_the_bound_policy_is_what_the_code_below_reads(self, store: Configured) -> None:
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
            with tenant_policy(policy):
                assert current_policy() is policy

    async def test_nothing_is_bound_outside_a_block(self) -> None:
        assert policy_here() is None

    async def test_reading_a_policy_where_none_is_bound_refuses(self) -> None:
        """A missing policy is not an unlimited one, so it cannot read as absence."""
        with pytest.raises(ConfigurationError):
            current_policy()

    async def test_a_limit_lowered_mid_block_does_not_reach_the_work_already_bound(
        self, store: Configured
    ) -> None:
        """Completed work is not retroactively over a ceiling that arrived after it."""
        cached = CachingTenantConfig(store, ttl=timedelta(hours=1), clock=FakeClock())
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(cached)
            with tenant_policy(policy):
                store.known["acme"] = CHEAP_TENANT.model_copy(
                    update={"limits": TenantLimits(models=frozenset())}
                )
                cached.invalidate("acme")
                current_policy().check_model("gpt-4o-mini")
                after = await resolve_tenant_policy(cached)
        with pytest.raises(TenantLimitError):
            after.check_model("gpt-4o-mini")

    async def test_the_block_restores_what_was_bound_before_it(self, store: Configured) -> None:
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
            with tenant_policy(policy):
                with tenant_policy(policy.model_copy()):
                    pass
                assert current_policy() is policy


class TestTheRunLoopHoldsARunToTheBoundPolicy:
    """The acceptance scenario: one agent, two plans, and no conditional in between."""

    @staticmethod
    def _runner() -> AgentRunner:
        provider = ScriptedProvider(
            _CALLING,
            _CALLING,
            ModelResponse(
                content="Kyoto, four nights.", usage=Usage(input_tokens=10, output_tokens=5)
            ),
            capabilities=ModelCapabilities(tool_calling=True, context_window_tokens=200_000),
        )
        return AgentRunner(
            provider=provider,
            clock=FakeClock(),
            tools=FakeToolRegistry({"lookup": lambda: {"trains": 4}}),
        )

    async def _run_for(self, tenant: str, store: Configured) -> Run:
        agent = Agent(
            name="planner",
            instructions="Plan trips.",
            free_text=True,
            model="scripted-1",
            tools=("lookup",),
        )
        with tenant_scope(tenant):
            policy = await resolve_tenant_policy(store)
            with tenant_policy(policy):
                return await self._runner().run(agent, "plan a trip", tenant=tenant)

    async def test_the_cheap_plan_halts_at_its_ceiling_and_the_richer_one_continues(self) -> None:
        store = Configured(
            TenantConfig(
                tenant="acme", limits=TenantLimits(budget=BudgetLimits(max_model_calls=1))
            ),
            TenantConfig(
                tenant="globex", limits=TenantLimits(budget=BudgetLimits(max_model_calls=5))
            ),
        )
        assert (await self._run_for("acme", store)).state is RunState.BUDGET_EXHAUSTED
        assert (await self._run_for("globex", store)).state is RunState.COMPLETED

    async def test_a_tenant_ceiling_cannot_widen_the_one_the_agent_states(self) -> None:
        """The tenant layer is highest everywhere except money, where the tightest wins."""
        store = Configured(
            TenantConfig(
                tenant="acme", limits=TenantLimits(budget=BudgetLimits(max_model_calls=99))
            )
        )
        agent = Agent(
            name="planner",
            instructions="Plan trips.",
            free_text=True,
            model="scripted-1",
            tools=("lookup",),
            budget=BudgetLimits(max_model_calls=1),
        )
        with tenant_scope("acme"):
            policy = await resolve_tenant_policy(store)
            with tenant_policy(policy):
                run = await self._runner().run(agent, "plan a trip", tenant="acme")
        assert run.state is RunState.BUDGET_EXHAUSTED


class TestSecretsInTenantConfiguration:
    """A tenant file is a file somebody commits; a literal in one is a leaked credential."""

    def test_a_secret_is_named_rather_than_carried(self) -> None:
        ref = SecretRef(name="ACME_PROVIDER_KEY")
        assert ref.resolve(FakeSecrets({"ACME_PROVIDER_KEY": "sk-real"})) == "sk-real"

    def test_a_named_secret_nobody_supplied_refuses_rather_than_reading_as_empty(self) -> None:
        with pytest.raises(ConfigurationError, match="ACME_PROVIDER_KEY"):
            SecretRef(name="ACME_PROVIDER_KEY").resolve(FakeSecrets({}))

    def test_the_reference_does_not_carry_the_value_into_a_log_line(self) -> None:
        ref = SecretRef(name="ACME_PROVIDER_KEY")
        ref.resolve(FakeSecrets({"ACME_PROVIDER_KEY": "sk-real"}))
        assert "sk-real" not in repr(ref)

    def test_a_secret_bearing_setting_will_not_take_a_literal(self) -> None:
        """The type is the enforcement: a string where a reference belongs is refused."""
        with pytest.raises(ValueError, match="secrets"):
            TenantConfig(tenant="acme", limits=TenantLimits(), secrets={"key": "sk-literal"})  # type: ignore[dict-item]


class TestTheFileBackedProvider:
    """The simplest deployment: a directory somebody edits, read one tenant at a time."""

    @pytest.fixture
    def directory(self, tmp_path: Path) -> Path:
        (tmp_path / "acme.toml").write_text(
            '[limits]\nmodels = ["gpt-4o-mini"]\nregion = "eu-west-1"\n'
            "[limits.budget]\nmax_cost = 1.0\n",
            encoding="utf-8",
        )
        return tmp_path

    async def test_a_tenant_is_read_from_its_own_file(self, directory: Path) -> None:
        config = await FileTenantConfig(directory).config_for("acme")
        assert config.limits.region == "eu-west-1"
        assert config.limits.budget is not None
        assert config.limits.budget.max_cost == Decimal("1.0")

    async def test_a_tenant_with_no_file_is_unknown_rather_than_default(
        self, directory: Path
    ) -> None:
        with pytest.raises(UnknownTenantError, match="globex"):
            await FileTenantConfig(directory).config_for("globex")

    async def test_a_tenant_name_cannot_reach_outside_the_directory(self, directory: Path) -> None:
        """The name arrives from a request header on somebody's deployment."""
        with pytest.raises(UnknownTenantError):
            await FileTenantConfig(directory).config_for("../../etc/passwd")

    async def test_a_file_that_does_not_parse_refuses_rather_than_resolving_to_nothing(
        self, directory: Path
    ) -> None:
        (directory / "broken.toml").write_text("limits = [unclosed\n", encoding="utf-8")
        with pytest.raises(TenantUnconfiguredError, match="broken"):
            await FileTenantConfig(directory).config_for("broken")

    async def test_the_whole_directory_is_not_read_to_answer_for_one_tenant(
        self, directory: Path
    ) -> None:
        """A tenant per file and a hundred thousand tenants is still one read."""
        for n in range(50):
            (directory / f"t{n}.toml").write_text("[limits]\n", encoding="utf-8")
        provider = FileTenantConfig(directory)
        await provider.config_for("acme")
        assert provider.files_read == 1


class TestThePolicyItself:
    """Constructed from a config, and narrow enough to be worth reading off a run."""

    def test_the_policy_names_the_tenant_it_is_for(self) -> None:
        policy = TenantPolicy.of(CHEAP_TENANT)
        assert policy.tenant == "acme"

    def test_the_retention_window_is_carried_for_the_stores_that_apply_it(self) -> None:
        assert TenantPolicy.of(CHEAP_TENANT).retention == timedelta(days=30)

    def test_a_task_class_the_tenant_has_not_mapped_resolves_to_nothing(self) -> None:
        assert TenantPolicy.of(RICH_TENANT).model_for("transcription") is None

    def test_a_mapped_model_outside_the_allowlist_is_a_configuration_error(self) -> None:
        """Two settings that contradict each other are caught where they are written."""
        with pytest.raises(ConfigurationError, match="o3"):
            TenantLimits(models=frozenset({"gpt-4o-mini"}), task_class_models={"reasoning": "o3"})

    def test_the_allowlisted_model_for_a_class_passes_its_own_check(self) -> None:
        policy = TenantPolicy.of(CHEAP_TENANT)
        chosen = policy.model_for(CHEAP)
        assert chosen is not None
        policy.check_model(chosen)
