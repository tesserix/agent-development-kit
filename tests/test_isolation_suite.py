"""Proving isolation rather than assuming it, with two tenants and a marker each."""

from __future__ import annotations

import asyncio

import pytest

from tesserix_adk.core import MissingTenantContextError, current_tenant, tenant_scope
from tesserix_adk.testing import (
    CONFUSABLE_FIXTURES,
    IsolationScenario,
    LeakReport,
    Observed,
    Step,
    Surface,
    assert_no_leak,
    interleaved,
    sentinel_for,
)

pytestmark = pytest.mark.anyio

ACME = "acme"
RIVAL = "rival"


def _scenario() -> IsolationScenario:
    return IsolationScenario.confusable(ACME, RIVAL)


class TestSentinels:
    def test_a_sentinel_is_unique_to_its_tenant_and_its_kind(self) -> None:
        assert sentinel_for(ACME, "document") != sentinel_for(RIVAL, "document")
        assert sentinel_for(ACME, "document") != sentinel_for(ACME, "profile")

    def test_a_sentinel_is_stable_so_a_failure_names_the_same_marker_twice(self) -> None:
        assert sentinel_for(ACME, "document") == sentinel_for(ACME, "document")

    def test_a_sentinel_survives_the_mangling_a_summary_does(self) -> None:
        """Lowercasing, splitting and rejoining must not lose the marker."""
        marker = sentinel_for(ACME, "document")
        summarised = " ".join(f"a {marker} b".upper().lower().split())
        assert marker in summarised

    def test_a_sentinel_is_one_token_a_tokeniser_will_not_split_on_punctuation(self) -> None:
        assert sentinel_for(ACME, "document").isalnum()


class TestTheFixtures:
    def test_the_two_tenants_hold_deliberately_confusable_data(self) -> None:
        scenario = _scenario()
        theirs = scenario.fixture(RIVAL)
        ours = scenario.fixture(ACME)
        assert ours.documents[0].body != theirs.documents[0].body
        assert ours.documents[0].title == theirs.documents[0].title

    def test_every_seeded_document_carries_its_tenants_sentinel(self) -> None:
        scenario = _scenario()
        for tenant in (ACME, RIVAL):
            for document in scenario.fixture(tenant).documents:
                assert sentinel_for(tenant, "document") in document.body

    def test_the_profile_keys_collide_across_tenants_on_purpose(self) -> None:
        scenario = _scenario()
        assert set(scenario.fixture(ACME).profile) == set(scenario.fixture(RIVAL).profile)

    def test_the_cache_inputs_collide_so_a_shared_key_would_show(self) -> None:
        scenario = _scenario()
        assert scenario.fixture(ACME).cache_input == scenario.fixture(RIVAL).cache_input

    def test_the_shipped_fixtures_carry_nothing_real(self) -> None:
        for fixture in CONFUSABLE_FIXTURES:
            for document in fixture.documents:
                assert "@" not in document.body or "example.com" in document.body

    def test_a_scenario_with_one_tenant_is_refused(self) -> None:
        """One tenant is exactly the suite that misses this class of defect."""
        with pytest.raises(ValueError, match="two tenants"):
            IsolationScenario.confusable(ACME)

    def test_a_tenant_the_scenario_does_not_cover_is_a_mistake_not_an_empty_fixture(self) -> None:
        with pytest.raises(KeyError, match="third"):
            _scenario().fixture("third")


class TestDetectingALeak:
    def test_a_clean_run_reports_no_leak(self) -> None:
        report = assert_no_leak(
            _scenario(),
            tenant=ACME,
            observed=[
                Observed(Surface.OUTPUT, f"{sentinel_for(ACME, 'document')} is ours"),
                Observed(Surface.MEMORY, "nothing of theirs"),
                Observed(Surface.SEARCH, ""),
                Observed(Surface.CACHE, ""),
                Observed(Surface.SPANS, ""),
                Observed(Surface.EVENTS, ""),
            ],
        )
        assert report.clean is True
        assert report.leaks == ()

    def test_a_neighbours_sentinel_in_the_output_fails_and_names_the_surface(self) -> None:
        with pytest.raises(AssertionError, match="output") as leaked:
            assert_no_leak(
                _scenario(),
                tenant=ACME,
                observed=[
                    Observed(Surface.OUTPUT, f"leaked {sentinel_for(RIVAL, 'document')}"),
                    *_blank_except(Surface.OUTPUT),
                ],
            )
        assert sentinel_for(RIVAL, "document") in str(leaked.value)

    def test_a_leak_in_a_derived_artefact_is_caught_the_same_way(self) -> None:
        """A summary is where a leak hides; the sentinel travels into it."""
        theirs = sentinel_for(RIVAL, "profile")
        with pytest.raises(AssertionError, match="memory"):
            assert_no_leak(
                _scenario(),
                tenant=ACME,
                observed=[
                    Observed(Surface.MEMORY, f"summary mentioning {theirs}"),
                    *_blank_except(Surface.MEMORY),
                ],
            )

    def test_a_leak_in_a_span_attribute_is_a_leak(self) -> None:
        with pytest.raises(AssertionError, match="spans"):
            assert_no_leak(
                _scenario(),
                tenant=ACME,
                observed=[
                    Observed(Surface.SPANS, sentinel_for(RIVAL, "document")),
                    *_blank_except(Surface.SPANS),
                ],
            )

    def test_every_leaking_surface_is_named_not_only_the_first(self) -> None:
        report = LeakReport.over(
            _scenario(),
            tenant=ACME,
            observed=[
                Observed(Surface.OUTPUT, sentinel_for(RIVAL, "document")),
                Observed(Surface.CACHE, sentinel_for(RIVAL, "document")),
                *_blank_except(Surface.OUTPUT, Surface.CACHE),
            ],
        )
        assert {leak.surface for leak in report.leaks} == {Surface.OUTPUT, Surface.CACHE}


class TestUninspectedSurfaces:
    def test_a_surface_that_was_not_inspected_is_not_a_pass(self) -> None:
        """A suite reporting green over surfaces it never read is worse than no suite."""
        with pytest.raises(AssertionError, match="not inspected"):
            assert_no_leak(
                _scenario(),
                tenant=ACME,
                observed=[Observed(Surface.OUTPUT, "clean")],
            )

    def test_the_refusal_names_the_surfaces_that_were_missed(self) -> None:
        with pytest.raises(AssertionError, match="events") as unread:
            assert_no_leak(_scenario(), tenant=ACME, observed=[Observed(Surface.OUTPUT, "")])
        assert "cache" in str(unread.value)

    def test_a_surface_a_deployment_genuinely_has_none_of_is_declared_not_skipped(self) -> None:
        report = assert_no_leak(
            _scenario(),
            tenant=ACME,
            observed=[Observed(Surface.OUTPUT, "clean")],
            absent=(Surface.MEMORY, Surface.SEARCH, Surface.CACHE, Surface.SPANS, Surface.EVENTS),
        )
        assert report.clean is True
        assert report.declared_absent == (
            Surface.CACHE,
            Surface.EVENTS,
            Surface.MEMORY,
            Surface.SEARCH,
            Surface.SPANS,
        )


class TestConcurrency:
    async def test_two_interleaved_runs_each_see_their_own_tenant(self) -> None:
        seen: dict[str, list[str]] = {ACME: [], RIVAL: []}

        async def run(tenant: str, step: Step) -> None:
            for _ in range(3):
                seen[tenant].append(current_tenant().tenant)
                await step()

        await interleaved(_scenario(), run)
        assert seen == {ACME: [ACME] * 3, RIVAL: [RIVAL] * 3}

    async def test_the_two_runs_are_interleaved_rather_than_run_one_after_the_other(self) -> None:
        """A context-bleed bug that only shows under interleaving must not be intermittent."""
        order: list[str] = []

        async def run(tenant: str, step: Step) -> None:
            for _ in range(3):
                order.append(tenant)
                await step()

        await interleaved(_scenario(), run)
        assert order == [ACME, RIVAL, ACME, RIVAL, ACME, RIVAL]

    async def test_a_leak_between_interleaved_runs_is_reported_per_tenant(self) -> None:
        scenario = _scenario()

        async def run(tenant: str, step: Step) -> list[LeakReport]:
            await step()
            return [
                LeakReport.over(
                    scenario,
                    tenant=tenant,
                    observed=[Observed(surface, "") for surface in Surface],
                )
            ]

        reports = await interleaved(scenario, run)
        assert set(reports) == {ACME, RIVAL}
        assert all(report.clean for tenant in reports for report in reports[tenant])

    async def test_a_background_task_spawned_outside_a_scope_is_refused(self) -> None:
        async def work() -> str:
            return current_tenant().tenant

        with tenant_scope(ACME):
            pass
        task = asyncio.ensure_future(work())
        with pytest.raises(MissingTenantContextError):
            await task

    async def test_a_task_spawned_inside_a_scope_inherits_it(self) -> None:
        async def work() -> str:
            return current_tenant().tenant

        with tenant_scope(ACME):
            task = asyncio.ensure_future(work())
        assert await task == ACME


def _blank_except(*inspected: Surface) -> list[Observed]:
    """Blank observations for every surface the caller did not supply."""
    return [Observed(surface, "") for surface in Surface if surface not in inspected]
