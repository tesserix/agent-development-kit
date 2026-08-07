"""Choosing a model by naming the job rather than the model.

A model id written at a call site is a cost decision compiled into a product: changing it
means a code change in every consumer, and there is no way to say that classification
wants the cheap model while planning wants the reasoning one. A task class is the name of
the job; the routing table is where an operator answers it.

The two things asserted hardest are that resolution never silently downgrades — a
candidate that cannot do the work is rejected by name, and an empty set is an error rather
than a fallback — and that the answer is explainable afterwards.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    CHEAP,
    REASONING,
    SMART,
    Capability,
    ConfigurationError,
    ModelCapabilities,
    ModelRef,
    ModelRequirements,
    ModelSpec,
    NoEligibleModelError,
    RoutingDecision,
    TaskClass,
)
from tesserix_adk.models.routing import RoutingRule, RoutingTable, TableRouter, routing_table

if TYPE_CHECKING:
    from pathlib import Path

SEES = ModelRequirements(capabilities=frozenset({Capability.VISION}))
SMALL = ModelCapabilities(tool_calling=True, streaming=True, context_window_tokens=8_000)
BIG = ModelCapabilities(
    tool_calling=True,
    structured_output=True,
    vision=True,
    streaming=True,
    context_window_tokens=200_000,
)


def spec(ref: str, capabilities: ModelCapabilities = SMALL) -> ModelSpec:
    provider, _, model = ref.partition(":")
    return ModelSpec(provider=provider, model=model, capabilities=capabilities)


def rule(task_class: TaskClass, *candidates: ModelSpec, **scope: str) -> RoutingRule:
    return RoutingRule(task_class=task_class, candidates=tuple(candidates), **scope)


def router(*rules: RoutingRule) -> TableRouter:
    return TableRouter(RoutingTable(rules=rules))


class TestNamingTheJobRatherThanTheModel:
    def test_a_task_class_resolves_to_the_first_candidate_that_can_do_it(self) -> None:
        decided = router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(CHEAP)
        assert decided.chosen == ModelRef(provider="openai", model="gpt-4o-mini")

    def test_the_table_decides_the_vendor_so_retuning_is_not_a_code_change(self) -> None:
        """The same call site, two tables, two vendors."""
        for candidate in ("openai:gpt-4o-mini", "anthropic:claude-haiku-4-5"):
            decided = router(rule(CHEAP, spec(candidate))).resolve(CHEAP)
            assert str(decided.chosen) == candidate

    def test_a_consumer_may_name_a_class_the_kit_has_never_heard_of(self) -> None:
        """The three standard classes are a starting set, not the whole vocabulary."""
        transcription = TaskClass("transcription")
        decided = router(rule(transcription, spec("openai:gpt-4o-mini"))).resolve(transcription)
        assert decided.task_class == transcription

    def test_the_three_standard_classes_are_the_strings_configuration_writes(self) -> None:
        assert (CHEAP, SMART, REASONING) == ("cheap", "smart", "reasoning")


class TestACandidateThatCannotDoTheJob:
    def test_a_candidate_missing_a_required_capability_is_passed_over(self) -> None:
        decided = router(
            rule(SMART, spec("openai:gpt-4o-mini"), spec("openai:gpt-4o", BIG))
        ).resolve(SMART, requirements=SEES)
        assert decided.chosen.model == "gpt-4o"

    def test_a_candidate_whose_window_is_too_small_is_passed_over(self) -> None:
        decided = router(
            rule(SMART, spec("openai:gpt-4o-mini"), spec("openai:gpt-4o", BIG))
        ).resolve(SMART, requirements=ModelRequirements(min_context_window_tokens=100_000))
        assert decided.chosen.model == "gpt-4o"

    def test_a_candidate_that_declares_no_window_at_all_cannot_promise_one(self) -> None:
        """Silence is not a claim: an unknown window is not evidence of a large one."""
        unknown = spec("vllm:qwen", ModelCapabilities(tool_calling=True))
        with pytest.raises(NoEligibleModelError):
            router(rule(CHEAP, unknown)).resolve(
                CHEAP, requirements=ModelRequirements(min_context_window_tokens=1)
            )

    def test_nothing_eligible_is_an_error_rather_than_the_nearest_thing(self) -> None:
        with pytest.raises(NoEligibleModelError) as refused:
            router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(CHEAP, requirements=SEES)
        assert refused.value.task_class == CHEAP
        assert refused.value.unsatisfied == (Capability.VISION.value,)

    def test_the_error_names_every_candidate_it_rejected_and_why(self) -> None:
        with pytest.raises(NoEligibleModelError) as refused:
            router(rule(CHEAP, spec("openai:gpt-4o-mini"), spec("vllm:qwen"))).resolve(
                CHEAP, requirements=SEES
            )
        rejected = {candidate.ref: candidate.reason for candidate in refused.value.rejected}
        assert set(rejected) == {"openai:gpt-4o-mini", "vllm:qwen"}
        assert all("vision" in reason for reason in rejected.values())

    def test_a_class_the_table_does_not_route_is_an_error_not_a_default(self) -> None:
        """Falling through to whatever is configured is how a cheap step bills as smart."""
        with pytest.raises(NoEligibleModelError, match="reasoning"):
            router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(REASONING)


class TestWhoIsAsking:
    def test_a_tenant_override_wins_over_the_general_rule(self) -> None:
        decided = router(
            rule(CHEAP, spec("openai:gpt-4o-mini")),
            rule(CHEAP, spec("anthropic:claude-haiku-4-5"), tenant="acme"),
        ).resolve(CHEAP, tenant="acme")
        assert decided.chosen.provider == "anthropic"

    def test_a_tenant_without_an_override_gets_the_general_rule(self) -> None:
        decided = router(
            rule(CHEAP, spec("openai:gpt-4o-mini")),
            rule(CHEAP, spec("anthropic:claude-haiku-4-5"), tenant="acme"),
        ).resolve(CHEAP, tenant="other")
        assert decided.chosen.provider == "openai"

    def test_an_agent_override_wins_over_the_general_rule(self) -> None:
        decided = router(
            rule(SMART, spec("openai:gpt-4o-mini")),
            rule(SMART, spec("anthropic:claude-sonnet-4-5"), agent="planner"),
        ).resolve(SMART, agent="planner")
        assert decided.chosen.provider == "anthropic"

    def test_a_rule_naming_both_wins_over_a_rule_naming_one(self) -> None:
        """Most specific first, so an escalation for one tenant's one agent is expressible."""
        decided = router(
            rule(SMART, spec("openai:gpt-4o-mini")),
            rule(SMART, spec("openai:gpt-4o", BIG), tenant="acme"),
            rule(SMART, spec("anthropic:claude-sonnet-4-5"), tenant="acme", agent="planner"),
        ).resolve(SMART, tenant="acme", agent="planner")
        assert decided.chosen.provider == "anthropic"

    def test_the_decision_says_which_rule_answered(self) -> None:
        decided = router(
            rule(CHEAP, spec("openai:gpt-4o-mini")),
            rule(CHEAP, spec("anthropic:claude-haiku-4-5"), tenant="acme"),
        ).resolve(CHEAP, tenant="acme")
        assert decided.rule == "cheap@acme"


class TestExplainingItAfterwards:
    def test_the_decision_carries_the_class_the_candidates_and_the_choice(self) -> None:
        decided = router(
            rule(SMART, spec("openai:gpt-4o-mini"), spec("openai:gpt-4o", BIG))
        ).resolve(SMART, requirements=SEES)
        assert decided.task_class == SMART
        assert decided.considered == ("openai:gpt-4o-mini", "openai:gpt-4o")
        assert decided.chosen.model == "gpt-4o"

    def test_it_says_why_the_earlier_candidates_lost(self) -> None:
        decided = router(
            rule(SMART, spec("openai:gpt-4o-mini"), spec("openai:gpt-4o", BIG))
        ).resolve(SMART, requirements=SEES)
        assert [candidate.ref for candidate in decided.rejected] == ["openai:gpt-4o-mini"]
        assert "vision" in decided.rejected[0].reason

    def test_it_reads_as_one_line_in_a_run_record(self) -> None:
        decided = router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(CHEAP)
        assert decided.explain() == (
            "cheap -> openai:gpt-4o-mini (rule cheap, 1 considered; asked for no capability floor)"
        )


class TestPinningAModelOnPurpose:
    """Reproducing a run means running the model that ran, not the one the table names
    today. The pin is the escape hatch, and it is recorded as one."""

    def test_a_pinned_model_is_used_whatever_the_table_says(self) -> None:
        decided = router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(
            CHEAP, pinned=ModelRef(provider="anthropic", model="claude-haiku-4-5")
        )
        assert decided.chosen.provider == "anthropic"
        assert decided.pinned

    def test_a_pin_still_has_to_be_a_model_the_table_knows(self) -> None:
        """A pin is a choice between configured models, not a way past configuration."""
        with pytest.raises(NoEligibleModelError, match="not in the routing table"):
            router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(
                CHEAP, pinned=ModelRef(provider="openai", model="gpt-3.5-turbo")
            )

    def test_a_pin_that_cannot_do_the_job_is_refused_rather_than_downgraded(self) -> None:
        with pytest.raises(NoEligibleModelError):
            router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(
                CHEAP,
                requirements=SEES,
                pinned=ModelRef(provider="openai", model="gpt-4o-mini"),
            )


class TestATableThatWouldFailLater:
    """Every one of these is caught when the table is built, which is boot, rather than by
    the first request that happened to need it."""

    def test_an_empty_table_is_refused_rather_than_defaulting_to_anything(self) -> None:
        with pytest.raises(ConfigurationError, match="no rules"):
            RoutingTable(rules=())

    def test_a_rule_with_no_candidates_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="no candidates"):
            RoutingTable(rules=(rule(CHEAP),))

    def test_two_rules_at_the_same_scope_are_refused_rather_than_one_winning_quietly(
        self,
    ) -> None:
        with pytest.raises(ConfigurationError, match="twice"):
            RoutingTable(
                rules=(rule(CHEAP, spec("openai:gpt-4o-mini")), rule(CHEAP, spec("vllm:qwen")))
            )

    def test_a_candidate_that_declares_nothing_is_refused(self) -> None:
        """A row with no capabilities matches no requirement and is a row nobody meant."""
        with pytest.raises(ConfigurationError, match="declares no capabilities"):
            RoutingTable(rules=(rule(CHEAP, spec("vllm:qwen", ModelCapabilities())),))

    def test_the_schema_version_is_carried_so_a_table_can_be_read_by_version(self) -> None:
        assert RoutingTable(rules=(rule(CHEAP, spec("openai:gpt-4o-mini")),)).version == 1

    def test_a_table_from_a_version_the_kit_does_not_read_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="version"):
            RoutingTable(version=99, rules=(rule(CHEAP, spec("openai:gpt-4o-mini")),))


class TestATableThatChangesUnderneath:
    """A reload is a new table. A run that started against the old one keeps the model it
    started with, because a run that changed model halfway is two runs in one record."""

    def test_a_decision_does_not_change_when_the_router_is_given_a_new_table(self) -> None:
        first = router(rule(CHEAP, spec("openai:gpt-4o-mini")))
        decided = first.resolve(CHEAP)
        reloaded = first.reloaded(
            RoutingTable(rules=(rule(CHEAP, spec("anthropic:claude-haiku-4-5")),))
        )
        assert decided.chosen.provider == "openai"
        assert reloaded.resolve(CHEAP).chosen.provider == "anthropic"

    def test_reloading_leaves_the_router_that_a_run_is_holding_alone(self) -> None:
        first = router(rule(CHEAP, spec("openai:gpt-4o-mini")))
        first.reloaded(RoutingTable(rules=(rule(CHEAP, spec("vllm:qwen")),)))
        assert first.resolve(CHEAP).chosen.provider == "openai"


class TestAModelTheVendorNoLongerServes:
    """A decommissioned id is caught when the table is read, not by the production request
    that happened to be routed to it."""

    def test_a_model_the_catalogue_does_not_know_is_refused_for_a_vendor_it_covers(self) -> None:
        with pytest.raises(ConfigurationError, match="catalogue"):
            RoutingTable(rules=(rule(CHEAP, spec("openai:gpt-3.5-turbo")),))

    def test_a_self_hosted_model_is_not_measured_against_a_vendor_catalogue(self) -> None:
        """Nobody publishes a card for the weights a court runs on its own hardware."""
        assert RoutingTable(rules=(rule(CHEAP, spec("vllm:qwen")),)).rules[0].candidates[0].model


class TestATenantThatIsNotEntitledToTheModel:
    """An entitlement is not a preference. A tenant whose contract does not cover a model
    is refused it even where a rule names it, because honouring the rule bills work to a
    model the tenant never agreed to."""

    def test_a_model_the_tenant_is_not_entitled_to_is_passed_over(self) -> None:
        allowed = TableRouter(
            RoutingTable(
                rules=(rule(CHEAP, spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")),)
            ),
            entitlements={"acme": frozenset({"anthropic:claude-haiku-4-5"})},
        )
        assert allowed.resolve(CHEAP, tenant="acme").chosen.provider == "anthropic"

    def test_a_tenant_override_naming_a_model_it_cannot_have_is_refused(self) -> None:
        allowed = TableRouter(
            RoutingTable(
                rules=(
                    rule(CHEAP, spec("openai:gpt-4o-mini")),
                    rule(CHEAP, spec("anthropic:claude-haiku-4-5"), tenant="acme"),
                )
            ),
            entitlements={"acme": frozenset({"openai:gpt-4o-mini"})},
        )
        with pytest.raises(NoEligibleModelError, match="not entitled"):
            allowed.resolve(CHEAP, tenant="acme")

    def test_a_tenant_nobody_wrote_an_entitlement_for_is_not_narrowed(self) -> None:
        """An absent entry is no restriction; an empty one is a tenant entitled to nothing."""
        allowed = TableRouter(
            RoutingTable(rules=(rule(CHEAP, spec("openai:gpt-4o-mini")),)),
            entitlements={"acme": frozenset({"openai:gpt-4o-mini"})},
        )
        assert allowed.resolve(CHEAP, tenant="other").chosen.model == "gpt-4o-mini"

    def test_a_pin_does_not_get_past_an_entitlement_either(self) -> None:
        allowed = TableRouter(
            RoutingTable(
                rules=(rule(CHEAP, spec("openai:gpt-4o-mini"), spec("anthropic:claude-haiku-4-5")),)
            ),
            entitlements={"acme": frozenset({"openai:gpt-4o-mini"})},
        )
        with pytest.raises(NoEligibleModelError, match="not entitled"):
            allowed.resolve(
                CHEAP,
                tenant="acme",
                pinned=ModelRef(provider="anthropic", model="claude-haiku-4-5"),
            )


class TestReadingTheTableFromConfiguration:
    """The point of the feature is that retuning is an operational change, which means the
    table has to be readable from a file rather than only constructible in code."""

    def written(self, tmp_path: Path, text: str) -> Path:
        table = tmp_path / "routing.toml"
        table.write_text(text, encoding="utf-8")
        return table

    def test_a_table_is_read_from_a_toml_file(self, tmp_path: Path) -> None:
        path = self.written(
            tmp_path,
            """
            version = 1

            [[rules]]
            task_class = "cheap"

            [[rules.candidates]]
            provider = "openai"
            model = "gpt-4o-mini"
            capabilities = { tool_calling = true }
            """,
        )
        assert routing_table(path).rules[0].candidates[0].model == "gpt-4o-mini"

    def test_the_environment_names_the_file_when_the_caller_does_not(self, tmp_path: Path) -> None:
        path = self.written(
            tmp_path,
            """
            [[rules]]
            task_class = "smart"
            candidates = [
              { provider = "openai", model = "gpt-4o", capabilities = { vision = true } },
            ]
            """,
        )
        loaded = routing_table(env={"ADK_ROUTING_TABLE": str(path)})
        assert loaded.rules[0].task_class == SMART

    def test_an_explicit_path_beats_the_environment(self, tmp_path: Path) -> None:
        chosen = self.written(
            tmp_path,
            """
            [[rules]]
            task_class = "cheap"
            candidates = [
              { provider = "vllm", model = "qwen", capabilities = { tool_calling = true } },
            ]
            """,
        )
        loaded = routing_table(chosen, env={"ADK_ROUTING_TABLE": "/does/not/exist.toml"})
        assert loaded.rules[0].candidates[0].provider == "vllm"

    def test_a_file_nobody_named_is_a_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="ADK_ROUTING_TABLE"):
            routing_table(env={})

    def test_a_file_that_is_not_there_is_named_in_the_failure(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match=r"routing\.toml"):
            routing_table(tmp_path / "routing.toml")

    def test_a_table_that_would_fail_later_fails_at_load(self, tmp_path: Path) -> None:
        """The whole point of loading at boot: a bad file stops the process starting."""
        path = self.written(tmp_path, "version = 1\nrules = []\n")
        with pytest.raises(ConfigurationError, match="no rules"):
            routing_table(path)

    def test_a_file_that_is_not_toml_names_the_file_rather_than_the_parser(
        self, tmp_path: Path
    ) -> None:
        path = self.written(tmp_path, "task_class = cheap\n")
        with pytest.raises(ConfigurationError, match="not readable TOML"):
            routing_table(path)

    def test_readable_toml_of_the_wrong_shape_is_a_configuration_error(
        self, tmp_path: Path
    ) -> None:
        """A file tomllib reads happily can still say nothing a table can be built from."""
        path = self.written(tmp_path, 'version = "one"\n')
        with pytest.raises(ConfigurationError, match="is not a table"):
            routing_table(path)

    def test_the_router_says_which_table_it_reads(self) -> None:
        """An operator asking a running process what it routes by should not have to guess."""
        loaded = RoutingTable(rules=(rule(CHEAP, spec("openai:gpt-4o-mini")),))
        assert TableRouter(loaded).table is loaded


class TestTheDecisionIsData:
    def test_it_round_trips_so_a_run_record_can_be_stored_and_read_back(self) -> None:
        decided = router(rule(CHEAP, spec("openai:gpt-4o-mini"))).resolve(CHEAP)
        assert RoutingDecision.model_validate(decided.model_dump()) == decided
