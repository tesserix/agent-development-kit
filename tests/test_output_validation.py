"""Typing the answer a caller is handed, the rules it has to satisfy, and bounded repair."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from tesserix_adk.core import (
    Abstention,
    Bounded,
    GuardrailEvaluationError,
    GuardVerdict,
    Invariant,
    OneOf,
    OutputValidationError,
    Policy,
    PolicyViolation,
    RequiresCitation,
    evaluate,
    reach,
)
from tesserix_adk.guardrails import PolicyGuard, SchemaGuard, validated
from tesserix_adk.runtime.structured import OutputContract

pytestmark = pytest.mark.anyio


class Quote(BaseModel):
    total: int
    currency: str = "AUD"
    citations: tuple[str, ...] = ()
    note: str | None = None


class Nested(BaseModel):
    quote: Quote | None = None


CONTRACT = OutputContract.of(Quote)
IN_BAND = '{"total": 90}'
OUT_OF_BAND = '{"total": 4000}'
BAND = Bounded("total", minimum=10, maximum=100)


class Breaks:
    name = "breaks"

    def check(self, result: BaseModel) -> tuple[PolicyViolation, ...]:
        raise RuntimeError(type(result).__name__)


class TestReachingIntoAnAnswer:
    def test_a_dotted_path_walks_into_a_nested_model(self) -> None:
        assert reach(Nested(quote=Quote(total=90)), "quote.total") == 90

    def test_a_path_through_an_unanswered_field_is_not_a_value(self) -> None:
        assert reach(Nested(), "quote.total") is None

    def test_a_path_that_does_not_exist_is_a_rule_that_stopped_running(self) -> None:
        with pytest.raises(GuardrailEvaluationError) as refused:
            reach(Quote(total=90), "subtotal")

        assert refused.value.details["reason"] == "unknown_path"


class TestTheDeclarativeRules:
    def test_a_total_inside_the_band_violates_nothing(self) -> None:
        assert BAND.check(Quote(total=90)) == ()

    @pytest.mark.parametrize("total", [4, 4000])
    def test_a_total_outside_it_names_the_policy_and_not_the_value(self, total: int) -> None:
        (violation,) = BAND.check(Quote(total=total))

        assert violation.policy == "total_within_band"
        assert violation.path == "total"
        assert str(total) not in violation.detail

    def test_a_missing_value_is_expressible_unless_the_rule_requires_one(self) -> None:
        assert Bounded("note").check(Quote(total=90)) == ()
        assert Bounded("note", required=True).check(Quote(total=90))[0].path == "note"

    def test_a_field_that_is_not_a_number_is_a_violation_rather_than_a_crash(self) -> None:
        (violation,) = Bounded("currency", maximum=100).check(Quote(total=90))

        assert violation.detail == "is not a number"

    def test_a_named_rule_keeps_the_identifier_a_caller_matches_on(self) -> None:
        rule = Bounded("total", minimum=10, name="house_limit")

        assert rule.check(Quote(total=1))[0].policy == "house_limit"

    def test_membership_is_compared_in_one_normal_form(self) -> None:
        allowed = OneOf("currency", ("SEK", "AUD"))

        assert allowed.check(Quote(total=1, currency="AUD")) == ()
        assert allowed.check(Quote(total=1, currency="ÅUD"))  # a decomposed A-ring

    def test_a_composed_and_a_decomposed_answer_are_judged_alike(self) -> None:
        allowed = OneOf("currency", ("ÅRE",))

        assert allowed.check(Quote(total=1, currency="ÅRE")) == ()

    def test_an_unanswered_field_only_breaks_membership_where_it_was_required(self) -> None:
        assert OneOf("note", ("a",)).check(Quote(total=1)) == ()
        assert OneOf("note", ("a",), required=True).check(Quote(total=1))[0].path == "note"

    def test_an_assertion_with_nothing_behind_it_is_rejected(self) -> None:
        rule = RequiresCitation("note", "citations")

        assert rule.check(Quote(total=1, note="it closes at six"))[0].policy == "note_is_cited"

    def test_a_cited_assertion_passes_and_so_does_an_absent_one(self) -> None:
        rule = RequiresCitation("note", "citations")

        assert rule.check(Quote(total=1, note="it closes at six", citations=("a",))) == ()
        assert rule.check(Quote(total=1)) == ()
        assert rule.check(Quote(total=1, note="")) == ()

    def test_a_cross_field_rule_is_about_the_whole_answer(self) -> None:
        rule = Invariant[Quote](
            "no_free_quotes", lambda quote: bool(quote.citations), "nothing behind it"
        )

        (violation,) = rule.check(Quote(total=1))

        assert violation.path == ""
        assert rule.check(Quote(total=1, citations=("a",))) == ()


class TestEvaluatingThemTogether:
    def test_violations_come_back_in_declaration_order(self) -> None:
        report = evaluate(Quote(total=4000, currency="XXX"), (BAND, OneOf("currency", ("AUD",))))

        assert report.rejected
        assert report.policies == ("currency_is_known", "total_within_band")

    def test_an_abstention_is_not_judged_because_it_quoted_nothing(self) -> None:
        assert not evaluate(Abstention(abstained=True, reason="no fare data"), (BAND,)).rejected

    def test_a_rule_that_raises_fails_closed(self) -> None:
        with pytest.raises(GuardrailEvaluationError) as refused:
            evaluate(Quote(total=90), (Breaks(),))

        assert refused.value.details["reason"] == "raised"

    def test_a_rule_that_reaches_a_renamed_field_keeps_its_own_reason(self) -> None:
        with pytest.raises(GuardrailEvaluationError) as refused:
            evaluate(Quote(total=90), (Bounded("subtotal"),))

        assert refused.value.details["reason"] == "unknown_path"

    def test_a_rule_is_recognised_structurally(self) -> None:
        assert isinstance(BAND, Policy)


class TestTypingWhatCameBack:
    def test_a_well_formed_answer_becomes_the_declared_type(self) -> None:
        assert SchemaGuard(CONTRACT).parse(IN_BAND) == Quote(total=90)

    def test_a_fenced_answer_is_unwrapped_rather_than_scraped(self) -> None:
        assert SchemaGuard(CONTRACT).parse(f"```json\n{IN_BAND}\n```") == Quote(total=90)

    def test_prose_is_a_typed_error_carrying_the_payload_for_a_debugger(self) -> None:
        with pytest.raises(OutputValidationError) as refused:
            SchemaGuard(CONTRACT).parse("I think about ninety dollars")

        assert refused.value.model == "Quote"
        assert refused.value.payload == "I think about ninety dollars"
        assert refused.value.attempts == 1

    def test_a_missing_field_is_never_filled_in_to_make_it_validate(self) -> None:
        with pytest.raises(OutputValidationError) as refused:
            SchemaGuard(CONTRACT).parse('{"currency": "AUD"}')

        assert refused.value.paths == ("total",)

    async def test_the_guard_blocks_without_quoting_what_came_back(self) -> None:
        result = await SchemaGuard(CONTRACT).check_output('{"currency": "AUD"}')

        assert result.verdict is GuardVerdict.BLOCK
        assert result.code == "schema_violation"
        assert result.detail == "Quote: total"

    async def test_prose_reads_as_not_having_parsed_at_all(self) -> None:
        result = await SchemaGuard(CONTRACT).check_output("about ninety")

        assert result.detail == "Quote: did not parse"

    async def test_a_valid_answer_passes_the_guard(self) -> None:
        assert (await SchemaGuard(CONTRACT).check_output(IN_BAND)).verdict is GuardVerdict.ALLOW


class TestSayingItDoesNotKnow:
    def test_an_abstention_is_refused_where_the_agent_did_not_allow_one(self) -> None:
        with pytest.raises(OutputValidationError):
            SchemaGuard(CONTRACT).parse('{"abstained": true, "reason": "no fare data"}')

    def test_it_is_a_typed_outcome_where_the_agent_did(self) -> None:
        answer = SchemaGuard(CONTRACT, abstention=True).parse(
            '{"abstained": true, "reason": "no fare data"}'
        )

        assert isinstance(answer, Abstention)
        assert answer.reason == "no fare data"

    def test_a_real_answer_still_parses_with_abstention_allowed(self) -> None:
        assert SchemaGuard(CONTRACT, abstention=True).parse(IN_BAND) == Quote(total=90)


class TestApplyingTheRulesToAParsedAnswer:
    def test_an_answer_inside_the_rules_raises_nothing(self) -> None:
        PolicyGuard((BAND,)).raise_for(Quote(total=90))

    def test_one_outside_them_names_the_rule_and_never_the_value(self) -> None:
        with pytest.raises(OutputValidationError) as refused:
            PolicyGuard((BAND,)).raise_for(Quote(total=4000))

        assert refused.value.policies == ("total_within_band",)
        assert refused.value.paths == ("total",)
        assert "4000" not in str(refused.value)

    def test_the_correction_says_what_failed_and_supplies_no_value(self) -> None:
        guard = PolicyGuard((BAND,))

        correction = guard.correction(guard.check(Quote(total=4000)))

        assert "total_within_band" in correction
        assert "4000" not in correction


class TestBoundedRepair:
    async def test_without_a_re_ask_the_first_answer_is_the_only_one(self) -> None:
        asked: list[str] = []

        with pytest.raises(OutputValidationError) as refused:
            await validated(OUT_OF_BAND, schema=SchemaGuard(CONTRACT), policy=PolicyGuard((BAND,)))

        assert refused.value.attempts == 1
        assert asked == []

    async def test_a_corrected_answer_is_accepted(self) -> None:
        asked: list[str] = []

        async def reask(correction: str) -> str:
            asked.append(correction)
            return IN_BAND

        answer = await validated(
            OUT_OF_BAND,
            schema=SchemaGuard(CONTRACT),
            policy=PolicyGuard((BAND,)),
            reask=reask,
            attempts=3,
        )

        assert answer == Quote(total=90)
        assert len(asked) == 1

    async def test_a_model_that_keeps_failing_stops_at_the_cap(self) -> None:
        asked: list[str] = []

        async def reask(correction: str) -> str:
            asked.append(correction)
            return OUT_OF_BAND

        with pytest.raises(OutputValidationError) as refused:
            await validated(
                OUT_OF_BAND,
                schema=SchemaGuard(CONTRACT),
                policy=PolicyGuard((BAND,)),
                reask=reask,
                attempts=3,
            )

        assert len(asked) == 2
        assert refused.value.attempts == 3
        assert refused.value.details["attempts"] == "3"

    async def test_a_schema_failure_is_re_asked_with_what_actually_failed(self) -> None:
        asked: list[str] = []

        async def reask(correction: str) -> str:
            asked.append(correction)
            return IN_BAND

        answer = await validated(
            "about ninety", schema=SchemaGuard(CONTRACT), reask=reask, attempts=2
        )

        assert answer == Quote(total=90)
        assert "JSON" in asked[0]

    async def test_an_answer_that_needs_nothing_costs_no_attempt(self) -> None:
        async def reask(correction: str) -> str:
            raise AssertionError(correction)

        assert await validated(IN_BAND, schema=SchemaGuard(CONTRACT), reask=reask) == Quote(
            total=90
        )

    async def test_an_abstention_ends_the_loop_rather_than_being_repaired(self) -> None:
        async def reask(correction: str) -> str:
            raise AssertionError(correction)

        answer = await validated(
            '{"abstained": true, "reason": "no fare data"}',
            schema=SchemaGuard(CONTRACT, abstention=True),
            policy=PolicyGuard((BAND,)),
            reask=reask,
            attempts=3,
        )

        assert isinstance(answer, Abstention)

    async def test_an_attempt_cap_below_one_still_asks_once(self) -> None:
        with pytest.raises(OutputValidationError) as refused:
            await validated(
                OUT_OF_BAND, schema=SchemaGuard(CONTRACT), policy=PolicyGuard((BAND,)), attempts=0
            )

        assert refused.value.attempts == 1
