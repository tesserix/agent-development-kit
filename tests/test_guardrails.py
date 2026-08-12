"""What a guard may say, the order guards are asked in, and what happens when one cannot."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

import pytest

from tesserix_adk.core import (
    ConfigurationError,
    GuardrailEvaluationError,
    GuardrailPipeline,
    GuardrailViolationError,
    GuardResult,
    GuardStage,
    GuardVerdict,
    ProtocolConformanceError,
)
from tesserix_adk.guardrails import Guard
from tesserix_adk.testing import FakeTracer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

CARD = "the card number is 4111 1111 1111 1111"


class Watching(Guard):
    """A guard that records what it was shown and answers with a fixed result."""

    def __init__(self, name: str, result: GuardResult | None = None) -> None:
        self.name = name
        self._result = result or GuardResult.allow()
        self.seen: list[str] = []

    async def check_input(self, content: str) -> GuardResult:
        """Record and answer."""
        self.seen.append(content)
        return self._result

    async def check_output(self, content: str) -> GuardResult:
        """Record and answer, the same way on the way out."""
        self.seen.append(content)
        return self._result


class Breaking(Guard):
    """A guard that cannot reach a verdict."""

    def __init__(self, name: str = "broken", failure: BaseException | None = None) -> None:
        self.name = name
        self._failure = failure or RuntimeError("classifier down")

    async def check_input(self, content: str) -> GuardResult:
        """Fail rather than decide."""
        del content
        raise self._failure


class Stalling(Guard):
    """A guard that never answers, so a test can watch the pipeline give up on it."""

    def __init__(self, name: str = "slow") -> None:
        self.name = name
        self.entered = asyncio.Event()

    async def check_input(self, content: str) -> GuardResult:
        """Wait to be cancelled."""
        del content
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError


class Nonsense(Guard):
    """A guard whose verdict cannot be read."""

    name = "nonsense"

    async def check_input(self, content: str) -> GuardResult:
        """Answer with something that is not a verdict."""
        del content
        return "probably fine"  # type: ignore[return-value]


async def _chunks(*parts: str) -> AsyncIterator[str]:
    """A streamed answer, one part at a time."""
    for part in parts:
        yield part


class TestWhatAGuardMaySay:
    def test_a_block_carries_a_code_a_caller_can_match_on(self) -> None:
        result = GuardResult.blocked(code="pii_detected", detail="one card number")

        assert result.verdict is GuardVerdict.BLOCK
        assert result.code == "pii_detected"

    def test_a_block_without_a_code_is_refused_where_it_is_written(self) -> None:
        with pytest.raises(ConfigurationError, match="code"):
            GuardResult.blocked(code="")

    def test_a_redaction_carries_what_should_be_used_instead(self) -> None:
        result = GuardResult.redacted("the card number is ****", code="pii_masked")

        assert result.verdict is GuardVerdict.REDACT
        assert result.content == "the card number is ****"

    def test_an_allow_carries_no_replacement_content(self) -> None:
        assert GuardResult.allow().content is None

    def test_a_redaction_with_nothing_to_use_instead_is_refused(self) -> None:
        """A redaction the run cannot apply would silently become an allow."""
        with pytest.raises(ConfigurationError, match="content"):
            GuardResult(verdict=GuardVerdict.REDACT, code="pii_masked")

    def test_an_allow_that_also_replaces_the_content_is_refused(self) -> None:
        """Two readings of one verdict is one reading too many."""
        with pytest.raises(ConfigurationError, match="carries no content"):
            GuardResult(verdict=GuardVerdict.ALLOW, content="something else")


class TestTheOrderGuardsAreAskedIn:
    async def test_guards_are_asked_in_the_order_they_were_declared(self) -> None:
        order: list[str] = []

        class Recording(Watching):
            async def check_input(self, content: str) -> GuardResult:
                order.append(self.name)
                return await super().check_input(content)

        pipeline = GuardrailPipeline((Recording("first"), Recording("second")))

        await pipeline.check_input("ask")

        assert order == ["first", "second"]

    async def test_a_redaction_is_what_the_next_guard_sees_and_what_comes_back(self) -> None:
        masking = Watching("mask", GuardResult.redacted("the card number is ****", code="pii"))
        after = Watching("after")
        pipeline = GuardrailPipeline((masking, after))

        checked = await pipeline.check_input(CARD)

        assert checked == "the card number is ****"
        assert after.seen == ["the card number is ****"]

    async def test_content_nobody_objected_to_comes_back_unchanged(self) -> None:
        pipeline = GuardrailPipeline((Watching("one"), Watching("two")))

        assert await pipeline.check_output("an answer") == "an answer"

    async def test_an_empty_pipeline_is_legal_and_changes_nothing(self) -> None:
        assert await GuardrailPipeline(()).check_input("ask") == "ask"

    async def test_the_two_stages_are_asked_separately(self) -> None:
        guard = Watching("both")
        pipeline = GuardrailPipeline((guard,))

        await pipeline.check_input("ask")
        await pipeline.check_output("answer")

        assert guard.seen == ["ask", "answer"]

    async def test_a_guard_that_does_not_care_about_a_stage_allows_it(self) -> None:
        pipeline = GuardrailPipeline((Breaking("input_only"),))

        assert await pipeline.check_output("an answer") == "an answer"


class TestWhenAGuardBlocks:
    async def test_the_run_stops_with_the_code_stage_and_guard_that_stopped_it(self) -> None:
        pipeline = GuardrailPipeline(
            (Watching("pii", GuardResult.blocked(code="pii_detected", detail="one card number")),)
        )

        with pytest.raises(GuardrailViolationError) as refused:
            await pipeline.check_input(CARD)

        assert refused.value.code == "pii_detected"
        assert refused.value.stage is GuardStage.INPUT
        assert refused.value.guard == "pii"
        assert refused.value.detail == "one card number"

    async def test_a_block_is_not_reconsidered_by_the_guards_after_it(self) -> None:
        later = Watching("later")
        pipeline = GuardrailPipeline((Watching("pii", GuardResult.blocked(code="pii")), later))

        with pytest.raises(GuardrailViolationError):
            await pipeline.check_input(CARD)

        assert later.seen == []

    async def test_the_blocked_content_is_never_carried_on_the_error(self) -> None:
        pipeline = GuardrailPipeline((Watching("pii", GuardResult.blocked(code="pii")),))

        with pytest.raises(GuardrailViolationError) as refused:
            await pipeline.check_input(CARD)

        assert CARD not in str(refused.value)


class TestWhenAGuardCannotDecide:
    async def test_a_guard_that_raises_blocks_rather_than_being_treated_as_a_pass(self) -> None:
        pipeline = GuardrailPipeline((Breaking("toxicity"),))

        with pytest.raises(GuardrailEvaluationError) as refused:
            await pipeline.check_input("ask")

        assert refused.value.guard == "toxicity"
        assert refused.value.stage is GuardStage.INPUT
        assert refused.value.reason == "raised"
        assert isinstance(refused.value.__cause__, RuntimeError)

    async def test_a_guard_that_never_answers_is_given_up_on_and_blocks(self) -> None:
        pipeline = GuardrailPipeline((Stalling(),), timeout_seconds=0.01)

        with pytest.raises(GuardrailEvaluationError) as refused:
            await pipeline.check_input("ask")

        assert refused.value.reason == "timeout"

    async def test_a_verdict_that_cannot_be_read_blocks(self) -> None:
        pipeline = GuardrailPipeline((Nonsense(),))

        with pytest.raises(GuardrailEvaluationError) as refused:
            await pipeline.check_input("ask")

        assert refused.value.reason == "unreadable"

    async def test_the_guards_after_one_that_failed_are_not_asked(self) -> None:
        later = Watching("later")
        pipeline = GuardrailPipeline((Breaking(), later))

        with pytest.raises(GuardrailEvaluationError):
            await pipeline.check_input("ask")

        assert later.seen == []

    async def test_cancelling_the_check_is_not_a_verdict(self) -> None:
        stalling = Stalling()
        pipeline = GuardrailPipeline((stalling,))
        checking = asyncio.ensure_future(pipeline.check_input("ask"))
        await stalling.entered.wait()

        checking.cancel()

        with pytest.raises(asyncio.CancelledError):
            await checking


class TestWhatIsRecorded:
    async def test_every_verdict_is_recorded_with_its_guard_and_stage(self) -> None:
        tracer = FakeTracer()
        pipeline = GuardrailPipeline((Watching("pii"),), tracer=tracer)

        await pipeline.check_output("an answer")

        recorded = [event for event in tracer.recorded if event.name == "guardrail"]
        assert recorded[0].attributes["guard"] == "pii"
        assert recorded[0].attributes["stage"] == "output"
        assert recorded[0].attributes["verdict"] == "allow"
        assert float(cast("float", recorded[0].attributes["duration_ms"])) >= 0

    async def test_what_the_guard_looked_at_is_never_recorded(self) -> None:
        tracer = FakeTracer()
        pipeline = GuardrailPipeline(
            (Watching("pii", GuardResult.blocked(code="pii", detail=CARD)),), tracer=tracer
        )

        with pytest.raises(GuardrailViolationError):
            await pipeline.check_input(CARD)

        assert CARD not in repr(tracer.recorded)

    async def test_a_guard_that_could_not_decide_is_recorded_as_such(self) -> None:
        tracer = FakeTracer()
        pipeline = GuardrailPipeline((Breaking("toxicity"),), tracer=tracer)

        with pytest.raises(GuardrailEvaluationError):
            await pipeline.check_input("ask")

        assert tracer.recorded[0].attributes["verdict"] == "unevaluated"

    async def test_a_collector_that_fails_does_not_decide_anything(self) -> None:
        class Falling(FakeTracer):
            def event(self, name: str, **attributes: object) -> None:
                del name, attributes
                raise RuntimeError("collector down")

        pipeline = GuardrailPipeline((Watching("pii"),), tracer=Falling())

        assert await pipeline.check_output("an answer") == "an answer"


class TestBuildingOne:
    def test_two_guards_answering_to_one_name_are_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="pii"):
            GuardrailPipeline((Watching("pii"), Watching("pii")))

    def test_something_that_is_not_a_guard_is_refused_where_it_is_wired(self) -> None:
        with pytest.raises(ProtocolConformanceError):
            GuardrailPipeline((object(),))  # type: ignore[arg-type]

    def test_a_timeout_that_could_never_permit_a_check_is_refused(self) -> None:
        with pytest.raises(ConfigurationError, match="timeout"):
            GuardrailPipeline((), timeout_seconds=0)

    def test_the_pipeline_says_which_guards_it_holds_in_order(self) -> None:
        pipeline = GuardrailPipeline((Watching("pii"), Watching("toxicity")))

        assert pipeline.guards == ("pii", "toxicity")


class TestAStreamedAnswer:
    async def test_a_streamed_answer_is_checked_before_any_of_it_is_handed_on(self) -> None:
        pipeline = GuardrailPipeline((Watching("pii", GuardResult.blocked(code="pii")),))
        seen: list[str] = []

        with pytest.raises(GuardrailViolationError):
            seen.extend([part async for part in pipeline.check_stream(_chunks("the ", "card"))])

        assert seen == []

    async def test_a_redacted_stream_hands_on_what_the_guard_left(self) -> None:
        pipeline = GuardrailPipeline((Watching("pii", GuardResult.redacted("****", code="pii")),))

        assert [part async for part in pipeline.check_stream(_chunks("4111", " 1111"))] == ["****"]

    async def test_a_stream_nobody_objected_to_arrives_whole(self) -> None:
        pipeline = GuardrailPipeline((Watching("pii"),))

        parts = [part async for part in pipeline.check_stream(_chunks("an ", "answer"))]

        assert parts == ["an answer"]
