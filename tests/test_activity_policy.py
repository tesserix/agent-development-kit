"""What a slow model call and a non-idempotent tool are each allowed to do."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tesserix_adk.core import (
    DEFAULT_ACTIVITY_POLICIES,
    ActivityClass,
    ActivityPolicy,
    ApprovalDeniedError,
    Backoff,
    BudgetExceededError,
    GuardrailViolationError,
    Heartbeat,
    Idempotency,
    IdempotencyPolicy,
    ProviderUnavailableError,
    RetryConfig,
    SchemaViolationError,
    ScopeEscalationError,
)
from tesserix_adk.tools import tool as as_tool
from tesserix_adk.workflows import (
    RULES,
    WORKFLOW_MARKER,
    activity_policy_for,
    attempts_for_tool,
    guard_source,
    policy_for_tool,
)


class TestTheDefaultsPerActivityClass:
    def test_a_reasoning_call_is_given_the_quarter_hour_it_can_take(self) -> None:
        """Nine minutes of streaming is a healthy call, not a hung one."""
        assert DEFAULT_ACTIVITY_POLICIES[ActivityClass.MODEL].start_to_close_seconds >= 900

    def test_a_model_activity_is_kept_alive_by_heartbeats_rather_than_by_a_long_timeout(
        self,
    ) -> None:
        policy = DEFAULT_ACTIVITY_POLICIES[ActivityClass.MODEL]

        assert 0 < policy.heartbeat_timeout_seconds < policy.start_to_close_seconds

    def test_a_tool_is_tried_once_until_something_says_repeating_it_is_safe(self) -> None:
        assert DEFAULT_ACTIVITY_POLICIES[ActivityClass.TOOL].retry.max_attempts == 1

    def test_retrieval_and_memory_are_retried_because_repeating_them_changes_nothing(
        self,
    ) -> None:
        assert DEFAULT_ACTIVITY_POLICIES[ActivityClass.RETRIEVAL].retry.max_attempts > 1
        assert DEFAULT_ACTIVITY_POLICIES[ActivityClass.MEMORY].retry.max_attempts > 1

    def test_every_class_has_one(self) -> None:
        assert set(DEFAULT_ACTIVITY_POLICIES) == set(ActivityClass)


class TestWhatIsNeverRetried:
    @pytest.mark.parametrize(
        "failure",
        [
            SchemaViolationError("the model sent something else"),
            GuardrailViolationError("blocked"),
            BudgetExceededError("out of allowance"),
            ApprovalDeniedError("a person said no"),
            ScopeEscalationError("it asked for more than it was granted"),
        ],
    )
    def test_an_answer_is_never_mistaken_for_a_fault(self, failure: Exception) -> None:
        assert ActivityPolicy().retryable(failure) is False

    def test_not_even_where_a_consumer_asked_for_it(self) -> None:
        """A guardrail block retried is the guardrail evaluated until it gives in."""
        insistent = ActivityPolicy(retry=RetryConfig(max_attempts=5))

        assert insistent.retryable(GuardrailViolationError("blocked")) is False

    def test_a_provider_being_down_still_is(self) -> None:
        assert ActivityPolicy().retryable(ProviderUnavailableError("503")) is True

    def test_an_error_the_kit_did_not_raise_is_left_alone(self) -> None:
        """The kit does not know whether repeating someone else's call repeats its effect."""
        assert ActivityPolicy().retryable(RuntimeError("boom")) is False


class TestHowManyAttemptsOneToolGets:
    def test_a_read_only_tool_gets_the_policy_s_attempts(self) -> None:
        policy = ActivityPolicy(retry=RetryConfig(max_attempts=3))

        assert policy.attempts_for(IdempotencyPolicy(kind=Idempotency.READ_ONLY)) == 3

    def test_an_idempotent_tool_does_too(self) -> None:
        policy = ActivityPolicy(retry=RetryConfig(max_attempts=3))

        assert policy.attempts_for(IdempotencyPolicy(kind=Idempotency.IDEMPOTENT)) == 3

    def test_an_effectful_tool_gets_one_attempt(self) -> None:
        """Retrying it charges the card twice."""
        policy = ActivityPolicy(retry=RetryConfig(max_attempts=3))

        assert policy.attempts_for(IdempotencyPolicy(kind=Idempotency.EFFECTFUL)) == 1

    def test_an_effectful_tool_with_a_key_gets_them_back(self) -> None:
        """The key is what makes the second call land on the first call's result."""
        policy = ActivityPolicy(retry=RetryConfig(max_attempts=3))

        assert policy.attempts_for(IdempotencyPolicy(kind=Idempotency.EFFECTFUL), keyed=True) == 3

    def test_a_tool_that_declared_nothing_gets_one(self) -> None:
        assert ActivityPolicy(retry=RetryConfig(max_attempts=3)).attempts_for(None) == 1


class TestBackoffThatWouldOutlastTheActivity:
    def test_a_window_inside_the_timeout_is_left_alone(self) -> None:
        backoff = ActivityPolicy(start_to_close_seconds=60).backoff(2.0, elapsed=1.0)

        assert backoff == Backoff(seconds=2.0)
        assert backoff.truncated is False

    def test_one_that_would_run_past_it_is_cut_and_says_so(self) -> None:
        """Silently dropping it leaves an operator reading a delay that never happened."""
        backoff = ActivityPolicy(start_to_close_seconds=10).backoff(30.0, elapsed=8.0)

        assert backoff.truncated is True
        assert backoff.seconds == pytest.approx(2.0)
        assert "start_to_close" in backoff.reason

    def test_no_room_left_at_all_means_no_further_attempt(self) -> None:
        backoff = ActivityPolicy(start_to_close_seconds=10).backoff(30.0, elapsed=10.0)

        assert backoff.seconds == 0.0
        assert backoff.truncated is True


class TestWhatAHeartbeatIsAllowedToCarry:
    def test_it_reports_progress(self) -> None:
        beat = Heartbeat(step="step-3", tokens=120, chunks=8, at=4.5)

        assert beat.tokens == 120
        assert beat.chunks == 8

    def test_it_cannot_carry_the_partial_text(self) -> None:
        """A consumer that reads a heartbeat as an answer ships half a sentence."""
        with pytest.raises(ValidationError):
            Heartbeat(step="step-3", text="the fare is")  # type: ignore[call-arg]

    def test_progress_is_never_a_result(self) -> None:
        assert Heartbeat().is_result is False


class TestDerivingThePolicyFromWhatATooLDeclared:
    def test_the_tool_s_own_timeout_wins_over_the_class_default(self) -> None:
        policy = activity_policy_for(ActivityClass.TOOL, timeout=5.0)

        assert policy.start_to_close_seconds == 5.0

    def test_a_tool_that_declared_no_timeout_keeps_the_class_default(self) -> None:
        policy = activity_policy_for(ActivityClass.TOOL)

        assert policy == DEFAULT_ACTIVITY_POLICIES[ActivityClass.TOOL]

    def test_the_heartbeat_window_follows_the_timeout_it_was_derived_from(self) -> None:
        """A heartbeat timeout tied to a global constant kills exactly the slow calls."""
        policy = activity_policy_for(ActivityClass.MODEL, timeout=600.0)

        assert 0 < policy.heartbeat_timeout_seconds < 600.0

    def test_a_heartbeat_window_is_never_shorter_than_a_stalled_first_token(self) -> None:
        policy = activity_policy_for(ActivityClass.MODEL, timeout=12.0)

        assert policy.heartbeat_timeout_seconds >= 5.0

    def test_a_policy_a_consumer_tuned_is_not_overwritten(self) -> None:
        tuned = ActivityPolicy(retry=RetryConfig(max_attempts=7))

        assert activity_policy_for(ActivityClass.TOOL, timeout=5.0, base=tuned).retry == tuned.retry


class TestTakingTheNumbersOffTheToolItself:
    def test_a_declared_timeout_is_the_activity_s_timeout(self) -> None:
        @as_tool(timeout=5.0)
        def fare(leg: str) -> str:  # noqa: ARG001
            """Price a leg.

            Args:
                leg: The hop to price.
            """
            return "40 EUR"

        assert policy_for_tool(fare).start_to_close_seconds == 5.0

    def test_a_tool_that_changes_the_world_is_tried_once(self) -> None:
        @as_tool(idempotency=IdempotencyPolicy(kind=Idempotency.EFFECTFUL))
        def charge(card: str) -> str:  # noqa: ARG001
            """Charge a card.

            Args:
                card: The card to charge.
            """
            return "charged"

        assert attempts_for_tool(charge) == 1

    def test_an_idempotency_key_gives_the_attempts_back(self) -> None:
        @as_tool(idempotency=IdempotencyPolicy(kind=Idempotency.EFFECTFUL))
        def charge(card: str) -> str:  # noqa: ARG001
            """Charge a card.

            Args:
                card: The card to charge.
            """
            return "charged"

        tuned = ActivityPolicy(retry=RetryConfig(max_attempts=3))

        assert attempts_for_tool(charge, keyed=True, base=tuned) == 3


class TestAPolicyThatCouldNotBeFollowed:
    def test_a_heartbeat_window_wider_than_the_activity_is_refused(self) -> None:
        """It is not a timeout: nothing could ever breach it."""
        with pytest.raises(ValidationError, match="heartbeat"):
            ActivityPolicy(start_to_close_seconds=10.0, heartbeat_timeout_seconds=30.0)

    def test_an_activity_with_no_time_at_all_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ActivityPolicy(start_to_close_seconds=0.0)


class TestJitterOnTheWorkflowPath:
    def test_the_replay_guard_names_the_call_the_backoff_would_have_used(self) -> None:
        """Backoff is drawn inside the activity, or two replays wait different amounts."""
        findings = guard_source(
            f"{WORKFLOW_MARKER} = True\nimport random\n\n"
            "def plan():\n    return random.uniform(0, 2)\n"
        )

        assert [one.code for one in findings] == ["ADK-W002"]

    def test_the_rule_lists_every_way_the_jitter_could_be_drawn(self) -> None:
        rule = next(one for one in RULES if one.code == "ADK-W002")

        assert "random.uniform" in rule.calls
