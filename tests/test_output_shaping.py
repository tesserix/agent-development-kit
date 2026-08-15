"""Effort clamped downward on resumption turns, terseness steered by suffix only."""

from __future__ import annotations

import pytest

from tesserix_adk.core import ConfigurationError, Message, ModelRequest, TextPart, ToolCall
from tesserix_adk.models import (
    Effort,
    Shaping,
    TurnKind,
    classify,
    errored,
    provider_effort,
)

SYSTEM = Message(role="system", content=[TextPart(text="You answer from tool output only.")])
ASKED = Message(role="user", content=[TextPart(text="Which hosts are unreachable?")])
CALLED = Message(
    role="assistant",
    content=[TextPart(text="")],
    tool_calls=(ToolCall(id="c-1", name="hosts", arguments={}),),
)
RETURNED = Message(role="tool", tool_call_id="c-1", content=[TextPart(text="node-004")])


def asking(*messages: Message) -> ModelRequest:
    """One request, with whatever conversation the scenario needs."""
    return ModelRequest(model="gpt-x", messages=messages or (SYSTEM, ASKED))


class TestWhatKindOfTurnThisIs:
    """Only a turn that consumes a successful tool result is cheap to think less about."""

    def test_a_question_from_the_user_is_a_new_question(self) -> None:
        assert classify([SYSTEM, ASKED]) is TurnKind.NEW_QUESTION

    def test_consuming_a_tool_result_is_a_resumption(self) -> None:
        assert classify([SYSTEM, ASKED, CALLED, RETURNED]) is TurnKind.TOOL_RESUMPTION

    def test_consuming_a_failed_tool_result_is_error_recovery(self) -> None:
        assert classify([SYSTEM, ASKED, CALLED, errored(RETURNED)]) is TurnKind.ERROR_RECOVERY

    def test_an_empty_conversation_is_a_new_question(self) -> None:
        assert classify([]) is TurnKind.NEW_QUESTION


class TestClampingDownAndOnlyDown:
    """Shaping can only ever cost less than what the caller asked for."""

    def test_a_resumption_turn_is_clamped_to_the_configured_bound(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.LOW)

        shaped = shaping.shape(asking(SYSTEM, ASKED, CALLED, RETURNED), effort=Effort.HIGH)

        assert shaped.effort is Effort.LOW
        assert shaped.clamped is True

    def test_a_new_question_keeps_the_effort_the_caller_asked_for(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.LOW)

        shaped = shaping.shape(asking(), effort=Effort.HIGH)

        assert shaped.effort is Effort.HIGH
        assert shaped.clamped is False

    def test_error_recovery_is_never_clamped(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.MINIMAL)

        shaped = shaping.shape(asking(SYSTEM, ASKED, CALLED, errored(RETURNED)), effort=Effort.HIGH)

        assert shaped.effort is Effort.HIGH
        assert "recovery" in shaped.reason

    def test_a_caller_asking_for_less_than_the_bound_keeps_their_own_value(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.MEDIUM)

        shaped = shaping.shape(asking(SYSTEM, ASKED, CALLED, RETURNED), effort=Effort.MINIMAL)

        assert shaped.effort is Effort.MINIMAL
        assert shaped.clamped is False

    def test_the_final_turn_that_answers_the_user_is_not_clamped(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.MINIMAL)

        shaped = shaping.shape(
            asking(SYSTEM, ASKED, CALLED, RETURNED), effort=Effort.HIGH, final=True
        )

        assert shaped.effort is Effort.HIGH
        assert "final" in shaped.reason

    def test_a_structured_output_call_is_left_alone(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.MINIMAL)
        structured = ModelRequest(
            model="gpt-x",
            messages=(SYSTEM, ASKED, CALLED, RETURNED),
            output_schema={"type": "object"},
        )

        shaped = shaping.shape(structured, effort=Effort.HIGH)

        assert shaped.effort is Effort.HIGH
        assert "schema" in shaped.reason

    def test_shaping_is_off_until_it_is_turned_on(self) -> None:
        shaped = Shaping().shape(asking(SYSTEM, ASKED, CALLED, RETURNED), effort=Effort.HIGH)

        assert shaped.effort is Effort.HIGH
        assert shaped.clamped is False

    def test_with_no_effort_asked_for_the_baseline_is_what_is_clamped(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.LOW)

        shaped = shaping.shape(asking(SYSTEM, ASKED, CALLED, RETURNED))

        assert shaped.requested is Effort.HIGH
        assert shaped.effort is Effort.LOW


class TestAPolicyThatCouldCostMore:
    """A policy that can increase spend or latency is rejected before it runs at all."""

    def test_a_policy_that_would_raise_effort_is_refused_at_configuration_time(self) -> None:
        with pytest.raises(ConfigurationError):
            Shaping(enabled=True, baseline=Effort.LOW, resumption=Effort.HIGH)

    def test_a_policy_clamping_to_the_baseline_is_allowed(self) -> None:
        assert Shaping(baseline=Effort.LOW, resumption=Effort.LOW).resumption is Effort.LOW


class TestSteeringWithoutRewriting:
    """A rewrite invalidates the cached prefix and gives back more than it saves."""

    def test_the_instruction_is_appended_after_what_was_already_there(self) -> None:
        shaping = Shaping(enabled=True, terseness="Answer directly.")

        steered = shaping.steer(asking())

        assert steered.messages[0].content[0] == SYSTEM.content[0]
        assert steered.messages[0].content[-1] == TextPart(text="Answer directly.")

    def test_steering_twice_produces_the_same_bytes(self) -> None:
        shaping = Shaping(enabled=True, terseness="Answer directly.")

        once = shaping.steer(asking())

        assert shaping.steer(once) == once

    def test_nothing_is_appended_when_there_is_nothing_to_say(self) -> None:
        assert Shaping(enabled=True).steer(asking()) == asking()

    def test_steering_is_off_until_it_is_turned_on(self) -> None:
        assert Shaping(terseness="Answer directly.").steer(asking()) == asking()

    def test_a_prompt_with_no_system_message_gets_one(self) -> None:
        shaping = Shaping(enabled=True, terseness="Answer directly.")

        steered = shaping.steer(ModelRequest(model="gpt-x", messages=(ASKED,)))

        assert steered.messages[0].role == "system"
        assert steered.messages[1] == ASKED

    def test_the_conversation_below_the_system_prompt_is_untouched(self) -> None:
        shaping = Shaping(enabled=True, terseness="Answer directly.")

        steered = shaping.steer(asking(SYSTEM, ASKED, CALLED, RETURNED))

        assert steered.messages[1:] == (ASKED, CALLED, RETURNED)


class TestWhatEachProviderIsToldAndWhatIsRecorded:
    """One expression of effort, mapped per provider, ignored where there is no parameter."""

    def test_openai_is_told_in_its_own_parameter(self) -> None:
        assert provider_effort(Effort.LOW, provider="openai") == {"reasoning_effort": "low"}

    def test_anthropic_is_told_in_its_own_parameter(self) -> None:
        assert provider_effort(Effort.HIGH, provider="anthropic") == {"effort": "high"}

    def test_a_provider_with_no_effort_parameter_ignores_it_without_erroring(self) -> None:
        assert provider_effort(Effort.MINIMAL, provider="llama_cpp") == {}

    def test_the_record_says_what_was_applied_and_why(self) -> None:
        shaping = Shaping(enabled=True, baseline=Effort.HIGH, resumption=Effort.LOW)

        shaped = shaping.shape(asking(SYSTEM, ASKED, CALLED, RETURNED), effort=Effort.HIGH)

        assert shaped.attributes() == {
            "adk.shaping.effort": "low",
            "adk.shaping.requested": "high",
            "adk.shaping.clamped": "true",
            "adk.shaping.turn": "tool_resumption",
        }

    def test_marking_a_failed_result_leaves_the_original_alone(self) -> None:
        errored(RETURNED)

        assert classify([SYSTEM, ASKED, CALLED, RETURNED]) is TurnKind.TOOL_RESUMPTION
