"""The vocabulary every other layer speaks.

These types are the shared shape of a message, a tool call and what a step cost. They
are frozen, they validate at construction rather than at first use, and they survive a
round trip through JSON — a run checkpointed by one process is rehydrated by another.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from tesserix_adk.core import (
    BinaryPart,
    Cost,
    Message,
    TextPart,
    ToolCall,
    Usage,
    deduplicate,
)


def money(amount: str, currency: str = "USD") -> Cost:
    return Cost(input=Decimal(amount), currency=currency)


class TestUsage:
    def test_tokens_and_cost_round_trip(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=20, cost=money("0.004"))
        assert Usage.model_validate_json(usage.model_dump_json()) == usage

    def test_cost_is_none_for_a_model_that_has_no_price(self) -> None:
        """A self-hosted model costs something; it does not cost zero."""
        usage = Usage(input_tokens=100, output_tokens=20)
        assert usage.cost is None

    def test_a_negative_token_count_is_refused_and_the_field_is_named(self) -> None:
        with pytest.raises(ValidationError, match="input_tokens"):
            Usage(input_tokens=-1, output_tokens=0)

    def test_a_provider_field_the_kit_does_not_model_is_kept(self) -> None:
        """Dropping it loses the evidence; promoting it makes a provider quirk public."""
        usage = Usage(input_tokens=1, output_tokens=1, extras={"audio_seconds": 512})
        assert usage.extras["audio_seconds"] == 512

    def test_usage_is_frozen(self) -> None:
        usage = Usage(input_tokens=1, output_tokens=1)
        with pytest.raises(ValidationError):
            usage.input_tokens = 2

    def test_adding_two_usages_sums_the_tokens(self) -> None:
        total = Usage(input_tokens=10, output_tokens=2) + Usage(input_tokens=5, output_tokens=1)
        assert (total.input_tokens, total.output_tokens) == (15, 3)

    def test_adding_a_priced_and_an_unpriced_usage_gives_an_unknown_cost(self) -> None:
        """Unknown is not zero. A total that silently omits a step understates the bill."""
        total = Usage(input_tokens=1, output_tokens=1, cost=money("0.5")) + Usage(
            input_tokens=1, output_tokens=1
        )
        assert total.cost is None

    def test_adding_two_priced_usages_sums_the_cost(self) -> None:
        total = Usage(input_tokens=1, output_tokens=1, cost=money("0.5")) + Usage(
            input_tokens=1, output_tokens=1, cost=money("0.25")
        )
        assert total.cost is not None
        assert total.cost.total == Decimal("0.75")

    def test_adding_across_currencies_is_refused(self) -> None:
        """Summing USD and EUR produces a number that is true in neither."""
        priced = Usage(input_tokens=1, output_tokens=1, cost=money("1.0"))
        other = Usage(input_tokens=1, output_tokens=1, cost=money("1.0", currency="EUR"))
        with pytest.raises(ValueError, match="neither currency"):
            priced + other

    def test_nothing_spent_yet_does_not_make_the_total_unknown(self) -> None:
        """A run starts on an empty usage. If that counted as an unknown price, no run
        could ever report a cost."""
        priced = Usage(input_tokens=1, output_tokens=1, cost=money("0.5"))
        assert (Usage(input_tokens=0, output_tokens=0) + priced).cost == priced.cost
        assert (priced + Usage(input_tokens=0, output_tokens=0)).cost == priced.cost

    def test_nothing_spent_yet_keeps_the_currency(self) -> None:
        priced = Usage(input_tokens=1, output_tokens=1, cost=money("0.5", currency="EUR"))
        total = (Usage(input_tokens=0, output_tokens=0) + priced).cost
        assert total is not None
        assert total.currency == "EUR"

    def test_an_empty_usage_that_reports_something_is_not_nothing(self) -> None:
        """It carries a field the kit does not model, so its price is genuinely unknown."""
        priced = Usage(input_tokens=1, output_tokens=1, cost=money("0.5"))
        reported = Usage(input_tokens=0, output_tokens=0, extras={"audio_seconds": 8})
        assert (priced + reported).cost is None


class TestMessage:
    def test_a_text_message_round_trips(self) -> None:
        message = Message(role="user", content=[TextPart(text="hello")])
        assert Message.model_validate_json(message.model_dump_json()) == message

    def test_a_role_the_kit_does_not_model_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            Message(role="oracle", content=[TextPart(text="hi")])  # type: ignore[arg-type]

    def test_a_message_is_frozen(self) -> None:
        message = Message(role="user", content=[TextPart(text="hello")])
        with pytest.raises(ValidationError):
            message.role = "assistant"

    def test_a_tool_result_must_say_which_call_it_answers(self) -> None:
        """Without the id, a result cannot be matched to its call after a parallel step."""
        with pytest.raises(ValidationError, match="tool_call_id"):
            Message(role="tool", content=[TextPart(text="42")])

    def test_a_non_tool_message_may_not_carry_a_tool_call_id(self) -> None:
        with pytest.raises(ValidationError, match="tool_call_id"):
            Message(role="user", content=[TextPart(text="hi")], tool_call_id="call_1")

    def test_binary_content_round_trips(self) -> None:
        message = Message(
            role="user", content=[BinaryPart(media_type="image/png", data=b"\x89PNG\r\n")]
        )
        assert Message.model_validate_json(message.model_dump_json()) == message

    def test_binary_content_is_not_in_its_repr(self) -> None:
        """A repr reaches a log or a trace attribute; an image payload must not follow it."""
        part = BinaryPart(media_type="image/png", data=b"\x89PNG\r\nsecret")
        assert "secret" not in repr(part)
        assert "image/png" in repr(part)

    def test_a_binary_repr_says_how_much_was_withheld(self) -> None:
        part = BinaryPart(media_type="image/png", data=b"0123456789")
        assert "10 bytes" in repr(part)

    def test_text_content_is_in_its_repr(self) -> None:
        """Redaction of prompt text belongs to the telemetry exporter, not to the type."""
        assert "hello" in repr(TextPart(text="hello"))

    def test_a_message_with_no_content_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            Message(role="user", content=[])


class TestToolCall:
    def test_a_tool_call_round_trips(self) -> None:
        call = ToolCall(id="call_1", name="search", arguments={"q": "trains"})
        assert ToolCall.model_validate_json(call.model_dump_json()) == call

    def test_a_tool_call_is_frozen(self) -> None:
        call = ToolCall(id="call_1", name="search", arguments={})
        with pytest.raises(ValidationError):
            call.name = "other"

    def test_a_call_is_not_assumed_safe_to_repeat(self) -> None:
        """A retry that re-sends a payment is worse than a retry that does nothing."""
        assert ToolCall(id="call_1", name="pay", arguments={}).idempotent is False

    def test_a_call_with_an_empty_id_is_refused(self) -> None:
        """An empty id makes every call the same call once they are deduplicated."""
        with pytest.raises(ValidationError, match="id"):
            ToolCall(id="", name="search", arguments={})


class TestDeduplication:
    def test_a_repeated_id_is_dropped(self) -> None:
        """A retried provider response repeats the call; running it twice is the bug."""
        first = ToolCall(id="call_1", name="search", arguments={"q": "a"})
        again = ToolCall(id="call_1", name="search", arguments={"q": "a"})
        assert deduplicate([first, again]) == (first,)

    def test_the_first_occurrence_wins(self) -> None:
        first = ToolCall(id="call_1", name="search", arguments={"q": "a"})
        differing = ToolCall(id="call_1", name="search", arguments={"q": "b"})
        assert deduplicate([first, differing]) == (first,)

    def test_distinct_ids_are_all_kept_in_order(self) -> None:
        """Deduplication is by id, never by position: parallel calls repeat names."""
        one = ToolCall(id="call_1", name="search", arguments={"q": "a"})
        two = ToolCall(id="call_2", name="search", arguments={"q": "a"})
        assert deduplicate([one, two]) == (one, two)
