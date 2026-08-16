"""Nothing sensitive leaves the process, whatever a caller attached to a span."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pydantic import SecretStr

from tesserix_adk.core import MASK
from tesserix_adk.observability import (
    DENIED_KEYS,
    PAYLOAD_ATTRIBUTES,
    ExportedSpan,
    PendingSpan,
    RedactingSpanProcessor,
    RedactionPolicy,
    SpanEvent,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

CARD = "4111 1111 1111 1111"
TOKEN = "sk-test-0123456789abcdef"  # noqa: S105 — synthetic, and marked so by the prefix


def _span(**overrides: object) -> PendingSpan:
    fields: dict[str, object] = {
        "name": "adk.model",
        "attributes": {"adk.tenant": "acme", "adk.model": "gpt-5"},
    }
    return PendingSpan(**(fields | overrides))  # type: ignore[arg-type]


def _processed(span: PendingSpan, policy: RedactionPolicy | None = None) -> ExportedSpan:
    return RedactingSpanProcessor(policy).process(span)


class TestTheExportPath:
    def test_a_sensitive_value_a_caller_attached_never_reaches_the_backend(self) -> None:
        exported = _processed(_span(attributes={"checkout.card": CARD}))
        assert CARD not in exported.attributes["checkout.card"]
        assert MASK in exported.attributes["checkout.card"]

    def test_the_kits_own_identity_attributes_are_left_alone(self) -> None:
        """Redacting the tenant would leave the spend attributed to nobody."""
        exported = _processed(_span())
        assert exported.attributes["adk.tenant"] == "acme"

    def test_a_secret_str_renders_as_a_mask_rather_than_its_value(self) -> None:
        exported = _processed(_span(attributes={"checkout.key": SecretStr(TOKEN)}))
        assert exported.attributes["checkout.key"] == MASK

    def test_a_denylisted_key_is_dropped_whatever_its_value_looks_like(self) -> None:
        exported = _processed(_span(attributes={"http.authorization": "opaque"}))
        assert "http.authorization" not in exported.attributes
        assert "http.authorization" in exported.redaction.dropped

    def test_the_denylist_covers_the_names_a_credential_usually_travels_under(self) -> None:
        assert {"authorization", "api_key", "password", "cookie"} <= DENIED_KEYS


class TestPayloadAttributes:
    def test_payload_capture_is_off_by_default(self) -> None:
        """Prompt text is the most useful attribute and the most likely to hold a passport."""
        exported = _processed(_span(attributes={"adk.prompt": "hello ada"}))
        assert "adk.prompt" not in exported.attributes

    def test_an_allowlisted_payload_travels_but_still_goes_through_redaction(self) -> None:
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.prompt"}))
        exported = _processed(_span(attributes={"adk.prompt": f"pay with {CARD}"}), policy)
        assert exported.attributes["adk.prompt"] == f"pay with {MASK}"

    def test_the_payload_names_the_kit_knows_about_are_declared(self) -> None:
        assert "adk.prompt" in PAYLOAD_ATTRIBUTES
        assert "adk.tool.arguments" in PAYLOAD_ATTRIBUTES

    def test_a_dropped_payload_leaves_a_reference_a_developer_can_correlate_on(self) -> None:
        exported = _processed(_span(attributes={"adk.prompt": "hello ada"}))
        assert exported.attributes["adk.prompt.ref"].startswith("sha256:")

    def test_the_same_payload_produces_the_same_reference(self) -> None:
        first = _processed(_span(attributes={"adk.prompt": "hello ada"}))
        second = _processed(_span(attributes={"adk.prompt": "hello ada"}))
        assert first.attributes["adk.prompt.ref"] == second.attributes["adk.prompt.ref"]

    def test_a_different_payload_produces_a_different_reference(self) -> None:
        first = _processed(_span(attributes={"adk.prompt": "hello ada"}))
        second = _processed(_span(attributes={"adk.prompt": "hello grace"}))
        assert first.attributes["adk.prompt.ref"] != second.attributes["adk.prompt.ref"]

    def test_references_can_be_turned_off_where_even_a_hash_is_unwelcome(self) -> None:
        policy = RedactionPolicy(content_references=False)
        exported = _processed(_span(attributes={"adk.prompt": "hello ada"}), policy)
        assert "adk.prompt.ref" not in exported.attributes


class TestNestedValues:
    def test_a_secret_inside_a_json_tool_argument_is_found(self) -> None:
        """A top-level string match misses everything a tool actually sends."""
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.tool.arguments"}))
        arguments = json.dumps({"customer": {"card": CARD}, "amount": 12})
        exported = _processed(_span(attributes={"adk.tool.arguments": arguments}), policy)
        assert CARD not in exported.attributes["adk.tool.arguments"]
        assert json.loads(exported.attributes["adk.tool.arguments"])["amount"] == 12

    def test_a_secret_in_a_json_list_is_found(self) -> None:
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.tool.arguments"}))
        arguments = json.dumps({"tokens": [TOKEN, "fine"]})
        exported = _processed(_span(attributes={"adk.tool.arguments": arguments}), policy)
        assert TOKEN not in exported.attributes["adk.tool.arguments"]

    def test_a_denylisted_key_nested_in_json_is_dropped_too(self) -> None:
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.tool.arguments"}))
        arguments = json.dumps({"headers": {"authorization": "opaque"}})
        exported = _processed(_span(attributes={"adk.tool.arguments": arguments}), policy)
        assert "opaque" not in exported.attributes["adk.tool.arguments"]

    def test_a_bare_json_scalar_is_scrubbed_as_text_rather_than_walked(self) -> None:
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.prompt"}))
        exported = _processed(_span(attributes={"adk.prompt": json.dumps(CARD)}), policy)
        assert CARD not in exported.attributes["adk.prompt"]

    def test_a_value_that_only_looks_like_json_is_scrubbed_as_text(self) -> None:
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.prompt"}))
        exported = _processed(_span(attributes={"adk.prompt": f"{{oops {CARD}"}), policy)
        assert CARD not in exported.attributes["adk.prompt"]


class TestEventsAndExceptions:
    def test_a_span_event_goes_through_the_same_path(self) -> None:
        event = SpanEvent(name="tool.called", attributes={"checkout.card": CARD})
        exported = _processed(_span(events=(event,)))
        assert CARD not in exported.events[0].attributes["checkout.card"]

    def test_a_traceback_frame_carrying_an_argument_value_is_scrubbed(self) -> None:
        """The common bypass: the value never was an attribute, it was in the stack."""
        span = _span(exception=f"ValueError: charge failed for {CARD}")
        exported = _processed(span)
        assert exported.exception is not None
        assert CARD not in exported.exception

    def test_a_span_with_no_exception_stays_that_way(self) -> None:
        assert _processed(_span()).exception is None


class TestFailClosed:
    def test_a_detector_that_raises_drops_the_attribute_rather_than_exporting_it(self) -> None:
        processor = RedactingSpanProcessor(detector=_exploding)
        exported = processor.process(_span(attributes={"checkout.note": "anything"}))
        assert "checkout.note" not in exported.attributes
        assert processor.stats.failures == 1

    def test_a_detector_failure_never_costs_the_whole_span(self) -> None:
        """Dropping the trace loses the causality that explains the failure."""
        processor = RedactingSpanProcessor(detector=_exploding)
        attributes = {"adk.tenant": "acme", "checkout.note": "anything"}
        exported = processor.process(_span(attributes=attributes))
        assert exported.name == "adk.model"
        assert exported.attributes["adk.tenant"] == "acme"

    def test_a_detector_that_recognises_a_value_the_shapes_missed_masks_it(self) -> None:
        processor = RedactingSpanProcessor(detector=lambda value: "passport" in value)
        exported = processor.process(_span(attributes={"checkout.note": "passport J123"}))
        assert exported.attributes["checkout.note"] == MASK

    def test_a_detector_that_falls_over_on_a_traceback_masks_the_whole_message(self) -> None:
        """A traceback is the last place to gamble on a partial scrub."""
        processor = RedactingSpanProcessor(detector=_exploding)
        exported = processor.process(_span(exception=f"ValueError: {CARD}"))
        assert exported.exception == MASK

    def test_the_failure_counter_is_zero_when_nothing_went_wrong(self) -> None:
        processor = RedactingSpanProcessor()
        processor.process(_span())
        assert processor.stats.failures == 0

    def test_what_was_redacted_is_counted(self) -> None:
        processor = RedactingSpanProcessor()
        processor.process(_span(attributes={"checkout.card": CARD}))
        assert processor.stats.redacted == 1


class TestSizeCaps:
    def test_a_payload_larger_than_the_cap_is_cut_before_it_is_scanned(self) -> None:
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.prompt"}), max_length=64)
        exported = _processed(_span(attributes={"adk.prompt": "x" * 5_000}), policy)
        assert len(exported.attributes["adk.prompt"]) <= 64

    def test_a_secret_beyond_the_cap_cannot_survive_by_being_far_enough_in(self) -> None:
        policy = RedactionPolicy(payload_attributes=frozenset({"adk.prompt"}), max_length=64)
        exported = _processed(_span(attributes={"adk.prompt": "x" * 5_000 + CARD}), policy)
        assert CARD not in exported.attributes["adk.prompt"]


class TestPolicy:
    def test_a_deployment_can_add_a_shape_the_kit_does_not_know(self) -> None:
        policy = RedactionPolicy(extra_patterns=("CASE-[0-9]{6}",))
        exported = _processed(_span(attributes={"support.ref": "CASE-123456"}), policy)
        assert exported.attributes["support.ref"] == MASK

    def test_a_cap_below_the_reference_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="max_length"):
            RedactionPolicy(max_length=0)


def _exploding(value: str) -> bool:
    message = f"detector fell over on {len(value)} characters"
    raise RuntimeError(message)


class TestTheLeakScan:
    def test_no_seeded_secret_survives_a_realistic_run(self) -> None:
        """The scan that runs in CI: seed every shape, export, and read what came out."""
        policy = RedactionPolicy(payload_attributes=frozenset(PAYLOAD_ATTRIBUTES))
        processor = RedactingSpanProcessor(policy)
        seeded = {
            "card": CARD,
            "token": TOKEN,
            "email": "ada@example.com",
            "jwt": "eyJhbGciOiJIUzI1NiJ9abcdef",
            "bearer": "Bearer opaque-value",
        }
        span = PendingSpan(
            name="adk.tool",
            attributes={
                "adk.tenant": "acme",
                "adk.prompt": f"pay with {CARD} for ada@example.com",
                "adk.tool.arguments": json.dumps({"auth": {"jwt": seeded["jwt"]}}),
                "checkout.bearer": seeded["bearer"],
                "checkout.key": SecretStr(TOKEN),
            },
            events=(SpanEvent(name="tool.called", attributes={"arg": TOKEN}),),
            exception=f"ValueError: {seeded['bearer']} rejected",
        )
        exported = processor.process(span)
        assert not _leaks(exported, seeded.values())

    def test_the_scan_would_notice_a_value_that_did_survive(self) -> None:
        """A leak scan that cannot fail is a leak scan nobody should trust."""
        exported = ExportedSpan(name="adk.tool", attributes={"checkout.card": CARD})
        assert _leaks(exported, [CARD])


def _leaks(exported: ExportedSpan, seeded: Iterable[str]) -> list[str]:
    """Every seeded value that appears anywhere in what would be sent."""
    emitted = exported.model_dump_json()
    return [value for value in seeded if value in emitted]
