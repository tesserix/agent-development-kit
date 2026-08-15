"""Finding identifiers, standing in for them, and where the stand-in has to happen."""

from __future__ import annotations

import pytest

from tesserix_adk.core import (
    DEFAULT_DETECTORS,
    GuardrailEvaluationError,
    GuardVerdict,
    PatternDetector,
    PIIDetector,
    PIIKind,
    PIIMatch,
    placeholder,
    redact,
)
from tesserix_adk.guardrails import PIIGuard
from tesserix_adk.memory.erasure import PIIRedactor
from tesserix_adk.observability.redaction import Redactor

pytestmark = pytest.mark.anyio

CARD_NUMBER = "4111 1111 1111 1111"


class Breaks:
    """A detector that cannot answer, which must never read as 'found nothing'."""

    name = "breaks"

    def detect(self, text: str) -> tuple[PIIMatch, ...]:
        raise RuntimeError(text)


class TestWhatTheBuiltInsRecognise:
    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            (f"pay with {CARD_NUMBER}", PIIKind.CARD),
            ("write to ada@example.test", PIIKind.EMAIL),
            ("account GB29NWBK60161331926819", PIIKind.IBAN),
            ("Authorization: Bearer abc.def-123", PIIKind.BEARER_TOKEN),
            ("key sk-live_0123456789abcdef", PIIKind.API_KEY),
            ("ssn 123-45-6789", PIIKind.NATIONAL_ID),
            ("passport X1234567", PIIKind.PASSPORT),
            ("call +44 20 7946 0958", PIIKind.PHONE),
        ],
    )
    def test_the_kind_is_named_rather_than_the_value_kept(self, text: str, kind: PIIKind) -> None:
        applied = redact(text, tenant="acme", threshold=0.5)

        assert applied.kinds == (kind,)
        assert applied.found

    def test_a_passage_with_nothing_in_it_comes_back_as_it_went_in(self) -> None:
        applied = redact("the booking is confirmed", tenant="acme")

        assert applied.text == "the booking is confirmed"
        assert not applied.found
        assert applied.count == 0

    def test_an_order_reference_is_not_a_card_because_it_fails_luhn(self) -> None:
        assert redact("order 1234567812345678", tenant="acme").kinds == ()

    def test_a_run_too_short_to_be_a_card_fails_the_checksum_outright(self) -> None:
        short = PatternDetector("short", kind=PIIKind.CARD, pattern=r"\d+", checksum="luhn")

        assert not short.detect("18")

    def test_a_detector_that_cannot_answer_refuses_rather_than_passing_content_through(
        self,
    ) -> None:
        with pytest.raises(GuardrailEvaluationError) as refused:
            redact("anything", tenant="acme", detectors=(Breaks(),))

        assert refused.value.details["guard"] == "pii"

    def test_a_detector_is_recognised_structurally(self) -> None:
        assert isinstance(Breaks(), PIIDetector)

    def test_a_detector_reads_as_its_name_rather_than_an_address(self) -> None:
        assert repr(DEFAULT_DETECTORS[0]) == "PatternDetector('card')"

    def test_a_match_reports_its_span_rather_than_its_text(self) -> None:
        match = PIIMatch(kind=PIIKind.EMAIL, start=4, end=20)

        assert match.length == 16


class TestWhatAPlaceholderStandsFor:
    def test_it_keeps_the_type_so_the_agent_still_knows_what_it_is_reasoning_about(self) -> None:
        stood_in = placeholder(PIIKind.PASSPORT, "X1234567", tenant="acme")

        assert stood_in.startswith("[passport:")

    def test_the_same_value_is_the_same_pseudonym_within_one_tenant(self) -> None:
        assert placeholder(PIIKind.EMAIL, "ada@example.test", tenant="acme") == placeholder(
            PIIKind.EMAIL, "ada@example.test", tenant="acme"
        )

    def test_two_tenants_never_share_one_so_neither_can_probe_the_other(self) -> None:
        assert placeholder(PIIKind.EMAIL, "ada@example.test", tenant="acme") != placeholder(
            PIIKind.EMAIL, "ada@example.test", tenant="globex"
        )

    def test_the_value_is_gone_from_the_rewritten_passage(self) -> None:
        applied = redact(f"pay with {CARD_NUMBER} today", tenant="acme")

        assert CARD_NUMBER not in applied.text
        assert applied.text.endswith(" today")


class TestWhereOverRedactionIsTheCost:
    def test_a_low_confidence_shape_is_left_alone_at_the_default_bar(self) -> None:
        assert redact("flight X1234567", tenant="acme", threshold=0.9).kinds == ()

    def test_raising_the_bar_is_how_a_tenant_stops_it(self) -> None:
        loose = redact("flight X1234567", tenant="acme", threshold=0.5)

        assert loose.kinds == (PIIKind.PASSPORT,)

    def test_a_tenant_can_say_a_shape_is_not_an_identifier_after_all(self) -> None:
        applied = redact("ref X1234567", tenant="acme", threshold=0.5, allow=(r"X\d{7}",))

        assert applied.kinds == ()

    def test_the_earlier_detector_wins_an_overlap_so_a_card_stays_a_card(self) -> None:
        loose = PatternDetector("loose", kind=PIIKind.PHONE, pattern=r"\d[\d ]+\d")
        applied = redact(
            f"pay with {CARD_NUMBER}", tenant="acme", detectors=(*DEFAULT_DETECTORS, loose)
        )

        assert applied.kinds == (PIIKind.CARD,)
        assert applied.count == 1


class TestTheGuardOnBothStages:
    async def test_it_redacts_on_the_way_in_so_nothing_reaches_prompt_assembly(self) -> None:
        result = await PIIGuard(tenant="acme").check_input(f"pay with {CARD_NUMBER}")

        assert result.verdict is GuardVerdict.REDACT
        assert result.content is not None
        assert CARD_NUMBER not in result.content

    async def test_it_redacts_on_the_way_out_because_the_model_can_reconstruct_it(self) -> None:
        result = await PIIGuard(tenant="acme").check_output("mail ada@example.test")

        assert result.code == "pii_redacted"
        assert "email" in result.detail

    async def test_untouched_content_is_allowed_rather_than_rewritten(self) -> None:
        result = await PIIGuard(tenant="acme").check_input("the booking is confirmed")

        assert result.verdict is GuardVerdict.ALLOW
        assert result.content is None

    async def test_a_tenant_allow_shape_reaches_the_detectors(self) -> None:
        guard = PIIGuard(tenant="acme", threshold=0.5, allow=(r"X\d{7}",))

        assert (await guard.check_input("ref X1234567")).verdict is GuardVerdict.ALLOW

    async def test_a_custom_detector_stands_beside_the_built_ins(self) -> None:
        cases = PatternDetector("case", kind=PIIKind.NATIONAL_ID, pattern=r"\bCASE-\d{4}\b")
        guard = PIIGuard(tenant="acme", detectors=(*DEFAULT_DETECTORS, cases))

        assert (await guard.check_input("see CASE-4417")).verdict is GuardVerdict.REDACT


class TestTheOtherTwoPathsThatNeedTheSameDetectors:
    def test_the_memory_write_path_stands_in_at_any_depth(self) -> None:
        stored, paths = PIIRedactor(tenant="acme").redact(
            {"turns": [{"text": "mail ada@example.test"}]}
        )

        assert paths == ("turns.0.text",)
        assert "ada@example.test" not in str(stored)

    def test_it_leaves_a_record_with_nothing_in_it_untouched(self) -> None:
        stored, paths = PIIRedactor(tenant="acme").redact({"ok": True, "note": "confirmed"})

        assert stored == {"ok": True, "note": "confirmed"}
        assert paths == ()

    def test_a_tenant_setting_reaches_the_memory_path_too(self) -> None:
        _, paths = PIIRedactor(tenant="acme", threshold=0.5, allow=(r"X\d{7}",)).redact("X1234567")

        assert paths == ()

    def test_the_exporter_masks_flatly_where_no_tenant_was_named(self) -> None:
        kept, _ = Redactor().scrub({"who": "ada@example.test"})

        assert kept["who"] == "[redacted]"

    def test_it_stands_in_typed_where_one_was_so_a_trace_and_a_record_can_be_joined(self) -> None:
        kept, dropped = Redactor(pii_tenant="acme").scrub({"who": "ada@example.test"})

        assert kept["who"] == placeholder(PIIKind.EMAIL, "ada@example.test", tenant="acme")
        assert dropped.dropped == ("who",)
