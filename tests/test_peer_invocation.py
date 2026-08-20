"""Calling a peer: typed both ways, inside one trace, against one budget."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.a2a import (
    AgentCard,
    AgentLimits,
    AgentSkill,
    PeerCall,
    PeerClient,
    PeerInvocationError,
    PeerInvocationReason,
    PeerProgress,
    PeerReply,
    PeerResult,
)
from tesserix_adk.a2a.invocation import UnsupportedSchemaError, checkable, conforms
from tesserix_adk.core import (
    AgentIdentity,
    BudgetExceededError,
    BudgetLimits,
    BudgetScope,
    CapabilityError,
    CountSource,
    DelegationLimitError,
    Principal,
    RunBudget,
    ScopedLimits,
    Usage,
    most_restrictive,
)
from tesserix_adk.observability import W3CContext
from tesserix_adk.runtime import CancellationToken
from tesserix_adk.testing import FakeClock
from tesserix_adk.tools import CredentialBroker, CredentialRequest, ExchangedCredentials

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

READ = "itinerary:read"
WRITE = "payments:write"
PRICE = "price_leg"
REFUND = "refund"

_INPUT = {
    "type": "object",
    "properties": {"leg": {"type": "string"}, "seats": {"type": "integer"}},
    "required": ["leg"],
    "additionalProperties": False,
}
_OUTPUT = {
    "type": "object",
    "properties": {"eur": {"type": "number"}},
    "required": ["eur"],
    "additionalProperties": False,
}


def _card(*, available: bool = True, **overrides: Any) -> AgentCard:
    """A peer that prices legs and, with a scope, refunds them."""
    fields: dict[str, Any] = {
        "agent": "booker",
        "audience": "https://booker.example.gov",
        "declared": (READ, WRITE),
        "available": available,
        "limits": AgentLimits(max_payload_bytes=4096),
        "skills": (
            AgentSkill(
                name=PRICE,
                description="Price one leg.",
                input_schema=_INPUT,
                output_schema=_OUTPUT,
                idempotent=True,
            ),
            AgentSkill(
                name=REFUND,
                description="Refund an order.",
                input_schema={"type": "object", "properties": {"order": {"type": "string"}}},
                required_scopes=(WRITE,),
            ),
        ),
    }
    return AgentCard(**{**fields, **overrides})


class _Exchange:
    """A token endpoint that mints one token per call."""

    async def exchange(self, request: CredentialRequest) -> tuple[str, float]:
        return f"tok-{request.audience}", 300.0


class _Peer:
    """The other agent, as a transport the test drives."""

    def __init__(self, reply: PeerReply | None = None) -> None:
        self.reply = reply or PeerReply(output={"eur": 40.0})
        self.calls: list[PeerCall] = []
        self.cancelled: list[str] = []
        self.fails: Exception | None = None
        self.progress: tuple[PeerProgress, ...] = ()
        self.started = asyncio.Event()
        self.hangs = False

    async def invoke(self, call: PeerCall) -> PeerReply:
        self.calls.append(call)
        self.started.set()
        if self.hangs:
            await asyncio.Event().wait()
        if self.fails is not None:
            raise self.fails
        return self.reply

    async def stream(self, call: PeerCall) -> AsyncIterator[PeerProgress | PeerReply]:
        self.calls.append(call)
        for event in self.progress:
            yield event
        yield self.reply

    async def cancel(self, call: PeerCall) -> None:
        self.cancelled.append(call.call_id)


def _identity(*held: str, tenant: str = "acme") -> AgentIdentity:
    return AgentIdentity.resolve(
        agent="desk",
        declared=(READ, WRITE),
        principal=Principal(subject="ada", tenant=tenant, scopes=frozenset(held or (READ,))),
    )


def _budget(**stated: Any) -> RunBudget:
    return RunBudget(
        resolved=most_restrictive(
            ScopedLimits(scope=BudgetScope.RUN, limits=BudgetLimits(**stated))
        ),
        clock=FakeClock(),
    )


def _client(
    peer: _Peer | None = None,
    *,
    card: AgentCard | None = None,
    identity: AgentIdentity | None = None,
    **options: Any,
) -> tuple[PeerClient, _Peer]:
    """A client for one peer, with a working credential source and a clock a test moves."""
    clock = FakeClock()
    transport = peer or _Peer()
    client = PeerClient(
        card or _card(),
        transport,
        credentials=CredentialBroker(ExchangedCredentials(_Exchange(), clock=clock), clock=clock),
        identity=identity or _identity(READ),
        run_id="run_1",
        clock=clock,
        **options,
    )
    return client, transport


class TestCallingAPeerWithTypes:
    async def test_a_valid_call_returns_the_peer_s_answer(self) -> None:
        client, _ = _client()
        result = await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert result.output == {"eur": 40.0}

    async def test_a_skill_the_card_does_not_publish_is_not_called(self) -> None:
        client, peer = _client()
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke("book_hotel", {})
        assert (refused.value.reason, peer.calls) == (PeerInvocationReason.UNKNOWN_SKILL, [])

    async def test_a_payload_the_peer_would_reject_is_not_sent(self) -> None:
        client, peer = _client()
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK", "cabin": "first"})
        assert (refused.value.reason, peer.calls) == (PeerInvocationReason.INPUT_SCHEMA, [])

    async def test_a_missing_required_argument_is_refused_rather_than_filled_in(self) -> None:
        client, _ = _client()
        with pytest.raises(PeerInvocationError, match="leg"):
            await client.invoke(PRICE, {"seats": 2})

    async def test_an_answer_violating_the_declared_output_is_not_coerced(self) -> None:
        client, _ = _client(_Peer(PeerReply(output={"eur": "forty"})))
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert refused.value.reason is PeerInvocationReason.OUTPUT_SCHEMA

    async def test_a_refusal_carries_the_peer_the_skill_and_what_arrived(self) -> None:
        client, _ = _client(_Peer(PeerReply(output={"eur": "forty"})))
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert (refused.value.peer, refused.value.skill) == ("booker", PRICE)
        assert "forty" in refused.value.payload

    async def test_a_degraded_peer_is_not_called(self) -> None:
        client, peer = _client(card=_card(available=False))
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert (refused.value.reason, peer.calls) == (PeerInvocationReason.UNAVAILABLE, [])

    async def test_a_payload_beyond_what_the_peer_accepts_is_refused_here(self) -> None:
        client, peer = _client()
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "x" * 8192})
        assert (refused.value.reason, peer.calls) == (PeerInvocationReason.TOO_LARGE, [])


class TestWhatTravelsWithTheCall:
    async def test_the_call_carries_the_user_and_the_tenant_not_the_service(self) -> None:
        client, peer = _client()
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        meta = peer.calls[0].meta
        assert "ada" in " ".join(meta.values())
        assert "acme" in " ".join(meta.values())

    async def test_both_agents_appear_in_one_trace(self) -> None:
        trace = W3CContext.rooted("run_1")
        client, peer = _client(trace=trace)
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert peer.calls[0].meta["traceparent"].split("-")[1] == trace.trace_id

    async def test_the_hop_is_a_child_span_rather_than_the_caller_s_own(self) -> None:
        trace = W3CContext.rooted("run_1")
        client, peer = _client(trace=trace)
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert peer.calls[0].meta["traceparent"].split("-")[2] != trace.span_id

    async def test_the_credential_travels_in_headers_and_never_in_the_metadata(self) -> None:
        client, peer = _client()
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        call = peer.calls[0]
        assert "authorization" in {name.lower() for name in call.headers}
        assert "tok-" not in " ".join(call.meta.values())

    async def test_a_call_never_renders_the_credential_it_carries(self) -> None:
        client, peer = _client()
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert "tok-" not in repr(peer.calls[0])


class TestScopeIsAttenuatedNeverWidened:
    async def test_a_skill_needing_more_than_the_caller_holds_is_refused(self) -> None:
        client, peer = _client(identity=_identity(READ))
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(REFUND, {"order": "o-1"})
        assert (refused.value.reason, peer.calls) == (PeerInvocationReason.SCOPE_ESCALATION, [])

    async def test_the_peer_is_sent_no_more_than_the_caller_holds(self) -> None:
        client, peer = _client(identity=_identity(READ))
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        carried = peer.calls[0].meta["tesserix/adk/delegation/scopes"].split()
        assert carried == [READ]

    async def test_a_caller_holding_the_scope_may_use_the_skill_that_needs_it(self) -> None:
        client, peer = _client(identity=_identity(READ, WRITE), peer=None)
        result = await client.invoke(REFUND, {"order": "o-1"})
        assert result.skill == REFUND
        assert peer.calls[0].meta["tesserix/adk/delegation/scopes"].split() == [WRITE]


class TestDelegationCannotLoop:
    async def test_an_agent_already_in_the_chain_is_not_called_again(self) -> None:
        client, _ = _client(card=_card(agent="desk"))
        with pytest.raises(DelegationLimitError, match="cycle"):
            await client.invoke(PRICE, {"leg": "LHR-JFK"})

    async def test_the_chain_the_call_carried_is_on_the_result(self) -> None:
        client, _ = _client()
        result = await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert result.chain == ("desk",)


class TestUsageIsChargedToTheCallingRun:
    async def test_what_the_peer_reports_is_spent_from_the_caller_s_budget(self) -> None:
        reply = PeerReply(output={"eur": 40.0}, usage=Usage(input_tokens=900, output_tokens=100))
        budget = _budget(max_input_tokens=2000)
        client, _ = _client(_Peer(reply), budget=budget)
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert budget.spent.usage.input_tokens == 900

    async def test_a_peer_call_counts_against_the_run_s_peer_ceiling(self) -> None:
        budget = _budget(max_peer_invocations=1)
        client, _ = _client(budget=budget)
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert refused.value.reason is PeerInvocationReason.BUDGET

    async def test_a_call_the_remaining_budget_cannot_cover_is_not_made(self) -> None:
        reply = PeerReply(output={"eur": 40.0}, usage=Usage(input_tokens=1500, output_tokens=0))
        budget = _budget(max_input_tokens=1000)
        client, peer = _client(_Peer(reply), budget=budget)
        with pytest.raises(BudgetExceededError):
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert (refused.value.reason, len(peer.calls)) == (PeerInvocationReason.BUDGET, 1)

    async def test_a_peer_reporting_implausible_usage_cannot_spend_the_run(self) -> None:
        reply = PeerReply(
            output={"eur": 40.0}, usage=Usage(input_tokens=10**9, output_tokens=10**9)
        )
        budget = _budget(max_input_tokens=1000)
        client, _ = _client(_Peer(reply), budget=budget, max_reported_tokens=100)
        result = await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert (result.usage.input_tokens, result.usage_trusted) == (100, False)

    async def test_a_peer_reporting_nothing_is_recorded_as_unmeasured_not_as_free(self) -> None:
        client, _ = _client(_Peer(PeerReply(output={"eur": 40.0})))
        result = await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert (result.usage.source, result.usage_trusted) == (CountSource.HEURISTIC, False)

    async def test_the_result_is_attributable_without_carrying_the_answer(self) -> None:
        client, _ = _client()
        attributes = (await client.invoke(PRICE, {"leg": "LHR-JFK"})).attributes()
        assert attributes["a2a.peer"] == "booker"
        assert "LHR-JFK" not in " ".join(attributes.values())


class TestARetryDoesNotDuplicateTheEffect:
    async def test_an_effectful_skill_carries_a_key_the_peer_can_deduplicate_on(self) -> None:
        client, peer = _client(identity=_identity(READ, WRITE))
        await client.invoke(REFUND, {"order": "o-1"})
        assert peer.calls[0].idempotency_key

    async def test_the_same_call_retried_carries_the_same_key(self) -> None:
        client, peer = _client(identity=_identity(READ, WRITE))
        await client.invoke(REFUND, {"order": "o-1"})
        await client.invoke(REFUND, {"order": "o-1"})
        assert peer.calls[0].idempotency_key == peer.calls[1].idempotency_key

    async def test_a_different_call_does_not(self) -> None:
        client, peer = _client(identity=_identity(READ, WRITE))
        await client.invoke(REFUND, {"order": "o-1"})
        await client.invoke(REFUND, {"order": "o-2"})
        assert peer.calls[0].idempotency_key != peer.calls[1].idempotency_key

    async def test_a_skill_that_declared_itself_idempotent_needs_no_key(self) -> None:
        client, peer = _client()
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert peer.calls[0].idempotency_key is None

    async def test_a_caller_resuming_work_may_supply_the_key_itself(self) -> None:
        client, peer = _client(identity=_identity(READ, WRITE))
        await client.invoke(REFUND, {"order": "o-1"}, idempotency_key="settled-2026-08")
        assert peer.calls[0].idempotency_key == "settled-2026-08"


class TestWhenThePeerDoesNotAnswer:
    async def test_a_transport_failure_is_typed_and_does_not_invent_an_answer(self) -> None:
        peer = _Peer()
        peer.fails = ConnectionError("no route to booker")
        client, _ = _client(peer)
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert refused.value.reason is PeerInvocationReason.TRANSPORT

    async def test_a_peer_that_never_answers_stops_at_the_caller_s_deadline(self) -> None:
        peer = _Peer()
        peer.hangs = True
        client, _ = _client(peer, timeout_seconds=0.01)
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert refused.value.reason is PeerInvocationReason.TIMED_OUT

    async def test_a_deadline_that_has_passed_tells_the_peer_to_stop(self) -> None:
        peer = _Peer()
        peer.hangs = True
        client, _ = _client(peer, timeout_seconds=0.01)
        with pytest.raises(PeerInvocationError):
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert peer.cancelled == [peer.calls[0].call_id]

    async def test_a_call_cannot_be_given_longer_than_the_run_has_left(self) -> None:
        peer = _Peer()
        peer.hangs = True
        client, _ = _client(peer, timeout_seconds=0.01)
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"}, deadline_seconds=30.0)
        assert refused.value.reason is PeerInvocationReason.TIMED_OUT


class TestCancellationReachesThePeer:
    async def test_a_cancelled_run_stops_waiting_for_the_peer(self) -> None:
        peer = _Peer()
        peer.hangs = True
        client, _ = _client(peer)
        token = CancellationToken()
        call = asyncio.create_task(client.invoke(PRICE, {"leg": "LHR-JFK"}, cancellation=token))
        await peer.started.wait()
        token.cancel("caller went away")
        with pytest.raises(PeerInvocationError) as refused:
            await call
        assert refused.value.reason is PeerInvocationReason.CANCELLED

    async def test_the_peer_is_told_rather_than_left_working(self) -> None:
        peer = _Peer()
        peer.hangs = True
        client, _ = _client(peer)
        token = CancellationToken()
        call = asyncio.create_task(client.invoke(PRICE, {"leg": "LHR-JFK"}, cancellation=token))
        await peer.started.wait()
        token.cancel()
        with pytest.raises(PeerInvocationError):
            await call
        assert peer.cancelled == [peer.calls[0].call_id]

    async def test_a_run_cancelled_before_the_call_never_starts_it(self) -> None:
        client, peer = _client()
        token = CancellationToken()
        token.cancel()
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"}, cancellation=token)
        assert (refused.value.reason, peer.calls) == (PeerInvocationReason.CANCELLED, [])


class TestWorkThatTakesAWhile:
    async def test_progress_arrives_before_the_answer_does(self) -> None:
        peer = _Peer()
        peer.progress = (PeerProgress(note="pricing"), PeerProgress(note="checking fares"))
        client, _ = _client(peer)
        seen = [event async for event in client.stream(PRICE, {"leg": "LHR-JFK"})]
        assert [event.note for event in seen if isinstance(event, PeerProgress)] == [
            "pricing",
            "checking fares",
        ]

    async def test_the_last_thing_streamed_is_the_validated_result(self) -> None:
        peer = _Peer()
        peer.progress = (PeerProgress(note="pricing"),)
        client, _ = _client(peer)
        seen = [event async for event in client.stream(PRICE, {"leg": "LHR-JFK"})]
        assert isinstance(seen[-1], PeerResult)
        assert seen[-1].output == {"eur": 40.0}

    async def test_a_streamed_answer_is_held_to_the_same_output_schema(self) -> None:
        client, _ = _client(_Peer(PeerReply(output={"eur": "forty"})))
        with pytest.raises(PeerInvocationError) as refused:
            _ = [event async for event in client.stream(PRICE, {"leg": "LHR-JFK"})]
        assert refused.value.reason is PeerInvocationReason.OUTPUT_SCHEMA

    async def test_a_run_cancelled_mid_answer_stops_and_tells_the_peer(self) -> None:
        peer = _Peer()
        peer.progress = (PeerProgress(note="pricing"), PeerProgress(note="checking fares"))
        client, _ = _client(peer)
        token = CancellationToken()

        async def consume() -> None:
            async for event in client.stream(PRICE, {"leg": "LHR-JFK"}, cancellation=token):
                del event
                token.cancel("caller went away")

        with pytest.raises(PeerInvocationError) as refused:
            await consume()
        assert (refused.value.reason, peer.cancelled) == (
            PeerInvocationReason.CANCELLED,
            [peer.calls[0].call_id],
        )

    async def test_a_streamed_call_is_validated_before_anything_is_sent(self) -> None:
        client, peer = _client()
        with pytest.raises(PeerInvocationError):
            _ = [event async for event in client.stream(PRICE, {"cabin": "first"})]
        assert peer.calls == []


class TestPersonalDataInAPeerAnswer:
    async def test_the_answer_is_redacted_before_anything_records_it(self) -> None:
        reply = PeerReply(output={"eur": 40.0, "who": "ada@example.gov"})
        client, _ = _client(
            _Peer(reply),
            card=_card(skills=(_card().skills[0].model_copy(update={"output_schema": None}),)),
        )
        result = await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert "ada@example.gov" not in str(result.redacted())


class TestTheSchemaCheckIsTheOneThePeerPublished:
    async def test_a_schema_keyword_the_kit_cannot_check_refuses_the_call(self) -> None:
        card = _card(
            skills=(
                AgentSkill(
                    name=PRICE,
                    description="Price one leg.",
                    input_schema={"type": "object", "not": {"type": "string"}},
                    idempotent=True,
                ),
            )
        )
        client, peer = _client(card=card)
        with pytest.raises(PeerInvocationError) as refused:
            await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert (refused.value.reason, peer.calls) == (PeerInvocationReason.SCHEMA_UNSUPPORTED, [])

    @pytest.mark.parametrize(
        ("schema", "payload", "accepted"),
        [
            ({"type": "object"}, {"anything": 1}, True),
            ({"type": "object", "required": ["a"]}, {}, False),
            ({"type": "array", "items": {"type": "integer"}}, [1, 2], True),
            ({"type": "array", "items": {"type": "integer"}}, [1, "2"], False),
            ({"type": "string", "minLength": 2}, "ab", True),
            ({"type": "string", "minLength": 2}, "a", False),
            ({"type": ["string", "null"]}, None, True),
            ({"enum": ["a", "b"]}, "c", False),
            ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, 3, True),
            ({"anyOf": [{"type": "string"}, {"type": "integer"}]}, 3.5, False),
            ({"type": "integer"}, True, False),
            ({"type": "number"}, 3, True),
        ],
    )
    def test_the_subset_agrees_with_a_real_json_schema_validator(
        self, schema: dict[str, Any], payload: Any, accepted: bool
    ) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        from tesserix_adk.a2a.invocation import conforms

        assert (conforms(payload, schema) == "") is accepted
        validator = jsonschema.validators.validator_for(schema)(schema)
        assert validator.is_valid(payload) is accepted


class TestPricedPeerWork:
    async def test_a_cost_the_peer_reports_lands_on_the_caller_s_bill(self) -> None:
        from tesserix_adk.core import Cost

        reply = PeerReply(
            output={"eur": 40.0},
            usage=Usage(
                input_tokens=10, output_tokens=1, cost=Cost(input=Decimal("0.02"), currency="USD")
            ),
        )
        budget = _budget(max_cost=Decimal("1.00"))
        client, _ = _client(_Peer(reply), budget=budget)
        await client.invoke(PRICE, {"leg": "LHR-JFK"})
        assert budget.spent.usage.cost is not None
        assert budget.spent.usage.cost.total == Decimal("0.02")


class TestTheClientRefusesToBeUseless:
    def test_a_client_that_would_never_wait_is_not_built(self) -> None:
        with pytest.raises(ValueError, match="timeout_seconds"):
            _client(timeout_seconds=0.0)

    async def test_a_transport_that_cannot_stream_says_so_rather_than_blocking(self) -> None:
        class _Plain:
            async def invoke(self, call: PeerCall) -> PeerReply:
                del call
                return PeerReply(output={"eur": 40.0})

            async def cancel(self, call: PeerCall) -> None:
                del call

        client, _ = _client()
        plain = PeerClient(
            _card(),
            _Plain(),
            credentials=CredentialBroker(
                ExchangedCredentials(_Exchange(), clock=FakeClock()), clock=FakeClock()
            ),
            identity=_identity(READ),
            run_id="run_1",
            clock=FakeClock(),
        )
        del client
        with pytest.raises(CapabilityError, match="does not stream"):
            _ = [event async for event in plain.stream(PRICE, {"leg": "LHR-JFK"})]

    async def test_a_call_that_answers_stops_watching_for_cancellation(self) -> None:
        client, _ = _client()
        result = await client.invoke(PRICE, {"leg": "LHR-JFK"}, cancellation=CancellationToken())
        assert result.output == {"eur": 40.0}


class TestTheSchemaSubset:
    def test_a_schema_within_the_subset_is_accepted_whole(self) -> None:
        within = {"type": "object", "properties": {"leg": {"type": "string"}}}
        checkable(within)
        assert conforms({"leg": "LHR"}, within) == ""

    def test_a_keyword_buried_in_a_property_is_found_before_any_call(self) -> None:
        buried = {"type": "object", "properties": {"leg": {"pattern": "^L"}}}
        with pytest.raises(UnsupportedSchemaError) as refused:
            checkable(buried)
        assert refused.value.keyword == "pattern"

    def test_a_keyword_in_an_alternative_or_an_item_is_found_too(self) -> None:
        for schema in (
            {"anyOf": [{"type": "string"}, {"not": {"type": "null"}}]},
            {"type": "array", "items": {"uniqueItems": True}},
            {"type": "object", "additionalProperties": {"multipleOf": 2}},
            {"$defs": {"leg": {"pattern": "^L"}}, "$ref": "#/$defs/leg"},
        ):
            with pytest.raises(UnsupportedSchemaError):
                checkable(schema)

    def test_a_local_definition_is_followed(self) -> None:
        schema = {
            "$defs": {"Leg": {"type": "string"}},
            "type": "object",
            "properties": {"leg": {"$ref": "#/$defs/Leg"}},
        }
        assert conforms({"leg": "LHR"}, schema) == ""
        assert conforms({"leg": 1}, schema) != ""

    def test_a_definition_somewhere_else_is_not_fetched(self) -> None:
        with pytest.raises(UnsupportedSchemaError, match="outside"):
            conforms("x", {"$ref": "https://schemas.example.gov/leg.json"})

    def test_a_definition_the_schema_does_not_carry_is_refused(self) -> None:
        with pytest.raises(UnsupportedSchemaError, match="does not carry"):
            conforms("x", {"$ref": "#/$defs/Missing"})

    def test_a_type_nobody_defined_is_refused_rather_than_assumed(self) -> None:
        with pytest.raises(UnsupportedSchemaError, match="JSON type"):
            conforms("x", {"type": "date"})

    @pytest.mark.parametrize(
        ("schema", "payload", "accepted"),
        [
            ({"const": "eur"}, "eur", True),
            ({"const": "eur"}, "usd", False),
            ({"type": "integer", "maximum": 4}, 5, False),
            ({"type": "string", "maxLength": 2}, "abc", False),
            ({"type": "array", "minItems": 2}, [1], False),
            ({"type": "array", "maxItems": 1}, [1, 2], False),
            ({"type": "boolean", "minimum": 1}, True, True),
            ({"type": "array", "items": {"type": "object"}}, [{"a": 1}], True),
            ({"type": "array"}, ["anything"], True),
            ({"type": "object", "properties": {"a": {"type": "integer"}}}, {"b": 1}, True),
        ],
    )
    def test_the_bounds_agree_with_a_real_json_schema_validator(
        self, schema: dict[str, Any], payload: Any, accepted: bool
    ) -> None:
        jsonschema = pytest.importorskip("jsonschema")
        assert (conforms(payload, schema) == "") is accepted
        if "minimum" not in schema:
            validator = jsonschema.validators.validator_for(schema)(schema)
            assert validator.is_valid(payload) is accepted


class TestRedactingWhatCameBack:
    async def test_identifiers_nested_in_the_answer_are_replaced_too(self) -> None:
        reply = PeerReply(output={"legs": [{"who": "ada@example.gov"}], "eur": 40.0, "note": None})
        skill = _card().skills[0].model_copy(update={"output_schema": None})
        client, _ = _client(_Peer(reply), card=_card(skills=(skill,)))
        redacted = (await client.invoke(PRICE, {"leg": "LHR-JFK"})).redacted(tenant="acme")
        assert "ada@example.gov" not in str(redacted)
        assert redacted["eur"] == 40.0
