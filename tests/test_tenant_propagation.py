"""The tenant surviving a hop that leaves the process, under one contract rather than six.

In-process the tenant is a contextvar. The moment work crosses to a queue, a peer or a
durable workflow it is whatever somebody remembered to put in the payload, and every
product has invented a different field name for it. These are about the one contract: what
goes on the wire, what is refused on the way in, and what a consumer re-establishes.

The refusals are the point. A message with no tenant is refused rather than run under the
worker's own; a tenant that contradicts the authenticated credential is refused before any
store is touched; a version nobody here understands is refused rather than guessed at.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tesserix_adk.core import (
    HEADER,
    PAYLOAD_KEY,
    VERSION,
    TenantContext,
    TenantContextError,
    TenantCrossingError,
    arriving,
    carried,
    current_tenant,
    header_of,
    in_payload,
    of_payload,
    restored,
    tenant_here,
    tenant_scope,
)
from tesserix_adk.core.propagation import MAX_HEADER_BYTES
from tesserix_adk.core.queue import WorkItem, WorkState
from tesserix_adk.testing import TenantPropagationConformance

if TYPE_CHECKING:
    from collections.abc import Mapping

ACME = TenantContext(tenant="acme", user="ada", locale="en-GB", region="eu-west-1")


class TestWhatGoesOnTheWire:
    def test_one_header_carries_the_whole_context(self) -> None:
        value = header_of(ACME)

        assert value.startswith(f"{VERSION} ")
        assert "tenant=acme" in value
        assert "user=ada" in value
        assert "locale=en-GB" in value

    def test_the_header_has_one_name_every_transport_uses(self) -> None:
        """A shared service cannot honour a field each product named differently."""
        assert carried(ACME) == {HEADER: header_of(ACME)}

    def test_a_value_carrying_a_separator_cannot_rewrite_the_field_beside_it(self) -> None:
        smuggled = TenantContext(tenant="acme", user="ada;tenant=globex")

        assert restored(carried(smuggled)).tenant == "acme"
        assert restored(carried(smuggled)).user == "ada;tenant=globex"

    def test_it_round_trips(self) -> None:
        assert restored(carried(ACME)) == ACME

    def test_a_broker_that_normalises_header_case_has_not_broken_the_contract(self) -> None:
        assert restored({HEADER.upper(): header_of(ACME)}).tenant == "acme"


class TestWhatIsRefusedOnTheWayIn:
    def test_a_message_with_no_context_is_refused_rather_than_run_under_a_default(self) -> None:
        """The consuming worker's own tenant is the wrong answer, not a fallback."""
        with pytest.raises(TenantContextError) as refused:
            restored({})
        assert refused.value.reason == "missing"

    def test_an_unreadable_header_is_refused_rather_than_half_read(self) -> None:
        with pytest.raises(TenantContextError) as refused:
            restored({HEADER: f"{VERSION} user=ada"})
        assert refused.value.reason == "malformed"

    def test_a_blank_tenant_is_not_a_tenant(self) -> None:
        with pytest.raises(TenantContextError) as refused:
            restored({HEADER: f"{VERSION} tenant=;user=ada"})
        assert refused.value.reason == "malformed"

    def test_a_version_this_side_does_not_know_is_refused_rather_than_guessed_at(self) -> None:
        """Reading unknown fields by position is how a tenant becomes a locale."""
        with pytest.raises(TenantContextError) as refused:
            restored({HEADER: "adk/99 tenant=acme"})
        assert refused.value.reason == "version"

    def test_a_context_contradicting_the_credential_is_refused(self) -> None:
        """The payload never outranks what the caller authenticated as."""
        with pytest.raises(TenantContextError) as refused:
            restored(carried(ACME), authenticated="globex")
        assert refused.value.reason == "contradicted"
        assert refused.value.tenant == "globex"

    def test_a_context_agreeing_with_the_credential_is_accepted(self) -> None:
        assert restored(carried(ACME), authenticated="acme").user == "ada"

    def test_a_credential_with_no_context_beside_it_is_still_refused(self) -> None:
        """An authenticated caller who sent no context has sent no context."""
        with pytest.raises(TenantContextError) as refused:
            restored({}, authenticated="acme")
        assert refused.value.reason == "missing"


class TestWorkThatTravelsAsInputRatherThanAsAHeader:
    def test_a_payload_carries_the_context_under_one_key(self) -> None:
        payload = in_payload(ACME, {"booking": "AB-1"})

        assert payload["booking"] == "AB-1"
        assert PAYLOAD_KEY in payload
        assert of_payload(payload) == ACME

    def test_the_payload_the_caller_passed_is_not_mutated(self) -> None:
        original = {"booking": "AB-1"}
        in_payload(ACME, original)

        assert original == {"booking": "AB-1"}

    def test_a_replay_reconstructs_the_same_tenant_rather_than_the_replayer_s(self) -> None:
        """A workflow replayed on another worker reads its input, never ambient state."""
        recorded = in_payload(ACME, {"booking": "AB-1"})

        with tenant_scope("globex"):
            assert of_payload(recorded).tenant == "acme"

    def test_a_payload_with_no_context_is_refused(self) -> None:
        with pytest.raises(TenantContextError) as refused:
            of_payload({"booking": "AB-1"})
        assert refused.value.reason == "missing"

    def test_redelivery_of_the_same_payload_reads_the_same_tenant(self) -> None:
        """A retry and a dead-letter redelivery are the same bytes arriving again."""
        recorded = in_payload(ACME, {"booking": "AB-1"})

        assert {of_payload(recorded) for _ in range(3)} == {ACME}


class TestATransportWithAHeaderCeiling:
    def test_a_context_that_does_not_fit_sheds_what_is_optional_and_says_so(self) -> None:
        wordy = ACME.model_copy(update={"correlation_id": "c" * MAX_HEADER_BYTES})
        value = header_of(wordy)

        assert len(value.encode()) <= MAX_HEADER_BYTES
        assert restored({HEADER: value}).tenant == "acme"
        assert restored({HEADER: value}).user == "ada"
        assert restored({HEADER: value}).correlation_id is None

    def test_what_cannot_be_shed_is_refused_rather_than_truncated(self) -> None:
        """Half a tenant name is a different tenant, so an oversized one is an error."""
        with pytest.raises(TenantContextError) as refused:
            header_of(TenantContext(tenant="t" * MAX_HEADER_BYTES))
        assert refused.value.reason == "oversized"

    def test_a_shed_context_is_marked_partial_so_the_far_side_knows(self) -> None:
        wordy = ACME.model_copy(update={"correlation_id": "c" * MAX_HEADER_BYTES})

        assert restored({HEADER: header_of(wordy)}).partial is True
        assert restored(carried(ACME)).partial is False


class TestIngressBindsIt:
    def test_the_consumer_reads_the_tenant_the_producer_sent(self) -> None:
        with arriving(carried(ACME)) as here:
            assert current_tenant().tenant == "acme"
            assert here.user == "ada"

    def test_the_binding_does_not_outlive_the_message(self) -> None:
        with arriving(carried(ACME)):
            pass
        assert tenant_here() is None

    def test_a_worker_holding_its_own_tenant_cannot_silently_take_another_s_work(self) -> None:
        """The worker's default is exactly the wrong answer, so it is loud."""
        with (
            tenant_scope("globex"),
            pytest.raises(TenantCrossingError),
            arriving(carried(ACME)),
        ):
            pass  # pragma: no cover — refused on entry

    def test_a_message_with_no_context_is_refused_before_anything_is_bound(self) -> None:
        with pytest.raises(TenantContextError), arriving({}):
            pass  # pragma: no cover — refused on entry
        assert tenant_here() is None

    def test_a_frame_carrying_two_tenants_binds_each_message_separately(self) -> None:
        """A multiplexed transport is several messages, never one scope over the batch."""
        globex = TenantContext(tenant="globex", user="grace")
        seen = []
        for message in (carried(ACME), carried(globex)):
            with arriving(message):
                seen.append(current_tenant().tenant)

        assert seen == ["acme", "globex"]
        assert tenant_here() is None


class TestQueuedWorkClaimedMuchLater:
    """The criterion the contract exists for: a worker never runs work under its own tenant."""

    def test_an_item_claimed_days_later_runs_under_the_tenant_it_was_enqueued_under(
        self,
    ) -> None:
        item = WorkItem(id="w1", tenant="acme", payload=in_payload(ACME, {"booking": "AB-1"}))

        later = item.model_copy(update={"state": WorkState.CLAIMED, "worker": "w-7"})
        with arriving(carried(of_payload(later.payload))) as here:
            assert here.user == "ada"
            assert current_tenant().tenant == "acme"

    def test_a_worker_with_its_own_tenant_bound_is_refused_rather_than_served(self) -> None:
        item = WorkItem(id="w1", tenant="acme", payload=in_payload(ACME, {}))

        with (
            tenant_scope("globex"),
            pytest.raises(TenantCrossingError),
            arriving(carried(of_payload(item.payload))),
        ):
            pass  # pragma: no cover — refused on entry


class TestATransportRunsTheConformanceSuite(TenantPropagationConformance):
    """A broker that lower-cases header names and truncates long values, as many do."""

    def round_trip(self, headers: Mapping[str, str]) -> Mapping[str, str]:
        return {name.lower(): value[:MAX_HEADER_BYTES] for name, value in headers.items()}
