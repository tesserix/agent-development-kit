"""One trace across a peer hop, a queue and a workflow that suspends for a day."""

from __future__ import annotations

import pytest

from tesserix_adk.observability import (
    MAX_STATE_ENTRIES,
    MAX_STATE_LENGTH,
    TRACEPARENT,
    TRACESTATE,
    VENDOR,
    Link,
    LinkKind,
    W3CContext,
    joined,
    span_id_of,
    trace_id_of,
)

PEER = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


def _root() -> W3CContext:
    return W3CContext.rooted("run-1")


class TestTheWireFormat:
    def test_a_context_renders_as_a_w3c_traceparent(self) -> None:
        header = _root().carried()[TRACEPARENT]
        version, trace, span, flags = header.split("-")
        assert (version, len(trace), len(span)) == ("00", 32, 16)
        assert flags == "01"

    def test_an_unsampled_context_says_so_in_the_flags(self) -> None:
        assert W3CContext.rooted("run-1", sampled=False).carried()[TRACEPARENT].endswith("-00")

    def test_a_traceparent_from_another_vendor_is_understood(self) -> None:
        extracted = W3CContext.extracted({TRACEPARENT: PEER})
        assert extracted is not None
        assert extracted.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
        assert extracted.span_id == "00f067aa0ba902b7"

    def test_headers_are_read_whatever_case_the_broker_normalised_them_to(self) -> None:
        assert W3CContext.extracted({"TraceParent": PEER}) is not None

    def test_a_future_version_is_still_read_rather_than_discarded(self) -> None:
        """A newer version keeps the first three fields; refusing it breaks the trace."""
        assert W3CContext.extracted({TRACEPARENT: PEER.replace("00-", "cc-", 1)}) is not None

    def test_a_malformed_traceparent_is_not_read(self) -> None:
        assert W3CContext.extracted({TRACEPARENT: "nonsense"}) is None

    def test_an_all_zero_trace_id_is_refused_as_the_spec_requires(self) -> None:
        assert W3CContext.extracted({TRACEPARENT: PEER.replace(PEER[3:35], "0" * 32)}) is None

    def test_a_context_with_no_headers_at_all_is_not_read(self) -> None:
        assert W3CContext.extracted({}) is None


class TestTraceState:
    def test_the_kits_own_entry_travels_under_its_vendor_key(self) -> None:
        carried = _root().carried()
        assert carried[TRACESTATE].startswith(f"{VENDOR}=")

    def test_another_vendors_state_survives_the_hop(self) -> None:
        headers = {TRACEPARENT: PEER, TRACESTATE: "congo=t61rcWkgMzE"}
        extracted = W3CContext.extracted(headers)
        assert extracted is not None
        assert extracted.state["congo"] == "t61rcWkgMzE"

    def test_this_process_puts_its_entry_first_as_the_spec_requires(self) -> None:
        headers = {TRACEPARENT: PEER, TRACESTATE: "congo=t61rcWkgMzE"}
        extracted = W3CContext.extracted(headers)
        assert extracted is not None
        assert extracted.child("model").carried()[TRACESTATE].startswith(f"{VENDOR}=")

    def test_state_is_capped_so_a_constrained_transport_still_carries_it(self) -> None:
        crowded = ",".join(f"v{index}=x" for index in range(MAX_STATE_ENTRIES * 2))
        extracted = W3CContext.extracted({TRACEPARENT: PEER, TRACESTATE: crowded})
        assert extracted is not None
        rendered = extracted.carried()[TRACESTATE]
        assert len(rendered.split(",")) <= MAX_STATE_ENTRIES
        assert len(rendered) <= MAX_STATE_LENGTH

    def test_the_oldest_entries_are_the_ones_dropped(self) -> None:
        """The spec's ordering: the nearest participant is the most useful one."""
        crowded = ",".join(f"v{index}=x" for index in range(MAX_STATE_ENTRIES * 2))
        extracted = W3CContext.extracted({TRACEPARENT: PEER, TRACESTATE: crowded})
        assert extracted is not None
        assert "v0=x" in extracted.carried()[TRACESTATE]

    def test_an_unparseable_state_entry_does_not_cost_the_trace(self) -> None:
        extracted = W3CContext.extracted({TRACEPARENT: PEER, TRACESTATE: "junk"})
        assert extracted is not None
        assert extracted.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"


class TestReplaySafety:
    def test_the_same_run_produces_the_same_trace_id_on_every_replay(self) -> None:
        """A workflow replayed after a worker restart must rebuild the same trace."""
        assert W3CContext.rooted("run-1").trace_id == W3CContext.rooted("run-1").trace_id

    def test_two_runs_do_not_share_a_trace_id(self) -> None:
        assert W3CContext.rooted("run-1").trace_id != W3CContext.rooted("run-2").trace_id

    def test_the_same_activity_produces_the_same_span_id_on_every_replay(self) -> None:
        first = _root().child("charge", attempt=1)
        second = _root().child("charge", attempt=1)
        assert first.span_id == second.span_id

    def test_a_retry_of_an_activity_is_a_different_span(self) -> None:
        assert _root().child("charge", attempt=1) != _root().child("charge", attempt=2)

    def test_nothing_here_reads_a_clock_or_a_random_source(self) -> None:
        """Both would make replay reconstruct a different trace than the first run did."""
        assert trace_id_of("run-1") == trace_id_of("run-1")
        assert span_id_of("run-1", "charge") == span_id_of("run-1", "charge")

    def test_an_identifier_is_never_the_all_zero_value_the_spec_forbids(self) -> None:
        assert set(trace_id_of("")) != {"0"}
        assert set(span_id_of("")) != {"0"}


class TestSamplingAcrossHops:
    def test_a_sampled_trace_stays_sampled_over_a_hop(self) -> None:
        """A peer deciding for itself is how a trace arrives truncated at the boundary."""
        downstream, _ = joined(_root().carried(), run_id="run-2")
        assert downstream.sampled

    def test_an_unsampled_trace_is_not_revived_downstream(self) -> None:
        upstream = W3CContext.rooted("run-1", sampled=False)
        downstream, _ = joined(upstream.carried(), run_id="run-2")
        assert not downstream.sampled

    def test_a_hop_keeps_the_trace_id_so_both_sides_land_in_one_trace(self) -> None:
        downstream, _ = joined(_root().carried(), run_id="run-2")
        assert downstream.trace_id == _root().trace_id

    def test_the_downstream_span_is_a_child_of_the_calling_span(self) -> None:
        upstream = _root()
        downstream, _ = joined(upstream.carried(), run_id="run-2")
        assert downstream.parent_span_id == upstream.span_id


class TestABrokenHop:
    def test_a_peer_that_sent_no_context_starts_a_new_root(self) -> None:
        downstream, link = joined({}, run_id="run-2")
        assert downstream.trace_id == trace_id_of("run-2")
        assert link is None

    def test_a_correlated_but_unreadable_context_leaves_a_link_back(self) -> None:
        """A gap somebody can see beats a trace that quietly looks unrelated."""
        _, link = joined({TRACEPARENT: "nonsense", "adk-correlation": "run-1"}, run_id="run-2")
        assert link is not None
        assert link.kind is LinkKind.BROKEN
        assert link.attributes["adk.correlation_id"] == "run-1"

    def test_the_new_root_is_still_usable_when_the_hop_broke(self) -> None:
        downstream, _ = joined({TRACEPARENT: "nonsense"}, run_id="run-2")
        assert downstream.carried()[TRACEPARENT].split("-")[1] == trace_id_of("run-2")


class TestLinks:
    def test_a_fan_out_branch_links_to_the_span_that_started_it(self) -> None:
        link = _root().link(LinkKind.FAN_OUT, branch="north")
        assert (link.trace_id, link.span_id) == (_root().trace_id, _root().span_id)
        assert link.attributes["branch"] == "north"

    def test_a_delayed_continuation_follows_from_a_parent_that_has_ended(self) -> None:
        """Queue delivery hours later is not a child of a span that closed hours ago."""
        link = _root().link(LinkKind.FOLLOWS_FROM)
        assert link.kind is LinkKind.FOLLOWS_FROM

    def test_a_link_renders_as_attributes_a_backend_can_store(self) -> None:
        rendered = _root().link(LinkKind.CONTINUATION).rendered()
        assert rendered["adk.link.kind"] == "continuation"
        assert rendered["adk.link.trace_id"] == _root().trace_id

    def test_a_link_can_be_built_for_a_context_this_process_never_held(self) -> None:
        link = Link(trace_id=trace_id_of("run-9"), span_id=span_id_of("run-9"), kind=LinkKind.FAN_OUT)
        assert link.rendered()["adk.link.span_id"] == span_id_of("run-9")


class TestValidation:
    def test_a_trace_id_that_is_not_hexadecimal_is_refused(self) -> None:
        with pytest.raises(ValueError, match="trace_id"):
            W3CContext(trace_id="z" * 32, span_id="0" * 15 + "1")

    def test_a_span_id_of_the_wrong_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="span_id"):
            W3CContext(trace_id="a" * 32, span_id="abc")
