"""Carrying the tenant across a hop, and refusing the messages that cannot be trusted.

Run: uv run python examples/tenant_propagation.py
"""

from __future__ import annotations

from tesserix_adk.core import (
    TenantContext,
    TenantContextError,
    TenantCrossingError,
    arriving,
    carried,
    current_tenant,
    in_payload,
    of_payload,
    tenant_scope,
)
from tesserix_adk.core.propagation import MAX_HEADER_BYTES


def a_peer_call() -> None:
    """The producer attaches the context it is running under; the consumer binds it."""
    with tenant_scope(TenantContext(tenant="acme", user="ada", locale="en-GB")):
        request = {"content-type": "application/json", **carried(current_tenant())}

    print(f"on the wire: {request['adk-tenant']}")  # noqa: T201
    with arriving(request) as here:
        print(f"the far side runs as: {here.tenant} / {here.user}")  # noqa: T201


def queued_work_claimed_much_later() -> None:
    """A worker reads the tenant off the item, never its own."""
    recorded = in_payload(TenantContext(tenant="acme", user="ada"), {"booking": "AB-1"})

    with tenant_scope("globex"):
        print(f"worker is running as globex, item says: {of_payload(recorded).tenant}")  # noqa: T201

    try:
        with tenant_scope("globex"), arriving(carried(of_payload(recorded))):
            pass
    except TenantCrossingError as refused:
        print(f"and taking it anyway is refused: {refused}")  # noqa: T201


def what_is_refused() -> None:
    """Each refusal names itself, so a consumer branches on a value not on message text."""
    acme = carried(TenantContext(tenant="acme", user="ada"))
    for what, headers, credential in (
        ("nothing sent", {}, None),
        ("no tenant named", {"adk-tenant": "adk/1 user=ada"}, None),
        ("a version we do not read", {"adk-tenant": "adk/9 tenant=acme"}, None),
        ("disagrees with the credential", acme, "globex"),
    ):
        try:
            with arriving(headers, authenticated=credential):
                pass
        except TenantContextError as refused:
            print(f"{what}: {refused.reason}")  # noqa: T201


def a_transport_with_a_ceiling() -> None:
    """What does not fit is shed, and the far side is told it was."""
    wordy = TenantContext(tenant="acme", user="ada", correlation_id="c" * MAX_HEADER_BYTES)

    with arriving(carried(wordy)) as here:
        print(f"tenant {here.tenant} and user {here.user} survived")  # noqa: T201
        print(f"correlation_id shed: {here.correlation_id is None}, partial: {here.partial}")  # noqa: T201


if __name__ == "__main__":
    a_peer_call()
    queued_work_claimed_much_later()
    what_is_refused()
    a_transport_with_a_ceiling()
