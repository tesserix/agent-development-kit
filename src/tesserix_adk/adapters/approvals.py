"""Where an approval question actually goes: a queue, a webhook, or a terminal.

Each of these is only delivery. None of them decides anything, none of them holds the run,
and a question that could not be delivered raises rather than returning silence — a gate
that reads an outage as an answer is a gate that opens on the day the queue is down.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import sys
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import ValidationError

from tesserix_adk.core.errors import ApprovalDeliveryError, ConfigurationError
from tesserix_adk.core.hooks import ApprovalDecision

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pydantic import SecretStr

    from tesserix_adk.core.hooks import ApprovalRecord
    from tesserix_adk.core.protocols import Clock

__all__ = [
    "DEFAULT_APPROVAL_SUBJECT",
    "ConsoleApprovals",
    "HttpPoster",
    "MessagePublisher",
    "NatsApprovals",
    "WebhookApprovals",
]

DEFAULT_APPROVAL_SUBJECT = "adk.approvals"

_YES = frozenset({"y", "yes", "approve", "approved"})


@runtime_checkable
class MessagePublisher(Protocol):
    """The one method this adapter uses from a NATS client."""

    async def publish(self, subject: str, payload: bytes) -> None:
        """Publish `payload` on `subject`."""
        ...


@runtime_checkable
class HttpPoster(Protocol):
    """The one method this adapter uses from an HTTP client."""

    async def post(
        self, url: str, *, content: bytes, headers: Mapping[str, str]
    ) -> tuple[int, bytes]:
        """POST `content` and return the status and body."""
        ...


class NatsApprovals:
    """Publishes the question on a per-tenant subject; the answer arrives out of band.

    The subject carries the tenant so a subscriber can be authorised for its own and no
    other, which is the isolation the queue can enforce and the payload cannot.

    Args:
        publisher: A connected NATS client.
        subject: The subject root. The tenant is appended to it.

    Raises:
        ConfigurationError: If the root is not a plain subject token.
    """

    def __init__(
        self, publisher: MessagePublisher, *, subject: str = DEFAULT_APPROVAL_SUBJECT
    ) -> None:
        self._publisher = publisher
        self._subject = _token(subject, allow_dots=True)

    async def deliver(self, record: ApprovalRecord) -> None:
        """Publish `record` for whoever is subscribed to its tenant.

        Raises:
            ConfigurationError: If the tenant could not be used as a subject token, which
                would publish the question to a wider audience than the tenant.
        """
        await self._publisher.publish(
            f"{self._subject}.{_token(record.tenant)}", record.model_dump_json().encode()
        )


class WebhookApprovals:
    """POSTs the question to a signed endpoint, and reads an answer only if one came back.

    The body is signed so the receiver can tell the court of one agent from anybody else
    who found the URL. A 2xx with no decision in it means the answer will arrive out of
    band; anything else is a delivery failure.

    Args:
        poster: An HTTP client.
        url: Where to post. HTTPS only.
        secret: The shared secret the signature is taken under.
        headers: Anything else the receiver needs, e.g. a bearer token.

    Raises:
        ConfigurationError: If the URL is not HTTPS.
    """

    def __init__(
        self,
        poster: HttpPoster,
        *,
        url: str,
        secret: SecretStr,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        if not url.startswith("https://"):
            raise ConfigurationError(
                f"an approval webhook must be https, not {url!r}; the question names the "
                f"tenant, the run and the tool"
            )
        self._poster = poster
        self._url = url
        self._secret = secret
        self._headers = dict(headers or {})

    async def deliver(self, record: ApprovalRecord) -> ApprovalDecision | None:
        """POST `record`, returning the decision where the receiver gave one.

        Raises:
            ApprovalDeliveryError: If the receiver refused the post, or answered with
                something that is not a decision about this request. Both fail closed.
        """
        body = record.model_dump_json().encode()
        status, answer = await self._poster.post(
            self._url, content=body, headers=self._signed(body)
        )
        if not 200 <= status < 300:
            raise ApprovalDeliveryError(
                f"the approval endpoint answered {status} for {record.tool_name!r}; "
                f"nobody was asked"
            )
        return self._read(record, answer)

    def _signed(self, body: bytes) -> dict[str, str]:
        """The headers for one post, including a signature over exactly this body."""
        signature = hmac.new(
            self._secret.get_secret_value().encode(), body, hashlib.sha256
        ).hexdigest()
        return {
            **self._headers,
            "Content-Type": "application/json",
            "X-Adk-Signature": f"sha256={signature}",
        }

    def _read(self, record: ApprovalRecord, answer: bytes) -> ApprovalDecision | None:
        """An answer in the response, where there is one that answers this question."""
        if not answer.strip():
            return None
        try:
            decision = ApprovalDecision.model_validate_json(answer)
        except ValidationError as malformed:
            raise ApprovalDeliveryError(
                f"the approval endpoint's answer for {record.tool_name!r} is not a decision"
            ) from malformed
        if decision.record_id != record.id:
            raise ApprovalDeliveryError(
                f"the approval endpoint answered a different request than {record.id!r}"
            )
        return decision


class ConsoleApprovals:
    """Asks whoever is at the terminal, for a single-operator deployment or a local run.

    Anything that is not a yes is a no, because a gate that reads an empty line or a
    disconnected terminal as consent is not a gate.

    Args:
        approver: Who is at the terminal. Recorded on the decision.
        ask: Reads one line. Runs off the event loop, so a run is not blocked by it.
        show: Writes one line.
        clock: Time source for the decision.

    Raises:
        ConfigurationError: If the approver cannot be named.
    """

    def __init__(
        self,
        *,
        approver: str,
        ask: Callable[[], str] | None = None,
        show: Callable[[str], None] | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not approver.strip():
            raise ConfigurationError("a console approval needs an approver it can record")
        self._approver = approver.strip()
        self._ask = ask or sys.stdin.readline
        self._show = show or _to_stdout
        self._clock = clock

    async def deliver(self, record: ApprovalRecord) -> ApprovalDecision:
        """Show the held call and read one line back as the answer."""
        self._show(f"{record.tool_name} needs approval: {record.reason}")
        self._show(f"  run {record.run_id} · tenant {record.tenant} · {record.summary}")
        self._show(f"  digest {record.arguments_digest[:12]} — approve? [y/N] ")
        answer = await asyncio.to_thread(self._ask)
        granted = answer.strip().casefold() in _YES
        return ApprovalDecision(
            record_id=record.id,
            granted=granted,
            decided_by=self._approver,
            decided_at=self._clock.now() if self._clock else record.requested_at,
            reason="" if granted else f"declined at the console by {self._approver}",
        )


def _to_stdout(line: str) -> None:
    """Write one line where a console approval writes by default."""
    sys.stdout.write(f"{line}\n")


def _token(value: str, *, allow_dots: bool = False) -> str:
    """A value safe to put in a subject, or a refusal to publish a wider one.

    Raises:
        ConfigurationError: If it carries a wildcard, a separator or whitespace.
    """
    allowed = set(".") if allow_dots else set()
    if not value or any(
        character.isspace() or (not character.isalnum() and character not in {"-", "_"} | allowed)
        for character in value
    ):
        raise ConfigurationError(
            f"{value!r} is not a plain subject token; a wildcard or a separator here "
            f"publishes the question to a wider audience"
        )
    return value
