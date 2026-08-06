"""Recorded HTTP, so a vendor adapter is tested without a network or a key.

`Cassette` records at the provider boundary and proves what a run did with a response.
This records one layer lower, at the wire, and proves what the adapter *sent* — which is
half of an adapter's job and the half a provider-level recording cannot see.

A replay serves the exchanges in the order they were recorded and refuses anything else.
There is no nearest match: a replay that quietly served the closest recording would be a
green test asserting nothing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import Field

from tesserix_adk.core.models import AdkModel
from tesserix_adk.testing.cassette import CassetteMissError, CassetteVersionError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["HTTP_CASSETTE_FORMAT", "HttpCassette", "HttpExchange", "HttpReplay", "SentRequest"]

HTTP_CASSETTE_FORMAT = "1"


class HttpExchange(AdkModel):
    """One recorded round trip.

    Args:
        method: The verb the recording was made with.
        path: The path it was made against, with any query. A request to another path is
            a miss rather than a near-enough match.
        status: The status to answer with.
        headers: Response headers worth keeping — `retry-after` and a request id are the
            ones the adapters read.
        body: The JSON body, for an ordinary response.
        stream: The raw server-sent event lines, for a streaming response. A cassette
            carries one or the other; `stream` wins where both are set.
    """

    method: str = "POST"
    path: str = Field(min_length=1)
    status: int = 200
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None
    stream: tuple[str, ...] = ()


class HttpCassette(AdkModel):
    """Recorded exchanges and what they were recorded against.

    Args:
        format: The cassette format, so an old file is refused rather than misread.
        provider: Which vendor's wire format this holds.
        recorded: When it was taken, as a date. Vendors change their wire format, and a
            recording with no date gives a reader nothing to judge it by.
        exchanges: The round trips, in the order they happened.
    """

    format: str = HTTP_CASSETTE_FORMAT
    provider: str = Field(min_length=1)
    recorded: str = ""
    exchanges: tuple[HttpExchange, ...] = ()

    def save(self, path: Path) -> None:
        """Write the cassette to `path` as indented JSON, so a diff is readable in review."""
        path.write_text(json.dumps(self.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> HttpCassette:
        """Read a cassette from `path`.

        Raises:
            CassetteVersionError: If it was written in a format this kit does not read.
        """
        loaded = cls.model_validate_json(path.read_text())
        if loaded.format != HTTP_CASSETTE_FORMAT:
            raise CassetteVersionError(
                f"cassette at {path} is format {loaded.format!r}, and this kit reads "
                f"{HTTP_CASSETTE_FORMAT!r}; re-record it rather than reading it on trust"
            )
        return loaded


class SentRequest(AdkModel):
    """What the adapter put on the wire, kept so a test can assert on the translation.

    Args:
        method: The verb it used.
        path: The path with its query.
        headers: The headers it set. In memory only — a `SentRequest` is never saved, so
            the key a test supplies does not reach a committed file.
        body: The JSON it sent, or `None` where it sent something else.
    """

    method: str
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: dict[str, Any] | None = None


class HttpReplay:
    """Serves a cassette through an `httpx` transport and refuses to invent an exchange.

    Args:
        cassette: The recording to serve.
        expect_provider: The vendor the caller believes it is testing against.

    Raises:
        CassetteVersionError: If the cassette was recorded against another vendor.
    """

    def __init__(self, cassette: HttpCassette, *, expect_provider: str | None = None) -> None:
        if expect_provider is not None and cassette.provider != expect_provider:
            raise CassetteVersionError(
                f"cassette was recorded against {cassette.provider!r}, not "
                f"{expect_provider!r}; re-record it rather than replaying it on trust"
            )
        self._cassette = cassette
        self._served = 0
        self.sent: list[SentRequest] = []

    @property
    def transport(self) -> httpx.MockTransport:
        """The transport to hand a provider, in place of a real connection."""
        return httpx.MockTransport(self._answer)

    @property
    def remaining(self) -> int:
        """Recordings not yet served, so a test can assert the adapter used them all."""
        return len(self._cassette.exchanges) - self._served

    def _answer(self, request: httpx.Request) -> httpx.Response:
        self.sent.append(_asked(request))
        exchange = self._next(request)
        self._served += 1
        if exchange.stream:
            return httpx.Response(
                exchange.status,
                content="\n\n".join(exchange.stream).encode() + b"\n\n",
                headers={"content-type": "text/event-stream", **exchange.headers},
            )
        return httpx.Response(exchange.status, json=exchange.body or {}, headers=exchange.headers)

    def _next(self, request: httpx.Request) -> HttpExchange:
        asked = _path_of(request)
        if self._served >= len(self._cassette.exchanges):
            raise CassetteMissError(
                f"cassette holds no further recording for {request.method} {asked}; the "
                f"adapter called more times than it did when recorded"
                if self._cassette.exchanges
                else f"cassette is empty, so there is nothing to serve for {asked}"
            )
        exchange = self._cassette.exchanges[self._served]
        if request.method != exchange.method or not asked.startswith(exchange.path):
            raise CassetteMissError(
                f"cassette expected {exchange.method} {exchange.path} next, and the "
                f"adapter sent {request.method} {asked}"
            )
        return exchange


def _asked(request: httpx.Request) -> SentRequest:
    return SentRequest(
        method=request.method,
        path=_path_of(request),
        headers=dict(request.headers.items()),
        body=_json_of(request),
    )


def _path_of(request: httpx.Request) -> str:
    query = request.url.query.decode()
    return request.url.path + (f"?{query}" if query else "")


def _json_of(request: httpx.Request) -> dict[str, Any] | None:
    try:
        parsed = json.loads(request.content)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
