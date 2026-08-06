"""A recorded round trip is only useful if a replay refuses to invent the ones it lacks.

The vendor adapters are tested against recorded HTTP, so the whole matrix runs in CI with
no network and no key. That is worth nothing if a replay serves the nearest recording it
can find, so these tests are mostly about what the replay refuses.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest

from tesserix_adk.testing import CassetteMissError, CassetteVersionError, HttpCassette, HttpReplay
from tesserix_adk.testing.http_cassette import HttpExchange

if TYPE_CHECKING:
    from pathlib import Path


def cassette(*exchanges: HttpExchange) -> HttpCassette:
    return HttpCassette(provider="acme", exchanges=exchanges)


ANSWERED = HttpExchange(path="/v1/messages", body={"ok": True})


async def sent(replay: HttpReplay, path: str = "/v1/messages") -> httpx.Response:
    async with httpx.AsyncClient(
        base_url="https://api.example.test", transport=replay.transport
    ) as client:
        return await client.post(path, json={"model": "acme-1"})


class TestAReplayServesWhatWasRecorded:
    async def test_the_recorded_body_comes_back(self) -> None:
        response = await sent(HttpReplay(cassette(ANSWERED)))
        assert response.json() == {"ok": True}

    async def test_the_recorded_status_comes_back(self) -> None:
        replay = HttpReplay(cassette(HttpExchange(path="/v1/messages", status=429, body={})))
        assert (await sent(replay)).status_code == 429

    async def test_the_recorded_headers_come_back(self) -> None:
        replay = HttpReplay(
            cassette(
                HttpExchange(path="/v1/messages", headers={"retry-after": "3"}, body={}),
            )
        )
        assert (await sent(replay)).headers["retry-after"] == "3"

    async def test_exchanges_are_served_in_the_order_they_were_recorded(self) -> None:
        replay = HttpReplay(
            cassette(
                HttpExchange(path="/v1/messages", body={"turn": 1}),
                HttpExchange(path="/v1/messages", body={"turn": 2}),
            )
        )
        first = (await sent(replay)).json()
        second = (await sent(replay)).json()
        assert (first, second) == ({"turn": 1}, {"turn": 2})

    async def test_a_recorded_stream_arrives_as_server_sent_events(self) -> None:
        replay = HttpReplay(
            cassette(
                HttpExchange(
                    path="/v1/messages",
                    stream=('data: {"n": 1}', 'data: {"n": 2}', "data: [DONE]"),
                )
            )
        )
        response = await sent(replay)
        assert response.headers["content-type"] == "text/event-stream"
        assert response.text.count("data:") == 3


class TestAReplayRefusesWhatItDoesNotHold:
    async def test_a_request_to_another_path_is_a_miss(self) -> None:
        with pytest.raises(CassetteMissError, match="/v1/other"):
            await sent(HttpReplay(cassette(ANSWERED)), "/v1/other")

    async def test_a_request_past_the_end_of_the_recording_is_a_miss(self) -> None:
        replay = HttpReplay(cassette(ANSWERED))
        await sent(replay)
        with pytest.raises(CassetteMissError, match="no further"):
            await sent(replay)

    async def test_an_empty_cassette_says_so(self) -> None:
        with pytest.raises(CassetteMissError, match="empty"):
            await sent(HttpReplay(cassette()))

    async def test_a_request_by_another_method_is_a_miss(self) -> None:
        replay = HttpReplay(cassette(ANSWERED))
        async with httpx.AsyncClient(
            base_url="https://api.example.test", transport=replay.transport
        ) as client:
            with pytest.raises(CassetteMissError, match="GET"):
                await client.get("/v1/messages")


class TestTheReplayKeepsWhatWasAskedOfIt:
    async def test_the_request_body_is_available_to_assert_on(self) -> None:
        """Request translation is half of what a provider does, so it is half of the test."""
        replay = HttpReplay(cassette(ANSWERED))
        await sent(replay)
        assert replay.sent[0].body == {"model": "acme-1"}

    async def test_the_request_headers_are_available_to_assert_on(self) -> None:
        replay = HttpReplay(cassette(ANSWERED))
        await sent(replay)
        assert replay.sent[0].headers["content-type"] == "application/json"

    async def test_the_path_is_kept_with_its_query(self) -> None:
        replay = HttpReplay(cassette(HttpExchange(path="/v1/models/x:stream", body={})))
        await sent(replay, "/v1/models/x:stream?alt=sse")
        assert replay.sent[0].path == "/v1/models/x:stream?alt=sse"

    async def test_a_body_that_is_not_json_is_kept_as_none(self) -> None:
        replay = HttpReplay(cassette(HttpExchange(method="PUT", path="/v1/raw", body={})))
        async with httpx.AsyncClient(
            base_url="https://api.example.test", transport=replay.transport
        ) as client:
            await client.put("/v1/raw", content=b"not json")
        assert replay.sent[0].body is None


class TestACassetteIsAFilePeopleCommit:
    def test_it_round_trips_through_a_file(self, tmp_path: Path) -> None:
        original = cassette(ANSWERED)
        original.save(tmp_path / "acme.json")
        assert HttpCassette.load(tmp_path / "acme.json") == original

    def test_it_is_written_as_readable_json(self, tmp_path: Path) -> None:
        """A cassette is reviewed in a diff, so it is not written on one line."""
        cassette(ANSWERED).save(tmp_path / "acme.json")
        assert "\n  " in (tmp_path / "acme.json").read_text()

    def test_a_cassette_from_another_format_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "acme.json"
        path.write_text(json.dumps({"format": "0", "provider": "acme", "exchanges": []}))
        with pytest.raises(CassetteVersionError, match="format"):
            HttpCassette.load(path)

    def test_a_cassette_recorded_against_another_provider_is_refused(self) -> None:
        with pytest.raises(CassetteVersionError, match="acme"):
            HttpReplay(cassette(ANSWERED), expect_provider="other")
