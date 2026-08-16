"""Pytest plugin shipped for consumers: network isolation and bounded quarantine.

Enable it from a conftest with `pytest_plugins = ["tesserix_adk.testing.pytest_plugin"]`.
"""

from __future__ import annotations

import datetime as dt
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tesserix_adk.core.errors import AdkError, ConfigurationError
from tesserix_adk.testing.cassette import CassetteHarness, mode_from_env
from tesserix_adk.testing.fake_model import FakeModelProvider

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

__all__ = ["NetworkAccessInTestError", "QuarantineError"]

_REQUIRED_QUARANTINE_FIELDS = ("owner", "expires", "reason")


class NetworkAccessInTestError(AdkError):
    """Raised when a test reaches for a real network connection.

    A unit suite that can talk to a provider passes or fails on that provider's
    availability, which makes every failure ambiguous.
    """

    def __init__(self, target: str) -> None:
        self.target = target
        super().__init__(
            f"network access to {target!r} blocked in tests. Inject a fake, or mark the "
            f"test with @pytest.mark.allow_network if it is a real integration test."
        )


class QuarantineError(AdkError):
    """Raised when a quarantine marker is missing provenance or has expired."""


def _target(address: object) -> str:
    if isinstance(address, tuple) and len(address) >= 2:
        return f"{address[0]}:{address[1]}"
    return str(address)


@pytest.fixture(autouse=True)
def _block_network(request: pytest.FixtureRequest) -> Iterator[None]:
    """Block outbound TCP and DNS unless the test opts in."""
    if request.node.get_closest_marker("allow_network"):
        yield
        return

    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self: socket.socket, address: Any) -> None:  # noqa: ANN401
        if self.family == socket.AF_UNIX:
            real_connect(self, address)
            return
        raise NetworkAccessInTestError(_target(address))

    def guarded_getaddrinfo(host: Any, port: Any, *_a: Any, **_kw: Any) -> Any:  # noqa: ANN401
        raise NetworkAccessInTestError(_target((host, port)))

    socket.socket.connect = guarded_connect  # type: ignore[method-assign, assignment]
    socket.getaddrinfo = guarded_getaddrinfo
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]
        socket.getaddrinfo = real_getaddrinfo


@pytest.fixture
def fake_model_factory() -> Callable[..., FakeModelProvider]:
    """Build a scripted model provider, one per run.

    Two concurrent runs sharing one fake consume each other's turns, so a test that
    needs more than one asks for more than one rather than reusing the fixture value.
    """
    return FakeModelProvider


@pytest.fixture
def fake_model(fake_model_factory: Callable[..., FakeModelProvider]) -> FakeModelProvider:
    """An empty strict fake: the script is written by the test that needs one."""
    return fake_model_factory()


@pytest.fixture
def cassette_dir() -> Path:
    """Where cassettes live. Override it in a conftest to move them."""
    return Path("tests/cassettes")


@pytest.fixture
def cassettes(request: pytest.FixtureRequest, cassette_dir: Path) -> Iterator[CassetteHarness]:
    """The cassette named by `@pytest.mark.cassette(...)`, in the mode the environment asks for.

    Replay is the default, so a suite cannot start spending because somebody forgot a
    flag. What was recorded is written back only when this run recorded it.

    Raises:
        ConfigurationError: If the test carries no cassette marker.
    """
    marker = request.node.get_closest_marker("cassette")
    if marker is None or not marker.args:
        message = (
            f"{request.node.name} asked for the cassettes fixture without "
            f"@pytest.mark.cassette('<name>'), so there is no cassette to load"
        )
        raise ConfigurationError(message)
    harness = CassetteHarness(
        cassette_dir / f"{marker.args[0]}.json",
        mode=mode_from_env(),
        **marker.kwargs,
    )
    yield harness
    harness.save()


def pytest_configure(config: pytest.Config) -> None:
    """Register the markers this plugin owns."""
    config.addinivalue_line(
        "markers", "allow_network: this test is an integration test and may use the network"
    )
    config.addinivalue_line(
        "markers", "cassette(name, **options): replay provider traffic recorded under `name`"
    )
    config.addinivalue_line(
        "markers",
        "quarantine(owner, expires, reason): tolerate failures until the expiry date",
    )


def _quarantine_reason(fields: Mapping[str, object]) -> str:
    """Return the skip reason for a valid quarantine, or raise explaining why it is not."""
    missing = [f for f in _REQUIRED_QUARANTINE_FIELDS if f not in fields]
    if missing:
        raise QuarantineError(
            f"quarantine marker is missing {', '.join(missing)}; a quarantine with no "
            f"owner is a test nobody will fix"
        )
    expires = dt.date.fromisoformat(str(fields["expires"]))
    if expires < dt.date.today():
        raise QuarantineError(
            f"quarantine expired on {expires.isoformat()} "
            f"(owner {fields['owner']}): fix the test or renew it deliberately"
        )
    return f"quarantined until {expires.isoformat()} by {fields['owner']}: {fields['reason']}"


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Skip a live quarantine; fail an expired or undocumented one."""
    marker = item.get_closest_marker("quarantine")
    if marker is None:
        return
    try:
        reason = _quarantine_reason(marker.kwargs)
    except (QuarantineError, ValueError) as err:
        raise QuarantineError(str(err)) from err
    pytest.skip(reason)
