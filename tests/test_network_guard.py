"""A unit suite that can reach the network is a suite whose result depends on the weather.

The guard ships in `tesserix_adk.testing` rather than living in this repository's
conftest, so a consuming product inherits the same guarantee by enabling one plugin.
"""

import socket
from pathlib import Path

import pytest

from tesserix_adk.core import AdkError
from tesserix_adk.testing import NetworkAccessInTestError

pytest_plugins = ["pytester"]


def test_network_access_error_is_an_adk_error() -> None:
    assert issubclass(NetworkAccessInTestError, AdkError)


def test_a_tcp_connect_is_blocked_and_names_the_host() -> None:
    with pytest.raises(NetworkAccessInTestError) as exc:
        socket.create_connection(("models.example.invalid", 443), timeout=0.01)

    assert "models.example.invalid" in str(exc.value)
    assert "443" in str(exc.value)


def test_a_raw_socket_connect_is_blocked_and_names_the_host() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(NetworkAccessInTestError) as exc,
    ):
        sock.connect(("registry.example.invalid", 80))

    assert "registry.example.invalid" in str(exc.value)


def test_name_resolution_is_blocked() -> None:
    """Blocking connect alone still lets a test hang on DNS in an offline runner."""
    with pytest.raises(NetworkAccessInTestError) as exc:
        socket.getaddrinfo("provider.example.invalid", 443)

    assert "provider.example.invalid" in str(exc.value)


def test_unix_sockets_stay_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Local IPC is not the hazard, and blocking it breaks unrelated tooling."""
    monkeypatch.chdir(tmp_path)  # AF_UNIX paths are length-limited; stay relative
    address = "adk.sock"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(address)
        server.listen(1)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(address)
            assert client.getpeername() == address


def test_a_missing_unix_socket_still_reports_the_real_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with (
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock,
        pytest.raises(FileNotFoundError) as exc,
    ):
        sock.connect("absent.sock")

    assert not isinstance(exc.value, NetworkAccessInTestError)


@pytest.mark.allow_network
def test_the_opt_in_marker_lifts_the_guard_in_process() -> None:
    """The marked path must leave the real socket module untouched."""
    assert socket.socket.connect is not None
    with pytest.raises(NetworkAccessInTestError):
        raise NetworkAccessInTestError("proof the error type is still importable")


def test_a_non_tuple_address_is_reported_verbatim() -> None:
    from tesserix_adk.testing.pytest_plugin import _target

    assert _target("/var/run/adk.sock") == "/var/run/adk.sock"


def test_a_test_may_opt_in_to_the_network_explicitly(pytester: pytest.Pytester) -> None:
    """Integration suites need a documented escape hatch, not a monkeypatched one.

    Run out of process: this suite's own guard is active in the parent interpreter.
    """
    pytester.makeconftest('pytest_plugins = ["tesserix_adk.testing.pytest_plugin"]')
    pytester.makepyfile(
        """
        import socket
        import pytest

        @pytest.mark.allow_network
        def test_opted_in():
            with pytest.raises(OSError) as exc:
                socket.create_connection(("localhost", 1), timeout=0.01)
            assert type(exc.value).__name__ != "NetworkAccessInTestError"
        """
    )
    pytester.runpytest_subprocess("-p", "no:randomly").assert_outcomes(passed=1)
