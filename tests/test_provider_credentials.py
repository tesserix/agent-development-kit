"""A key is read when it is used, from the environment or an injected secret provider.

A key captured at construction is a key that outlives its rotation: the process keeps
presenting the revoked one until somebody restarts it, which is at best an outage and at
worst a revoked credential still in flight.
"""

from __future__ import annotations

import pytest

from tesserix_adk.core import ConfigurationError
from tesserix_adk.models import Credential, EnvironmentSecrets


class RotatingSecrets:
    """A secret provider whose answer changes, as a real one's does."""

    def __init__(self, value: str) -> None:
        self.value = value
        self.reads = 0

    def secret(self, name: str) -> str | None:
        self.reads += 1
        return f"{self.value}-{name}"


class TestAKeyIsReadAtTheMomentItIsUsed:
    def test_the_environment_is_read_on_every_use(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VENDOR_API_KEY", "first")
        credential = Credential("VENDOR_API_KEY")
        assert credential.value() == "first"
        monkeypatch.setenv("VENDOR_API_KEY", "rotated")
        assert credential.value() == "rotated"

    def test_an_injected_provider_is_asked_instead_of_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VENDOR_API_KEY", "from-environment")
        secrets = RotatingSecrets("vault")
        assert Credential("VENDOR_API_KEY", secrets=secrets).value() == "vault-VENDOR_API_KEY"

    def test_an_injected_provider_is_asked_every_time_too(self) -> None:
        secrets = RotatingSecrets("vault")
        credential = Credential("VENDOR_API_KEY", secrets=secrets)
        credential.value()
        credential.value()
        assert secrets.reads == 2

    def test_the_environment_answers_when_the_provider_has_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VENDOR_API_KEY", "from-environment")
        assert Credential("VENDOR_API_KEY", secrets=EnvironmentSecrets()).value() == (
            "from-environment"
        )


class TestAKeyThatIsNotThereFailsBeforeTheRequest:
    def test_an_absent_variable_names_itself(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VENDOR_API_KEY", raising=False)
        with pytest.raises(ConfigurationError) as refused:
            Credential("VENDOR_API_KEY").value()
        assert "VENDOR_API_KEY" in str(refused.value)

    def test_a_blank_variable_counts_as_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An empty key is a deployment that thinks it is configured and is not."""
        monkeypatch.setenv("VENDOR_API_KEY", "   ")
        with pytest.raises(ConfigurationError):
            Credential("VENDOR_API_KEY").value()


class TestAKeyDoesNotLeakByBeingHandled:
    def test_the_credential_does_not_print_the_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VENDOR_API_KEY", "sk-live-do-not-print")
        credential = Credential("VENDOR_API_KEY")
        assert "sk-live-do-not-print" not in repr(credential)
        assert "sk-live-do-not-print" not in str(credential)

    def test_the_variable_name_is_reported_because_it_is_not_the_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VENDOR_API_KEY", "sk-live")
        assert "VENDOR_API_KEY" in repr(Credential("VENDOR_API_KEY"))

    def test_the_key_is_never_kept_on_the_credential(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing to cache is nothing to leak, and nothing to serve after a rotation."""
        monkeypatch.setenv("VENDOR_API_KEY", "sk-live")
        credential = Credential("VENDOR_API_KEY")
        credential.value()
        assert "sk-live" not in str(vars(credential))
