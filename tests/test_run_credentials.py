"""Surviving credential rotation without restarting the run."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable  # noqa: TC003 — used at runtime by the doubles
from dataclasses import dataclass, field
from random import Random

import pytest
from pydantic import SecretStr

from tesserix_adk.core import (
    AgentIdentity,
    AuthorisationError,
    AuthorityRevokedError,
    CredentialExpiredError,
    ExpiringCredential,
    McpAuthError,
    McpAuthReason,
    Principal,
)
from tesserix_adk.runtime import Reauthoriser, RefreshPolicy, RunCredentials
from tesserix_adk.testing import FakeClock

pytestmark = pytest.mark.anyio

READ = "payments:read"
WRITE = "payments:write"
PAYMENTS = "https://payments.internal"
LIFETIME = 300.0


@dataclass(frozen=True, slots=True)
class _Credential:
    """What a mint returns, in the shape the run loop reads."""

    token: SecretStr
    scopes: frozenset[str]
    expires_at: float
    attribution: tuple[str, ...] = ()

    def expired(self, now: float, *, skew: float = 30.0) -> bool:
        return now >= self.expires_at - skew

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token.get_secret_value()}"}


@dataclass
class _Source:
    """A mint that counts, can be made to fail, and records what it was asked for."""

    clock: FakeClock
    lifetime: float = LIFETIME
    failures: int = 0
    delay: float = 0.0
    minted: int = 0
    asked: list[tuple[str, tuple[str, ...], str]] = field(default_factory=list)

    async def for_tool(
        self,
        *,
        identity: AgentIdentity,
        audience: str,
        needs: Iterable[str],
        run_id: str,
        agent_version: str = "1.0.0",
    ) -> _Credential:
        wanted = tuple(sorted(needs))
        self.asked.append((audience, wanted, agent_version))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failures:
            self.failures -= 1
            raise TimeoutError("the token endpoint is unreachable")
        self.minted += 1
        return _Credential(
            token=SecretStr(f"tok-{self.minted}-{identity.principal.tenant}-{run_id}"),
            scopes=frozenset(wanted),
            expires_at=self.clock.now() + self.lifetime,
        )


def _identity(*held: str, tenant: str = "acme", expires_at: float | None = None) -> AgentIdentity:
    return AgentIdentity.resolve(
        agent="desk",
        declared=(READ, WRITE),
        principal=Principal(
            subject="ada",
            tenant=tenant,
            scopes=frozenset(held or (READ, WRITE)),
            expires_at=expires_at,
        ),
    )


def _credentials(
    source: _Source,
    *,
    clock: FakeClock,
    identity: AgentIdentity | None = None,
    policy: RefreshPolicy | None = None,
    reauthorise: Reauthoriser | None = None,
) -> RunCredentials:
    return RunCredentials(
        source,
        identity=identity or _identity(),
        clock=clock,
        policy=policy or RefreshPolicy(),
        reauthorise=reauthorise,
        random=Random(7),  # noqa: S311 — a seeded jitter source, asserted not waited out
    )


async def _for_call(credentials: RunCredentials, *needs: str) -> _Credential:
    got = await credentials.for_call(audience=PAYMENTS, needs=needs or (READ,), run_id="run_1")
    assert isinstance(got, _Credential)
    return got


class TestThePolicy:
    def test_a_skew_that_is_negative_would_refresh_after_the_fact(self) -> None:
        with pytest.raises(ValueError, match="skew_seconds"):
            RefreshPolicy(skew_seconds=-1.0)

    def test_a_skew_wider_than_the_lifetime_is_the_callers_business(self) -> None:
        assert RefreshPolicy(skew_seconds=600.0).skew_seconds == 600.0


class TestProactiveRefresh:
    async def test_a_credential_still_good_is_reused(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)
        first = await _for_call(credentials)
        assert await _for_call(credentials) is first
        assert source.minted == 1

    async def test_the_skew_window_refreshes_before_the_call_is_made(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)
        first = await _for_call(credentials)
        clock.advance(LIFETIME - RefreshPolicy().skew_seconds)
        second = await _for_call(credentials)
        assert second is not first
        assert source.minted == 2

    async def test_the_run_carries_on_where_it_was(self) -> None:
        """Nothing about a refresh re-runs work: it is one mint between two calls."""
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)
        await _for_call(credentials)
        clock.advance(LIFETIME)
        refreshed = await _for_call(credentials)
        assert refreshed.scopes == frozenset({READ})
        assert source.asked == [(PAYMENTS, (READ,), "1.0.0"), (PAYMENTS, (READ,), "1.0.0")]

    async def test_a_refresh_is_bounded_by_what_the_caller_still_holds(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock, identity=_identity(READ))
        with pytest.raises(AuthorisationError, match=WRITE):
            await _for_call(credentials, READ, WRITE)

    async def test_a_credential_is_kept_per_audience(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)
        await _for_call(credentials)
        await credentials.for_call(audience="https://other.internal", needs=(READ,), run_id="run_1")
        assert source.minted == 2


class TestSingleFlight:
    async def test_a_fan_out_inside_the_window_refreshes_once(self) -> None:
        clock = FakeClock()
        source = _Source(clock, delay=0.01)
        credentials = _credentials(source, clock=clock)
        await _for_call(credentials)
        clock.advance(LIFETIME)
        got = await asyncio.gather(*(_for_call(credentials) for _ in range(6)))
        assert source.minted == 2
        assert len({credential.token.get_secret_value() for credential in got}) == 1

    async def test_two_audiences_refresh_in_parallel(self) -> None:
        clock = FakeClock()
        source = _Source(clock, delay=0.01)
        credentials = _credentials(source, clock=clock)
        await asyncio.gather(
            credentials.for_call(audience=PAYMENTS, needs=(READ,), run_id="run_1"),
            credentials.for_call(audience="https://other.internal", needs=(READ,), run_id="run_1"),
        )
        assert source.minted == 2

    async def test_a_cancelled_refresh_leaves_nothing_behind(self) -> None:
        clock = FakeClock()
        source = _Source(clock, delay=0.05)
        credentials = _credentials(source, clock=clock)
        pending = asyncio.ensure_future(_for_call(credentials))
        await asyncio.sleep(0.01)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert credentials.held == 0
        assert (await _for_call(credentials)).scopes == frozenset({READ})


class TestReactiveRefresh:
    async def test_a_downstream_expiry_rejection_refreshes_and_retries(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)
        seen: list[str] = []

        async def call(credential: ExpiringCredential, key: str) -> str:
            seen.append(credential.token.get_secret_value())
            if len(seen) == 1:
                raise CredentialExpiredError("the server says it lapsed", audience=PAYMENTS)
            return key

        answer = await credentials.call(
            call, audience=PAYMENTS, needs=(READ,), run_id="run_1", idempotency_key="key-1"
        )
        assert answer == "key-1"
        assert seen[0] != seen[1]
        assert source.minted == 2

    async def test_the_retry_reuses_the_original_idempotency_key(self) -> None:
        clock = FakeClock()
        credentials = _credentials(_Source(clock), clock=clock)
        keys: list[str] = []

        async def call(_credential: ExpiringCredential, key: str) -> None:
            keys.append(key)
            if len(keys) == 1:
                raise CredentialExpiredError("lapsed", audience=PAYMENTS)

        await credentials.call(
            call, audience=PAYMENTS, needs=(READ,), run_id="run_1", idempotency_key="key-1"
        )
        assert keys == ["key-1", "key-1"]

    async def test_a_server_saying_expired_in_its_own_words_is_the_same_rejection(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)
        attempts = 0

        async def call(_credential: ExpiringCredential, key: str) -> str:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise McpAuthError("lapsed", reason=McpAuthReason.EXPIRED, server="payments")
            return key

        assert (
            await credentials.call(
                call, audience=PAYMENTS, needs=(READ,), run_id="run_1", idempotency_key="key-1"
            )
            == "key-1"
        )
        assert source.minted == 2

    async def test_a_call_with_no_key_is_not_retried_and_its_outcome_is_unknown(self) -> None:
        """A non-idempotent call that may have landed is not quietly made twice."""
        clock = FakeClock()
        credentials = _credentials(_Source(clock), clock=clock)
        attempts = 0

        async def call(_credential: ExpiringCredential, _key: str) -> None:
            nonlocal attempts
            attempts += 1
            raise CredentialExpiredError("lapsed", audience=PAYMENTS)

        with pytest.raises(CredentialExpiredError) as refused:
            await credentials.call(call, audience=PAYMENTS, needs=(READ,), run_id="run_1")
        assert refused.value.outcome == "unknown"
        assert refused.value.retryable is False
        assert attempts == 1

    async def test_a_failure_that_is_not_an_expiry_is_left_alone(self) -> None:
        clock = FakeClock()
        credentials = _credentials(_Source(clock), clock=clock)

        async def call(_credential: ExpiringCredential, _key: str) -> None:
            raise ValueError("the tool itself failed")

        with pytest.raises(ValueError, match="the tool itself"):
            await credentials.call(
                call, audience=PAYMENTS, needs=(READ,), run_id="run_1", idempotency_key="key-1"
            )

    async def test_a_server_refusing_the_scope_is_not_answered_with_a_fresh_token(self) -> None:
        """A narrower token would be refused for the same reason, so nothing is minted."""
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)

        async def call(_credential: ExpiringCredential, _key: str) -> None:
            raise McpAuthError(
                "that scope is not yours",
                reason=McpAuthReason.INSUFFICIENT_SCOPE,
                server="payments",
            )

        with pytest.raises(McpAuthError, match="not yours"):
            await credentials.call(
                call, audience=PAYMENTS, needs=(READ,), run_id="run_1", idempotency_key="key-1"
            )
        assert source.minted == 1

    async def test_a_second_rejection_is_not_refreshed_a_third_time(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)

        async def call(_credential: ExpiringCredential, _key: str) -> None:
            raise CredentialExpiredError("lapsed", audience=PAYMENTS)

        with pytest.raises(CredentialExpiredError):
            await credentials.call(
                call, audience=PAYMENTS, needs=(READ,), run_id="run_1", idempotency_key="key-1"
            )
        assert source.minted == 2


class TestRevocation:
    async def test_a_revoked_caller_halts_the_run(self) -> None:
        clock = FakeClock()
        source = _Source(clock)

        class _Revoked:
            async def reauthorise(self, identity: AgentIdentity) -> AgentIdentity:  # noqa: ARG002
                raise AuthorisationError("the grant was withdrawn")

        credentials = _credentials(source, clock=clock, reauthorise=_Revoked())
        with pytest.raises(AuthorityRevokedError, match="withdrawn"):
            await _for_call(credentials)
        assert credentials.halted

    async def test_a_halted_run_dispatches_nothing_further(self) -> None:
        clock = FakeClock()
        source = _Source(clock)

        class _Revoked:
            async def reauthorise(self, identity: AgentIdentity) -> AgentIdentity:  # noqa: ARG002
                raise AuthorisationError("the grant was withdrawn")

        credentials = _credentials(source, clock=clock, reauthorise=_Revoked())
        with pytest.raises(AuthorityRevokedError):
            await _for_call(credentials)
        with pytest.raises(AuthorityRevokedError, match="halted"):
            await _for_call(credentials)
        assert source.minted == 0

    async def test_an_authority_that_lapsed_halts_rather_than_renewing(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(
            source, clock=clock, identity=_identity(READ, expires_at=clock.now() + 10.0)
        )
        await _for_call(credentials)
        clock.advance(LIFETIME)
        with pytest.raises(AuthorityRevokedError):
            await _for_call(credentials)
        assert credentials.halted

    async def test_a_re_derived_authority_may_narrow(self) -> None:
        clock = FakeClock()
        source = _Source(clock)

        class _Narrowing:
            async def reauthorise(self, identity: AgentIdentity) -> AgentIdentity:
                return AgentIdentity.resolve(
                    agent=identity.agent,
                    declared=identity.declared,
                    principal=identity.principal.model_copy(update={"scopes": frozenset({READ})}),
                )

        credentials = _credentials(source, clock=clock, reauthorise=_Narrowing())
        await _for_call(credentials, READ)
        assert credentials.identity.effective.names == frozenset({READ})
        with pytest.raises(AuthorisationError, match=WRITE):
            await credentials.for_call(audience=PAYMENTS, needs=(WRITE,), run_id="run_1")

    async def test_a_re_derived_authority_may_not_widen(self) -> None:
        clock = FakeClock()
        source = _Source(clock)

        class _Widening:
            async def reauthorise(self, identity: AgentIdentity) -> AgentIdentity:
                return AgentIdentity.resolve(
                    agent=identity.agent,
                    declared=(READ, WRITE),
                    principal=identity.principal.model_copy(
                        update={"scopes": frozenset({READ, WRITE})}
                    ),
                )

        credentials = _credentials(
            source, clock=clock, identity=_identity(READ), reauthorise=_Widening()
        )
        await _for_call(credentials, READ)
        assert credentials.identity.effective.names == frozenset({READ})


class TestSuspension:
    async def test_a_suspended_run_carries_no_credential_across(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        credentials = _credentials(source, clock=clock)
        await _for_call(credentials)
        credentials.suspend()
        assert credentials.held == 0
        await _for_call(credentials)
        assert source.minted == 2

    async def test_a_resumed_run_re_derives_its_authority(self) -> None:
        clock = FakeClock()
        source = _Source(clock)
        derived = 0

        class _Recording:
            async def reauthorise(self, identity: AgentIdentity) -> AgentIdentity:
                nonlocal derived
                derived += 1
                return identity

        credentials = _credentials(source, clock=clock, reauthorise=_Recording())
        await _for_call(credentials)
        credentials.suspend()
        clock.advance(86_400.0)
        await _for_call(credentials)
        assert derived == 2

    async def test_a_run_whose_authority_lapsed_while_suspended_does_not_resume(self) -> None:
        clock = FakeClock()
        credentials = _credentials(
            _Source(clock), clock=clock, identity=_identity(READ, expires_at=clock.now() + 60.0)
        )
        credentials.suspend()
        clock.advance(86_400.0)
        with pytest.raises(AuthorityRevokedError):
            await _for_call(credentials)


class TestTransientFailure:
    async def test_a_flaky_mint_is_retried_with_backoff(self) -> None:
        clock = FakeClock()
        source = _Source(clock, failures=2)
        credentials = _credentials(source, clock=clock)
        assert (await _for_call(credentials)).scopes == frozenset({READ})
        assert len(clock.slept) == 2

    async def test_the_backoff_is_jittered_rather_than_in_unison(self) -> None:
        clock = FakeClock()
        credentials = _credentials(_Source(clock, failures=2), clock=clock)
        await _for_call(credentials)
        assert clock.slept[0] != clock.slept[1]
        assert all(delay > 0 for delay in clock.slept)

    async def test_a_mint_that_stays_down_is_transient_not_a_revocation(self) -> None:
        clock = FakeClock()
        credentials = _credentials(_Source(clock, failures=99), clock=clock)
        with pytest.raises(CredentialExpiredError) as refused:
            await _for_call(credentials)
        assert refused.value.outcome == "not_started"
        assert refused.value.retryable is True
        assert not credentials.halted

    async def test_a_refusal_from_the_mint_is_a_revocation_not_a_blip(self) -> None:
        clock = FakeClock()

        class _Refusing(_Source):
            async def for_tool(self, **_: object) -> _Credential:
                raise AuthorisationError("that audience is no longer permitted")

        credentials = _credentials(_Refusing(clock), clock=clock)
        with pytest.raises(AuthorityRevokedError, match="no longer permitted"):
            await _for_call(credentials)
        assert credentials.halted
