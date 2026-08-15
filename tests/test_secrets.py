"""What configuration may hold, and what never reaches a rendering of it."""

from __future__ import annotations

import asyncio
import copy
import json
import pickle
from collections.abc import Mapping  # noqa: TC003 — pydantic needs the runtime type

import pytest
from pydantic import SecretStr

from tesserix_adk.core import AdkModel
from tesserix_adk.core.errors import SecretResolutionError
from tesserix_adk.core.secrets import (
    CachingSecrets,
    ChainedSecrets,
    EnvironmentSecrets,
    ProvidedSecrets,
    SecretResolver,
    literal_credentials,
)
from tesserix_adk.core.tenant_config import SecretRef
from tesserix_adk.testing import FakeClock, FakeSecrets

pytestmark = pytest.mark.anyio

VALUE = "sk-test-a1b2c3"


class _Counting:
    """A resolver that answers a fixed value and says how often it was asked."""

    def __init__(self, value: str = VALUE, *, delay: float = 0.0) -> None:
        self.value = value
        self.delay = delay
        self.calls = 0

    async def resolve(self, ref: SecretRef) -> SecretStr:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return SecretStr(f"{self.value}-{ref.name}")


class _Broken:
    """A resolver whose backend is down, which is not the same as holding nothing."""

    async def resolve(self, ref: SecretRef) -> SecretStr:
        raise SecretResolutionError("backend unreachable", ref=ref.describe())


class TestAReference:
    def test_names_a_secret_rather_than_carrying_one(self) -> None:
        ref = SecretRef(name="openai-key", version="7")
        assert VALUE not in repr(ref)
        assert ref.describe() == "openai-key@7"

    def test_without_a_version_reads_whatever_rotation_last_wrote(self) -> None:
        assert SecretRef(name="openai-key").describe() == "openai-key@latest"

    def test_that_names_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least 1 character"):
            SecretRef(name="")

    def test_fills_a_tenant_placeholder(self) -> None:
        bound = SecretRef(name="{tenant}-openai-key").for_tenant("acme")
        assert bound.name == "acme-openai-key"
        assert bound.tenant == "acme"

    def test_bound_to_one_tenant_cannot_be_read_for_another(self) -> None:
        bound = SecretRef(name="{tenant}-key").for_tenant("acme")
        with pytest.raises(SecretResolutionError, match="cannot be read for 'globex'"):
            bound.for_tenant("globex")

    def test_rebinding_to_the_same_tenant_is_not_a_second_binding(self) -> None:
        bound = SecretRef(name="{tenant}-key").for_tenant("acme")
        assert bound.for_tenant("acme").name == "acme-key"

    def test_binding_leaves_the_original_alone(self) -> None:
        ref = SecretRef(name="{tenant}-key")
        ref.for_tenant("acme")
        assert ref.name == "{tenant}-key"


class TestResolvingFromTheEnvironment:
    async def test_reads_the_folded_upper_case_variable(self) -> None:
        secrets = EnvironmentSecrets(environ={"OPENAI_KEY": VALUE})
        held = await secrets.resolve(SecretRef(name="openai-key"))
        assert held.get_secret_value() == VALUE

    async def test_applies_a_prefix_for_a_process_holding_more_than_one_kit(self) -> None:
        secrets = EnvironmentSecrets(prefix="ADK_", environ={"ADK_OPENAI_KEY": VALUE})
        assert (await secrets.resolve(SecretRef(name="openai.key"))).get_secret_value() == VALUE

    async def test_an_unset_variable_is_refused_by_name(self) -> None:
        secrets = EnvironmentSecrets(environ={})
        with pytest.raises(SecretResolutionError, match="looked for OPENAI_KEY"):
            await secrets.resolve(SecretRef(name="openai-key"))

    async def test_an_empty_variable_is_refused_rather_than_served(self) -> None:
        secrets = EnvironmentSecrets(environ={"OPENAI_KEY": ""})
        with pytest.raises(SecretResolutionError):
            await secrets.resolve(SecretRef(name="openai-key"))

    def test_defaults_to_the_real_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_KEY", VALUE)
        secrets = EnvironmentSecrets()
        assert secrets.environ["OPENAI_KEY"] == VALUE

    def test_says_which_variable_a_reference_reads(self) -> None:
        assert EnvironmentSecrets().variable_for(SecretRef(name="openai-key")) == "OPENAI_KEY"


class TestResolvingFromTheKitsProvider:
    async def test_reads_what_the_provider_holds(self) -> None:
        secrets = ProvidedSecrets(FakeSecrets({"openai-key": VALUE}))
        assert (await secrets.resolve(SecretRef(name="openai-key"))).get_secret_value() == VALUE

    async def test_what_it_does_not_hold_is_refused_naming_the_provider(self) -> None:
        secrets = ProvidedSecrets(FakeSecrets({}))
        with pytest.raises(SecretResolutionError, match="FakeSecrets"):
            await secrets.resolve(SecretRef(name="openai-key"))


class TestAChain:
    async def test_the_first_resolver_that_holds_it_answers(self) -> None:
        chain = ChainedSecrets(
            (
                EnvironmentSecrets(environ={"K": "deployed"}),
                EnvironmentSecrets(environ={"K": "laptop"}),
            )
        )
        assert (await chain.resolve(SecretRef(name="k"))).get_secret_value() == "deployed"

    async def test_falls_through_a_resolver_that_holds_nothing(self) -> None:
        chain = ChainedSecrets(
            (EnvironmentSecrets(environ={}), EnvironmentSecrets(environ={"K": VALUE}))
        )
        assert (await chain.resolve(SecretRef(name="k"))).get_secret_value() == VALUE

    async def test_none_of_them_holding_it_says_how_many_were_asked(self) -> None:
        chain = ChainedSecrets((_Broken(), _Broken()))
        with pytest.raises(SecretResolutionError, match="any of the 2 resolvers"):
            await chain.resolve(SecretRef(name="k"))

    async def test_an_empty_chain_resolves_nothing_loudly(self) -> None:
        with pytest.raises(SecretResolutionError, match="any of the 0 resolvers"):
            await ChainedSecrets().resolve(SecretRef(name="k"))


class TestCaching:
    def _cache(self, inner: SecretResolver, *, ttl: float = 60.0) -> CachingSecrets:
        return CachingSecrets(inner, clock=FakeClock(), ttl_seconds=ttl)

    async def test_a_second_ask_inside_the_ttl_does_not_fetch_again(self) -> None:
        inner = _Counting()
        cache = self._cache(inner)
        ref = SecretRef(name="k")
        await cache.resolve(ref)
        await cache.resolve(ref)
        assert inner.calls == 1

    async def test_a_rotated_secret_is_picked_up_once_the_ttl_lapses(self) -> None:
        inner = _Counting()
        clock = FakeClock()
        cache = CachingSecrets(inner, clock=clock, ttl_seconds=60.0)
        ref = SecretRef(name="k")
        await cache.resolve(ref)
        clock.advance(61.0)
        await cache.resolve(ref)
        assert inner.calls == 2

    async def test_an_expired_entry_is_not_served_when_the_backend_is_down(self) -> None:
        clock = FakeClock()
        cache = CachingSecrets(_Counting(), clock=clock, ttl_seconds=60.0)
        ref = SecretRef(name="k")
        await cache.resolve(ref)
        clock.advance(61.0)
        cache._inner = _Broken()
        with pytest.raises(SecretResolutionError):
            await cache.resolve(ref)

    async def test_different_references_are_held_apart(self) -> None:
        inner = _Counting()
        cache = self._cache(inner)
        first = await cache.resolve(SecretRef(name="a"))
        second = await cache.resolve(SecretRef(name="b"))
        assert first.get_secret_value() != second.get_secret_value()
        assert cache.cached == 2

    async def test_one_tenants_entry_is_not_another_tenants(self) -> None:
        cache = self._cache(_Counting())
        ref = SecretRef(name="{tenant}-k")
        await cache.resolve(ref.for_tenant("acme"))
        await cache.resolve(ref.for_tenant("globex"))
        assert cache.cached == 2

    async def test_concurrent_cold_asks_share_one_fetch(self) -> None:
        inner = _Counting(delay=0.01)
        cache = self._cache(inner)
        ref = SecretRef(name="k")
        await asyncio.gather(*(cache.resolve(ref) for _ in range(5)))
        assert inner.calls == 1

    async def test_different_references_still_fetch_in_parallel(self) -> None:
        inner = _Counting(delay=0.05)
        cache = self._cache(inner)
        started = asyncio.get_running_loop().time()
        await asyncio.gather(*(cache.resolve(SecretRef(name=f"k{n}")) for n in range(4)))
        assert asyncio.get_running_loop().time() - started < 0.15

    async def test_invalidating_one_reference_refetches_only_that_one(self) -> None:
        inner = _Counting()
        cache = self._cache(inner)
        await cache.resolve(SecretRef(name="a"))
        await cache.resolve(SecretRef(name="b"))
        cache.invalidate(SecretRef(name="a"))
        await cache.resolve(SecretRef(name="a"))
        await cache.resolve(SecretRef(name="b"))
        assert inner.calls == 3

    def test_invalidating_something_never_cached_is_not_an_error(self) -> None:
        self._cache(_Counting()).invalidate(SecretRef(name="never"))

    async def test_invalidating_everything_is_what_a_rotation_needs(self) -> None:
        inner = _Counting()
        cache = self._cache(inner)
        await cache.resolve(SecretRef(name="a"))
        cache.invalidate_all()
        assert cache.cached == 0
        await cache.resolve(SecretRef(name="a"))
        assert inner.calls == 2

    def test_a_ttl_that_caches_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="ttl_seconds must be positive"):
            CachingSecrets(_Counting(), clock=FakeClock(), ttl_seconds=0)


class _Nested(AdkModel):
    api_key: str = ""
    endpoint: str = ""


class _Config(AdkModel):
    name: str = "svc"
    password: str = ""
    token: SecretStr | None = None
    ref: SecretRef | None = None
    nested: _Nested | None = None
    providers: tuple[_Nested, ...] = ()
    hosts: tuple[str, ...] = ()
    by_name: Mapping[str, str] = {}
    by_provider: Mapping[str, _Nested] = {}


class TestLintingConfigForLiterals:
    def test_a_credential_field_holding_a_literal_is_named(self) -> None:
        assert literal_credentials(_Config(password=VALUE)) == ("password",)

    def test_a_secret_str_passes_because_it_redacts_itself(self) -> None:
        assert literal_credentials(_Config(token=SecretStr(VALUE))) == ()

    def test_a_reference_passes_because_it_is_not_a_value(self) -> None:
        assert literal_credentials(_Config(ref=SecretRef(name="k"))) == ()

    def test_a_field_that_is_not_credential_shaped_passes(self) -> None:
        assert literal_credentials(_Config(name="svc")) == ()

    def test_an_empty_credential_field_is_not_a_leak(self) -> None:
        assert literal_credentials(_Config(password="")) == ()

    def test_nested_models_are_reached(self) -> None:
        config = _Config(nested=_Nested(api_key=VALUE, endpoint="https://example.invalid"))
        assert literal_credentials(config) == ("nested.api_key",)

    def test_models_inside_a_sequence_are_reached(self) -> None:
        config = _Config(providers=(_Nested(), _Nested(api_key=VALUE)))
        assert literal_credentials(config) == ("providers[1].api_key",)

    def test_a_credential_shaped_mapping_key_is_reached(self) -> None:
        found = literal_credentials(_Config(by_name={"db_password": VALUE}))
        assert found == ("by_name['db_password']",)

    def test_a_model_inside_a_mapping_is_reached(self) -> None:
        config = _Config(by_provider={"openai": _Nested(api_key=VALUE), "local": _Nested()})
        assert literal_credentials(config) == ("by_provider['openai'].api_key",)

    def test_a_sequence_of_plain_values_passes(self) -> None:
        assert literal_credentials(_Config(hosts=("a", "b"))) == ()

    def test_a_plain_mapping_entry_passes(self) -> None:
        assert literal_credentials(_Config(by_name={"region": "asia-south1"})) == ()

    def test_findings_come_back_in_a_stable_order(self) -> None:
        config = _Config(password=VALUE, nested=_Nested(api_key=VALUE))
        assert literal_credentials(config) == ("nested.api_key", "password")


class TestNothingRevealedReachesARendering:
    def _populated(self) -> _Config:
        return _Config(
            name="svc",
            token=SecretStr(VALUE),
            ref=SecretRef(name="openai-key", version="7"),
            nested=_Nested(endpoint="https://example.invalid"),
        )

    def test_not_the_repr(self) -> None:
        assert VALUE not in repr(self._populated())

    def test_not_the_str(self) -> None:
        assert VALUE not in str(self._populated())

    def test_not_a_json_dump(self) -> None:
        assert VALUE not in self._populated().model_dump_json()

    def test_not_a_dict_dump_rendered_for_a_log(self) -> None:
        assert VALUE not in json.dumps(self._populated().model_dump(mode="json"))

    def test_not_a_deep_copy_of_it(self) -> None:
        assert VALUE not in repr(copy.deepcopy(self._populated()))

    def test_not_a_pickled_round_trip_of_it(self) -> None:
        restored = pickle.loads(pickle.dumps(self._populated()))  # noqa: S301 — the kit's own bytes
        assert VALUE not in repr(restored)
        assert restored.token is not None
        assert restored.token.get_secret_value() == VALUE

    def test_and_a_refusal_names_the_reference_not_the_value(self) -> None:
        refusal = SecretResolutionError("unreachable", ref="openai-key@7", backend="env")
        assert VALUE not in str(refusal)
        assert refusal.ref == "openai-key@7"
        assert refusal.backend == "env"

    def test_the_value_is_revealed_only_where_it_is_asked_for(self) -> None:
        config = self._populated()
        assert config.token is not None
        assert config.token.get_secret_value() == VALUE
