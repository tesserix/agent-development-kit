"""What a model can do is data the kit reads, not something it finds out by failing.

A capability discovered from a provider's error message is a capability discovered after
paying for the call, in production, on the first request that needed it. So a provider
declares what it supports as a record, the kit checks the record before the request goes
out, and the error names the capability, the provider and the model.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tesserix_adk.core import (
    Capability,
    CapabilityError,
    ModelCapabilities,
    ModelRef,
    ModelSpec,
)

FULL = ModelCapabilities(
    structured_output=True,
    tool_calling=True,
    parallel_tool_calls=True,
    vision=True,
    streaming=True,
    context_window_tokens=200_000,
    max_output_tokens=8_192,
)


class TestACapabilityRecordIsData:
    def test_a_provider_supports_only_what_it_declared(self) -> None:
        text_only = ModelCapabilities(tool_calling=True)
        assert text_only.supports(Capability.TOOL_CALLING)
        assert not text_only.supports(Capability.VISION)

    def test_nothing_is_supported_by_default(self) -> None:
        """Silence is not consent: an undeclared capability is one the model may not have."""
        assert ModelCapabilities().declared == frozenset()

    def test_the_declared_set_names_every_capability_turned_on(self) -> None:
        assert FULL.declared == frozenset(Capability)

    def test_the_record_cannot_be_edited_after_it_is_read(self) -> None:
        """A capability a caller can flip is a capability nothing can be checked against."""
        with pytest.raises(ValidationError):
            FULL.vision = False

    def test_an_unknown_capability_field_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ModelCapabilities(tool_callling=True)  # type: ignore[call-arg]

    def test_every_field_has_a_default_so_a_new_one_is_additive(self) -> None:
        """A capability added as a required argument breaks every provider in existence."""
        required = [n for n, f in ModelCapabilities.model_fields.items() if f.is_required()]
        assert required == []

    def test_a_window_must_be_a_positive_number_of_tokens(self) -> None:
        with pytest.raises(ValidationError):
            ModelCapabilities(context_window_tokens=0)


class TestRequiringACapability:
    def test_requiring_one_that_is_declared_passes_quietly(self) -> None:
        FULL.require(Capability.VISION, provider="acme", model="acme-1")

    def test_requiring_one_that_is_missing_names_all_three(self) -> None:
        """ "Unsupported" without the capability, provider and model is a bug hunt."""
        with pytest.raises(CapabilityError) as raised:
            ModelCapabilities().require(Capability.TOOL_CALLING, provider="acme", model="acme-1")
        assert raised.value.capability == Capability.TOOL_CALLING
        assert raised.value.provider == "acme"
        assert raised.value.model == "acme-1"
        assert "tool_calling" in str(raised.value)
        assert "acme-1" in str(raised.value)


class TestAModelIsAddressableByName:
    def test_a_reference_is_the_provider_and_the_model(self) -> None:
        assert str(ModelRef(provider="acme", model="acme-1")) == "acme:acme-1"

    def test_two_providers_serving_the_same_model_id_stay_distinct(self) -> None:
        """A vendor API and an OpenAI-compatible proxy answer to the same model name."""
        assert ModelRef(provider="acme", model="gpt-4o") != ModelRef(
            provider="proxy", model="gpt-4o"
        )

    def test_a_reference_parses_back_from_its_own_text(self) -> None:
        assert ModelRef.parse("acme:acme-1") == ModelRef(provider="acme", model="acme-1")

    def test_a_reference_without_a_provider_is_refused(self) -> None:
        """Defaulting the provider is how the proxy gets billed as the vendor."""
        with pytest.raises(ValueError, match="provider:model"):
            ModelRef.parse("acme-1")

    def test_a_spec_carries_the_capability_record_with_the_name(self) -> None:
        spec = ModelSpec(provider="acme", model="acme-1", capabilities=FULL)
        assert spec.ref == ModelRef(provider="acme", model="acme-1")
        assert spec.capabilities.supports(Capability.STREAMING)


class TestCapabilitiesAreOverridableFromConfiguration:
    def test_a_deployment_narrows_the_record_without_subclassing(self) -> None:
        """A self-hosted endpoint serves the weights it was given, not the ones on the card."""
        served = ModelSpec(provider="acme", model="acme-1", capabilities=FULL).with_capabilities(
            vision=False, context_window_tokens=32_000
        )
        assert not served.capabilities.supports(Capability.VISION)
        assert served.capabilities.context_window_tokens == 32_000
        assert served.capabilities.tool_calling

    def test_the_original_record_is_untouched(self) -> None:
        spec = ModelSpec(provider="acme", model="acme-1", capabilities=FULL)
        spec.with_capabilities(vision=False)
        assert spec.capabilities.supports(Capability.VISION)

    def test_an_override_of_a_field_that_does_not_exist_is_refused(self) -> None:
        spec = ModelSpec(provider="acme", model="acme-1", capabilities=FULL)
        with pytest.raises(ValidationError):
            spec.with_capabilities(visionn=False)


class TestOneRecordIsNarrowedFromAnother:
    def test_it_returns_a_copy_carrying_the_override(self) -> None:
        talks = ModelCapabilities(context_window_tokens=1_000)
        sees = talks.declaring(vision=True)
        assert sees.vision
        assert sees.context_window_tokens == 1_000
        assert not talks.vision

    def test_an_override_naming_nothing_real_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            ModelCapabilities().declaring(telepathy=True)
