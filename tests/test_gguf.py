"""What a quantized model will need before anybody tries to load it.

A model that does not fit is an OOM kill: no error, no message, a container restarting.
The arithmetic is simple enough to do in advance, so it is done in advance.
"""

from __future__ import annotations

import pytest

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.models.gguf import (
    GgufModel,
    ModelTooLargeError,
    Quantization,
    quantization_for,
    refuse_if_it_will_not_fit,
)

GIB = 1024**3


def llama_8b(quantization: Quantization = Quantization.Q4_K_M) -> GgufModel:
    return GgufModel(name="llama-3.1-8b-instruct", parameters_b=8.03, quantization=quantization)


class TestWhatAQuantizedModelWeighs:
    def test_a_quantization_carries_its_bits_per_weight(self) -> None:
        assert Quantization.Q4_K_M.bits_per_weight == 4.83
        assert Quantization.F16.bits_per_weight == 16.0

    def test_weights_are_parameters_times_bits(self) -> None:
        """8.03B at 4.83 bits is about 4.5 GiB, which is what the published file weighs."""
        weights = llama_8b().footprint(context_tokens=0).weights
        assert 4.4 * GIB < weights < 4.7 * GIB

    def test_a_heavier_quantization_weighs_more(self) -> None:
        light = llama_8b(Quantization.Q4_K_M).footprint(context_tokens=0).weights
        heavy = llama_8b(Quantization.Q8_0).footprint(context_tokens=0).weights
        assert heavy > light

    def test_the_kv_cache_grows_with_the_context(self) -> None:
        """Context is not free, and a 32k window is where a model that fitted stops fitting."""
        short = llama_8b().footprint(context_tokens=4_096)
        long = llama_8b().footprint(context_tokens=32_768)
        assert long.kv_cache == short.kv_cache * 8

    def test_the_total_is_weights_plus_cache_plus_the_compute_buffers(self) -> None:
        estimate = llama_8b().footprint(context_tokens=8_192)
        assert estimate.total == estimate.weights + estimate.kv_cache + estimate.overhead

    def test_a_footprint_reads_as_gibibytes_because_that_is_how_ram_is_sold(self) -> None:
        assert llama_8b().footprint(context_tokens=8_192).readable.endswith(" GiB")


class TestAModelThatWillNotFit:
    def test_it_is_refused_before_loading_rather_than_killed_during_it(self) -> None:
        """The OOM killer explains nothing. This does."""
        with pytest.raises(ModelTooLargeError) as refused:
            refuse_if_it_will_not_fit(llama_8b(), context_tokens=8_192, available_bytes=2 * GIB)
        assert "llama-3.1-8b-instruct" in str(refused.value)

    def test_the_refusal_says_what_is_needed_and_what_there_is(self) -> None:
        with pytest.raises(ModelTooLargeError) as refused:
            refuse_if_it_will_not_fit(llama_8b(), context_tokens=8_192, available_bytes=2 * GIB)
        message = str(refused.value)
        assert "GiB" in message
        assert "2.00 GiB" in message

    def test_the_refusal_names_a_quantization_that_would_have_fitted(self) -> None:
        """An operator who is told only "no" goes and guesses."""
        with pytest.raises(ModelTooLargeError) as refused:
            refuse_if_it_will_not_fit(llama_8b(), context_tokens=4_096, available_bytes=5 * GIB)
        assert "q3_k_m" in str(refused.value).lower()

    def test_a_model_nothing_would_fit_says_so_rather_than_naming_one(self) -> None:
        with pytest.raises(ModelTooLargeError) as refused:
            refuse_if_it_will_not_fit(llama_8b(), context_tokens=4_096, available_bytes=GIB // 2)
        assert "no quantization" in str(refused.value).lower()

    def test_it_is_a_configuration_error_because_that_is_what_it_is(self) -> None:
        """Caught by anyone already handling assembly failures, and raised before any call."""
        assert issubclass(ModelTooLargeError, ConfigurationError)

    def test_a_model_that_fits_passes_silently(self) -> None:
        refuse_if_it_will_not_fit(llama_8b(), context_tokens=8_192, available_bytes=16 * GIB)


class TestChoosingAQuantization:
    def test_the_default_is_the_one_that_holds_quality(self) -> None:
        """Q4_K_M is the published trade-off point, not the smallest thing that loads."""
        assert quantization_for(8.03, context_tokens=8_192, available_bytes=16 * GIB) is (
            Quantization.Q4_K_M
        )

    def test_a_generous_machine_is_not_given_a_heavier_file_than_it_needs(self) -> None:
        """More bits per weight costs memory bandwidth, which on CPU is the whole budget."""
        assert quantization_for(8.03, context_tokens=8_192, available_bytes=64 * GIB) is (
            Quantization.Q4_K_M
        )

    def test_a_tight_machine_drops_to_what_fits(self) -> None:
        chosen = quantization_for(8.03, context_tokens=4_096, available_bytes=5 * GIB)
        assert chosen.bits_per_weight < Quantization.Q4_K_M.bits_per_weight

    def test_a_machine_nothing_fits_on_is_refused_rather_than_given_the_smallest(self) -> None:
        with pytest.raises(ModelTooLargeError):
            quantization_for(70.0, context_tokens=8_192, available_bytes=8 * GIB)


class TestTheModelDescription:
    def test_a_kv_cost_per_token_can_be_stated_because_architectures_differ(self) -> None:
        """Grouped-query attention changes it by an order of magnitude between two 8B models."""
        wide = GgufModel(
            name="mistral-7b",
            parameters_b=7.24,
            quantization=Quantization.Q4_K_M,
            kv_bytes_per_token=262_144,
        )
        assert wide.footprint(context_tokens=1_024).kv_cache == 262_144 * 1_024

    def test_a_parameter_count_has_to_be_positive(self) -> None:
        with pytest.raises(ValueError, match="greater than 0"):
            GgufModel(name="nothing", parameters_b=0, quantization=Quantization.Q4_K_M)
