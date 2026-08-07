"""What a GGUF model will need, worked out before anything tries to load it.

A model that does not fit in RAM is not an error on a CPU box: it is an OOM kill, which
is a container restarting with nothing in the log to explain why. The arithmetic is
simple, so it is done in advance and the answer is a typed refusal that names the shortfall.

Quantization is the other half of the same question. Fewer bits per weight means a smaller
file and less memory bandwidth per token, which on CPU is the entire performance budget —
and worse output. Q4_K_M is where the published trade-off sits, so it is the default and
the kit drops below it only when the machine leaves no choice.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from tesserix_adk.core.errors import ConfigurationError
from tesserix_adk.core.models import AdkModel

__all__ = [
    "DEFAULT_KV_BYTES_PER_TOKEN",
    "DEFAULT_QUANTIZATION",
    "GgufModel",
    "MemoryEstimate",
    "ModelTooLargeError",
    "Quantization",
    "quantization_for",
    "refuse_if_it_will_not_fit",
]

# One token of KV cache for an 8B llama-architecture model at f16: two tensors, 32 layers,
# 8 KV heads, 128 head dimensions, 2 bytes each. Grouped-query attention makes this vary by
# an order of magnitude between models of the same size, so it is a field, not a constant.
DEFAULT_KV_BYTES_PER_TOKEN = 131_072

# Compute buffers, the tokenizer, and the server's own working set. Measured as roughly
# constant across model sizes on the CPU backend, which is why it is not a fraction.
_OVERHEAD_BYTES = 512 * 1024**2

_GIB = 1024**3
_BITS_PER_BYTE = 8


class Quantization(StrEnum):
    """A GGUF quantization, named as the file names it.

    The K-quants (`_K_M`, `_K_S`) mix precisions across tensors, which is why their bits
    per weight are not round numbers.
    """

    Q2_K = "q2_k"
    Q3_K_M = "q3_k_m"
    Q4_0 = "q4_0"
    Q4_K_M = "q4_k_m"
    Q5_K_M = "q5_k_m"
    Q6_K = "q6_k"
    Q8_0 = "q8_0"
    F16 = "f16"

    @property
    def bits_per_weight(self) -> float:
        """Average bits each parameter costs in this format, as llama.cpp publishes them."""
        return _BITS[self]


_BITS = {
    Quantization.Q2_K: 3.35,
    Quantization.Q3_K_M: 3.91,
    Quantization.Q4_0: 4.55,
    Quantization.Q4_K_M: 4.83,
    Quantization.Q5_K_M: 5.67,
    Quantization.Q6_K: 6.56,
    Quantization.Q8_0: 8.5,
    Quantization.F16: 16.0,
}

DEFAULT_QUANTIZATION = Quantization.Q4_K_M

# Smallest first, so the search stops at the lightest file that fits and the refusal can
# name the heaviest one that would have.
_LADDER = (
    Quantization.Q2_K,
    Quantization.Q3_K_M,
    Quantization.Q4_0,
    Quantization.Q4_K_M,
    Quantization.Q5_K_M,
    Quantization.Q6_K,
    Quantization.Q8_0,
    Quantization.F16,
)


class ModelTooLargeError(ConfigurationError):
    """Raised when a model cannot fit in the memory the machine has.

    A `ConfigurationError` because that is what it is — the wrong file for this box, found
    at assembly rather than by the OOM killer halfway through a run.
    """


class MemoryEstimate(AdkModel):
    """What loading a model will cost, broken into the parts that move for different reasons.

    Args:
        weights: The model file, which is fixed by the quantization.
        kv_cache: The attention cache, which grows linearly with the context length.
        overhead: Compute buffers and the server's own working set.
    """

    weights: int = Field(ge=0)
    kv_cache: int = Field(ge=0)
    overhead: int = Field(ge=0)

    @property
    def total(self) -> int:
        """Every part added up, which is what has to fit."""
        return self.weights + self.kv_cache + self.overhead

    @property
    def readable(self) -> str:
        """The total in the unit RAM is sold in, for a message a person has to act on."""
        return _gib(self.total)


class GgufModel(AdkModel):
    """A quantized model as it will be loaded.

    Args:
        name: What the file is, as an operator would recognise it.
        parameters_b: Parameter count in billions. 8.03 for Llama 3.1 8B, not 8.
        quantization: Which GGUF format.
        kv_bytes_per_token: One token of KV cache for this architecture. The default is an
            8B llama-architecture model with grouped-query attention; a model without it
            costs several times more, and using the default there under-counts badly.
    """

    name: str = Field(min_length=1)
    parameters_b: float = Field(gt=0)
    quantization: Quantization = DEFAULT_QUANTIZATION
    kv_bytes_per_token: int = Field(default=DEFAULT_KV_BYTES_PER_TOKEN, gt=0)

    def footprint(self, *, context_tokens: int) -> MemoryEstimate:
        """What this model needs with a context window of `context_tokens`."""
        return MemoryEstimate(
            weights=_weights(self.parameters_b, self.quantization),
            kv_cache=self.kv_bytes_per_token * context_tokens,
            overhead=_OVERHEAD_BYTES,
        )

    def at(self, quantization: Quantization) -> GgufModel:
        """The same model in another format."""
        return self.model_copy(update={"quantization": quantization})


def quantization_for(
    parameters_b: float,
    *,
    context_tokens: int,
    available_bytes: int,
    kv_bytes_per_token: int = DEFAULT_KV_BYTES_PER_TOKEN,
) -> Quantization:
    """The format to serve this model in on a machine with `available_bytes`.

    Never heavier than `DEFAULT_QUANTIZATION`: more bits per weight past that point buys
    little quality and costs memory bandwidth, which is the whole budget on CPU.

    Raises:
        ModelTooLargeError: If nothing in the ladder fits.
    """
    model = GgufModel(
        name="candidate", parameters_b=parameters_b, kv_bytes_per_token=kv_bytes_per_token
    )
    best = _heaviest_that_fits(model, context_tokens, available_bytes, ceiling=DEFAULT_QUANTIZATION)
    if best is None:
        needed = model.at(_LADDER[0]).footprint(context_tokens=context_tokens)
        raise ModelTooLargeError(
            f"a {parameters_b}B model needs at least {needed.readable} at "
            f"{_LADDER[0].value} with a {context_tokens}-token context, and this machine "
            f"has {_gib(available_bytes)}: no quantization fits"
        )
    return best


def refuse_if_it_will_not_fit(
    model: GgufModel, *, context_tokens: int, available_bytes: int
) -> None:
    """Check a model against the memory there is, before anything tries to load it.

    Raises:
        ModelTooLargeError: If it will not fit, naming the shortfall and — where there is
            one — a lighter quantization that would have.
    """
    estimate = model.footprint(context_tokens=context_tokens)
    if estimate.total <= available_bytes:
        return
    lighter = _heaviest_that_fits(model, context_tokens, available_bytes, ceiling=None)
    advice = f"{lighter.value} would fit" if lighter is not None else "no quantization of it fits"
    raise ModelTooLargeError(
        f"{model.name} at {model.quantization.value} with a {context_tokens}-token context "
        f"needs {estimate.readable} ({_gib(estimate.weights)} of weights, "
        f"{_gib(estimate.kv_cache)} of KV cache, {_gib(estimate.overhead)} of buffers) and "
        f"this machine has {_gib(available_bytes)}: {advice}"
    )


def _heaviest_that_fits(
    model: GgufModel, context_tokens: int, available_bytes: int, *, ceiling: Quantization | None
) -> Quantization | None:
    limit = _BITS[ceiling] if ceiling is not None else _BITS[Quantization.F16]
    fitting = [
        candidate
        for candidate in _LADDER
        if candidate.bits_per_weight <= limit
        and model.at(candidate).footprint(context_tokens=context_tokens).total <= available_bytes
    ]
    return fitting[-1] if fitting else None


def _weights(parameters_b: float, quantization: Quantization) -> int:
    return int(parameters_b * 1e9 * quantization.bits_per_weight / _BITS_PER_BYTE)


def _gib(value: int) -> str:
    return f"{value / _GIB:.2f} GiB"
