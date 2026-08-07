"""`llama-server`, which is how an operator without a GPU runs anything at all.

llama.cpp with GGUF quantization is the mature CPU inference path, and it serves OpenAI's
wire format, so it arrives through the same adapter as vLLM — an agent does not become a
different agent because the weights moved onto a CPU.

Three things it needs that a hosted vendor does not. Its first token waits for the weights
to load, so the timeout is minutes rather than seconds. Its prompt cache is opt-in per
request, and without it every turn re-evaluates the whole prefix, which on CPU is the run.
And it will happily be started with a model larger than the machine, which is not an error
but an OOM kill — so the fit is checked here, before anything loads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, model_validator

from tesserix_adk.core.models import AdkModel
from tesserix_adk.models.gguf import refuse_if_it_will_not_fit
from tesserix_adk.models.providers.compatible import (
    CompatibilityPreset,
    OpenAICompatibleProvider,
)

if TYPE_CHECKING:
    from typing import Self

    import httpx

    from tesserix_adk.core.capabilities import ModelCapabilities
    from tesserix_adk.core.protocols import SecretProvider
    from tesserix_adk.core.provider import ModelRequest
    from tesserix_adk.models.gguf import GgufModel

__all__ = ["LLAMA_CPP", "LlamaCppProvider", "LlamaCppTuning"]

LLAMA_CPP = CompatibilityPreset(
    name="llama.cpp",
    strict_schemas=False,
    stream_usage_option=False,
    mints_tool_call_ids=True,
    timeout=600.0,
)


class LlamaCppTuning(AdkModel):
    """How the server was started, in the form both the operator and the kit read.

    The flags below are the ones that move CPU throughput. They are described here rather
    than left in a deployment's shell history so the fit check and the launch command
    cannot disagree about the context length.

    Args:
        threads: Generation threads. Physical cores, not hyperthreads — oversubscribing a
            CPU backend makes it slower, reliably and counter-intuitively. `None` leaves
            llama.cpp's own default.
        batch_size: Prompt tokens evaluated per pass. Larger fills memory bandwidth better
            on prompt ingestion, which is where a CPU run spends its first seconds.
        micro_batch_size: The physical batch inside a logical one, where a deployment tunes
            it separately.
        context_tokens: What the server was started with. The KV cache is sized from this
            whether or not a request uses it, so it is what the fit is checked against.
        prompt_cache: Whether to ask the server to keep the evaluated prefix between turns.
    """

    threads: int | None = Field(default=None, gt=0)
    batch_size: int | None = Field(default=None, gt=0)
    micro_batch_size: int | None = Field(default=None, gt=0)
    context_tokens: int = Field(default=4_096, gt=0)
    prompt_cache: bool = True

    @model_validator(mode="after")
    def _batches_agree(self) -> Self:
        if (
            self.batch_size is not None
            and self.micro_batch_size is not None
            and self.batch_size < self.micro_batch_size
        ):
            raise ValueError(
                f"a batch of {self.batch_size} is smaller than its micro-batch of "
                f"{self.micro_batch_size}, which llama.cpp reads as a batch of "
                f"{self.micro_batch_size}"
            )
        return self

    def server_arguments(self) -> tuple[str, ...]:
        """The `llama-server` flags this describes, in the order the docs list them."""
        flags: list[str] = []
        if self.threads is not None:
            flags += ["--threads", str(self.threads)]
        if self.batch_size is not None:
            flags += ["--batch-size", str(self.batch_size)]
        if self.micro_batch_size is not None:
            flags += ["--ubatch-size", str(self.micro_batch_size)]
        flags += ["--ctx-size", str(self.context_tokens)]
        return tuple(flags)


class LlamaCppProvider(OpenAICompatibleProvider):
    """A `llama-server` endpoint, with the CPU backend's own concerns handled.

    Args:
        model: The model id the server was started with.
        base_url: Where it answers. Usually a loopback address or a cluster-local service.
        capabilities: What this build and this model can do. Required, as for any
            self-hosted endpoint.
        tuning: How the server was started.
        weights: The GGUF file being served, where the fit is to be checked.
        available_bytes: Memory the machine has for it. Given with `weights`, the model is
            refused here rather than by the OOM killer halfway through a run.
        name: Overrides `llama.cpp`, for one of several local servers.
        secrets: Where a key comes from, on the rare server that wants one.
        api_key_variable: The variable holding that key. `None` sends no header, which is
            the local case.
        timeout: Seconds for one request. Defaults to ten minutes.
        transport: An injected `httpx` transport, for tests.

    Raises:
        ConfigurationError: If the address or the capabilities are missing.
        ModelTooLargeError: If `weights` will not fit in `available_bytes`.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        capabilities: ModelCapabilities | None,
        tuning: LlamaCppTuning | None = None,
        weights: GgufModel | None = None,
        available_bytes: int | None = None,
        name: str | None = None,
        secrets: SecretProvider | None = None,
        api_key_variable: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._tuning = tuning or LlamaCppTuning()
        if weights is not None and available_bytes is not None:
            refuse_if_it_will_not_fit(
                weights,
                context_tokens=self._tuning.context_tokens,
                available_bytes=available_bytes,
            )
        super().__init__(
            model,
            base_url=base_url,
            capabilities=capabilities,
            preset=LLAMA_CPP,
            name=name,
            api_key_variable=api_key_variable,
            secrets=secrets,
            timeout=timeout,
            transport=transport,
        )

    @property
    def tuning(self) -> LlamaCppTuning:
        """How the server this talks to was started."""
        return self._tuning

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        """Ask for the prompt cache, which llama.cpp does not turn on by itself."""
        payload = super()._payload(request)
        if self._tuning.prompt_cache:
            payload["cache_prompt"] = True
        return payload
