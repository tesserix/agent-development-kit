"""What a model can do, declared as data.

Duck typing a capability means finding out from the first request that needed it: the
tool call the model ignored, the image part it rejected, the schema it never enforced.
Each of those is a failure in production, on someone's budget, in the middle of a run.

So a provider declares a `ModelCapabilities` record, the kit checks it before the request
goes out, and a missing capability is a `CapabilityError` naming the capability, the
provider and the model. New capabilities are added as fields with defaults — a required
one would break every provider that already exists.

Every name exported here is semver-governed: it appears in `docs/api-surface.txt`, so a
change to it shows up in a pull request's diff and follows `docs/versioning.md`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BeforeValidator, Field

from tesserix_adk.core.errors import CapabilityError
from tesserix_adk.core.models import AdkModel
from tesserix_adk.core.trust import TrustBoundary

__all__ = ["Capability", "CapabilitySet", "ModelCapabilities", "ModelRef", "ModelSpec"]

_SEPARATOR = ":"


class Capability(StrEnum):
    """A thing a model either does or does not do, named the same way everywhere."""

    STRUCTURED_OUTPUT = "structured_output"
    TOOL_CALLING = "tool_calling"
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    VISION = "vision"
    STREAMING = "streaming"


CapabilitySet = Annotated[frozenset[Capability], BeforeValidator(frozenset)]
"""A set of capabilities, readable from the set or list a config file writes."""


class ModelCapabilities(AdkModel):
    """What a provider says its model supports.

    Every field defaults to off or unknown: a capability nobody declared is one the kit
    will not assume, because assuming it moves the failure to the first request that
    needed it.
    """

    structured_output: bool = False
    tool_calling: bool = False
    parallel_tool_calls: bool = False
    vision: bool = False
    streaming: bool = False
    context_window_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)

    @property
    def declared(self) -> frozenset[Capability]:
        """The capabilities turned on, as a set, for recording and for routing."""
        return frozenset(name for name in Capability if getattr(self, name.value))

    def declaring(self, **overrides: object) -> Self:
        """Return a copy carrying `overrides`, for a deployment that differs in one axis.

        Raises:
            ValidationError: If an override names a capability that does not exist.
        """
        return type(self).model_validate(self.model_dump() | overrides)

    def supports(self, requirement: Capability) -> bool:
        """Whether `requirement` is declared."""
        return requirement in self.declared

    def require(self, requirement: Capability, *, provider: str, model: str) -> None:
        """Raise unless `requirement` is declared.

        Raises:
            CapabilityError: Naming the capability, the provider and the model.
                "Unsupported" on its own leaves the caller to work out which of the three
                to change.
        """
        if not self.supports(requirement):
            raise CapabilityError(
                f"{provider}:{model} does not declare {requirement.value}. Declared: "
                f"{', '.join(sorted(c.value for c in self.declared)) or 'nothing'}",
                capability=requirement,
                provider=provider,
                model=model,
            )


class ModelRef(AdkModel):
    """A model, addressable by name from configuration.

    The provider is part of the identity rather than a lookup: a vendor API and an
    OpenAI-compatible proxy serve the same model ids, and they are not the same model.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)

    def __str__(self) -> str:
        """Return `provider:model`, the form configuration writes."""
        return f"{self.provider}{_SEPARATOR}{self.model}"

    @classmethod
    def parse(cls, text: str) -> Self:
        """Read a `provider:model` reference.

        Raises:
            ValueError: If the provider is absent. Defaulting it is how a proxy's
                traffic ends up recorded against the vendor.
        """
        provider, separator, model = text.partition(_SEPARATOR)
        if not separator:
            raise ValueError(f"a model reference is provider:model, got {text!r}")
        return cls(provider=provider, model=model)


class ModelSpec(AdkModel):
    """A named model together with what it can do and where it may be used.

    Args:
        provider: Who serves it.
        model: Which model they serve.
        capabilities: What it can do.
        trust: The boundary it sits inside. A fallback is legal only between models that
            share one; an unstated boundary constrains nothing and is admitted by nothing.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    capabilities: ModelCapabilities = Field(default_factory=ModelCapabilities)
    trust: TrustBoundary = Field(default_factory=TrustBoundary)

    @property
    def ref(self) -> ModelRef:
        """This model's reference."""
        return ModelRef(provider=self.provider, model=self.model)

    def with_capabilities(self, **overrides: object) -> Self:
        """Return a copy whose capability record carries `overrides`.

        A self-hosted endpoint serves the weights it was given rather than the ones on
        the model card, so a deployment must be able to narrow the record from
        configuration without writing a subclass to do it.

        Raises:
            ValidationError: If an override names a capability that does not exist.
        """
        return self.model_copy(update={"capabilities": self.capabilities.declaring(**overrides)})
