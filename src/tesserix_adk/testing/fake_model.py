"""A model provider a test scripts turn by turn, with no vendor SDK and no network.

A suite that reaches a real provider is a suite that is slow, costs money, and fails on
somebody else's outage. This one answers from a script, counts exactly the tokens the
script names, and raises the kit's own errors on demand, so the retry, budget and
degradation paths can be exercised without waiting for a real provider to misbehave.

An unscripted call raises rather than returning a polite default: a runaway loop that
gets an answer every time it asks fails the test in the shape of a hang, and the script
is the only record of how many calls the run was supposed to make.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from tesserix_adk.core import (
    AdkError,
    Capability,
    ModelResponse,
    ModelResponseError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
    StopReason,
    ToolCall,
    Usage,
)
from tesserix_adk.testing.fakes import CAPABLE, estimate_tokens, replayed

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence

    from tesserix_adk.core import Cost, Message, ModelCapabilities, ModelRequest, StreamEvent

__all__ = ["FakeModelProvider", "Fault", "ScriptExhaustedError", "ScriptedTurn"]


class ScriptExhaustedError(AdkError):
    """Raised when a run asked the fake for a turn the script does not have.

    The failure a runaway loop should produce: a test whose fake answers forever passes
    while the agent under test never stops, and the bill arrives in production.
    """


class Fault(StrEnum):
    """A failure a provider really produces, scripted so the recovery path is testable."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    TRANSPORT = "transport"
    MALFORMED = "malformed"


def _raised(fault: Fault, payload: str, *, provider: str, model: str) -> Exception:
    """The kit-vocabulary error `fault` stands for."""
    detail = f": {payload}" if payload else ""
    if fault is Fault.TIMEOUT:
        message = f"{provider} timed out{detail}"
        return ProviderTimeoutError(message, provider=provider, model=model)
    if fault is Fault.RATE_LIMIT:
        message = f"{provider} refused the call, rate limited{detail}"
        return RateLimitError(message, status=429, retry_after=1.0, provider=provider, model=model)
    if fault is Fault.TRANSPORT:
        message = f"{provider} transport failure{detail}"
        return ProviderError(message, provider=provider, model=model)
    message = f"{provider} sent a malformed body{detail}"
    return ModelResponseError(message, payload=payload, provider=provider)


@dataclass(frozen=True, slots=True)
class ScriptedTurn:
    """One turn the fake will answer with, or one failure it will raise.

    Built through `saying`, `calling`, `returning` and `failing` rather than directly, so
    a script reads as the conversation it stands for.

    Args:
        response: What to answer with, where this turn answers.
        fault: What to raise instead, where this turn fails.
        payload: The raw body a fault carries, kept so a failure report has something in
            it beyond the name of the error.
    """

    response: ModelResponse | None = None
    fault: Fault | None = None
    payload: str = ""

    @classmethod
    def saying(
        cls,
        content: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        cost: Cost | None = None,
        reasoning: str = "",
        stop_reason: StopReason = StopReason.END_TURN,
    ) -> ScriptedTurn:
        """A turn that answers in prose.

        Args:
            content: The answer.
            input_tokens: Prompt tokens to report. Reported exactly, never estimated: an
                assertion about a budget compared against an approximation is an
                assertion that flakes.
            output_tokens: Generated tokens to report.
            cached_tokens: Prompt tokens served from cache.
            cost: What the call cost, where the test asserts about money.
            reasoning: The model's own working out, where the test needs one.
            stop_reason: Why generation ended.
        """
        usage = Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            cost=cost,
        )
        return cls(
            response=ModelResponse(
                content=content, reasoning=reasoning, usage=usage, stop_reason=stop_reason
            )
        )

    @classmethod
    def calling(
        cls,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        call_id: str = "call_1",
        content: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: Cost | None = None,
    ) -> ScriptedTurn:
        """A turn that asks for a tool.

        Naming a tool the registry does not hold is allowed on purpose: what the runtime
        does with an unknown tool is exactly what a test needs to pin down.
        """
        return cls(
            response=ModelResponse(
                content=content,
                tool_calls=(ToolCall(id=call_id, name=name, arguments=dict(arguments or {})),),
                usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost),
                stop_reason=StopReason.TOOL_CALLS,
            )
        )

    @classmethod
    def returning(
        cls,
        payload: object,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: Cost | None = None,
        stop_reason: StopReason = StopReason.END_TURN,
    ) -> ScriptedTurn:
        """A turn that answers with a structured payload, serialised as the content.

        A payload that violates the requested schema is returned rather than raised: what
        an invalid answer means is the runtime's decision, and a fake that refuses to
        produce one is a fake that hides the repair path.
        """
        return cls.saying(
            json.dumps(payload, sort_keys=True),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            stop_reason=stop_reason,
        )

    @classmethod
    def failing(cls, fault: Fault, *, payload: str = "") -> ScriptedTurn:
        """A turn that fails, in the kit's own vocabulary."""
        return cls(fault=fault, payload=payload)


_DEFAULT = ModelResponse(content="", stop_reason=StopReason.END_TURN)


class FakeModelProvider:
    """A `ModelProvider` that replays scripted turns and records what it was asked.

    Args:
        turns: The script, in order.
        name: The provider name recorded on the run.
        capabilities: What this fake declares. Declaring a capability the fake lacks is
            the point of the argument: the refusal path needs a provider that refuses.
        strict: Whether an unscripted call raises `ScriptExhaustedError`. Off, the fake
            answers with an empty end-of-turn reply, which suits a test about something
            else entirely.

    Example:
        >>> provider = FakeModelProvider(ScriptedTurn.saying("hello"))
        >>> provider.remaining
        1
    """

    def __init__(
        self,
        *turns: ScriptedTurn,
        name: str = "fake",
        capabilities: ModelCapabilities | None = None,
        strict: bool = True,
    ) -> None:
        self._turns = list(turns)
        self._name = name
        self._capabilities = capabilities if capabilities is not None else CAPABLE
        self._strict = strict
        self._turn = 0
        self._lock = asyncio.Lock()
        self.requests: list[ModelRequest] = []

    @classmethod
    def factory(
        cls,
        *turns: ScriptedTurn,
        name: str = "fake",
        capabilities: ModelCapabilities | None = None,
        strict: bool = True,
    ) -> Callable[[], FakeModelProvider]:
        """A callable handing each caller its own fake on the same script.

        Concurrent runs sharing one fake consume each other's turns, and both tests then
        assert about a conversation neither of them had.
        """
        return lambda: cls(*turns, name=name, capabilities=capabilities, strict=strict)

    def script(self, *turns: ScriptedTurn) -> FakeModelProvider:
        """Append turns, so a fake handed over by a fixture can still be scripted."""
        self._turns.extend(turns)
        return self

    @property
    def name(self) -> str:
        """The provider name."""
        return self._name

    @property
    def capabilities(self) -> ModelCapabilities:
        """What this fake declares."""
        return self._capabilities

    @property
    def calls(self) -> int:
        """How many times the run asked for a completion."""
        return len(self.requests)

    @property
    def remaining(self) -> int:
        """Turns the run never reached — above zero, it stopped earlier than scripted."""
        return max(len(self._turns) - self._turn, 0)

    def count_tokens(self, messages: Sequence[Message]) -> int:
        """Count by characters, which is deterministic and therefore assertable."""
        return estimate_tokens(messages)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Answer the next scripted turn.

        Raises:
            ScriptExhaustedError: In strict mode, when the script has no turn left.
            ProviderError: When the next turn scripts a failure.
            ModelResponseError: When the next turn scripts a malformed body.
        """
        async with self._lock:
            self.requests.append(request)
            turn = self._next()
        if turn is None:
            return _DEFAULT
        if turn.fault is not None:
            raise _raised(turn.fault, turn.payload, provider=self._name, model=request.model)
        return turn.response if turn.response is not None else _DEFAULT

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Replay the next scripted turn as the stream a vendor would have sent.

        A consumer that stops part way leaves the script where it is: the turn it
        abandoned was still spent, and the next call gets the next turn.

        Raises:
            CapabilityError: If this fake does not declare `streaming`.
        """
        self._capabilities.require(Capability.STREAMING, provider=self._name, model=request.model)
        return replayed(await self.complete(request))

    def _next(self) -> ScriptedTurn | None:
        """The turn owed to this call, or None where a lenient fake should improvise."""
        if self._turn >= len(self._turns):
            if self._strict:
                message = (
                    f"{self._name} was called {len(self.requests)} times; the script has "
                    f"{len(self._turns)} turns"
                )
                raise ScriptExhaustedError(message)
            return None
        turn = self._turns[self._turn]
        self._turn += 1
        return turn
