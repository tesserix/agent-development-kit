"""Endpoints the operator runs, speaking OpenAI's wire format with their own accents.

vLLM, Ollama, TGI and friends implement enough of Chat Completions to be called through
the same adapter, and then each leaves something out: the usage object, the tool-call id,
the stop reason, the `strict` flag they never implemented. Left alone, every one of those
becomes a wrong answer rather than an error — a free call in the ledger, a tool result
matched back to nothing, a run that ends while the model was asking for a tool.

Two things are required rather than defaulted. The address, because there is no host to
guess for a service only the operator has named, and the capabilities, because a
deployment's flags decide them and no endpoint reports them honestly. A capability the
kit assumed is one it finds out about mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tesserix_adk.core.capabilities import Capability, ModelCapabilities
from tesserix_adk.core.cost import CountSource
from tesserix_adk.core.errors import ConfigurationError, ProviderError
from tesserix_adk.core.primitives import Message, TextPart, Usage
from tesserix_adk.core.provider import ModelResponse, StopReason
from tesserix_adk.core.streaming import StreamEnd
from tesserix_adk.models.providers.openai import COMPLETIONS_PATH, OpenAIProvider, _read, _Stream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping

    import httpx

    from tesserix_adk.core.protocols import SecretProvider
    from tesserix_adk.core.provider import ModelRequest
    from tesserix_adk.core.streaming import StreamEvent

__all__ = [
    "GROK",
    "GROQ",
    "OLLAMA",
    "OPENROUTER",
    "TGI",
    "VLLM",
    "XAI",
    "CompatibilityPreset",
    "OpenAICompatibleProvider",
]


@dataclass(frozen=True, slots=True)
class CompatibilityPreset:
    """One server's deviations from the format it claims to speak.

    Args:
        name: What the server is, as a run records it and routing selects it. Not
            `openai`: traffic to a box in the cluster billed to the vendor is traffic
            attributed to the wrong budget.
        strict_schemas: Whether the server implements the vendor's `strict` flag. Where
            it does not, claiming it is a 400 on the first structured request.
        stream_usage_option: Whether it understands `stream_options.include_usage`.
            Servers that do not may reject the whole body over the unknown field.
        mints_tool_call_ids: Whether the server omits ids on tool calls. It has to have
            one to match a result back to, so the adapter supplies what the server did not.
        completions_path: The provider's Chat Completions endpoint. Hosted compatible
            APIs commonly add a prefix that an absolute `/v1` path would otherwise drop.
        base_url: The provider's usual public endpoint. Empty for an operator-run service
            whose address cannot be guessed.
        api_key_variable: The provider's usual environment variable. `None` means the
            endpoint is unauthenticated unless the caller names one.
        timeout: Seconds for one request. Longer than a hosted vendor's, because the
            first token from a cold self-hosted model waits for the weights to load.
    """

    name: str
    strict_schemas: bool = False
    stream_usage_option: bool = True
    mints_tool_call_ids: bool = False
    completions_path: str = COMPLETIONS_PATH
    base_url: str = ""
    api_key_variable: str | None = None
    timeout: float = 120.0


VLLM = CompatibilityPreset(name="vllm", stream_usage_option=True, timeout=120.0)
OLLAMA = CompatibilityPreset(
    name="ollama", stream_usage_option=False, mints_tool_call_ids=True, timeout=300.0
)
TGI = CompatibilityPreset(name="tgi", stream_usage_option=False, timeout=120.0)
GROQ = CompatibilityPreset(
    name="groq",
    completions_path="/openai/v1/chat/completions",
    base_url="https://api.groq.com",
    api_key_variable="GROQ_API_KEY",
    timeout=60.0,
)
XAI = CompatibilityPreset(
    name="xai",
    base_url="https://api.x.ai",
    api_key_variable="XAI_API_KEY",
    timeout=60.0,
)
GROK = XAI
OPENROUTER = CompatibilityPreset(
    name="openrouter",
    completions_path="/api/v1/chat/completions",
    base_url="https://openrouter.ai",
    api_key_variable="OPENROUTER_API_KEY",
    timeout=60.0,
)

_DEFAULT = CompatibilityPreset(name="openai-compatible")


class OpenAICompatibleProvider(OpenAIProvider):
    """An OpenAI-compatible endpoint, described rather than probed.

    Args:
        model: The model id the server was started with.
        base_url: Where it answers. Hosted presets supply their public endpoint; an
            operator-run service still requires its in-cluster or gateway address.
        capabilities: What the deployment can do. Required.
        preset: Which server it is, and so which deviations to expect.
        name: Overrides the preset's name, for one of several deployments.
        api_key_variable: The variable holding the key, where the endpoint wants one.
            `None` uses the preset's variable; an empty string disables authentication.
        emulates: Whether the kit may stand in for a capability the endpoint has not
            declared. Off, because asking a small model for JSON in the prompt produces
            a schema enforced by nobody.
        secrets: Where the key comes from. Defaults to the environment.
        timeout: Seconds for one request. Defaults to the preset's.
        headers: Static gateway or attribution headers. `Authorization` and
            `Content-Type` remain owned by the adapter and cannot be overridden here.
        transport: An injected `httpx` transport, for tests and for a caller's own proxy.

    Raises:
        ConfigurationError: If the address or the capabilities are missing.
    """

    provider_name = _DEFAULT.name
    default_base_url = ""
    default_key_variable = ""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        capabilities: ModelCapabilities | None,
        preset: CompatibilityPreset = _DEFAULT,
        name: str | None = None,
        api_key_variable: str | None = None,
        emulates: bool = False,
        secrets: SecretProvider | None = None,
        timeout: float | None = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        resolved_base_url = preset.base_url if base_url is None else base_url
        if not resolved_base_url.strip():
            raise ConfigurationError(
                "an OpenAI-compatible provider needs a base_url; there is no default host "
                "for a service only the operator has named"
            )
        if capabilities is None:
            raise ConfigurationError(
                f"{name or preset.name} needs explicit capabilities: a self-hosted endpoint "
                f"reports none reliably, and a capability the kit assumed is one it finds "
                f"out about in the middle of a run"
            )
        self.provider_name = name or preset.name
        self._preset = preset
        self._emulates = emulates
        resolved_key_variable = (
            preset.api_key_variable if api_key_variable is None else api_key_variable
        )
        self._authenticated = bool(resolved_key_variable)
        self._extra_headers = {
            key: value
            for key, value in (headers or {}).items()
            if key.lower() not in {"authorization", "content-type"}
        }
        super().__init__(
            model,
            capabilities=capabilities,
            secrets=secrets,
            api_key_variable=resolved_key_variable or "",
            base_url=resolved_base_url,
            timeout=preset.timeout if timeout is None else timeout,
            transport=transport,
        )

    @property
    def emulates(self) -> bool:
        """Whether the kit may stand in for what this endpoint has not declared."""
        return self._emulates

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Send one request and return the whole answer, deviations reconciled.

        Raises:
            CapabilityError: If the request needs something the endpoint has not declared.
            ProviderError: On any transport or upstream failure, including an error the
                server returned under a 200.
            ModelResponseError: If the body cannot be read as a completion.
        """
        body = await self._post(
            self._preset.completions_path,
            self._payload(request),
            cost=self.count_tokens(request.messages),
        )
        _refuse_an_error_in_the_body(body, self.provider_name)
        answered = self._settled(self._completion(body), request)
        return self._reconciled(answered, request)

    async def stream(self, request: ModelRequest) -> AsyncIterator[StreamEvent]:
        """Send one request and return its events as they arrive.

        Raises:
            CapabilityError: If the endpoint does not declare streaming.
            StreamInterruptedError: If the stream ends before the model had finished.
        """
        self._capabilities.require(
            Capability.STREAMING, provider=self.provider_name, model=request.model
        )
        payload: dict[str, Any] = {**self._payload(request), "stream": True}
        if self._preset.stream_usage_option:
            payload["stream_options"] = {"include_usage": True}
        return self._counted(
            self._streamed(
                self._preset.completions_path,
                payload,
                request=request,
                state=_Stream(self.provider_name),
            ),
            request,
        )

    def _payload(self, request: ModelRequest) -> dict[str, Any]:
        payload = super()._payload(request)
        schema = payload.get("response_format")
        if schema is not None and not self._preset.strict_schemas:
            schema["json_schema"]["strict"] = False
        return payload

    def _completion(self, body: Mapping[str, Any]) -> ModelResponse:
        """Read the body, supplying the ids the server left off its tool calls."""
        return _read(_with_ids(body) if self._preset.mints_tool_call_ids else body, self.name)

    def _headers(self) -> dict[str, str]:
        """Send a key only where one was named: an in-cluster endpoint wants none."""
        headers = {**self._extra_headers, "content-type": "application/json"}
        if self._authenticated:
            headers["authorization"] = f"Bearer {self._credential.value()}"
        return headers

    async def _counted(
        self, events: AsyncIterator[StreamEvent], request: ModelRequest
    ) -> AsyncIterator[StreamEvent]:
        """Pass the stream through, estimating the usage on the end where none arrived."""
        async for event in events:
            if isinstance(event, StreamEnd):
                yield StreamEnd(response=self._reconciled(event.response, request))
            else:
                yield event

    def _reconciled(self, response: ModelResponse, request: ModelRequest) -> ModelResponse:
        """Fill in what the server left out: the stop reason and the token counts."""
        return response.model_copy(
            update={
                "stop_reason": _inferred(response),
                "usage": self._estimated(response, request),
            }
        )

    def _estimated(self, response: ModelResponse, request: ModelRequest) -> Usage:
        """Count the tokens ourselves where the server reported none.

        Zero is not a free call, it is an unmeasured one, and a ledger holding zeros for
        a GPU somebody is paying for is a ledger nobody can reconcile. The estimate is
        marked as one so that a report can say which is which.
        """
        usage = response.usage
        if usage.input_tokens or usage.output_tokens:
            return usage
        answered = Message(role="assistant", content=[TextPart(text=response.content)])
        return usage.model_copy(
            update={
                "input_tokens": self.count_tokens(request.messages),
                "output_tokens": self.count_tokens([answered]),
                "source": CountSource.TOKENISER,
            }
        )


def _inferred(response: ModelResponse) -> StopReason:
    """The stop reason the server sent, or the one its answer implies.

    `unknown` on a turn that asked for a tool ends the run with the call never made, so
    an omitted reason is read off what actually came back rather than passed on.
    """
    if response.stop_reason is not StopReason.UNKNOWN:
        return response.stop_reason
    return StopReason.TOOL_CALLS if response.tool_calls else StopReason.END_TURN


def _with_ids(body: Mapping[str, Any]) -> dict[str, Any]:
    """Give every tool call an id, for the servers that send none.

    Positional, so the same answer read twice mints the same ids: a random one would make
    a recorded run unreplayable.
    """
    read = dict(body)
    choices = read.get("choices")
    if not isinstance(choices, list):
        return read
    read["choices"] = [_choice_with_ids(choice) for choice in choices]
    return read


def _choice_with_ids(choice: object) -> object:
    if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
        return choice
    message = dict(choice["message"])
    calls = message.get("tool_calls")
    if not isinstance(calls, list):
        return choice
    message["tool_calls"] = [
        {**call, "id": call.get("id") or f"call-{position}"} if isinstance(call, dict) else call
        for position, call in enumerate(calls)
    ]
    return {**choice, "message": message}


def _refuse_an_error_in_the_body(body: Mapping[str, Any], provider: str) -> None:
    """Refuse a failure the server dressed as a success.

    Several compatible servers answer 200 with `{"error": ...}` and mean it. Reading on
    produces a response assembled from a failure, which is the one outcome worse than
    the failure.

    Raises:
        ProviderError: If the body carries an error rather than a choice.
    """
    reported = body.get("error")
    if not reported:
        return
    message = reported.get("message") if isinstance(reported, dict) else reported
    raise ProviderError(
        f"{provider} answered 200 with an error: {message}",
        details={"body": str(reported)[:_BODY_IN_ERRORS]},
    )


_BODY_IN_ERRORS = 500
