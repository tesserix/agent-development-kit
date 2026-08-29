"""Refuse a bad payload at the boundary it enters, and say where it went wrong.

Four scenarios: a payload rejected with every failing field named at once, a provider field
the kit does not model kept rather than dropped, a sensitive field that stays out of
telemetry while still round-tripping through a checkpoint, and an environment string parsed
once at the only edge that has no types. Run it with `python examples/models.py`.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import SecretStr

from tesserix_adk.core import (
    AdkModel,
    BinaryPart,
    Message,
    ProviderConfig,
    SchemaViolationError,
    Sensitive,
    TextPart,
    Usage,
    parsed_from_strings,
    telemetry_dump,
    validated,
)


class ToolArguments(AdkModel):
    """What a tool was asked for — the shape a provider's arguments have to take."""

    query: str
    page: int = 1
    api_key: Annotated[str, Sensitive("a tool credential is not a span attribute")] = ""


def a_violation_names_every_field() -> None:
    """Refuse a payload once, naming every field that failed rather than the first."""
    payload = {"query": 12, "page": "2", "pages": 3}

    try:
        validated(ToolArguments, payload)
    except SchemaViolationError as failure:
        print(f"model:    {failure.model}")  # noqa: T201
        print(f"paths:    {failure.paths}")  # noqa: T201
        print(f"retry?    {failure.retryable}")  # noqa: T201
        print(f"payload:  {failure.payload}")  # noqa: T201

    # `"2"` is not 2. A coerced value hides a real integration defect until the day the
    # string is "two".
    nested = {"role": "user", "content": [{"kind": "binary", "media_type": 1, "data": ""}]}
    try:
        validated(Message, nested)
    except SchemaViolationError as failure:
        print(f"nested:   {failure.paths}")  # noqa: T201


def a_provider_can_still_evolve() -> None:
    """Keep a field the kit does not model, in the map that declares it as unmodelled."""
    usage = Usage(input_tokens=100, output_tokens=20, extras={"reasoning_tokens": 7})
    print(f"extras:   {usage.extras}")  # noqa: T201

    # The same field loose on the model is refused: an extras map says "the kit does not
    # understand this", where an accepted unknown field says "somebody reads this".
    try:
        validated(Usage, {"input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 7})
    except SchemaViolationError as failure:
        print(f"loose:    {failure.paths}")  # noqa: T201


def telemetry_sees_less_than_a_checkpoint() -> None:
    """Drop the credential and the exhibit from the span, keep both in the checkpoint."""
    arguments = ToolArguments(query="kyoto in november", api_key="a-fixture-not-a-credential")
    print(f"span:     {telemetry_dump(arguments)}")  # noqa: T201

    # The credential and the exhibit are both still there when the run is rehydrated.
    restored = ToolArguments.model_validate_json(arguments.model_dump_json())
    print(f"restored: {restored == arguments}")  # noqa: T201

    exhibit = BinaryPart(media_type="image/png", data=b"\x89PNG\r\n")
    print(f"exhibit:  {telemetry_dump(exhibit)}")  # noqa: T201

    config = ProviderConfig(
        endpoint="https://llm.court.internal", api_key=SecretStr("a-fixture-not-a-credential")
    )
    if telemetry_dump(config)["api_key"] != "**********":
        raise RuntimeError("telemetry redaction contract failed")
    print("secret:   **********")  # noqa: T201


def the_environment_is_the_one_edge_without_types() -> None:
    """Parse a string where the source only has strings, and refuse it where it is not."""
    print(f"parsed:   {parsed_from_strings(int, '222')!r}")  # noqa: T201

    try:
        parsed_from_strings(int, "not a number")
    except SchemaViolationError as failure:
        print(f"refused:  {failure.problems}")  # noqa: T201


def main() -> None:
    """Run all four."""
    a_violation_names_every_field()
    a_provider_can_still_evolve()
    telemetry_sees_less_than_a_checkpoint()
    the_environment_is_the_one_edge_without_types()

    message = Message(role="user", content=[TextPart(text="what did the witness say?")])
    print(f"message:  {telemetry_dump(message)}")  # noqa: T201


if __name__ == "__main__":
    main()
