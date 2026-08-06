"""Every boundary in the kit is a strict model, and a violation says where it happened.

Data crosses an agent's boundaries as provider payloads, tool arguments, config blocks and
records read back from a store. Where those arrive as loose dicts, the first call site to
touch a field decides what it means, and a typo in a tool argument becomes an
AttributeError deep inside a tool body. A model at each crossing moves the failure to the
crossing and names the field.
"""

from __future__ import annotations

import inspect
import pkgutil
from datetime import timedelta
from importlib import import_module
from typing import Annotated

import pytest
from pydantic import BaseModel, Field, SecretStr

import tesserix_adk
from tesserix_adk.core import (
    AdkConfig,
    AdkModel,
    Agent,
    BinaryPart,
    Message,
    ProviderConfig,
    SchemaViolationError,
    Sensitive,
    TextPart,
    ToolCall,
    Usage,
    parsed_from_strings,
    telemetry_dump,
    validated,
)
from tesserix_adk.core.config import Duration

# Credential-shaped, credential to nothing: the fixture that proves redaction works.
FAKE_SECRET = "hunter2"  # noqa: S105 — a fixture, not a credential; gitleaks:allow


class Sample(AdkModel):
    """A boundary model of the shape the kit's own models take."""

    name: str
    count: int = 0
    secret: Annotated[str, Sensitive()] = ""
    nested: Sample | None = None
    extras: dict[str, str] = Field(default_factory=dict)


class TestTheBaseIsStrict:
    def test_an_unknown_field_is_refused_by_name(self) -> None:
        """A misspelt field silently ignored is a setting that never took effect."""
        with pytest.raises(SchemaViolationError, match="ncount"):
            validated(Sample, {"name": "a", "ncount": 2})

    def test_a_string_is_not_quietly_read_as_a_number(self) -> None:
        with pytest.raises(SchemaViolationError, match="count"):
            validated(Sample, {"name": "a", "count": "2"})

    def test_a_number_is_not_quietly_read_as_a_string(self) -> None:
        with pytest.raises(SchemaViolationError, match="name"):
            validated(Sample, {"name": 2})

    def test_a_boundary_model_cannot_be_mutated_after_validation(self) -> None:
        """A record, not a builder: what validated is what the next layer reads."""
        sample = Sample(name="a")

        with pytest.raises(ValueError, match="frozen"):
            sample.name = "b"

    def test_a_nested_payload_still_validates_as_a_model(self) -> None:
        sample = validated(Sample, {"name": "a", "nested": {"name": "b", "count": 1}})

        assert sample.nested == Sample(name="b", count=1)


class TestAViolationSaysWhere:
    def test_the_error_carries_the_model_the_path_and_the_payload(self) -> None:
        payload = {"name": "a", "nested": {"name": "b", "count": "3"}}

        with pytest.raises(SchemaViolationError) as raised:
            validated(Sample, payload)

        assert raised.value.model == "Sample"
        assert raised.value.paths == ("nested.count",)
        assert raised.value.payload == payload

    def test_every_failing_field_is_reported_at_once(self) -> None:
        """One round trip per fix is how a five-field config takes five deploys."""
        with pytest.raises(SchemaViolationError) as raised:
            validated(Sample, {"name": 1, "count": "x"})

        assert raised.value.paths == ("count", "name")

    def test_the_path_carries_the_index_and_the_union_member(self) -> None:
        """`content.0.binary.media_type` says which part of which message failed."""
        payload = {"role": "user", "content": [{"kind": "binary", "media_type": 1, "data": b""}]}

        with pytest.raises(SchemaViolationError) as raised:
            validated(Message, payload)

        assert raised.value.paths == ("content.0.binary.media_type",)

    def test_the_message_names_the_model_and_the_fields(self) -> None:
        with pytest.raises(SchemaViolationError, match=r"Sample.+count"):
            validated(Sample, {"name": "a", "count": "2"})

    def test_a_violation_is_not_worth_retrying(self) -> None:
        """The same payload validates the same way; asking again spends more to be refused."""
        with pytest.raises(SchemaViolationError) as raised:
            validated(Sample, {"name": 1})

        assert raised.value.retryable is False

    def test_a_payload_that_is_not_a_mapping_is_still_a_violation(self) -> None:
        with pytest.raises(SchemaViolationError) as raised:
            validated(Sample, ["name"])

        assert raised.value.model == "Sample"

    def test_a_valid_payload_is_returned_as_the_model(self) -> None:
        assert validated(Sample, {"name": "a"}) == Sample(name="a")


class TestDiscriminatedUnions:
    def test_a_content_part_resolves_by_its_discriminator(self) -> None:
        message = validated(
            Message,
            {
                "role": "user",
                "content": [{"kind": "binary", "media_type": "image/png", "data": ""}],
            },
        )

        assert isinstance(message.content[0], BinaryPart)

    def test_an_unknown_kind_names_the_discriminator_rather_than_guessing(self) -> None:
        """First-match guessing turns a new part type into a wrong part type."""
        with pytest.raises(SchemaViolationError, match="kind"):
            validated(Message, {"role": "user", "content": [{"kind": "video", "text": "hi"}]})

    def test_a_part_missing_its_discriminator_is_refused(self) -> None:
        with pytest.raises(SchemaViolationError, match="kind"):
            validated(Message, {"role": "user", "content": [{"text": "hi"}]})


class TestSensitiveFields:
    def test_a_sensitive_field_never_reaches_a_telemetry_dump(self) -> None:
        dumped = telemetry_dump(Sample(name="a", secret=FAKE_SECRET))

        assert FAKE_SECRET not in repr(dumped)
        assert "secret" not in dumped

    def test_a_sensitive_field_nested_in_another_model_is_dropped_too(self) -> None:
        dumped = telemetry_dump(Sample(name="a", nested=Sample(name="b", secret=FAKE_SECRET)))

        assert FAKE_SECRET not in repr(dumped)

    def test_a_sensitive_field_still_round_trips_through_a_checkpoint(self) -> None:
        """A checkpoint rehydrates a run; a dropped credential rehydrates a broken one."""
        sample = Sample(name="a", secret=FAKE_SECRET)

        assert Sample.model_validate_json(sample.model_dump_json()) == sample

    def test_a_telemetry_dump_keeps_the_fields_that_are_not_sensitive(self) -> None:
        dumped = telemetry_dump(Sample(name="a", count=2))

        assert dumped == {"name": "a", "count": 2, "nested": None, "extras": {}}

    def test_a_telemetry_dump_orders_fields_as_declared(self) -> None:
        """A dump whose key order moves makes every diff of two spans unreadable."""
        dumped = telemetry_dump(Sample(name="a", count=2))

        assert list(dumped) == ["name", "count", "nested", "extras"]

    def test_a_secret_value_is_masked_rather_than_printed(self) -> None:
        dumped = telemetry_dump(ProviderConfig(endpoint="http://x", api_key=SecretStr("k")))

        assert "k" not in repr(dumped.get("api_key", ""))

    def test_a_binary_payload_never_reaches_a_span(self) -> None:
        """A scanned exhibit is evidence in one system and a retention problem in another."""
        dumped = telemetry_dump(BinaryPart(media_type="image/png", data=b"\x89PNG"))

        assert dumped == {"kind": "binary", "media_type": "image/png"}

    def test_a_sensitive_field_inside_a_sequence_is_dropped_too(self) -> None:
        """A payload is safe on its own model and still a payload inside a message."""
        part = BinaryPart(media_type="image/png", data=b"\x89PNG")
        message = Message(role="user", content=[part])

        assert telemetry_dump(message)["content"] == [{"kind": "binary", "media_type": "image/png"}]

    def test_a_model_with_nothing_sensitive_dumps_whole(self) -> None:
        assert telemetry_dump(Usage(input_tokens=1, output_tokens=2))["input_tokens"] == 1


class TestStringsFromTheEnvironment:
    def test_a_number_arriving_as_a_string_parses_at_the_boundary(self) -> None:
        """Strict inside; the environment is outside, and everything there is a string."""
        assert parsed_from_strings(int, "222") == 222

    def test_a_duration_accepts_bare_seconds_and_iso8601(self) -> None:
        assert parsed_from_strings(Duration, "45") == timedelta(seconds=45)
        assert parsed_from_strings(Duration, "PT45S") == timedelta(seconds=45)

    def test_a_value_that_is_not_a_string_is_left_alone(self) -> None:
        assert parsed_from_strings(int, 222) == 222

    def test_a_string_that_is_not_the_type_is_refused(self) -> None:
        with pytest.raises(SchemaViolationError, match="int"):
            parsed_from_strings(int, "not a number")


class TestExtrasArePolicy:
    def test_provider_fields_the_kit_does_not_model_go_in_a_declared_map(self) -> None:
        """Forbidding extras must never mean a provider cannot evolve."""
        usage = Usage(input_tokens=1, output_tokens=1, extras={"reasoning_tokens": 7})

        assert usage.extras["reasoning_tokens"] == 7

    def test_the_same_field_loose_on_the_model_is_refused(self) -> None:
        with pytest.raises(SchemaViolationError, match="reasoning_tokens"):
            validated(Usage, {"input_tokens": 1, "output_tokens": 1, "reasoning_tokens": 7})


class TestEveryBoundaryModelRoundTrips:
    @pytest.mark.parametrize(
        "instance",
        [
            TextPart(text="hello"),
            BinaryPart(media_type="image/png", data=b"\x89PNG"),
            Message(role="user", content=[TextPart(text="hi")]),
            ToolCall(id="call_1", name="search", arguments={"q": "kyoto"}),
            Usage(input_tokens=1, output_tokens=2, cost=0.5, currency="USD"),
            AdkConfig(provider=ProviderConfig(endpoint="http://x")),
            Agent(name="a", instructions="do", model="claude-sonnet-5"),
        ],
        ids=lambda instance: type(instance).__name__,
    )
    def test_json_out_and_back_is_the_same_value(self, instance: BaseModel) -> None:
        restored = type(instance).model_validate_json(instance.model_dump_json())

        assert restored == instance


class TestThePolicyHolds:
    def test_every_model_in_the_kit_is_an_adk_model(self) -> None:
        """One base, so strictness cannot be forgotten on the model added next week."""
        loose = sorted(
            f"{model.__module__}.{model.__qualname__}"
            for model in _models_in_the_kit()
            if not issubclass(model, AdkModel)
        )

        assert loose == []

    def test_no_model_declares_an_alias(self) -> None:
        """Two names for one field is two spellings in every config file and payload."""
        aliased = sorted(
            f"{model.__qualname__}.{name}"
            for model in _models_in_the_kit()
            for name, field in model.model_fields.items()
            if field.alias or field.validation_alias or field.serialization_alias
        )

        assert aliased == []

    def test_the_base_forbids_extras_and_freezes_and_stays_strict(self) -> None:
        assert AdkModel.model_config["extra"] == "forbid"
        assert AdkModel.model_config["frozen"] is True
        assert AdkModel.model_config["strict"] is True


def _models_in_the_kit() -> list[type[BaseModel]]:
    found: dict[str, type[BaseModel]] = {}
    for module in pkgutil.walk_packages(tesserix_adk.__path__, f"{tesserix_adk.__name__}."):
        for _, member in inspect.getmembers(import_module(module.name), inspect.isclass):
            if issubclass(member, BaseModel) and member.__module__.startswith("tesserix_adk."):
                found[f"{member.__module__}.{member.__qualname__}"] = member
    return [model for model in found.values() if model is not AdkModel]
