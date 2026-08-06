"""A tool result is only readable next to the call it answers.

Every vendor wants the assistant turn that asked for a tool before the result that
answers it, and matches the two by id. A history that records only the results is a
history no vendor accepts and no reader can follow, so the turn is recorded as it
happened.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tesserix_adk.core import (
    Agent,
    Message,
    ModelResponse,
    NoOutput,
    StopReason,
    TextPart,
    ToolCall,
)
from tesserix_adk.runtime import AgentRunner
from tesserix_adk.testing import FakeClock, FakeToolRegistry, ScriptedProvider


def agent() -> Agent[NoOutput]:
    return Agent(
        name="clerk",
        instructions="Answer from sources.",
        model="scripted-1",
        free_text=True,
        tools=("lookup",),
    )


class TestAnAssistantTurnCanCarryTheCallsItMade:
    def test_an_assistant_message_carries_its_tool_calls(self) -> None:
        message = Message(
            role="assistant",
            content=[TextPart(text="looking that up")],
            tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),),
        )
        assert message.tool_calls[0].id == "call_1"

    def test_a_turn_that_only_calls_tools_says_nothing_in_words(self) -> None:
        message = Message(
            role="assistant",
            tool_calls=(ToolCall(id="call_1", name="lookup"),),
        )
        assert message.content == []

    def test_a_message_with_neither_content_nor_calls_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="content"):
            Message(role="assistant")

    def test_only_an_assistant_asks_for_a_tool(self) -> None:
        with pytest.raises(ValidationError, match="assistant"):
            Message(
                role="user",
                content=[TextPart(text="hi")],
                tool_calls=(ToolCall(id="call_1", name="lookup"),),
            )

    def test_the_turn_round_trips_through_json(self) -> None:
        """A run is checkpointed by one process and rehydrated by another."""
        message = Message(
            role="assistant",
            content=[TextPart(text="looking")],
            tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),),
        )
        assert Message.model_validate_json(message.model_dump_json()) == message


class TestTheLoopRecordsTheCallBeforeTheResult:
    async def test_the_assistant_turn_precedes_the_result_it_asked_for(self) -> None:
        provider = ScriptedProvider(
            ModelResponse(
                tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),),
                stop_reason=StopReason.TOOL_CALLS,
            ),
            ModelResponse(content="it rained"),
        )
        run = await AgentRunner(
            provider=provider,
            tools=FakeToolRegistry({"lookup": lambda q: f"{q}: yes"}),
            clock=FakeClock(),
        ).run(agent(), "did it rain", tenant="acme")
        roles = [message.role for message in run.messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]

    async def test_the_recorded_call_is_the_one_the_result_answers(self) -> None:
        provider = ScriptedProvider(
            ModelResponse(
                tool_calls=(ToolCall(id="call_1", name="lookup", arguments={"q": "rain"}),),
                stop_reason=StopReason.TOOL_CALLS,
            ),
            ModelResponse(content="it rained"),
        )
        run = await AgentRunner(
            provider=provider,
            tools=FakeToolRegistry({"lookup": lambda q: f"{q}: yes"}),
            clock=FakeClock(),
        ).run(agent(), "did it rain", tenant="acme")
        asked, answered = run.messages[2], run.messages[3]
        assert asked.tool_calls[0].id == answered.tool_call_id == "call_1"

    async def test_a_turn_with_no_tool_calls_records_no_calls(self) -> None:
        provider = ScriptedProvider(ModelResponse(content="no tools needed"))
        run = await AgentRunner(
            provider=provider,
            tools=FakeToolRegistry({"lookup": lambda q: f"{q}: yes"}),
            clock=FakeClock(),
        ).run(agent(), "hello", tenant="acme")
        assert all(message.tool_calls == () for message in run.messages)
