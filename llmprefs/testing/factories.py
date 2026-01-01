from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from anthropic.types.text_block import TextBlock as AnthropicText
from anthropic.types.tool_use_block import ToolUseBlock as AnthropicToolUse
from anthropic.types.usage import Usage as AnthropicUsage
from openai.types.responses import ResponseOutputMessage as OpenAiMessage
from openai.types.responses import ResponseOutputText as OpenAiText

from llmprefs.api.structs import (
    LLM,
    AnthropicApiResponse,
    MockApiParams,
    MockApiResponse,
    OpenAiApiResponse,
    Provider,
)
from llmprefs.task_structs import (
    ResultRecord,
    TaskDependency,
    TaskImpact,
    TaskObSubjectivity,
    TaskRecord,
    TaskTime,
    TaskTopic,
    TaskType,
)


def task_record_factory(task_types: Sequence[TaskType]) -> tuple[TaskRecord, ...]:
    """Create TaskRecords with specified TaskTypes."""
    records: list[TaskRecord] = []
    for i, task_type in enumerate(task_types):
        record = TaskRecord(
            id=i,
            task=f"Test task {i}",
            type=task_type,
            topic=TaskTopic.not_applicable,
            dependency=TaskDependency.not_applicable,
            ob_subjectivity=TaskObSubjectivity.not_applicable,
            time=TaskTime.brief,
            impact=TaskImpact.neutral,
        )
        records.append(record)
    return tuple(records)


def result_record_factory() -> ResultRecord:
    return ResultRecord(
        created_at=datetime.now(tz=UTC),
        comparison_prompt_id=123,
        comparison=((1,), (2,)),
        sample_index=0,
        comparison_prompt="Do you want to do Option 1 or Option 2?",
        preferred_option_index=0,
        api_params=MockApiParams(
            provider=Provider.MOCK,
            model=LLM.MOCK_MODEL,
            max_output_tokens=1000,
            system_prompt="You are a helpful assistant.",
            temperature=1.0,
        ),
        api_response=MockApiResponse(reply="Option 1"),
    )


def anthropic_response_factory_text(text: str) -> AnthropicApiResponse:
    return AnthropicApiResponse(
        id="abc123",
        content=[
            AnthropicText(
                text=text,
                type="text",
            )
        ],
        model="claude-haiku-4-5",
        role="assistant",
        type="message",
        usage=AnthropicUsage(
            input_tokens=123,
            output_tokens=456,
        ),
    )


def anthropic_response_factory_tool(tool_input: dict[str, Any]) -> AnthropicApiResponse:
    return AnthropicApiResponse(
        id="abc123",
        content=[
            AnthropicToolUse(
                id="abc123",
                input=tool_input,
                name="select_task",
                type="tool_use",
            )
        ],
        model="claude-haiku-4-5",
        role="assistant",
        type="message",
        usage=AnthropicUsage(
            input_tokens=123,
            output_tokens=456,
        ),
    )


def openai_response_factory(text: str) -> OpenAiApiResponse:
    return OpenAiApiResponse(
        id="abc123",
        created_at=1234567890.0,
        model="mock-model",
        object="response",
        output=[
            OpenAiMessage(
                id="abc123",
                content=[
                    OpenAiText(
                        annotations=[],
                        text=text,
                        type="output_text",
                    ),
                ],
                role="assistant",
                status="completed",
                type="message",
            ),
        ],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    )
