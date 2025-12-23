from collections.abc import Sequence
from datetime import UTC, datetime

from anthropic.types import TextBlock, Usage

from llmprefs.api.anthropic_api import AnthropicApiParams, AnthropicApiResponse
from llmprefs.api.structs import LLM, Provider
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
        preferred_option_index=0,
        api_params=AnthropicApiParams(
            provider=Provider.ANTHROPIC,
            model=LLM.MOCK_MODEL,
            max_output_tokens=1000,
            system_prompt="You are a helpful assistant.",
            temperature=1.0,
            thinking_budget=1000,
            structured_output=False,
        ),
        api_response=AnthropicApiResponse(
            id="123",
            type="message",
            model=LLM.MOCK_MODEL,
            content=[TextBlock(text="Option 1", type="text")],
            role="assistant",
            usage=Usage(
                input_tokens=10,
                output_tokens=5,
            ),
        ),
    )
