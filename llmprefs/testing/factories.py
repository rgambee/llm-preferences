from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from llmprefs.structs import (
    LLM,
    AnthropicApiParams,
    Provider,
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
        options=[[1], [2]],
        preferred_option_index=0,
        api_params=AnthropicApiParams(
            provider=Provider.ANTHROPIC,
            model=LLM.CLAUDE_SONNET_4_0_2025_05_14,
            max_tokens=1000,
            system_prompt="You are a helpful assistant.",
            temperature=1.0,
            thinking_budget=1000,
        ),
    )
