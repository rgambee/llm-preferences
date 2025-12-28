from collections.abc import Sequence
from datetime import UTC, datetime

from llmprefs.api.structs import LLM, MockApiParams, MockApiResponse, Provider
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
        api_params=MockApiParams(
            provider=Provider.MOCK,
            model=LLM.MOCK_MODEL,
            max_output_tokens=1000,
            system_prompt="You are a helpful assistant.",
            temperature=1.0,
        ),
        api_response=MockApiResponse(reply="Option 1"),
    )
