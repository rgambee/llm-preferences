from __future__ import annotations

from collections.abc import Sequence

from llmprefs.structs import (
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
