from __future__ import annotations

from collections.abc import Sequence

from llmprefs.structs import TaskDifficulty, TaskImpact, TaskRecord, TaskTopic, TaskType


def task_record_factory(task_types: Sequence[TaskType]) -> tuple[TaskRecord, ...]:
    """Create TaskRecords with specified TaskTypes."""
    records: list[TaskRecord] = []
    for i, task_type in enumerate(task_types):
        record = TaskRecord(
            id=i,
            type=task_type,
            task=f"Test task {i}",
            topic=TaskTopic.dummy,
            difficulty=TaskDifficulty.easy,
            impact=TaskImpact.neutral,
        )
        records.append(record)
    return tuple(records)
