import pytest

from llmprefs.comparisons import generate_options
from llmprefs.structs import TaskType
from llmprefs.testing.factories import task_record_factory


class TestGenerateOptions:
    def test_invalid_tasks_per_option(self) -> None:
        with pytest.raises(ValueError, match="tasks_per_option must be at least 1"):
            list(generate_options(records=[], tasks_per_option=0))

    def test_empty_records(self) -> None:
        options = generate_options(records=[], tasks_per_option=1)
        assert list(options) == []

    @pytest.mark.parametrize("task_type", [TaskType.dummy, TaskType.opt_out])
    def test_one_record(self, task_type: TaskType) -> None:
        records = task_record_factory([task_type])
        options = generate_options(records, tasks_per_option=1)
        assert list(options) == [records]
