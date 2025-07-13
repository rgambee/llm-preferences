import pytest

from llmprefs.comparisons import generate_options, is_opt_out_task
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

    def test_more_tasks_per_option_than_records(self) -> None:
        records = task_record_factory([TaskType.dummy, TaskType.opt_out])
        options = generate_options(records, tasks_per_option=3)
        # When given fewer regular records than `tasks_per_option`, the regular records
        # no not appear in the output options, though the opt out tasks still do. This
        # isn't ideal, but we live with it since fixing it would require consuming the
        # options iterable.
        assert list(options) == [(rec,) for rec in records if is_opt_out_task(rec)]

    def test_mixed_records(self) -> None:
        records = task_record_factory(
            [
                TaskType.dummy,
                TaskType.dummy,
                TaskType.opt_out,
                TaskType.opt_out,
            ]
        )
        options = list(generate_options(records, tasks_per_option=2))
        assert len(options) == 4
        assert options[0] == (records[0], records[1])
        assert options[1] == (records[1], records[0])
        # Opt outs by themselves
        assert options[2] == (records[2],)
        assert options[3] == (records[3],)
