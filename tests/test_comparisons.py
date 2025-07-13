import pytest

from llmprefs.comparisons import filter_comparisons, generate_options, is_opt_out_task
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

    def test_multiple_free_choice_records(self) -> None:
        records = task_record_factory(
            [TaskType.dummy, TaskType.free_choice, TaskType.free_choice],
        )
        regular, free_choice_a, free_choice_b = records
        options = list(generate_options(records, tasks_per_option=2))
        assert len(options) == 4
        assert options[0] == (regular, free_choice_a)
        assert options[1] == (regular, free_choice_b)
        assert options[2] == (free_choice_a, regular)
        assert options[3] == (free_choice_b, regular)

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


class TestFilterComparisons:
    @pytest.mark.parametrize(
        argnames=("type_a", "type_b"),
        argvalues=[
            (TaskType.dummy, TaskType.dummy),
            (TaskType.dummy, TaskType.opt_out),
            (TaskType.opt_out, TaskType.dummy),
            (TaskType.dummy, TaskType.free_choice),
            (TaskType.free_choice, TaskType.dummy),
            (TaskType.opt_out, TaskType.free_choice),
            (TaskType.free_choice, TaskType.opt_out),
        ],
    )
    def test_two_dissimilar_tasks(self, type_a: TaskType, type_b: TaskType) -> None:
        record_a, record_b = task_record_factory([type_a, type_b])
        comparisons = [((record_a,), (record_b,)), ((record_b,), (record_a,))]
        filtered = list(filter_comparisons(comparisons))
        assert filtered == comparisons

    @pytest.mark.parametrize(
        argnames="task_type",
        argvalues=[TaskType.opt_out, TaskType.free_choice],
    )
    def test_two_similar_tasks(self, task_type: TaskType) -> None:
        record_a, record_b = task_record_factory([task_type, task_type])
        comparisons = [((record_a,), (record_b,)), ((record_b,), (record_a,))]
        filtered = list(filter_comparisons(comparisons))
        assert filtered == []

    @pytest.mark.parametrize(
        argnames=("type_a", "type_b"),
        argvalues=[
            (TaskType.dummy, TaskType.dummy),
            (TaskType.dummy, TaskType.opt_out),
            (TaskType.opt_out, TaskType.dummy),
            (TaskType.dummy, TaskType.free_choice),
            (TaskType.free_choice, TaskType.dummy),
            (TaskType.opt_out, TaskType.free_choice),
            (TaskType.free_choice, TaskType.opt_out),
        ],
    )
    def test_option_ordering(self, type_a: TaskType, type_b: TaskType) -> None:
        record_a, record_b = task_record_factory([type_a, type_b])
        comparisons = [((record_a, record_b), (record_b, record_a))]
        filtered = list(filter_comparisons(comparisons))
        assert filtered == comparisons

    @pytest.mark.parametrize(
        argnames="irregular_type",
        argvalues=[TaskType.opt_out, TaskType.free_choice],
    )
    def test_different_regular_tasks(self, irregular_type: TaskType) -> None:
        regular_a, regular_b, irregular = task_record_factory(
            [TaskType.dummy, TaskType.dummy, irregular_type],
        )
        comparisons = [((regular_a, irregular), (regular_b, irregular))]
        filtered = list(filter_comparisons(comparisons))
        assert filtered == comparisons

    @pytest.mark.parametrize(
        argnames="irregular_type",
        argvalues=[TaskType.opt_out, TaskType.free_choice],
    )
    def test_different_irregular_tasks(self, irregular_type: TaskType) -> None:
        regular, irregular_a, irregular_b = task_record_factory(
            [TaskType.dummy, irregular_type, irregular_type],
        )
        comparisons = [((regular, irregular_a), (regular, irregular_b))]
        filtered = list(filter_comparisons(comparisons))
        assert filtered == []
