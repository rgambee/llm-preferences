import pytest

from llmprefs.comparisons import (
    count_comparisons_approx,
    filter_comparisons,
    generate_comparisons,
    generate_options,
    has_consecutive_free_choices,
    is_opt_out_task,
)
from llmprefs.task_structs import TaskType
from llmprefs.testing.factories import task_record_factory


class TestConsecutiveFreeChoices:
    def test_empty(self) -> None:
        assert not has_consecutive_free_choices(())

    def test_single_record(self) -> None:
        assert not has_consecutive_free_choices(task_record_factory([TaskType.regular]))
        assert not has_consecutive_free_choices(
            task_record_factory([TaskType.free_choice])
        )

    def test_no_consecutive_free_choices(self) -> None:
        assert not has_consecutive_free_choices(
            task_record_factory(
                [
                    TaskType.free_choice,
                    TaskType.regular,
                    TaskType.free_choice,
                ],
            )
        )

    def test_consecutive_free_choices(self) -> None:
        assert has_consecutive_free_choices(
            task_record_factory(
                [
                    TaskType.regular,
                    TaskType.free_choice,
                    TaskType.free_choice,
                ],
            )
        )


class TestGenerateOptions:
    def test_invalid_tasks_per_option(self) -> None:
        with pytest.raises(ValueError, match="tasks_per_option must be at least 1"):
            list(generate_options(records=[], tasks_per_option=0))

    def test_empty_records(self) -> None:
        options = generate_options(records=[], tasks_per_option=1)
        assert list(options) == []

    @pytest.mark.parametrize("task_type", [TaskType.regular, TaskType.opt_out])
    def test_one_record(self, task_type: TaskType) -> None:
        records = task_record_factory([task_type])
        options = generate_options(records, tasks_per_option=1)
        assert list(options) == [records]

    def test_more_tasks_per_option_than_records(self) -> None:
        records = task_record_factory([TaskType.regular, TaskType.opt_out])
        options = generate_options(records, tasks_per_option=3)
        # When given fewer regular records than `tasks_per_option`, the regular records
        # no not appear in the output options, though the opt out tasks still do. This
        # isn't ideal, but we live with it since fixing it would require consuming the
        # options iterable.
        assert list(options) == [(rec,) for rec in records if is_opt_out_task(rec)]

    def test_multiple_free_choice_records(self) -> None:
        records = task_record_factory(
            [TaskType.regular, TaskType.free_choice, TaskType.free_choice],
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
                TaskType.regular,
                TaskType.regular,
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
            (TaskType.regular, TaskType.regular),
            (TaskType.regular, TaskType.opt_out),
            (TaskType.opt_out, TaskType.regular),
            (TaskType.regular, TaskType.free_choice),
            (TaskType.free_choice, TaskType.regular),
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
        record_a, record_b = task_record_factory([task_type] * 2)
        comparisons = [((record_a,), (record_b,)), ((record_b,), (record_a,))]
        filtered = list(filter_comparisons(comparisons))
        assert filtered == []

    @pytest.mark.parametrize(
        argnames=("type_a", "type_b"),
        argvalues=[
            (TaskType.regular, TaskType.regular),
            (TaskType.regular, TaskType.opt_out),
            (TaskType.opt_out, TaskType.regular),
            (TaskType.regular, TaskType.free_choice),
            (TaskType.free_choice, TaskType.regular),
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
            [TaskType.regular, TaskType.regular, irregular_type],
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
            [TaskType.regular, irregular_type, irregular_type],
        )
        comparisons = [((regular, irregular_a), (regular, irregular_b))]
        filtered = list(filter_comparisons(comparisons))
        assert filtered == []


class TestGenerateComparisons:
    def test_empty_records(self) -> None:
        comparisons = list(generate_comparisons(records=[], tasks_per_option=1))
        assert comparisons == []

    def test_one_record(self) -> None:
        records = task_record_factory([TaskType.regular])
        comparisons = list(generate_comparisons(records, tasks_per_option=1))
        assert comparisons == []

    def test_multiple_regular_records(self) -> None:
        records = task_record_factory([TaskType.regular] * 3)
        comparisons = list(generate_comparisons(records, tasks_per_option=1))
        # With 3 records and tasks_per_option == 1, there are 3 options. The number of
        # comparisons (permutations of length 2) is 3 * 2 * 1 == 6.
        assert len(comparisons) == 3 * 2 * 1

    def test_mixed_records(self) -> None:
        records = task_record_factory(
            [
                TaskType.regular,
                TaskType.regular,
                TaskType.opt_out,
                TaskType.opt_out,
                TaskType.free_choice,
                TaskType.free_choice,
            ]
        )
        assert len(list(generate_options(records, tasks_per_option=2))) == 12
        comparisons = list(generate_comparisons(records, tasks_per_option=2))
        # With 6 records and tasks_per_option == 2, there are 6 * 5 == 30 naive options.
        # However, opt out tasks can only appear on their own. The same goes for free
        # choice tasks, at least in this case since tasks_per_option == 2. That means
        # there are 4 * 3 + 2 - 2 == 12 valid options.
        #
        # The number of comparisons (permutations of length 2) is 12 * 11 == 132.
        # From these we filter out the following unhelpful comparisons:
        #     - opt_out1 vs opt_out2
        #     - opt_out2 vs opt_out1
        #     - regular1 + free_choice1 vs regular1 + free_choice2
        #     - regular2 + free_choice1 vs regular2 + free_choice2
        #     - regular1 + free_choice2 vs regular1 + free_choice1
        #     - regular2 + free_choice2 vs regular2 + free_choice1
        #     - free_choice1 + regular1 vs free_choice2 + regular1
        #     - free_choice1 + regular2 vs free_choice2 + regular2
        #     - free_choice2 + regular1 vs free_choice1 + regular1
        #     - free_choice2 + regular2 vs free_choice1 + regular2
        assert len(comparisons) == 132 - 2 - 8


class TestCountComparisons:
    def test_empty_records(self) -> None:
        assert count_comparisons_approx(records=[], tasks_per_option=1) == 0

    def test_one_record(self) -> None:
        records = task_record_factory([TaskType.regular])
        assert count_comparisons_approx(records, tasks_per_option=1) == 0

    def test_one_task_per_comparison(self) -> None:
        records = task_record_factory(
            [
                TaskType.regular,
                TaskType.regular,
                TaskType.opt_out,
                TaskType.opt_out,
                TaskType.free_choice,
                TaskType.free_choice,
            ]
        )
        num_options = len(records)
        expected_count = num_options * (num_options - 1)
        assert count_comparisons_approx(records, tasks_per_option=1) == expected_count

    def test_two_tasks_per_comparison(self) -> None:
        records = task_record_factory(
            [
                TaskType.regular,
                TaskType.regular,
                TaskType.opt_out,
                TaskType.opt_out,
                TaskType.free_choice,
                TaskType.free_choice,
            ]
        )
        # To simplify, we pretend we can combine any two regular or free-choice tasks
        # into an option, giving us 4 * 3 == 12. We can also pick either opt-out task as
        # an option by itself.
        # Note that not all these options would actually be generated. See the tests for
        # generate_comparisons for a correct count.
        num_options = 4 * 3 + 2
        expected_count = num_options * (num_options - 1)
        assert count_comparisons_approx(records, tasks_per_option=2) == expected_count
