from collections.abc import Iterable
from itertools import chain

import numpy as np
import pytest

from llmprefs.analysis.rating import (
    ComparisonOutcomes,
    RatedOptions,
    ValueCI,
    compile_matrix,
    median_opt_out_rating,
    rate_options,
)
from llmprefs.task_structs import ResultRecord, TaskId, TaskRecord, TaskType
from llmprefs.testing.factories import result_record_factory, task_record_factory

RNG = np.random.default_rng(seed=123)


@pytest.fixture
def mock_results() -> list[ResultRecord]:
    results = [result_record_factory() for _ in range(3)]
    results[0].comparison = ((0,), (1,))
    results[1].comparison = ((0,), (2,))
    results[2].comparison = ((1,), (2,))
    return results


def tasks_to_match_results(results: Iterable[ResultRecord]) -> dict[TaskId, TaskRecord]:
    task_ids = {
        task_id
        for result in results
        for task_id in chain.from_iterable(result.comparison)
    }
    tasks: dict[TaskId, TaskRecord] = {}
    for task_id in task_ids:
        task = task_record_factory([TaskType.regular])[0]
        task.id = task_id
        tasks[task_id] = task
    return tasks


def highest_rating(ratings: RatedOptions) -> ValueCI:
    return max(ratings.values(), key=lambda vci: vci.value)


class TestRateOptions:
    def test_zero_results(self) -> None:
        outcomes = ComparisonOutcomes(options=(), counts=np.array([]))
        ratings = rate_options(
            outcomes,
            tasks={},
            num_resamples=1,
            confidence=0.0,
        )
        assert ratings == {}

    def test_one_result(self) -> None:
        result = result_record_factory()
        tasks = tasks_to_match_results([result])
        assert result.preferred_option_index is not None
        preferred_option = result.comparison[result.preferred_option_index]
        outcomes = compile_matrix([result])
        ratings = rate_options(
            outcomes,
            tasks,
            num_resamples=1,
            confidence=0.0,
        )

        assert len(ratings) == len(result.comparison)
        assert preferred_option in ratings
        assert highest_rating(ratings) == ratings[preferred_option]

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        outcomes = compile_matrix(mock_results)
        tasks = tasks_to_match_results(mock_results)
        ratings = rate_options(
            outcomes,
            tasks,
            num_resamples=1,
            confidence=0.0,
        )

        assert len(ratings) == len(mock_results)
        for i in range(1, len(mock_results)):
            assert ratings[(i - 1,)].value > ratings[(i,)].value


class TestMedianOptOutRating:
    def test_zero_results(self) -> None:
        median = median_opt_out_rating(
            resampled_ratings=np.zeros((0, 0)),
            options=(),
            tasks={},
        )
        assert median == 0.0

    def no_opt_outs(self) -> None:
        num_tasks = 3
        resampled_ratings = RNG.random((10, num_tasks))
        tasks = {
            task.id: task
            for task in task_record_factory([TaskType.regular] * num_tasks)
        }
        options = tuple((task_id,) for task_id in tasks)
        assert median_opt_out_rating(resampled_ratings, options, tasks) == 0.0

    def one_opt_out(self) -> None:
        resampled_ratings = np.array([[1.23]])
        tasks = {task.id: task for task in task_record_factory([TaskType.opt_out])}
        options = tuple((task_id,) for task_id in tasks)
        assert median_opt_out_rating(resampled_ratings, options, tasks) == 1.23

    def multiple_opt_outs(self) -> None:
        resampled_ratings = np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
        )
        tasks = {task.id: task for task in task_record_factory([TaskType.opt_out] * 2)}
        options = tuple((task_id,) for task_id in tasks)
        assert median_opt_out_rating(resampled_ratings, options, tasks) == 3.5

    def mixed_task_types(self) -> None:
        resampled_ratings = np.array(
            [  # reg   oo  reg   oo
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [9.0, 10.0, 11.0, 12.0],
            ],
        )
        tasks = {
            task.id: task
            for task in task_record_factory([TaskType.regular, TaskType.opt_out] * 2)
        }
        options = tuple((task_id,) for task_id in tasks)
        assert median_opt_out_rating(resampled_ratings, options, tasks) == 7.0


class TestCompileMatrix:
    def test_zero_results(self) -> None:
        outcomes = compile_matrix(())
        assert len(outcomes.options) == 0
        assert outcomes.counts.size == 0

    def test_one_result(self) -> None:
        result = result_record_factory()
        assert result.preferred_option_index is not None
        preferred_option = result.comparison[result.preferred_option_index]
        outcomes = compile_matrix([result])

        assert len(outcomes.options) == 2
        assert outcomes.counts.shape == (2, 2)
        assert outcomes.counts.sum() == 1
        for i, option in enumerate(outcomes.options):
            assert outcomes.counts[i, i] == 0
            if option == preferred_option:
                assert outcomes.counts[i, 1 - i] == 1
            else:
                assert outcomes.counts[i, 1 - i] == 0

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        outcomes = compile_matrix(mock_results)

        assert outcomes.options == ((0,), (1,), (2,))
        assert outcomes.counts.shape == (3, 3)
        expected_matrix = np.array(
            [
                [0, 1, 1],
                [0, 0, 1],
                [0, 0, 0],
            ]
        )
        assert (outcomes.counts == expected_matrix).all()
