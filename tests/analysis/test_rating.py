from collections.abc import Iterable, Mapping
from itertools import chain

import numpy as np
import pytest

from llmprefs.analysis.rating import (
    ComparisonOutcomes,
    ValueCI,
    compile_matrix,
    median_opt_out_rating,
    rate_options,
    resample_results,
)
from llmprefs.task_structs import OptionById, ResultRecord, TaskId, TaskRecord, TaskType
from llmprefs.testing.factories import result_record_factory, task_record_factory

RNG = np.random.default_rng(seed=123)


@pytest.fixture
def mock_results() -> list[ResultRecord]:
    results = [result_record_factory() for _ in range(4)]
    results[0].comparison = ((0,), (1,))
    results[1].comparison = ((0,), (2,))
    results[2].comparison = ((2,), (1,))
    results[2].preferred_option_index = 1
    results[3].comparison = ((1,), (0,))
    results[3].preferred_option_index = None
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


def highest_rating(rating_values: Mapping[OptionById, ValueCI]) -> ValueCI:
    return max(rating_values.values(), key=lambda vci: vci.value)


class TestComparisonOutcomes:
    def test_unfold_empty(self) -> None:
        outcomes = compile_matrix([])

        assert outcomes.unfold().shape == (0, 0)

    def test_unfold_one_result(self) -> None:
        outcomes = compile_matrix([result_record_factory()])
        unfolded = outcomes.unfold()

        assert unfolded.shape == (2, 2)
        assert unfolded.sum() == 1
        assert unfolded[0, 1] == 1

    def test_unfold_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        outcomes = compile_matrix(mock_results)
        num_options = len(outcomes.options)
        num_definite_outcomes = sum(
            1 for result in mock_results if result.preferred_option_index is not None
        )
        unfolded = outcomes.unfold()

        assert unfolded.shape == (num_options, num_options)
        assert unfolded.sum() == num_definite_outcomes
        assert unfolded[0, 1] == 1
        assert unfolded[0, 2] == 1
        assert unfolded[1, 2] == 1


class TestRateOptions:
    def test_zero_results(self) -> None:
        outcomes = ComparisonOutcomes(options=(), counts=np.array([]))
        ratings = rate_options(
            outcomes,
            tasks={},
            num_resamples=0,
        )
        assert ratings.options == ()
        assert ratings.ratings.shape == (0, 0)

    def test_one_result(self) -> None:
        result = result_record_factory()
        tasks = tasks_to_match_results([result])
        assert result.preferred_option_index is not None
        preferred_option = result.comparison[result.preferred_option_index]
        outcomes = compile_matrix([result])
        ratings = rate_options(
            outcomes,
            tasks,
            num_resamples=0,
        )

        assert len(ratings.options) == len(result.comparison)
        assert ratings.ratings.shape == (len(result.comparison), 1)
        assert preferred_option in ratings.options
        rating_values = ratings.values(confidence=0.0)
        assert highest_rating(rating_values) == rating_values[preferred_option]

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        outcomes = compile_matrix(mock_results)
        tasks = tasks_to_match_results(mock_results)
        ratings = rate_options(
            outcomes,
            tasks,
            num_resamples=0,
        )

        assert len(ratings.options) == len(outcomes.options)
        assert ratings.ratings.shape == (len(outcomes.options), 1)
        rating_values = ratings.values(confidence=0.0)
        for i in range(1, len(outcomes.options)):
            value_a = rating_values[outcomes.options[i - 1]].value
            value_b = rating_values[outcomes.options[i]].value
            assert value_a > value_b


class TestMedianOptOutRating:
    def test_zero_results(self) -> None:
        median = median_opt_out_rating(
            ratings=np.zeros((0, 0)),
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
        assert outcomes.counts.shape == (0, 0, 3)

    def test_one_result(self) -> None:
        result = result_record_factory()
        assert result.preferred_option_index == 0
        outcomes = compile_matrix([result])

        assert len(outcomes.options) == 2
        assert outcomes.counts.shape == (2, 2, 3)
        assert outcomes.counts.sum() == 1
        assert outcomes.counts[0, 1, 0] == 1

    def test_neither_option_preferred(self) -> None:
        result = result_record_factory()
        result.preferred_option_index = None
        outcomes = compile_matrix([result])

        assert len(outcomes.options) == 2
        assert outcomes.counts.shape == (2, 2, 3)
        assert outcomes.counts.sum() == 1
        assert outcomes.counts[0, 1, 2] == 1

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        outcomes = compile_matrix(mock_results)
        num_options = len(outcomes.options)

        assert outcomes.options == ((0,), (1,), (2,))
        assert outcomes.counts.shape == (num_options, num_options, 3)
        assert outcomes.counts.sum() == len(mock_results)
        assert outcomes.counts[0, 1, 0] == 1
        assert outcomes.counts[0, 2, 0] == 1
        assert outcomes.counts[1, 2, 0] == 1
        assert outcomes.counts[0, 1, 2] == 1


class TestResampleResults:
    def test_zero_results(self) -> None:
        outcomes = compile_matrix([])
        resample = resample_results(outcomes, RNG)

        assert resample.counts.shape == outcomes.counts.shape
        assert resample.counts.sum() == 0

    def test_one_result(self) -> None:
        outcomes = compile_matrix([result_record_factory()])
        resample = resample_results(outcomes, RNG)

        assert resample.counts.shape == outcomes.counts.shape
        assert np.all(resample.counts == outcomes.counts)

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        outcomes = compile_matrix(mock_results)

        resample = resample_results(outcomes, RNG)
        assert resample.counts.shape == outcomes.counts.shape
        assert resample.counts.sum() == len(mock_results)
        lower_triangle_indices = np.tril_indices(len(outcomes.options), k=0)
        # Resampled matrix should still be triangular: if i >= j,
        # then [i, j, :] should be zero.
        assert np.all(resample.counts[lower_triangle_indices] == 0)
