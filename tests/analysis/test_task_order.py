import warnings
from collections.abc import Iterable

import numpy as np
import pytest

from llmprefs.analysis.task_order import (
    OrderedOption,
    ReducedResult,
    TaskOrder,
    UnorderedOption,
    compute_delta,
    find_relevant_comparisons,
    task_order,
)
from llmprefs.task_structs import ResultRecord, TaskId
from llmprefs.testing.factories import result_record_factory


@pytest.fixture
def mock_results() -> list[ResultRecord]:
    results = [result_record_factory() for _ in range(5)]
    results[0].comparison = ((1, 2), (3, 4))
    results[1].comparison = ((3, 4), (1, 2))
    results[2].comparison = ((3, 4), (2, 1))
    results[3].comparison = ((1, 3), (2, 4))
    results[4].comparison = ((1, 2), (2, 1))
    return results


class TestUnorderedOption:
    @pytest.mark.parametrize(
        "tasks",
        [
            [1, 2],
            (1, 2),
            range(2),
        ],
    )
    def test_size_valid(self, tasks: Iterable[TaskId]) -> None:
        option = UnorderedOption(tasks)
        for task_id in option:
            assert task_id in tasks
        for task_id in tasks:
            assert task_id in option

    @pytest.mark.parametrize(
        "tasks",
        [
            [],
            [1],
            [1, 2, 3],
            range(1),
            range(3),
        ],
    )
    def test_size_invalid(self, tasks: Iterable[TaskId]) -> None:
        with pytest.raises(
            ValueError,
            match="Size of UnorderedOption must be 2",
        ):
            UnorderedOption(tasks)


class TestReducedResult:
    def test_unordered_options(self) -> None:
        result = ReducedResult(
            first_option=(1, 2),
            second_option=(3, 4),
            preferred_option_index=0,
        )
        assert result.unordered_first_option == {1, 2}
        assert result.unordered_second_option == {3, 4}

    def test_signed_outcome(self) -> None:
        result = ReducedResult(
            first_option=(1, 2),
            second_option=(3, 4),
            preferred_option_index=0,
        )
        assert result.signed_outcome(UnorderedOption((1, 2))) == 1
        assert result.signed_outcome(UnorderedOption((3, 4))) == -1
        with pytest.raises(
            ValueError,
            match="Result does not contain the desired pair",
        ):
            assert result.signed_outcome(UnorderedOption((1, 3)))


class TestComputeDelta:
    def test_empty_results(self) -> None:
        with (
            np.errstate(invalid="ignore"),
            warnings.catch_warnings(
                record=True,
                action="default",
                category=RuntimeWarning,
            ) as captured,
        ):
            delta = compute_delta(
                results=[],
                desired_pair=UnorderedOption((1, 2)),
            )
        assert np.isnan(delta)
        assert len(captured) == 1
        assert str(captured[0].message) == "Mean of empty slice"

    def test_one_result(self, mock_results: list[ResultRecord]) -> None:
        with pytest.raises(ValueError, match="Missing outcomes for one or more orders"):
            compute_delta(
                results=mock_results[:1],
                desired_pair=UnorderedOption((1, 2)),
            )

    def test_no_order_effect(self) -> None:
        result0 = result_record_factory()
        result0.comparison = ((1, 2), (3, 4))
        result1 = result_record_factory()
        result1.comparison = ((3, 4), (2, 1))
        result1.preferred_option_index = 1

        delta = compute_delta(
            results=[result0, result1],
            desired_pair=UnorderedOption((1, 2)),
        )
        assert delta == 0.0

    def test_perfect_order_effect(self) -> None:
        result0 = result_record_factory()
        result0.comparison = ((1, 2), (3, 4))
        result1 = result_record_factory()
        result1.comparison = ((3, 4), (2, 1))

        delta = compute_delta(
            results=[result0, result1],
            desired_pair=UnorderedOption((1, 2)),
        )
        assert delta == 1.0

        result0.preferred_option_index = 1
        result1.preferred_option_index = 1

        delta = compute_delta(
            results=[result0, result1],
            desired_pair=UnorderedOption((1, 2)),
        )
        assert delta == -1.0

    def test_partial_order_effect(self, mock_results: list[ResultRecord]) -> None:
        delta = compute_delta(
            results=mock_results,
            desired_pair=UnorderedOption((1, 2)),
        )
        # The ascending task order (1, 2) both won and lost. The descending order (2, 1)
        # lost. Therefore, the delta should be positive, indicating a slight preference
        # for ascending order.
        assert delta == 0.5


class TestFindRelevantComparisons:
    @pytest.mark.parametrize(
        "direct",
        [True, False],
    )
    def test_empty_results(self, direct: bool) -> None:  # noqa: FBT001
        relevant_results = list(
            find_relevant_comparisons(
                results=[],
                desired_pair=UnorderedOption((1, 2)),
                direct=direct,
            )
        )
        assert relevant_results == []

    def test_direct_comparisons(self, mock_results: list[ResultRecord]) -> None:
        desired_pair = UnorderedOption((1, 2))
        relevant_results = list(
            find_relevant_comparisons(
                results=mock_results,
                desired_pair=desired_pair,
                direct=False,
            )
        )
        assert len(relevant_results) == 3
        for res in relevant_results:
            assert desired_pair in {
                res.unordered_first_option,
                res.unordered_second_option,
            }
            assert not (
                res.unordered_first_option == desired_pair
                and res.unordered_second_option == desired_pair
            )

    def test_indirect_comparisons(self, mock_results: list[ResultRecord]) -> None:
        desired_pair = UnorderedOption((1, 2))
        relevant_results = list(
            find_relevant_comparisons(
                results=mock_results,
                desired_pair=desired_pair,
                direct=True,
            )
        )
        assert len(relevant_results) == 1
        for res in relevant_results:
            assert res.unordered_first_option == desired_pair
            assert res.unordered_second_option == desired_pair


class TestTaskOrder:
    @pytest.mark.parametrize(
        "option",
        [
            (0, 1),
            (1, 2),
            (0, 99),
        ],
    )
    def test_ascending(self, option: OrderedOption) -> None:
        assert task_order(option) == TaskOrder.ASCENDING

    @pytest.mark.parametrize(
        "option",
        [
            (0, 0),
            (1, 0),
            (123, 42),
        ],
    )
    def test_descending(self, option: OrderedOption) -> None:
        assert task_order(option) == TaskOrder.DESCENDING
