from collections.abc import Iterable

import numpy as np
import pytest

from llmprefs.analysis.task_order import (
    DirectComparison,
    IndirectComparison,
    OptionSizeError,
    OrderedTaskPair,
    ReducedResult,
    TaskOrder,
    UnorderedTaskPair,
    analyze_task_order,
    compute_delta_direct,
    compute_delta_indirect,
    find_relevant_comparisons,
    task_order,
)
from llmprefs.task_structs import ResultRecord, TaskId
from llmprefs.testing.factories import result_record_factory


@pytest.fixture
def mock_results() -> list[ResultRecord]:
    results = [result_record_factory() for _ in range(9)]
    results[0].comparison = ((1, 2), (3, 4))
    results[1].comparison = ((3, 4), (1, 2))
    results[2].comparison = ((4, 3), (2, 1))
    results[3].comparison = ((1, 3), (2, 4))
    results[4].comparison = ((4, 2), (3, 1))
    results[5].comparison = ((1, 2), (2, 1))
    results[6].comparison = ((3, 4), (4, 3))
    results[7].comparison = ((1, 3), (3, 1))
    results[8].comparison = ((4, 2), (2, 4))
    return results


class TestUnorderedTaskPair:
    @pytest.mark.parametrize(
        "tasks",
        [
            [1, 2],
            (1, 2),
            range(2),
        ],
    )
    def test_size_valid(self, tasks: Iterable[TaskId]) -> None:
        option = UnorderedTaskPair(tasks)
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
        with pytest.raises(OptionSizeError):
            UnorderedTaskPair(tasks)


class TestReducedResult:
    def test_ordered_pairs(self) -> None:
        result = ReducedResult(
            first_option=(1, 2),
            second_option=(3, 4),
            preferred_option_index=0,
        )
        assert result.first_pair_ordered == (1, 2)
        assert result.second_pair_ordered == (3, 4)

        result.second_option = (3,)
        with pytest.raises(OptionSizeError):
            _ = result.second_pair_ordered

    def test_unordered_pairs(self) -> None:
        result = ReducedResult(
            first_option=(1, 2),
            second_option=(3, 4),
            preferred_option_index=0,
        )
        assert result.first_pair_unordered == {1, 2}
        assert result.second_pair_unordered == {3, 4}

        result.second_option = (3,)
        with pytest.raises(OptionSizeError):
            _ = result.second_pair_unordered


class TestDirectComparison:
    def test_validation(self) -> None:
        DirectComparison(
            first_option=(1, 2),
            second_option=(2, 1),
            preferred_option_index=None,
        )
        with pytest.raises(ValueError, match="Comparison is not direct"):
            DirectComparison(
                first_option=(1, 2),
                second_option=(2, 3),
                preferred_option_index=None,
            )

    def test_signed_outcome(self) -> None:
        result = DirectComparison(
            first_option=(1, 2),
            second_option=(2, 1),
            preferred_option_index=0,
        )
        assert result.signed_outcome() == 1

        result.preferred_option_index = 1
        assert result.signed_outcome() == -1

        result.first_option = (2, 1)
        result.second_option = (1, 2)
        assert result.signed_outcome() == 1

        result.preferred_option_index = 0
        assert result.signed_outcome() == -1

        result.preferred_option_index = None
        assert result.signed_outcome() == 0


class TestIndirectComparison:
    def test_validation(self) -> None:
        IndirectComparison(
            first_option=(1, 2),
            second_option=(2, 3),
            preferred_option_index=None,
        )
        with pytest.raises(ValueError, match="Comparison is not indirect"):
            IndirectComparison(
                first_option=(1, 2),
                second_option=(2, 1),
                preferred_option_index=None,
            )

    def test_signed_outcome(self) -> None:
        result = IndirectComparison(
            first_option=(1, 2),
            second_option=(3, 4),
            preferred_option_index=0,
        )
        assert result.signed_outcome(UnorderedTaskPair((1, 2))) == 1
        assert result.signed_outcome(UnorderedTaskPair((3, 4))) == -1
        with pytest.raises(
            ValueError,
            match="Result does not contain the desired option",
        ):
            assert result.signed_outcome(UnorderedTaskPair((1, 3)))


class TestAnalyzeTaskOrder:
    def test_empty_results(self) -> None:
        analysis = analyze_task_order(results=[])

        assert len(analysis.tasks) == 0
        assert analysis.deltas_indirect.size == 0

    def test_multiple_results(self, mock_results: list[ResultRecord]) -> None:
        analysis = analyze_task_order(mock_results)

        assert analysis.tasks == (1, 2, 3, 4)
        assert analysis.deltas_indirect.shape == (4, 4)
        expected_indirect = np.array(
            [
                [np.nan, 0.5, 1.0, np.nan],
                [np.nan, np.nan, np.nan, -1.0],
                [np.nan, np.nan, np.nan, -0.5],
                [np.nan, np.nan, np.nan, np.nan],
            ],
        )
        for i in range(len(analysis.tasks)):
            for j in range(len(analysis.tasks)):
                if np.isnan(expected_indirect[i, j]):
                    assert np.isnan(analysis.deltas_indirect[i, j])
                else:
                    assert expected_indirect[i, j] == analysis.deltas_indirect[i, j]


class TestComputeDeltaDirect:
    def test_empty_results(self) -> None:
        delta = compute_delta_direct(
            results=[],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert np.isnan(delta)

    def test_one_result(self) -> None:
        result = result_record_factory()
        result.comparison = ((1, 2), (2, 1))
        delta = compute_delta_direct(
            results=[result],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == 1.0

    def test_no_order_effect(self) -> None:
        result0 = result_record_factory()
        result0.comparison = ((1, 2), (2, 1))
        result1 = result_record_factory()
        result1.comparison = ((2, 1), (1, 2))

        delta = compute_delta_direct(
            results=[result0, result1],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == 0.0

    def test_perfect_order_effect(self) -> None:
        result0 = result_record_factory()
        result0.comparison = ((1, 2), (2, 1))
        result1 = result_record_factory()
        result1.comparison = ((2, 1), (1, 2))
        result1.preferred_option_index = 1

        delta = compute_delta_direct(
            results=[result0, result1],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == 1.0

        result0.preferred_option_index = 1
        result1.preferred_option_index = 0

        delta = compute_delta_direct(
            results=[result0, result1],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == -1.0

    def test_partial_order_effect(self) -> None:
        results = [result_record_factory() for _ in range(3)]
        results[0].comparison = ((1, 2), (2, 1))
        results[1].comparison = ((2, 1), (1, 2))
        results[2].comparison = ((2, 1), (1, 2))
        delta = compute_delta_direct(
            results=results,
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == -1.0 / 3.0


class TestComputeDeltaIndirect:
    def test_empty_results(self) -> None:
        delta = compute_delta_indirect(
            results=[],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert np.isnan(delta)

    def test_one_result(self, mock_results: list[ResultRecord]) -> None:
        with pytest.raises(ValueError, match="Missing outcomes for one or more orders"):
            compute_delta_indirect(
                results=mock_results[:1],
                desired_option=UnorderedTaskPair((1, 2)),
            )

    def test_no_order_effect(self) -> None:
        result0 = result_record_factory()
        result0.comparison = ((1, 2), (3, 4))
        result1 = result_record_factory()
        result1.comparison = ((3, 4), (2, 1))
        result1.preferred_option_index = 1

        delta = compute_delta_indirect(
            results=[result0, result1],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == 0.0

    def test_perfect_order_effect(self) -> None:
        result0 = result_record_factory()
        result0.comparison = ((1, 2), (3, 4))
        result1 = result_record_factory()
        result1.comparison = ((3, 4), (2, 1))

        delta = compute_delta_indirect(
            results=[result0, result1],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == 1.0

        result0.preferred_option_index = 1
        result1.preferred_option_index = 1

        delta = compute_delta_indirect(
            results=[result0, result1],
            desired_option=UnorderedTaskPair((1, 2)),
        )
        assert delta == -1.0

    def test_partial_order_effect(self, mock_results: list[ResultRecord]) -> None:
        delta = compute_delta_indirect(
            results=mock_results,
            desired_option=UnorderedTaskPair((1, 2)),
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
                desired_option=UnorderedTaskPair((1, 2)),
                direct=direct,
            )
        )
        assert relevant_results == []

    def test_direct_comparisons(self, mock_results: list[ResultRecord]) -> None:
        desired_pair = UnorderedTaskPair((1, 2))
        relevant_results = list(
            find_relevant_comparisons(
                results=mock_results,
                desired_option=desired_pair,
                direct=False,
            )
        )
        assert len(relevant_results) == 3
        for res in relevant_results:
            assert desired_pair in {
                res.first_pair_unordered,
                res.second_pair_unordered,
            }
            assert not (
                res.first_pair_unordered == desired_pair
                and res.second_pair_unordered == desired_pair
            )

    def test_indirect_comparisons(self, mock_results: list[ResultRecord]) -> None:
        desired_pair = UnorderedTaskPair((1, 2))
        relevant_results = list(
            find_relevant_comparisons(
                results=mock_results,
                desired_option=desired_pair,
                direct=True,
            )
        )
        assert len(relevant_results) == 1
        for res in relevant_results:
            assert res.first_pair_unordered == desired_pair
            assert res.second_pair_unordered == desired_pair


class TestTaskOrder:
    @pytest.mark.parametrize(
        "option",
        [
            (0, 1),
            (1, 2),
            (0, 99),
        ],
    )
    def test_ascending(self, option: OrderedTaskPair) -> None:
        assert task_order(option) == TaskOrder.ASCENDING

    @pytest.mark.parametrize(
        "option",
        [
            (0, 0),
            (1, 0),
            (123, 42),
        ],
    )
    def test_descending(self, option: OrderedTaskPair) -> None:
        assert task_order(option) == TaskOrder.DESCENDING
