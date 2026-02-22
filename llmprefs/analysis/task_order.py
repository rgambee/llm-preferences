from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain, combinations
from typing import Literal, Self, overload

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from llmprefs.analysis.structs import ReducedResultBase
from llmprefs.analysis.visualization import (
    annotated_heatmap,
    construct_title,
    get_tick_labels,
)
from llmprefs.task_structs import ResultRecord, TaskId, TaskRecord


class OptionSizeError(Exception):
    pass


class OrderedTaskPair(tuple[TaskId, TaskId]):
    EXPECTED_SIZE = 2

    __slots__ = ()

    def __new__(cls, tasks: Iterable[TaskId]) -> Self:
        instance = super().__new__(cls, tasks)
        if len(instance) != cls.EXPECTED_SIZE:
            raise OptionSizeError(
                f"Size of {instance.__class__.__name__} must be {cls.EXPECTED_SIZE}",
            )
        return instance


class UnorderedTaskPair(frozenset[TaskId]):
    EXPECTED_SIZE = 2

    def __new__(cls, tasks: Iterable[TaskId]) -> Self:
        instance = super().__new__(cls, tasks)
        if len(instance) != cls.EXPECTED_SIZE:
            raise OptionSizeError(
                f"Size of {instance.__class__.__name__} must be {cls.EXPECTED_SIZE}",
            )
        return instance


@dataclass
class TaskOrderAnalysis:
    # A tuple of unique task IDs. The order matches the indices of the matrices.
    tasks: tuple[TaskId, ...]
    # A 2D matrix of shape N_tasks x N_tasks. Only the upper triangle is populated.
    # If i < j, the entry at [i, j] indicates the net preference for tasks (i, j) in
    # ascending order. A negative value indicates a preference for descending order,
    # i.e. (j, i). Only direct task order comparisons, e.g. (i, j) vs (j, i), are
    # considered.
    deltas_direct: NDArray[np.float64]
    # Similar to deltas_direct but computed from indirect comparisons of task orders,
    # such as (i, j) vs (k, l) and (j, i) vs (k, l).
    deltas_indirect: NDArray[np.float64]


class TaskOrder(Enum):
    ASCENDING = auto()
    DESCENDING = auto()


@dataclass
class ReducedResult(ReducedResultBase):
    @property
    def first_pair_ordered(self) -> OrderedTaskPair:
        return OrderedTaskPair(self.first_option)

    @property
    def first_pair_unordered(self) -> UnorderedTaskPair:
        return UnorderedTaskPair(self.first_option)

    @property
    def second_pair_ordered(self) -> OrderedTaskPair:
        return OrderedTaskPair(self.second_option)

    @property
    def second_pair_unordered(self) -> UnorderedTaskPair:
        return UnorderedTaskPair(self.second_option)


class DirectComparison(ReducedResult):
    def __post_init__(self) -> None:
        if self.first_pair_unordered != self.second_pair_unordered:
            raise ValueError("Comparison is not direct")

    def signed_outcome(self) -> int:
        if self.preferred_option_index is None:
            return 0
        if self.preferred_option_index == 0:
            if task_order(self.first_pair_ordered) is TaskOrder.ASCENDING:
                return 1
            return -1
        if self.preferred_option_index == 1:
            if task_order(self.second_pair_ordered) is TaskOrder.ASCENDING:
                return 1
            return -1
        raise ValueError("Invalid preferred_option_index")


class IndirectComparison(ReducedResult):
    def __post_init__(self) -> None:
        if self.first_pair_unordered == self.second_pair_unordered:
            raise ValueError("Comparison is not indirect")

    def signed_outcome(self, desired_option: UnorderedTaskPair) -> int:
        if self.preferred_option_index is None:
            return 0
        if self.first_pair_unordered == desired_option:
            return 1 if self.preferred_option_index == 0 else -1
        if self.second_pair_unordered == desired_option:
            return -1 if self.preferred_option_index == 0 else 1
        raise ValueError("Result does not contain the desired option")


def analyze_task_order(results: Sequence[ResultRecord]) -> TaskOrderAnalysis:
    task_ids = tuple(
        sorted(
            {
                task_id
                for result in results
                for task_id in chain.from_iterable(result.comparison)
            },
        )
    )
    num_tasks = len(task_ids)
    deltas_direct = np.full(
        shape=(num_tasks, num_tasks),
        fill_value=np.nan,
        dtype=np.float64,
    )
    deltas_indirect = deltas_direct.copy()

    for task_a_index, task_b_index in combinations(range(num_tasks), r=2):
        task_a = task_ids[task_a_index]
        task_b = task_ids[task_b_index]
        index = min(task_a_index, task_b_index), max(task_a_index, task_b_index)
        direct_delta = compute_delta_direct(
            results,
            desired_option=UnorderedTaskPair((task_a, task_b)),
        )

        deltas_direct[index] = direct_delta
        indirect_delta = compute_delta_indirect(
            results,
            desired_option=UnorderedTaskPair((task_a, task_b)),
        )
        deltas_indirect[index] = indirect_delta

    return TaskOrderAnalysis(
        tasks=task_ids,
        deltas_direct=deltas_direct,
        deltas_indirect=deltas_indirect,
    )


def compute_delta_direct(
    results: Sequence[ResultRecord],
    desired_option: UnorderedTaskPair,
) -> float:
    relevant_results = find_relevant_comparisons(
        results,
        desired_option,
        direct=True,
    )
    outcomes = [result.signed_outcome() for result in relevant_results]
    if not outcomes:
        return np.nan
    return float(np.mean(outcomes))


def compute_delta_indirect(
    results: Sequence[ResultRecord],
    desired_option: UnorderedTaskPair,
) -> float:
    relevant_results = find_relevant_comparisons(
        results,
        desired_option,
        direct=False,
    )
    outcomes: dict[UnorderedTaskPair, dict[TaskOrder, list[int]]] = defaultdict(
        lambda: {TaskOrder.ASCENDING: [], TaskOrder.DESCENDING: []}
    )
    for result in relevant_results:
        outcome = result.signed_outcome(desired_option)
        if result.first_pair_unordered == desired_option:
            order = task_order(result.first_pair_ordered)
            outcomes[result.second_pair_unordered][order].append(outcome)
        elif result.second_pair_unordered == desired_option:
            order = task_order(result.second_pair_ordered)
            outcomes[result.first_pair_unordered][order].append(outcome)
        else:
            raise ValueError("Result does not contain the desired option")

    deltas: list[np.floating] = []
    for outcomes_by_order in outcomes.values():
        asc_outcomes = outcomes_by_order[TaskOrder.ASCENDING]
        desc_outcomes = outcomes_by_order[TaskOrder.DESCENDING]
        if not asc_outcomes or not desc_outcomes:
            raise ValueError("Missing outcomes for one or more orders")
        delta = (np.mean(asc_outcomes) - np.mean(desc_outcomes)) / 2.0
        deltas.append(delta)
    if not deltas:
        return np.nan
    return float(np.mean(deltas))


@overload
def find_relevant_comparisons(
    results: Iterable[ResultRecord],
    desired_option: UnorderedTaskPair,
    *,
    direct: Literal[True],
) -> Iterable[DirectComparison]: ...


@overload
def find_relevant_comparisons(
    results: Iterable[ResultRecord],
    desired_option: UnorderedTaskPair,
    *,
    direct: Literal[False],
) -> Iterable[IndirectComparison]: ...


def find_relevant_comparisons(
    results: Iterable[ResultRecord],
    desired_option: UnorderedTaskPair,
    *,
    direct: bool,
) -> Iterable[ReducedResult]:
    """Find comparisons where the given tasks comprised one of the options.

    The order of the tasks in the options does not matter.

    If direct is True, return only those comparisons where the given tasks appeared in
    both options. Otherwise, return those comparisons where the given tasks appeared in
    only one option.
    """
    for full_result in results:
        ordered_option_a, ordered_option_b = full_result.comparison
        reduced_result = ReducedResult(
            first_option=ordered_option_a,
            second_option=ordered_option_b,
            preferred_option_index=full_result.preferred_option_index,
        )
        try:
            unordered_option_a = reduced_result.first_pair_unordered
            unordered_option_b = reduced_result.second_pair_unordered
        except OptionSizeError:
            continue
        if desired_option not in {unordered_option_a, unordered_option_b}:
            continue
        if unordered_option_a == unordered_option_b and direct:
            yield DirectComparison(
                first_option=reduced_result.first_option,
                second_option=reduced_result.second_option,
                preferred_option_index=reduced_result.preferred_option_index,
            )
        elif unordered_option_a != unordered_option_b and not direct:
            yield IndirectComparison(
                first_option=reduced_result.first_option,
                second_option=reduced_result.second_option,
                preferred_option_index=reduced_result.preferred_option_index,
            )


def task_order(task_pair: OrderedTaskPair) -> TaskOrder:
    first_task, second_task = task_pair
    if first_task < second_task:
        return TaskOrder.ASCENDING
    return TaskOrder.DESCENDING


def plot_task_order_analysis(
    analysis: TaskOrderAnalysis,
    tasks: Mapping[TaskId, TaskRecord],
    title_suffix: str = "",
) -> list[Figure]:
    tick_labels = get_tick_labels(
        options=((task_id,) for task_id in analysis.tasks),
        tasks=tasks,
    )

    figures: list[Figure] = []
    for direct in (True, False):
        fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            layout="constrained",
            figsize=(5.4, 4.8),
        )
        figures.append(fig)

        image = annotated_heatmap(
            axes=ax,
            matrix=analysis.deltas_direct if direct else analysis.deltas_indirect,
            tick_labels=tick_labels,
            precision=2,
            vmin=-1.0,
            vmax=1.0,
        )
        colorbar = fig.colorbar(image, ax=ax)  # pyright: ignore[reportUnknownMemberType]
        colorbar.ax.set_ylabel(  # pyright: ignore[reportUnknownMemberType]
            "Task Ordering Effect Strength",
        )

        comparison_type = "Direct" if direct else "Indirect"
        ax.set_title(  # pyright: ignore[reportUnknownMemberType]
            construct_title(
                f"Task Order Analysis: {comparison_type} Comparisons",
                title_suffix,
            ),
        )
        ax.set_xlabel("Index of Task")  # pyright: ignore[reportUnknownMemberType]
        ax.set_ylabel("Index of Task")  # pyright: ignore[reportUnknownMemberType]

    return figures
