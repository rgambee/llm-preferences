from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from itertools import chain, combinations
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from llmprefs.analysis.structs import ReducedResultBase
from llmprefs.analysis.visualization import annotated_heatmap, get_tick_labels
from llmprefs.task_structs import OptionById, ResultRecord, TaskId, TaskRecord

OrderedOption = OptionById
UnorderedOption = frozenset[TaskId]


class OptionSizeError(Exception):
    pass


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
    # A tuple of unique task IDs. The order matches the indices of the matrix.
    tasks: tuple[TaskId, ...]
    # A 2D matrix of shape N_tasks x N_tasks. Only the upper triangle is populated.
    # If i < j, the entry at [i, j] indicates the net preference for tasks (i, j) in
    # ascending order. A negative value indicates a preference for descending order,
    # i.e. (j, i).
    deltas_indirect: NDArray[np.float64]


class TaskOrder(Enum):
    ASCENDING = auto()
    DESCENDING = auto()


@dataclass
class ReducedResult(ReducedResultBase):
    @property
    def unordered_first_option(self) -> UnorderedOption:
        return UnorderedOption(self.first_option)

    @property
    def unordered_first_pair(self) -> UnorderedTaskPair:
        return UnorderedTaskPair(self.first_option)

    @property
    def unordered_second_option(self) -> UnorderedOption:
        return UnorderedOption(self.second_option)

    @property
    def unordered_second_pair(self) -> UnorderedTaskPair:
        return UnorderedTaskPair(self.second_option)

    def signed_outcome(self, desired_option: UnorderedOption) -> int:
        if self.preferred_option_index is None:
            return 0
        if self.unordered_first_option == desired_option:
            return 1 if self.preferred_option_index == 0 else -1
        if self.unordered_second_option == desired_option:
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
    deltas_indirect = np.full(
        shape=(num_tasks, num_tasks),
        fill_value=np.nan,
        dtype=np.float64,
    )
    for task_a_index, task_b_index in combinations(range(num_tasks), r=2):
        task_a = task_ids[task_a_index]
        task_b = task_ids[task_b_index]
        index = min(task_a_index, task_b_index), max(task_a_index, task_b_index)
        indirect_delta = compute_delta_indirect(
            results,
            desired_option=UnorderedTaskPair((task_a, task_b)),
        )
        deltas_indirect[index] = indirect_delta

    return TaskOrderAnalysis(
        tasks=task_ids,
        deltas_indirect=deltas_indirect,
    )


def compute_delta_indirect(
    results: Sequence[ResultRecord],
    desired_option: UnorderedOption,
) -> float:
    relevant_results = find_relevant_comparisons(
        results,
        desired_option,
        direct=False,
    )
    outcomes: dict[UnorderedOption, dict[TaskOrder, list[int]]] = defaultdict(
        lambda: {TaskOrder.ASCENDING: [], TaskOrder.DESCENDING: []}
    )
    for result in relevant_results:
        outcome = result.signed_outcome(desired_option)
        if result.unordered_first_option == desired_option:
            order = task_order(result.first_option)
            outcomes[result.unordered_second_option][order].append(outcome)
        elif result.unordered_second_option == desired_option:
            order = task_order(result.second_option)
            outcomes[result.unordered_first_option][order].append(outcome)
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


def find_relevant_comparisons(
    results: Iterable[ResultRecord],
    desired_option: UnorderedOption,
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
        unordered_option_a = reduced_result.unordered_first_option
        unordered_option_b = reduced_result.unordered_second_option
        if desired_option not in {unordered_option_a, unordered_option_b}:
            continue
        if (unordered_option_a == unordered_option_b and direct) or (
            unordered_option_a != unordered_option_b and not direct
        ):
            yield reduced_result


def task_order(option: OrderedOption) -> TaskOrder:
    first_task, second_task = option
    if first_task < second_task:
        return TaskOrder.ASCENDING
    return TaskOrder.DESCENDING


def plot_task_order_analysis(
    analysis: TaskOrderAnalysis,
    tasks: Mapping[TaskId, TaskRecord],
) -> Figure:
    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=1,
        squeeze=True,
    )
    tick_labels = get_tick_labels(
        options=((task_id,) for task_id in analysis.tasks),
        tasks=tasks,
    )
    image = annotated_heatmap(
        axes=ax,
        matrix=analysis.deltas_indirect,
        tick_labels=tick_labels,
        vmin=-1.0,
        vmax=1.0,
    )
    colorbar = fig.colorbar(image, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    colorbar.ax.set_ylabel(  # pyright: ignore[reportUnknownMemberType]
        "Task Ordering Effect Strength",
    )
    ax.set_title("Task Order Analysis")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Index of Task")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Index of Task")  # pyright: ignore[reportUnknownMemberType]
    return fig
