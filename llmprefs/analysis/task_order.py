from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Self

import numpy as np

from llmprefs.analysis.structs import ReducedResultBase
from llmprefs.task_structs import OptionById, ResultRecord, TaskId

OrderedOption = OptionById


class UnorderedOption(frozenset[TaskId]):
    EXPECTED_SIZE = 2

    def __new__(cls, tasks: Iterable[TaskId]) -> Self:
        instance = super().__new__(cls, tasks)
        if len(instance) != cls.EXPECTED_SIZE:
            raise ValueError(
                f"Size of {instance.__class__.__name__} must be {cls.EXPECTED_SIZE}",
            )
        return instance


class TaskOrder(Enum):
    ASCENDING = auto()
    DESCENDING = auto()


@dataclass
class ReducedResult(ReducedResultBase):
    @property
    def unordered_first_option(self) -> UnorderedOption:
        return UnorderedOption(self.first_option)

    @property
    def unordered_second_option(self) -> UnorderedOption:
        return UnorderedOption(self.second_option)

    def signed_outcome(self, desired_pair: UnorderedOption) -> int:
        if self.preferred_option_index is None:
            return 0
        if self.unordered_first_option == desired_pair:
            return 1 if self.preferred_option_index == 0 else -1
        if self.unordered_second_option == desired_pair:
            return -1 if self.preferred_option_index == 0 else 1
        raise ValueError("Result does not contain the desired pair")


def compute_delta(
    results: Sequence[ResultRecord],
    desired_pair: UnorderedOption,
) -> float:
    relevant_results = find_relevant_comparisons(
        results,
        desired_pair,
        direct=False,
    )
    outcomes: dict[UnorderedOption, dict[TaskOrder, list[int]]] = defaultdict(
        lambda: {TaskOrder.ASCENDING: [], TaskOrder.DESCENDING: []}
    )
    for result in relevant_results:
        outcome = result.signed_outcome(desired_pair)
        if result.unordered_first_option == desired_pair:
            order = task_order(result.first_option)
            outcomes[result.unordered_second_option][order].append(outcome)
        elif result.unordered_second_option == desired_pair:
            order = task_order(result.second_option)
            outcomes[result.unordered_first_option][order].append(outcome)
        else:
            raise ValueError("Result does not contain the desired pair")

    deltas: list[np.floating] = []
    for outcomes_by_order in outcomes.values():
        asc_outcomes = outcomes_by_order[TaskOrder.ASCENDING]
        desc_outcomes = outcomes_by_order[TaskOrder.DESCENDING]
        if not asc_outcomes or not desc_outcomes:
            raise ValueError("Missing outcomes for one or more orders")
        delta = (np.mean(asc_outcomes) - np.mean(desc_outcomes)) / 2.0
        deltas.append(delta)
    return float(np.mean(deltas))


def find_relevant_comparisons(
    results: Iterable[ResultRecord],
    desired_pair: UnorderedOption,
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
        if desired_pair not in {unordered_option_a, unordered_option_b}:
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
