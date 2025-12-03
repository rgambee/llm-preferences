from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import choix
import numpy as np
from numpy.typing import NDArray

from llmprefs.task_structs import OptionById, ResultRecord, TaskId


@dataclass
class OptionMatrix:
    # A tuple of unique options. The order matches the indices of the matrix.
    options: tuple[OptionById, ...]
    # A matrix of size N_opts x N_opts. The entry at [i, j] is the number of times
    # option i was preferred over option j.
    matrix: NDArray[np.float64]


RatedOptions = dict[OptionById, float]


def rate_options(
    results: Iterable[ResultRecord],
    alpha: float = 1e-6,
) -> RatedOptions:
    option_matrix = compile_matrix(results)

    if len(option_matrix.options) == 0:
        return {}

    ratings = choix.ilsr_pairwise_dense(comp_mat=option_matrix.matrix, alpha=alpha)
    rated_options: RatedOptions = {}
    for i, option in enumerate(option_matrix.options):
        rated_options[option] = ratings[i]
    return rated_options


def compile_matrix(results: Iterable[ResultRecord]) -> OptionMatrix:
    counts: dict[OptionById, dict[OptionById, int]] = defaultdict(
        lambda: defaultdict(int),
    )
    unique_options: set[OptionById] = set()
    for result in results:
        preferred_option = result.comparison[result.preferred_option_index]
        for i, option in enumerate(result.comparison):
            unique_options.add(option)
            if i != result.preferred_option_index:
                counts[preferred_option][option] += 1

    if len(unique_options) == 0:
        return OptionMatrix(options=(), matrix=np.array([]))

    sorted_options = tuple(sorted(unique_options))
    option_to_index: dict[tuple[TaskId, ...], int] = {
        option: i for i, option in enumerate(sorted_options)
    }

    n_options = len(unique_options)
    matrix: NDArray[np.float64] = np.zeros((n_options, n_options), dtype=np.float64)
    for preferred_option, beaten_options in counts.items():
        preferred_idx = option_to_index[preferred_option]
        for beaten_option, count in beaten_options.items():
            beaten_idx = option_to_index[beaten_option]
            matrix[preferred_idx, beaten_idx] = count
    return OptionMatrix(options=sorted_options, matrix=matrix)
