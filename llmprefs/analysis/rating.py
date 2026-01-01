import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import choix
import numpy as np
from numpy.typing import NDArray
from scipy import optimize

from llmprefs.task_structs import OptionById, ResultRecord, TaskId


@dataclass
class OptionMatrix:
    # A tuple of unique options. The order matches the indices of the matrix.
    options: tuple[OptionById, ...]
    # A matrix of size N_opts x N_opts. The entry at [i, j] is the number of times
    # option i was preferred over option j.
    matrix: NDArray[np.float64]


@dataclass
class ValueCI:
    value: float
    ci_lower: float
    ci_upper: float


RatedOptions = dict[OptionById, ValueCI]


def rate_options(
    option_matrix: OptionMatrix,
    num_resamples: int,
    confidence: float,
    generator: np.random.Generator | None = None,
    alpha: float = 1e-6,
) -> RatedOptions:
    """Return an estimate of each option's strength.

    Also compute a bootstrapped confidence interval for each rating.
    """
    if option_matrix.matrix.size == 0:
        return {}

    generator = generator or np.random.default_rng()
    resampled_ratings = np.full((num_resamples, len(option_matrix.options)), np.nan)
    for i in range(num_resamples):
        resample = resample_results(option_matrix=option_matrix, generator=generator)
        ratings = choix.ilsr_pairwise_dense(comp_mat=resample.matrix, alpha=alpha)
        resampled_ratings[i, :] = ratings

    medians = np.median(resampled_ratings, axis=0)  # mean?
    lower_quant = (1 - confidence) / 2
    upper_quant = (1 + confidence) / 2
    lower_bounds = np.quantile(resampled_ratings, q=lower_quant, axis=0)
    upper_bounds = np.quantile(resampled_ratings, q=upper_quant, axis=0)
    rated_options: RatedOptions = {}
    for option_index, option_id in enumerate(option_matrix.options):
        rated_options[option_id] = ValueCI(
            value=medians[option_index],
            ci_lower=lower_bounds[option_index],
            ci_upper=upper_bounds[option_index],
        )
    return rated_options


def compile_matrix(results: Iterable[ResultRecord]) -> OptionMatrix:
    """Compile comparison results into a square matrix.

    The matrix has size N_options x N_options. The entry at [i, j] is the number of
    times option i was preferred over option j.
    """
    counts: dict[OptionById, dict[OptionById, int]] = defaultdict(
        lambda: defaultdict(int),
    )
    unique_options: set[OptionById] = set()
    for result in results:
        if result.preferred_option_index is None:
            continue
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


def resample_results(
    option_matrix: OptionMatrix,
    generator: np.random.Generator,
) -> OptionMatrix:
    num_comparisons = int(np.round(cast(float, option_matrix.matrix.sum())))
    random_weights = generator.random(option_matrix.matrix.shape)
    resample = option_matrix.matrix * random_weights

    # Scale resample such that after rounding, its sum equals num_comparisons.
    # To do this, we solve the following equation for the scalar variable x:
    #   np.round(x * resample).sum() == num_comparisons.
    def error(x: float) -> int:
        return np.round(x * resample).sum() - num_comparisons

    x0 = num_comparisons / resample.sum()
    x = optimize.brentq(f=error, a=0.5 * x0, b=2.0 * x0)
    resample_sum = np.round(x * resample).sum()
    if resample_sum != num_comparisons:
        logging.getLogger().error(
            "Resample scaling failed to converge: "
            + f"{x=}, {resample_sum=}, {num_comparisons=}"
        )
        raise RuntimeError("Resample scaling failed to converge")
    return OptionMatrix(options=option_matrix.options, matrix=resample)
