from collections.abc import Iterable
from dataclasses import dataclass
from math import factorial
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from llmprefs.analysis.visualization import annotated_heatmap
from llmprefs.task_structs import OptionById, ResultRecord

NUM_OPTIONS_PER_COMPARISON = 2
NUM_OPTION_ORDERINGS = factorial(NUM_OPTIONS_PER_COMPARISON)
# The number of possible outcomes is one more than the number of options per
# comparison because it's possible neither option was chosen.
NUM_POSSIBLE_OUTCOMES = NUM_OPTIONS_PER_COMPARISON + 1


@dataclass
class Observations:
    # A tuple of unique options. The order matches the indices of the matrix.
    options: tuple[OptionById, ...]
    # A 4D tensor of shape N_opts x N_opts x N_orderings x N_outcomes. It is triangular
    # over the first two dimensions. Only elements where i < j are populated.
    matrix: NDArray[np.float64]


@dataclass
class OptionOrderAnalysis:
    # A tuple of unique options. The order matches the indices of the matrix.
    options: tuple[OptionById, ...]
    # A 2D matrix of shape N_opts x N_opts. Only the upper triangle is populated.
    # If i < j, the entry at [i, j] is Cramér's V, which indicates the strength of the
    # ordering effect for options i and j.
    cramer_v: NDArray[np.float64]


@dataclass
class ReducedResult:
    first_option: OptionById
    second_option: OptionById
    preferred_option_index: int | None

    @property
    def smaller_option(self) -> OptionById:
        return min(self.first_option, self.second_option)

    @property
    def larger_option(self) -> OptionById:
        return max(self.first_option, self.second_option)

    @property
    def option_order_index(self) -> Literal[0, 1]:
        if self.first_option < self.second_option:
            return 0
        return 1

    @property
    def outcome_index(self) -> int:
        if self.preferred_option_index is None:
            return NUM_OPTIONS_PER_COMPARISON
        return self.preferred_option_index


def analyze_option_order(results: Iterable[ResultRecord]) -> OptionOrderAnalysis:
    """Analyze whether the model is sensitive to the order of the options."""
    observations = compile_observations(results)
    return analyze_observations(observations)


def analyze_observations(observations: Observations) -> OptionOrderAnalysis:
    ordering_counts = observations.matrix.sum(axis=3)
    outcome_counts = observations.matrix.sum(axis=2)
    total_counts = observations.matrix.sum(axis=(2, 3))
    expected = np.zeros_like(observations.matrix)
    np.divide(
        ordering_counts[..., :, None] * outcome_counts[..., None, :],
        total_counts[..., None, None],
        out=expected,
        where=total_counts[..., None, None] != 0,
    )
    chi_squared_terms = np.zeros_like(observations.matrix)
    np.divide(
        (observations.matrix - expected) ** 2,
        expected,
        out=chi_squared_terms,
        where=expected != 0,
    )
    chi_squared = chi_squared_terms.sum(axis=(2, 3))
    # It's expected that total_counts will contain zeros, especially for the lower
    # triangle. Therefore, this division will produce NaNs. This is desired: NaN
    # indicates that we have no data for that cell.
    with np.errstate(divide="ignore", invalid="ignore"):
        cramer_v = np.sqrt(chi_squared / total_counts)
    return OptionOrderAnalysis(options=observations.options, cramer_v=cramer_v)


def compile_observations(results: Iterable[ResultRecord]) -> Observations:
    reduced_results: list[ReducedResult] = []
    unique_options: set[OptionById] = set()
    for result in results:
        for option in result.comparison:
            unique_options.add(option)
        reduced_results.append(
            ReducedResult(
                first_option=result.comparison[0],
                second_option=result.comparison[1],
                preferred_option_index=result.preferred_option_index,
            ),
        )

    sorted_options = tuple(sorted(unique_options))
    option_to_index: dict[OptionById, int] = {
        option: i for i, option in enumerate(sorted_options)
    }

    n_options = len(unique_options)
    # observations is triangular over the first two dimensions. Only elements where
    # i < j will be populated.
    observations = np.zeros(
        shape=(n_options, n_options, NUM_OPTION_ORDERINGS, NUM_POSSIBLE_OUTCOMES),
        dtype=np.float64,
    )
    for result in reduced_results:
        tensor_index = (
            option_to_index[result.smaller_option],
            option_to_index[result.larger_option],
            result.option_order_index,
            result.outcome_index,
        )
        observations[tensor_index] += 1
    return Observations(options=sorted_options, matrix=observations)


def plot_order_analysis(analysis: OptionOrderAnalysis) -> Figure:
    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=1,
        squeeze=True,
    )
    annotated_heatmap(ax, analysis.cramer_v)
    ax.set_title("Order Analysis")  # pyright: ignore[reportUnknownMemberType]
    return fig
