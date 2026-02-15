from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from llmprefs.analysis.structs import ReducedResultBase
from llmprefs.analysis.visualization import annotated_heatmap, get_tick_labels
from llmprefs.task_structs import OptionById, ResultRecord, TaskId, TaskRecord


@dataclass
class OptionOrderAnalysis:
    # A tuple of unique options. The order matches the indices of the matrices.
    options: tuple[OptionById, ...]
    # A 2D matrix of shape N_opts x N_opts. Only the upper triangle is populated.
    # If i < j, the entry at [i, j] indicates the net preference for the first presented
    # option in comparisons of option i vs. option j.
    deltas: NDArray[np.float64]


class OptionOrder(Enum):
    ASCENDING = auto()
    DESCENDING = auto()


@dataclass
class ReducedResult(ReducedResultBase):
    @property
    def smaller_option(self) -> OptionById:
        return min(self.first_option, self.second_option)

    @property
    def larger_option(self) -> OptionById:
        return max(self.first_option, self.second_option)

    @property
    def signed_outcome(self) -> int:
        if self.preferred_option_index is None:
            return 0
        if self.preferred_option_index == 0:
            return 1
        if self.preferred_option_index == 1:
            return -1
        raise ValueError("Invalid preferred option index")


def analyze_option_order(results: Sequence[ResultRecord]) -> OptionOrderAnalysis:
    """Analyze whether the model is sensitive to the order of the options."""
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
    # net_wins and totals are both triangular matrices. Only elements where
    # i < j will be populated.
    net_wins = np.zeros(shape=(n_options, n_options), dtype=np.float64)
    totals = net_wins.copy()
    for result in reduced_results:
        tensor_index = (
            option_to_index[result.smaller_option],
            option_to_index[result.larger_option],
        )
        net_wins[tensor_index] += result.signed_outcome
        totals[tensor_index] += 1
    # We expect totals to contain zeros. We ignore the division by zero warnings and
    # let the associated NaNs propagate.
    with np.errstate(divide="ignore", invalid="ignore"):
        deltas = np.divide(net_wins, totals)
    return OptionOrderAnalysis(options=sorted_options, deltas=deltas)


def plot_option_order_analysis(
    analysis: OptionOrderAnalysis,
    tasks: Mapping[TaskId, TaskRecord],
) -> Figure:
    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=1,
        squeeze=True,
    )
    tick_labels = get_tick_labels(analysis.options, tasks)
    image = annotated_heatmap(ax, analysis.deltas, tick_labels, vmin=-1.0, vmax=1.0)
    colorbar = fig.colorbar(image, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    colorbar.ax.set_ylabel(  # pyright: ignore[reportUnknownMemberType]
        "Option Ordering Effect Strength",
    )
    ax.set_title("Option Order Analysis")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Index of Option")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Index of Option")  # pyright: ignore[reportUnknownMemberType]
    return fig
