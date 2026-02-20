from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from llmprefs.analysis.rating import ComparisonOutcomes
from llmprefs.analysis.visualization import annotated_heatmap, get_tick_labels
from llmprefs.task_structs import TaskId, TaskRecord


def plot_comparison_outcomes_heatmap(
    outcomes: ComparisonOutcomes,
    tasks: Mapping[TaskId, TaskRecord],
) -> Figure:
    expected_dimensionality = 2
    if outcomes.counts.ndim != expected_dimensionality:
        raise ValueError("Option has wrong number of dimensions")
    if outcomes.counts.shape[0] != outcomes.counts.shape[1]:
        raise ValueError("Option matrix must be square")

    fig = plt.figure()  # pyright: ignore[reportUnknownMemberType]
    gridspec = fig.add_gridspec(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=2,
        width_ratios=[5, 1],
        wspace=0.05,
    )
    ax_main = fig.add_subplot(gridspec[0, 0])
    tick_labels = get_tick_labels(outcomes.options, tasks)
    annotated_heatmap(ax_main, outcomes.counts, tick_labels, precision=0)
    ax_main.set_title(  # pyright: ignore[reportUnknownMemberType]
        "Comparison Outcome Counts"
    )
    ax_main.set_xlabel(  # pyright: ignore[reportUnknownMemberType]
        "Index of Disfavored Option"
    )
    ax_main.set_ylabel(  # pyright: ignore[reportUnknownMemberType]
        "Index of Favored Option"
    )

    ax_right = fig.add_subplot(gridspec[0, 1])
    bar_height = 0.8
    spacing = 1.0 - bar_height
    ax_right.barh(  # pyright: ignore[reportUnknownMemberType]
        y=np.arange(len(outcomes.counts)),
        width=np.sum(outcomes.counts, axis=1)[::-1],
        height=bar_height,
    )
    ax_right.set_yticks([])  # pyright: ignore[reportUnknownMemberType]
    ax_right.set_ylim(
        -(bar_height + spacing) / 2.0,
        len(outcomes.counts) - 1.0 + (bar_height + spacing) / 2.0,
    )
    ax_right.set_xlabel(  # pyright: ignore[reportUnknownMemberType]
        "Sum of Wins"
    )

    return fig
