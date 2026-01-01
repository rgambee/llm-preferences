from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from llmprefs.analysis.rating import RatedOptions, ValueCI


def error_bars(values: Sequence[ValueCI]) -> tuple[list[float], list[float]]:
    """Return error bars compatible with Matplotlib."""
    diff_low = [vci.value - vci.ci_lower for vci in values]
    diff_high = [vci.ci_upper - vci.value for vci in values]
    return diff_low, diff_high


def plot_ratings_stem(rated_options: RatedOptions) -> Figure:
    xcoords = np.arange(len(rated_options))
    rating_values = [vci.value for vci in rated_options.values()]

    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=1,
        squeeze=True,
    )

    ax.stem(
        xcoords,
        rating_values,
    )
    ax.errorbar(  # pyright: ignore[reportUnknownMemberType]
        x=xcoords,
        y=rating_values,
        yerr=error_bars(list(rated_options.values())),
        marker="None",
        linestyle="None",
        ecolor="black",
        capsize=5.0,
        label="Bootstrapped CI",
    )

    return fig


def plot_ratings_heatmap(rated_options: RatedOptions) -> Figure:
    # Parts of this function are adapted from
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    expected_num_tasks = 2
    if any(len(option) != expected_num_tasks for option in rated_options):
        raise ValueError(
            f"Heatmap only accepts options containing {expected_num_tasks} tasks"
        )

    task_0_ids, task_1_ids = zip(*rated_options.keys(), strict=True)
    task_0_ids = sorted(set(task_0_ids))
    task_1_ids = sorted(set(task_1_ids))
    task_0_ids_to_index = {task_id: i for i, task_id in enumerate(task_0_ids)}
    task_1_ids_to_index = {task_id: i for i, task_id in enumerate(task_1_ids)}
    ratings = np.full((len(task_0_ids), len(task_1_ids)), np.nan, dtype=float)
    for (task0, task1), rating in rated_options.items():
        ratings[task_0_ids_to_index[task0], task_1_ids_to_index[task1]] = rating.value

    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=1,
        squeeze=True,
        figsize=(10, 10),
    )
    im = annotated_heatmap(ax, ratings)

    ax.set_ylabel("First Task ID")  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(  # pyright: ignore[reportUnknownMemberType]
        range(len(task_0_ids)), labels=task_0_ids
    )
    ax.set_xlabel("Second Task ID")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xticks(  # pyright: ignore[reportUnknownMemberType]
        range(len(task_1_ids)), labels=task_1_ids
    )

    colorbar = fig.colorbar(im, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    colorbar.ax.set_ylabel("Rating")  # pyright: ignore[reportUnknownMemberType]

    return fig


def annotated_heatmap(
    axes: Axes,
    matrix: NDArray[np.float64],
    precision: int = 3,
) -> AxesImage:
    expected_dimensionality = 2
    if matrix.ndim != expected_dimensionality:
        raise ValueError("Matrix has wrong number of dimensions")

    image = axes.imshow(  # pyright: ignore[reportUnknownMemberType]
        matrix,
        cmap="viridis",
    )

    threshold = image.norm(np.nanmax(matrix)) / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isnan(value):
                continue
            color = "white" if image.norm(value) < threshold else "black"
            axes.text(  # pyright: ignore[reportUnknownMemberType]
                x=j,
                y=i,
                s=f"{value:.{precision}f}",
                ha="center",
                va="center",
                color=color,
                size=6,
            )
    return image
