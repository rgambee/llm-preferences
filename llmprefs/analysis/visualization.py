from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from llmprefs.analysis.structs import ValueCI
from llmprefs.task_structs import OptionById, TaskId, TaskRecord


def error_bars(values: Sequence[ValueCI]) -> tuple[list[float], list[float]]:
    """Return error bars compatible with Matplotlib."""
    diff_low = [vci.value - vci.ci_lower for vci in values]
    diff_high = [vci.ci_upper - vci.value for vci in values]
    return diff_low, diff_high


def weights_from_ci(values: Iterable[ValueCI]) -> NDArray[np.float64]:
    """Compute weights inversely proportional to the confidence interval widths.

    If the width is zero, use a weight of 1.0.
    """
    ci_widths = np.array([val.ci_upper - val.ci_lower for val in values])
    weights = np.ones_like(ci_widths)
    np.divide(1.0, ci_widths, out=weights, where=ci_widths > 0)
    return weights


def get_tick_labels(
    options: Iterable[OptionById],
    tasks: Mapping[TaskId, TaskRecord],
    tick_label_length: int = 15,
) -> list[str]:
    labels: list[str] = []
    for option in options:
        tasks_in_option = [tasks[task_id] for task_id in option]
        task_labels = [
            f"{task.id}:{task.task}"[:tick_label_length] for task in tasks_in_option
        ]
        labels.append("\n".join(task_labels))
    return labels


def annotated_heatmap(
    axes: Axes,
    matrix: NDArray[np.float64],
    tick_labels: Sequence[str],
    precision: int = 3,
    **imshow_kwargs: Any,
) -> AxesImage:
    # Parts of this function are adapted from
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    expected_dimensionality = 2
    if matrix.ndim != expected_dimensionality:
        raise ValueError("Matrix has wrong number of dimensions")
    if len(tick_labels) != matrix.shape[0] or len(tick_labels) != matrix.shape[1]:
        raise ValueError("Length of tick labels must match matrix dimensions")

    image = axes.imshow(  # pyright: ignore[reportUnknownMemberType]
        matrix,
        cmap="viridis",
        **imshow_kwargs,
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

    axes.set_xticks(  # pyright: ignore[reportUnknownMemberType]
        ticks=range(len(tick_labels)),
        labels=tick_labels,
        fontsize="x-small",
        rotation="vertical",
    )
    axes.set_yticks(  # pyright: ignore[reportUnknownMemberType]
        ticks=range(len(tick_labels)),
        labels=tick_labels,
        fontsize="xx-small",
    )
    return image


def construct_title(title: str, title_suffix: str) -> str:
    if title_suffix:
        title = f"{title}\n{title_suffix}"
    return title
