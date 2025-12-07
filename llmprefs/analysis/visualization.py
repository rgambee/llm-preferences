from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from llmprefs.analysis.rating import RatedOptions


def plot_ratings_stem(rated_options: RatedOptions) -> Figure:
    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=1,
        squeeze=True,
    )

    ax.stem(
        [str(key) for key in rated_options],
        list(rated_options.values()),
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

    task_0_ids, task_1_ids = zip(*rated_options.keys())
    task_0_ids = sorted(set(task_0_ids))
    task_1_ids = sorted(set(task_1_ids))
    task_0_ids_to_index = {task_id: i for i, task_id in enumerate(task_0_ids)}
    task_1_ids_to_index = {task_id: i for i, task_id in enumerate(task_1_ids)}
    ratings = np.full((len(task_0_ids), len(task_1_ids)), np.nan, dtype=float)
    for (task0, task1), rating in rated_options.items():
        ratings[task_0_ids_to_index[task0], task_1_ids_to_index[task1]] = rating

    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        nrows=1,
        ncols=1,
        squeeze=True,
        figsize=(10, 10),
    )
    im = ax.imshow(  # pyright: ignore[reportUnknownMemberType]
        ratings,
        cmap="viridis",
    )

    threshold = cast(float, im.norm(np.nanmax(ratings))) / 2.0
    for task0 in task_0_ids:
        for task1 in task_1_ids:
            if task0 == task1:
                continue
            index0 = task_0_ids_to_index[task0]
            index1 = task_1_ids_to_index[task1]
            rating = ratings[index0, index1]
            color = "white" if im.norm(rating) < threshold else "black"
            ax.text(  # pyright: ignore[reportUnknownMemberType]
                x=index1,
                y=index0,
                s=f"{rating:.3f}",
                ha="center",
                va="center",
                color=color,
                size=6,
            )

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
