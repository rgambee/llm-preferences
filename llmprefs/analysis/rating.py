from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import choix
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from llmprefs.analysis.structs import ValueCI
from llmprefs.analysis.visualization import (
    annotated_heatmap,
    error_bars,
    get_tick_labels,
)
from llmprefs.comparisons import is_opt_out_task
from llmprefs.task_structs import OptionById, Outcome, ResultRecord, TaskId, TaskRecord


@dataclass
class ComparisonOutcomes:
    # A tuple of unique options. The order matches the indices of the matrix.
    options: tuple[OptionById, ...]
    # A 3D matrix of shape N_opts x N_opts x 3. It is triangular over the first two
    # dimensions. If i >= j, then [i, j, :] is zero. If i < j, then the entry at
    # [i, j, o] is the count of comparisons between options i and j with outcome o.
    # The three outcomes are:
    # - 0: option i beat j
    # - 1: option j beat i
    # - 2: neither option was preferred
    counts: NDArray[np.int64]

    def unfold(self) -> NDArray[np.int64]:
        """Unfold the triangular 3D matrix and slice it into a 2D matrix.

        In the output, the entry at [i, j] is the number of times that option i was
        preferred over option j.
        """
        return self.counts[:, :, 0] + self.counts[:, :, 1].transpose()


RatedOptions = dict[OptionById, ValueCI]


def rate_options(
    outcomes: ComparisonOutcomes,
    tasks: Mapping[TaskId, TaskRecord],
    num_resamples: int,
    confidence: float,
    alpha: float = 1e-6,
) -> RatedOptions:
    """Return an estimate of each option's strength.

    Also compute a bootstrapped confidence interval for each rating.
    """
    if num_resamples < 0:
        raise ValueError("Number of resamples must be non-negative")
    if outcomes.counts.size == 0:
        return {}

    # 2D matrix of shape (max(num_resamples, 1), N_opts)
    ratings: NDArray[np.float64]
    if num_resamples == 0:
        ratings_1d = choix.ilsr_pairwise_dense(
            comp_mat=outcomes.unfold().astype(np.float64),
            alpha=alpha,
        )
        ratings = np.expand_dims(ratings_1d, axis=0)
    else:
        generator = np.random.default_rng()
        ratings = np.full((num_resamples, len(outcomes.options)), np.nan)
        for i in range(num_resamples):
            resample = resample_results(outcomes=outcomes, generator=generator)
            sample_ratings = choix.ilsr_pairwise_dense(
                comp_mat=resample.unfold().astype(np.float64),
                alpha=alpha,
            )
            ratings[i, :] = sample_ratings

    medians = np.median(ratings, axis=0)
    lower_quant = (1 - confidence) / 2
    upper_quant = (1 + confidence) / 2
    lower_bounds = np.quantile(ratings, q=lower_quant, axis=0)
    upper_bounds = np.quantile(ratings, q=upper_quant, axis=0)
    # Apply an offset such that the opt-out tasks have a rating of 0.
    offset = median_opt_out_rating(ratings, outcomes.options, tasks)
    rated_options: RatedOptions = {}
    for option_index, option_id in enumerate(outcomes.options):
        rated_options[option_id] = ValueCI(
            value=medians[option_index] - offset,
            ci_lower=lower_bounds[option_index] - offset,
            ci_upper=upper_bounds[option_index] - offset,
        )
    return rated_options


def median_opt_out_rating(
    resampled_ratings: NDArray[np.float64],
    options: Sequence[OptionById],
    tasks: Mapping[TaskId, TaskRecord],
) -> float:
    opt_outs = [
        all(is_opt_out_task(tasks[task_id]) for task_id in option) for option in options
    ]
    if not any(opt_outs):
        return 0.0
    return np.median(resampled_ratings[:, opt_outs])


def compile_matrix(results: Iterable[ResultRecord]) -> ComparisonOutcomes:
    """Compile comparison results into a triangular 3D matrix."""
    counts: dict[OptionById, dict[OptionById, dict[Outcome, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0)),
    )
    unique_options: set[OptionById] = set()
    for result in results:
        option_0, option_1 = result.comparison
        smaller_option = min(option_0, option_1)
        larger_option = max(option_0, option_1)
        preferred_index = result.preferred_option_index
        if preferred_index is not None and option_0 == larger_option:
            preferred_index = 1 - preferred_index
        counts[smaller_option][larger_option][preferred_index] += 1
        unique_options.update(result.comparison)

    if len(unique_options) == 0:
        return ComparisonOutcomes(
            options=(),
            counts=np.zeros((0, 0, 3), dtype=np.int64),
        )

    sorted_options = tuple(sorted(unique_options))
    option_to_index: dict[OptionById, int] = {
        option: i for i, option in enumerate(sorted_options)
    }

    n_options = len(unique_options)
    matrix: NDArray[np.int64] = np.zeros((n_options, n_options, 3), dtype=np.int64)
    for option_0, counts_for_opt_0 in counts.items():
        for option_1, counts_for_opts_01 in counts_for_opt_0.items():
            for outcome, count in counts_for_opts_01.items():
                if outcome is None:
                    outcome = 2  # noqa: PLW2901
                i = option_to_index[option_0]
                j = option_to_index[option_1]
                matrix[i, j, outcome] = count

    return ComparisonOutcomes(options=sorted_options, counts=matrix)


def resample_results(
    outcomes: ComparisonOutcomes,
    generator: np.random.Generator,
) -> ComparisonOutcomes:
    if outcomes.counts.size == 0:
        return outcomes
    sums = outcomes.counts.sum(axis=2, keepdims=True)
    probabilities = np.zeros_like(outcomes.counts, dtype=np.float64)
    np.divide(
        outcomes.counts,
        sums,
        out=probabilities,
        where=sums > 0,
    )
    resample = generator.multinomial(
        n=sums.squeeze(axis=2),
        pvals=probabilities,
    )
    return ComparisonOutcomes(options=outcomes.options, counts=resample)


def plot_ratings_stem(
    rated_options: RatedOptions,
    tasks: Mapping[TaskId, TaskRecord],
) -> Figure:
    xcoords = np.arange(len(rated_options))
    rating_values = [vci.value for vci in rated_options.values()]

    fig, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]
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
    ax.set_title("Rated Options")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Index of Option")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xticks(  # pyright: ignore[reportUnknownMemberType]
        ticks=xcoords,
        labels=get_tick_labels(rated_options.keys(), tasks),
        rotation="vertical",
        fontsize="x-small",
    )
    ax.set_ylabel("Rating")  # pyright: ignore[reportUnknownMemberType]

    return fig


def plot_ratings_heatmap(
    rated_options: RatedOptions,
    tasks: Mapping[TaskId, TaskRecord],
) -> Figure:
    expected_num_tasks = 2
    if any(len(option) != expected_num_tasks for option in rated_options):
        raise ValueError(
            f"Heatmap only accepts options containing {expected_num_tasks} tasks"
        )

    task_0_and_1_ids: zip[tuple[TaskId, TaskId]] = zip(
        *rated_options.keys(),
        strict=True,
    )
    task_0_ids, task_1_ids = task_0_and_1_ids
    task_0_ids: Sequence[TaskId] = sorted(set(task_0_ids))
    task_1_ids: Sequence[TaskId] = sorted(set(task_1_ids))
    if not task_0_ids == task_1_ids:
        raise ValueError("Task IDs are not the same")
    del task_1_ids
    task_ids_to_index = {task_id: i for i, task_id in enumerate(task_0_ids)}
    ratings = np.full((len(task_0_ids), len(task_0_ids)), np.nan, dtype=float)
    for (task0, task1), rating in rated_options.items():
        ratings[task_ids_to_index[task0], task_ids_to_index[task1]] = rating.value

    fig, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]
    tick_labels = get_tick_labels(
        options=((task_id,) for task_id in task_0_ids),
        tasks=tasks,
    )
    im = annotated_heatmap(ax, ratings, tick_labels)

    ax.set_title("Rated Options")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("First Task ID")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Second Task ID")  # pyright: ignore[reportUnknownMemberType]

    colorbar = fig.colorbar(im, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    colorbar.ax.set_ylabel("Rating")  # pyright: ignore[reportUnknownMemberType]

    return fig


def plot_rating_additivity_scatter(
    rated_options_1tpo: RatedOptions,
    rated_options_2tpo: RatedOptions,
) -> Figure:
    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        subplot_kw={"aspect": "equal"}
    )

    x_values = [
        sum(rated_options_1tpo[(task_id,)].value for task_id in option)
        for option in rated_options_2tpo
    ]
    y_values = [rated_options_2tpo[option].value for option in rated_options_2tpo]

    # Add error bars
    ax.plot(  # pyright: ignore[reportUnknownMemberType]
        x_values,
        y_values,
        marker="o",
        linestyle="None",
        alpha=0.5,
        markeredgewidth=0,
        label="Data points",
    )

    ax.legend()  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Sum of Task Ratings")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Rating of Task Sequence")  # pyright: ignore[reportUnknownMemberType]
    ax.set_title("Task Rating Additivity")  # pyright: ignore[reportUnknownMemberType]
    return fig
