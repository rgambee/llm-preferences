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
    # A matrix of shape N_opts x N_opts x 3. The entry at [i, j, o] is the count of
    # comparisons between options i and j with outcome o. The three outcomes are:
    # - 0: option i beat j
    # - 1: option j beat i
    # - 2: neither option was preferred
    counts: NDArray[np.int64]


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
    if outcomes.counts.size == 0:
        return {}

    generator = np.random.default_rng()
    resampled_ratings = np.full((num_resamples, len(outcomes.options)), np.nan)
    for i in range(num_resamples):
        resample = resample_results(outcomes=outcomes, generator=generator)
        ratings = choix.ilsr_pairwise_dense(
            comp_mat=resample.counts.astype(np.float64),
            alpha=alpha,
        )
        resampled_ratings[i, :] = ratings

    medians = np.median(resampled_ratings, axis=0)  # mean?
    lower_quant = (1 - confidence) / 2
    upper_quant = (1 + confidence) / 2
    lower_bounds = np.quantile(resampled_ratings, q=lower_quant, axis=0)
    upper_bounds = np.quantile(resampled_ratings, q=upper_quant, axis=0)
    # Apply an offset such that the opt-out tasks have a rating of 0.
    offset = median_opt_out_rating(resampled_ratings, outcomes.options, tasks)
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
    """Compile comparison results into a square matrix.

    The matrix has shape N_options x N_options x 3. The entry at [i, j, o] is the number
    of comparisons between options i and j with outcome o. The three outcomes are:
    - 0: option i beat j
    - 1: option j beat i
    - 2: neither option was preferred
    """
    counts: dict[OptionById, dict[OptionById, dict[Outcome, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: 0)),
    )
    unique_options: set[OptionById] = set()
    for result in results:
        option_0, option_1 = result.comparison
        counts[option_0][option_1][result.preferred_option_index] += 1
        unique_options.update(result.comparison)

    if len(unique_options) == 0:
        return ComparisonOutcomes(options=(), counts=np.array([]))

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
                index_0 = option_to_index[option_0]
                index_1 = option_to_index[option_1]
                matrix[index_0, index_1, outcome] = count

    return ComparisonOutcomes(options=sorted_options, counts=matrix)


def resample_results(
    outcomes: ComparisonOutcomes,
    generator: np.random.Generator,
) -> ComparisonOutcomes:
    # FIXME: consider symmetry
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
