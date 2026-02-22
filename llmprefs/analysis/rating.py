from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import choix
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from matplotlib.axes import Axes
from matplotlib.collections import FillBetweenPolyCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure
from numpy.typing import NDArray
from odrpack import odr_fit

from llmprefs.analysis.structs import ValueCI
from llmprefs.analysis.visualization import (
    annotated_heatmap,
    construct_title,
    error_bars,
    get_tick_labels,
    weights_from_ci,
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


@dataclass
class RatedOptions:
    # A tuple of unique options. The order matches the indices of the matrix.
    options: tuple[OptionById, ...]
    # A 2D matrix of shape N_opts x N_resamples
    ratings: NDArray[np.float64]

    def values(self, confidence: float) -> dict[OptionById, ValueCI]:
        medians = np.median(self.ratings, axis=1)
        lower_quant = (1 - confidence) / 2
        upper_quant = (1 + confidence) / 2
        upper_bounds = np.quantile(self.ratings, q=upper_quant, axis=1)
        lower_bounds = np.quantile(self.ratings, q=lower_quant, axis=1)
        return {
            option: ValueCI(
                value=medians[i],
                ci_lower=lower_bounds[i],
                ci_upper=upper_bounds[i],
            )
            for i, option in enumerate(self.options)
        }


def rate_options(
    outcomes: ComparisonOutcomes,
    tasks: Mapping[TaskId, TaskRecord],
    num_resamples: int,
    alpha: float = 1e-6,
) -> RatedOptions:
    """Return an estimate of each option's strength.

    Also compute a bootstrapped confidence interval for each rating.
    """
    if num_resamples < 0:
        raise ValueError("Number of resamples must be non-negative")
    if outcomes.counts.size == 0:
        return RatedOptions(
            options=outcomes.options,
            ratings=np.zeros((0, num_resamples), dtype=np.float64),
        )

    # 2D matrix of shape (N_opts, max(num_resamples, 1))
    ratings: NDArray[np.float64]
    if num_resamples == 0:
        ratings_1d = choix.ilsr_pairwise_dense(
            comp_mat=outcomes.unfold().astype(np.float64),
            alpha=alpha,
        )
        ratings = np.expand_dims(ratings_1d, axis=1)
    else:
        generator = np.random.default_rng()
        ratings = np.full((len(outcomes.options), num_resamples), np.nan)
        for i in range(num_resamples):
            resample = resample_results(outcomes=outcomes, generator=generator)
            sample_ratings = choix.ilsr_pairwise_dense(
                comp_mat=resample.unfold().astype(np.float64),
                alpha=alpha,
            )
            ratings[:, i] = sample_ratings

    # Apply an offset such that the opt-out tasks have a rating of 0.
    offset = median_opt_out_rating(ratings, outcomes.options, tasks)
    ratings -= offset
    return RatedOptions(options=outcomes.options, ratings=ratings)


def median_opt_out_rating(
    ratings: NDArray[np.float64],
    options: Sequence[OptionById],
    tasks: Mapping[TaskId, TaskRecord],
) -> float:
    opt_outs = [
        all(is_opt_out_task(tasks[task_id]) for task_id in option) for option in options
    ]
    if not any(opt_outs):
        return 0.0
    return np.median(ratings[opt_outs, :])


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


def plot_ratings_scatter(
    rated_options: RatedOptions,
    tasks: Mapping[TaskId, TaskRecord],
    confidence: float,
    title_suffix: str = "",
) -> Figure:
    ycoords = np.arange(len(rated_options.options), 0, -1)
    rating_values = rated_options.values(confidence)
    medians = [vci.value for vci in rating_values.values()]

    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        layout="constrained",
    )
    ax.errorbar(  # pyright: ignore[reportUnknownMemberType]
        x=medians,
        y=ycoords,
        xerr=error_bars(list(rating_values.values())),
        marker="o",
        linestyle="None",
        ecolor="black",
        capsize=5.0,
        label=f"Data points with bootstrapped {confidence:.0%} CI",
    )
    ax.axvline(  # pyright: ignore[reportUnknownMemberType]
        x=0,
        linestyle="dashed",
        color="gray",
        zorder=0,
    )
    ax.set_title(  # pyright: ignore[reportUnknownMemberType]
        construct_title("Rated Options", title_suffix)
    )
    ax.set_xlabel("Rating")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Index of Option")  # pyright: ignore[reportUnknownMemberType]
    ax.set_yticks(  # pyright: ignore[reportUnknownMemberType]
        ticks=ycoords,
        labels=get_tick_labels(rated_options.options, tasks),
        fontsize="x-small",
    )
    ax.legend()  # pyright: ignore[reportUnknownMemberType]

    return fig


def plot_ratings_heatmap(
    rated_options: RatedOptions,
    tasks: Mapping[TaskId, TaskRecord],
    title_suffix: str = "",
) -> Figure:
    expected_num_tasks = 2
    options_to_plot = [
        option for option in rated_options.options if len(option) == expected_num_tasks
    ]
    if len(options_to_plot) == 0:
        raise ValueError(
            f"Heatmap requires options containing {expected_num_tasks} tasks"
        )

    task_0_and_1_ids: zip[tuple[TaskId, TaskId]] = zip(
        *options_to_plot,
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
    rated_values = rated_options.values(confidence=0.0)
    for task0, task1 in options_to_plot:
        rating = rated_values[(task0, task1)]
        ratings[task_ids_to_index[task0], task_ids_to_index[task1]] = rating.value

    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        layout="constrained",
        figsize=(5.2, 4.8),
    )
    tick_labels = get_tick_labels(
        options=((task_id,) for task_id in task_0_ids),
        tasks=tasks,
    )
    im = annotated_heatmap(ax, ratings, tick_labels)

    ax.set_title(  # pyright: ignore[reportUnknownMemberType]
        construct_title("Rated Options", title_suffix)
    )
    ax.set_ylabel("First Task ID")  # pyright: ignore[reportUnknownMemberType]
    ax.set_xlabel("Second Task ID")  # pyright: ignore[reportUnknownMemberType]

    colorbar = fig.colorbar(im, ax=ax)  # pyright: ignore[reportUnknownMemberType]
    colorbar.ax.set_ylabel("Rating")  # pyright: ignore[reportUnknownMemberType]

    return fig


def plot_rating_additivity_scatter(
    rated_options_1tpo: RatedOptions,
    rated_options_2tpo: RatedOptions,
    tasks: Mapping[TaskId, TaskRecord],
    confidence: float,
    title_suffix: str = "",
) -> Figure:
    if rated_options_1tpo.ratings.shape[1] != rated_options_2tpo.ratings.shape[1]:
        raise ValueError("Number of resamples must be the same")

    fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
        layout="constrained",
    )

    # Every option in rated_options_2tpo is a sequence of multiple tasks. Add the
    # ratings of the individual tasks from rated_options_1tpo to get the x coordinates.
    summed_ratings = compute_summed_ratings(
        rated_options_1tpo=rated_options_1tpo,
        rated_options_2tpo=rated_options_2tpo,
        tasks=tasks,
    )
    summed_values = summed_ratings.values(confidence)
    x_medians = np.array(
        [summed_values[option].value for option in summed_ratings.options]
    )

    # The y-coordinates are the ratings of the task sequences (from rated_options_2tpo).
    rating_values_2tpo = rated_options_2tpo.values(confidence)
    y_medians = np.array(
        [rating_values_2tpo[option].value for option in rated_options_2tpo.options]
    )

    data_markers = plot_medians_with_error_bars(
        ax=ax,
        x_values=[summed_values[opt] for opt in summed_ratings.options],
        y_values=[rating_values_2tpo[opt] for opt in rated_options_2tpo.options],
        confidence=confidence,
    )

    # Linear fit of the median x and y values
    fit_of_medians = odr_fit(
        f=lambda x, beta: beta[0] + beta[1] * x,
        xdata=x_medians,
        ydata=y_medians,
        beta0=(0.0, 1.0),
        weight_x=weights_from_ci(summed_values.values()),
        weight_y=weights_from_ci(rating_values_2tpo.values()),
    )
    linear_fit_line = ax.plot(  # pyright: ignore[reportUnknownMemberType]
        x_medians,
        fit_of_medians.beta[0] + fit_of_medians.beta[1] * x_medians,
        marker="None",
        color="C1",
        label="Linear fit",
    )

    confidence_band = show_fit_confidence_band(
        ax=ax,
        x_resample_ratings=summed_ratings.ratings,
        y_resample_ratings=rated_options_2tpo.ratings,
        confidence=confidence,
    )

    ax.legend(  # pyright: ignore[reportUnknownMemberType]
        handles=[
            data_markers,
            *linear_fit_line,
            confidence_band,
        ],
    )
    ax.set_xlabel("Sum of Task Ratings")  # pyright: ignore[reportUnknownMemberType]
    ax.set_ylabel("Rating of Task Sequence")  # pyright: ignore[reportUnknownMemberType]
    ax.set_title(  # pyright: ignore[reportUnknownMemberType]
        construct_title("Task Rating Additivity", title_suffix)
    )
    return fig


def compute_summed_ratings(
    rated_options_1tpo: RatedOptions,
    rated_options_2tpo: RatedOptions,
    tasks: Mapping[TaskId, TaskRecord],
) -> RatedOptions:
    # Say option o from rated_options_2tpo.options consists of tasks (i, j).
    # Then sums[o, :] == ratings_1tpo[i, :] + ratings_1tpo[j, :]
    sums = np.zeros_like(rated_options_2tpo.ratings, dtype=np.float64)
    expected_num_tasks = 2
    for o, option in enumerate(rated_options_2tpo.options):
        if len(option) != expected_num_tasks:
            if is_opt_out_task(tasks[option[0]]):
                continue
            raise ValueError(f"Option {option} is not a valid 2-task option")
        task_i, task_j = option
        index_i = rated_options_1tpo.options.index((task_i,))
        index_j = rated_options_1tpo.options.index((task_j,))
        sums[o, :] = (
            rated_options_1tpo.ratings[index_i, :]
            + rated_options_1tpo.ratings[index_j, :]
        )
    return RatedOptions(options=rated_options_2tpo.options, ratings=sums)


def plot_medians_with_error_bars(
    ax: Axes,
    x_values: Sequence[ValueCI],
    y_values: Sequence[ValueCI],
    confidence: float,
) -> ErrorbarContainer:
    x_medians = np.array([vci.value for vci in x_values])
    y_medians = np.array([vci.value for vci in y_values])
    return ax.errorbar(  # pyright: ignore[reportUnknownMemberType]
        x=x_medians,
        y=y_medians,
        xerr=error_bars(x_values),
        yerr=error_bars(y_values),
        marker="o",
        markeredgewidth=0,
        linestyle="None",
        ecolor="black",
        alpha=0.5,
        label=f"Data points with bootstrapped {confidence:.0%} CI",
    )


def show_fit_confidence_band(
    ax: Axes,
    x_resample_ratings: NDArray[np.float64],
    y_resample_ratings: NDArray[np.float64],
    confidence: float,
) -> FillBetweenPolyCollection:
    # Compute a linear fit of each sample to construct a confidence band
    x_medians = np.median(x_resample_ratings, axis=1)
    x_grid = np.linspace(
        start=x_medians.min(),
        stop=x_medians.max(),
        num=100,
        dtype=np.float64,
    )
    y_estimates = calculate_y_estimates(
        x_resample_ratings=x_resample_ratings,
        y_resample_ratings=y_resample_ratings,
        x_grid=x_grid,
    )
    y_lower = np.quantile(y_estimates, q=(1 - confidence) / 2, axis=1)
    y_upper = np.quantile(y_estimates, q=(1 + confidence) / 2, axis=1)
    return ax.fill_between(  # pyright: ignore[reportUnknownMemberType]
        x=x_grid,
        y1=y_lower,
        y2=y_upper,
        color="C1",
        alpha=0.2,
        label=f"{confidence:.0%} confidence band",
    )


def calculate_y_estimates(
    x_resample_ratings: NDArray[np.float64],
    y_resample_ratings: NDArray[np.float64],
    x_grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate the y-estimates at x_grid for each resample using a linear fit.

    x_resample_ratings and y_resample_ratings both have shape (N_options, N_resamples).
    The output has shape (len(x_grid), N_resamples).
    """
    num_resamples = x_resample_ratings.shape[1]
    y_estimates = np.zeros((len(x_grid), num_resamples), dtype=np.float64)
    for i in range(num_resamples):
        fit = scipy.stats.linregress(
            x=x_resample_ratings[:, i],
            y=y_resample_ratings[:, i],
        )
        y_estimates[:, i] = fit.intercept + fit.slope * x_grid
    return y_estimates
