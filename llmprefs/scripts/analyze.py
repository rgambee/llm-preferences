import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from llmprefs.analysis.option_order import (
    analyze_option_order,
    plot_option_order_analysis,
)
from llmprefs.analysis.outcomes import plot_comparison_outcomes_heatmap
from llmprefs.analysis.rating import (
    RatedOptions,
    compile_matrix,
    plot_multi_ratings_violin,
    plot_rating_additivity_scatter,
    plot_ratings_heatmap,
    plot_ratings_violin,
    rate_options,
)
from llmprefs.analysis.task_order import (
    analyze_task_order,
    plot_task_order_analysis,
)
from llmprefs.analysis.visualization import construct_figure_filename
from llmprefs.file_io.load_records import load_records
from llmprefs.task_structs import ResultRecord, TaskId, TaskRecord

CONFIDENCE = 0.95


def analyze_one_set_of_results(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    task_list = list(load_records(args.tasks_path, TaskRecord))
    tasks: dict[TaskId, TaskRecord] = {task.id: task for task in task_list}
    results = list(load_records(args.results_path, ResultRecord))
    outcomes = compile_matrix(results)
    logger.info(
        f"Analyzing {outcomes.counts.sum():.0f} preferences "
        + f"across {len(outcomes.options)} options"
    )
    rated_options = rate_options(
        outcomes,
        tasks,
        num_resamples=100,
    )

    fig = plot_comparison_outcomes_heatmap(
        outcomes,
        tasks,
        title_suffix=args.title_suffix,
    )
    save_figure(fig, args, "outcome-counts")

    fig = plot_ratings_violin(
        rated_options,
        tasks,
        title_suffix=args.title_suffix,
    )
    save_figure(fig, args, "rated-options")

    two_tasks_per_option = True
    try:
        fig = plot_ratings_heatmap(
            rated_options,
            tasks,
            title_suffix=args.title_suffix,
        )
    except ValueError as error:
        if error.args[0] == "Heatmap requires options containing 2 tasks":
            two_tasks_per_option = False
        else:
            raise
    else:
        save_figure(fig, args, "rated-options-heatmap")

    option_order_analysis = analyze_option_order(results)
    fig = plot_option_order_analysis(
        option_order_analysis,
        tasks,
        title_suffix=args.title_suffix,
    )
    save_figure(fig, args, "option-order-analysis")

    if two_tasks_per_option:
        task_order_analysis = analyze_task_order(results)
        figs = plot_task_order_analysis(
            task_order_analysis,
            tasks,
            title_suffix=args.title_suffix,
        )
        for i, fig in enumerate(figs):
            save_figure(fig, args, f"task-order-analysis-{i}")


def analyze_two_sets_of_results(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    task_list = list(load_records(args.tasks_path, TaskRecord))
    tasks: dict[TaskId, TaskRecord] = {task.id: task for task in task_list}
    results_1tpo = list(load_records(args.results_1tpo_path, ResultRecord))
    results_2tpo = list(load_records(args.results_2tpo_path, ResultRecord))
    options_1tpo = compile_matrix(results_1tpo)
    logger.info(
        f"Analyzing {options_1tpo.counts.sum():.0f} preferences "
        + f"across {len(options_1tpo.options)} options"
    )
    rated_options_1tpo = rate_options(
        options_1tpo,
        tasks,
        num_resamples=100,
    )
    options_2tpo = compile_matrix(results_2tpo)
    logger.info(
        f"Analyzing {options_2tpo.counts.sum():.0f} preferences "
        + f"across {len(options_2tpo.options)} options"
    )
    rated_options_2tpo = rate_options(
        options_2tpo,
        tasks,
        num_resamples=100,
    )
    fig = plot_rating_additivity_scatter(
        rated_options_1tpo,
        rated_options_2tpo,
        tasks,
        confidence=CONFIDENCE,
        title_suffix=args.title_suffix,
    )
    save_figure(
        fig,
        argparse.Namespace(
            output_dir=args.output_dir,
            results_path=args.results_1tpo_path,
        ),
        "rating-additivity",
    )


def analyze_multiple_sets_of_results(args: argparse.Namespace) -> None:
    logger = logging.getLogger(__name__)
    task_list = list(load_records(args.tasks_path, TaskRecord))
    tasks: dict[TaskId, TaskRecord] = {task.id: task for task in task_list}
    all_ratings: list[RatedOptions] = []
    for results_path in args.results_paths:
        results = list(load_records(results_path, ResultRecord))
        outcomes = compile_matrix(results)
        logger.info(
            f"Analyzing {outcomes.counts.sum():.0f} preferences "
            + f"across {len(outcomes.options)} options"
        )
        rated_options = rate_options(outcomes, tasks, num_resamples=100)
        all_ratings.append(rated_options)
    fig = plot_multi_ratings_violin(
        all_ratings,
        tasks,
        title_suffix=args.title_suffix,
        legend_labels=args.legend_labels,
    )
    save_figure(
        fig,
        args=argparse.Namespace(
            output_dir=args.output_dir,
            results_path=Path(),
        ),
        figure_name="rated-options",
    )


def save_figure(
    figure: Figure,
    args: argparse.Namespace,
    figure_name: str,
) -> Path | None:
    if args.output_dir is None:
        return None
    figure_filename = construct_figure_filename(
        output_dir=args.output_dir,
        results_path=args.results_path,
        figure_name=figure_name,
    )
    figure.savefig(figure_filename)  # pyright: ignore[reportUnknownMemberType]
    logging.getLogger(__name__).info(f"Saved figure to {figure_filename}")
    return figure_filename


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM task preference responses",
    )
    subparsers = parser.add_subparsers(required=True)

    analyze_one_parser = subparsers.add_parser(
        "analyze-one",
        help="Analyze a single set of results",
    )
    analyze_one_parser.add_argument(
        "--tasks-path",
        type=Path,
        required=True,
        help="Path to the tasks file",
    )
    analyze_one_parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to the results file",
    )
    analyze_one_parser.add_argument(
        "--title-suffix",
        type=str,
        help="Suffix to add to all figure titles",
    )
    analyze_one_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save figures as SVGs",
    )
    analyze_one_parser.set_defaults(func=analyze_one_set_of_results)

    analyze_two_parser = subparsers.add_parser(
        "analyze-two",
        help="""
            Analyze corresponding one-task-per-option and two-tasks-per-option results
        """,
    )
    analyze_two_parser.add_argument(
        "--tasks-path",
        type=Path,
        required=True,
        help="Path to the tasks file",
    )
    analyze_two_parser.add_argument(
        "--1-task-per-option-results-path",
        type=Path,
        required=True,
        dest="results_1tpo_path",
        help="Path to the results file for one task per option",
    )
    analyze_two_parser.add_argument(
        "--2-tasks-per-option-results-path",
        type=Path,
        required=True,
        dest="results_2tpo_path",
        help="Path to the results file for two tasks per option",
    )
    analyze_two_parser.add_argument(
        "--title-suffix",
        type=str,
        help="Suffix to add to all figure titles",
    )
    analyze_two_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save figures as SVGs",
    )
    analyze_two_parser.set_defaults(func=analyze_two_sets_of_results)

    analyze_multi_parser = subparsers.add_parser(
        "analyze-multi",
        help="Analyze multiple sets of one-task-per-option results",
    )
    analyze_multi_parser.add_argument(
        "--tasks-path",
        type=Path,
        required=True,
    )
    analyze_multi_parser.add_argument(
        "--results-paths",
        type=Path,
        required=True,
        nargs="+",
        help="Path to the results files",
    )
    analyze_multi_parser.add_argument(
        "--title-suffix",
        type=str,
        help="Suffix to add to all figure titles",
    )
    analyze_multi_parser.add_argument(
        "--legend-labels",
        type=str,
        nargs="+",
        help="Labels for the legend",
    )
    analyze_multi_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save figures as SVGs",
    )
    analyze_multi_parser.set_defaults(func=analyze_multiple_sets_of_results)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    args.func(args)

    plt.show()  # pyright: ignore[reportUnknownMemberType]
    if plt.isinteractive():
        breakpoint()  # noqa: T100


if __name__ == "__main__":
    main()
