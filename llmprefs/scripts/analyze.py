import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from llmprefs.analysis.option_order import (
    analyze_observations,
    compile_observations,
    plot_option_order_analysis,
)
from llmprefs.analysis.outcomes import plot_comparison_outcomes_heatmap
from llmprefs.analysis.rating import (
    compile_matrix,
    plot_ratings_heatmap,
    plot_ratings_stem,
    rate_options,
)
from llmprefs.analysis.task_order import (
    analyze_task_order,
    plot_task_order_analysis,
)
from llmprefs.file_io.load_records import load_records
from llmprefs.task_structs import ResultRecord, TaskId, TaskRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM responses",
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        required=True,
        help="Path to the tasks file",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to the results file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    task_list = list(load_records(args.tasks_path, TaskRecord))
    tasks: dict[TaskId, TaskRecord] = {task.id: task for task in task_list}
    results = list(load_records(args.results_path, ResultRecord))
    option_matrix = compile_matrix(results)
    logger.info(
        f"Analyzing {option_matrix.matrix.sum():.0f} preferences "
        + f"across {len(option_matrix.options)} options"
    )
    rated_options = rate_options(
        option_matrix,
        tasks,
        num_resamples=100,
        confidence=0.75,
    )

    plot_comparison_outcomes_heatmap(option_matrix, tasks)

    plot_ratings_stem(rated_options, tasks)
    desired_num_tasks = 2
    two_tasks_per_option = {
        key: value
        for key, value in rated_options.items()
        if len(key) == desired_num_tasks
    }
    if two_tasks_per_option:
        fig = plot_ratings_heatmap(two_tasks_per_option, tasks)
        fig.tight_layout()

    observations = compile_observations(results)
    option_order_analysis = analyze_observations(observations)
    plot_option_order_analysis(option_order_analysis, tasks)

    if two_tasks_per_option:
        task_order_analysis = analyze_task_order(results)
        plot_task_order_analysis(task_order_analysis, tasks)

    plt.show()  # pyright: ignore[reportUnknownMemberType]
    if plt.isinteractive():
        breakpoint()  # noqa: T100


if __name__ == "__main__":
    main()
