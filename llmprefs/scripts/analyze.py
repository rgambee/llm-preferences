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

    fig = plot_comparison_outcomes_heatmap(option_matrix, tasks)
    fig.show()

    fig = plot_ratings_stem(rated_options, tasks)
    fig.show()

    desired_num_tasks = 2
    selected_options = {
        key: value
        for key, value in rated_options.items()
        if len(key) == desired_num_tasks
    }
    if selected_options:
        fig = plot_ratings_heatmap(selected_options, tasks)
        fig.tight_layout()
        fig.show()

    observations = compile_observations(results)
    option_order_analysis = analyze_observations(observations)
    fig = plot_option_order_analysis(option_order_analysis, tasks)
    fig.show()

    if plt.isinteractive():
        breakpoint()  # noqa: T100


if __name__ == "__main__":
    main()
