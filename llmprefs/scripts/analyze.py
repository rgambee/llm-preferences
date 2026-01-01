import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt

from llmprefs.analysis.rating import compile_matrix, rate_options
from llmprefs.analysis.visualization import plot_ratings_heatmap, plot_ratings_stem
from llmprefs.file_io.load_records import load_records
from llmprefs.task_structs import ResultRecord


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LLM responses",
    )
    parser.add_argument("input_path", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = load_records(args.input_path, ResultRecord)
    option_matrix = compile_matrix(results)
    rated_options = rate_options(option_matrix, num_resamples=100, confidence=0.75)

    fig = plot_ratings_stem(rated_options)
    fig.show()

    desired_num_tasks = 2
    selected_options = {
        key: value
        for key, value in rated_options.items()
        if len(key) == desired_num_tasks
    }
    if selected_options:
        fig = plot_ratings_heatmap(selected_options)
        fig.tight_layout()
        fig.show()

    if plt.isinteractive():
        breakpoint()  # noqa: T100


if __name__ == "__main__":
    main()
