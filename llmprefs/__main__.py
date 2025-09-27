#!/usr/bin/env python3
import argparse
import asyncio
import logging
from pathlib import Path

from llmprefs.api.instantiate import get_api_for_llm
from llmprefs.comparisons import generate_comparisons
from llmprefs.file_io.load_records import load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.logs import configure_logging
from llmprefs.pipeline import run_pipeline
from llmprefs.structs import LLM, TaskRecord


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to input file containing task dataset",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="""
            Path to output file for saving results.
            If it already exists, new results will be appended.
        """,
    )
    parser.add_argument(
        "model",
        type=LLM,
        help="LLM to use. Choices are " + ", ".join(llm.value for llm in LLM),
    )
    parser.add_argument(
        "--tasks-per-option",
        default=2,
        type=int,
        help="Number of tasks per option (default: 2)",
    )
    parser.add_argument(
        "--concurrent-requests",
        default=10,
        type=int,
        help="Number of concurrent requests to make (default: 10)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to log file. By default, only log to the console.",
    )
    args = parser.parse_args()

    configure_logging(logging.INFO, args.log_file)

    tasks = load_records(args.input_path, TaskRecord)
    comparisons = generate_comparisons(tasks, args.tasks_per_option)

    api = get_api_for_llm(args.model)

    results = run_pipeline(api, comparisons, args.concurrent_requests)
    await save_results_jsonl(results, args.output_path)


if __name__ == "__main__":
    asyncio.run(main())
