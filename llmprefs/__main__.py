#!/usr/bin/env python3
import asyncio
import logging

from llmprefs.api.instantiate import get_api_for_llm
from llmprefs.comparisons import generate_comparisons
from llmprefs.file_io.load_records import load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.logs import configure_logging
from llmprefs.pipeline import run_pipeline
from llmprefs.settings import Settings
from llmprefs.task_structs import TaskRecord


async def main() -> None:
    settings = Settings()

    configure_logging(logging.INFO, settings.log_file)

    tasks = load_records(settings.input_path, TaskRecord)
    comparisons = generate_comparisons(tasks, settings.tasks_per_option)

    api = get_api_for_llm(settings)

    results = run_pipeline(api, comparisons, settings.concurrent_requests)
    await save_results_jsonl(results, settings.output_path)


if __name__ == "__main__":
    asyncio.run(main())
