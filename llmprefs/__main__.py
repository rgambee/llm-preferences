#!/usr/bin/env python3
import asyncio
import logging
from collections.abc import Iterable

from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from llmprefs.api.instantiate import instantiate_api
from llmprefs.comparisons import count_comparisons_approx, generate_comparisons
from llmprefs.file_io.load_records import load_existing_results, load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.logs import configure_logging
from llmprefs.pipeline import run_pipeline
from llmprefs.prompts import ENABLED_COMPARISON_TEMPLATES
from llmprefs.settings import Settings
from llmprefs.task_structs import TaskRecord


def count_samples(
    tasks: Iterable[TaskRecord],
    settings: Settings,
) -> int:
    approximate_comparison_count = count_comparisons_approx(
        records=tasks,
        tasks_per_option=settings.tasks_per_option,
    )
    return (
        approximate_comparison_count
        * len(ENABLED_COMPARISON_TEMPLATES)
        * settings.samples_per_comparison
    )


async def main() -> None:
    settings = Settings()

    configure_logging(logging.INFO, settings.log_file)
    logger = logging.getLogger(__name__)

    if not load_dotenv():
        logger.warning("No .env file found")

    logger.info(f"Loading tasks from {settings.input_path}")
    tasks = list(load_records(settings.input_path, TaskRecord))
    comparisons = generate_comparisons(tasks, settings.tasks_per_option)
    if settings.count_comparisons_only:
        count = sum(1 for _ in comparisons)
        logger.info(f"Comparison count: {count}")
        return
    sample_count_approx = count_samples(tasks, settings)

    api = instantiate_api(settings)

    results = run_pipeline(
        api=api,
        comparisons=comparisons,
        templates=ENABLED_COMPARISON_TEMPLATES,
        settings=settings,
        existing_results=load_existing_results(settings.output_path),
    )
    progress = tqdm(
        results,
        total=sample_count_approx,
    )
    with logging_redirect_tqdm():
        await save_results_jsonl(progress, settings.output_path)


if __name__ == "__main__":
    asyncio.run(main())
