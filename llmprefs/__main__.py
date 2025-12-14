#!/usr/bin/env python3
import asyncio
import logging

from dotenv import load_dotenv

from llmprefs.api.instantiate import instantiate_api
from llmprefs.comparisons import generate_comparisons
from llmprefs.file_io.load_records import load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.logs import configure_logging
from llmprefs.pipeline import run_pipeline
from llmprefs.prompts import ENABLED_COMPARISON_TEMPLATES
from llmprefs.settings import Settings
from llmprefs.task_structs import TaskRecord


async def main() -> None:
    settings = Settings()

    configure_logging(logging.INFO, settings.log_file)
    logger = logging.getLogger(__name__)

    if not load_dotenv():
        logger.warning("No .env file found")

    tasks = load_records(settings.input_path, TaskRecord)
    comparisons = generate_comparisons(tasks, settings.tasks_per_option)
    if settings.count_comparisons_only:
        count = sum(1 for _ in comparisons)
        logger.info(f"Comparison count: {count}")
        return

    api = instantiate_api(settings)

    results = run_pipeline(
        api=api,
        comparisons=comparisons,
        templates=ENABLED_COMPARISON_TEMPLATES,
        settings=settings,
    )
    await save_results_jsonl(results, settings.output_path)


if __name__ == "__main__":
    asyncio.run(main())
