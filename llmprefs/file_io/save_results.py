from __future__ import annotations

import logging
from collections.abc import AsyncIterable, Iterable
from io import TextIOWrapper
from pathlib import Path

from llmprefs.task_structs import ResultRecord


async def save_results_jsonl(
    results: AsyncIterable[ResultRecord] | Iterable[ResultRecord],
    path: Path,
) -> None:
    """Save the results to a JSONL file.

    Append to the file if it already exists.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Saving results to {path}")
    with path.open("a", encoding="utf-8") as f:
        if isinstance(results, AsyncIterable):
            async for result in results:
                write_single_result(f, result)
        else:
            for result in results:
                write_single_result(f, result)


def write_single_result(file: TextIOWrapper, result: ResultRecord) -> None:
    file.write(result.model_dump_json(by_alias=True) + "\n")
