import argparse
import asyncio
import logging
from collections.abc import Iterable
from pathlib import Path

from llmprefs.file_io.load_records import load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.parsing import parse_preference
from llmprefs.task_structs import ResultRecord


def load_results(path: Path) -> Iterable[ResultRecord]:
    for record in load_records(path, ResultRecord):
        record.preferred_option_index = parse_preference(
            num_options=len(record.comparison),
            llm_response=record.api_response.answer,
        )
        yield record


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reparse LLM responses using current parsing logic",
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    results = load_results(args.input_path)
    await save_results_jsonl(results, args.output_path)


if __name__ == "__main__":
    asyncio.run(main())
