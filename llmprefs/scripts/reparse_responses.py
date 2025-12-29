import argparse
import asyncio
import logging
from collections.abc import AsyncIterable
from pathlib import Path

from llmprefs.api.base import BaseApi
from llmprefs.api.instantiate import instantiate_api
from llmprefs.api.structs import LLM, AnyApiResponse, ApiStage
from llmprefs.file_io.load_records import load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.parsing import parse_preference
from llmprefs.settings import Settings
from llmprefs.task_structs import ResultRecord


async def process_results(
    path: Path,
    parsing_api: BaseApi[AnyApiResponse],
) -> AsyncIterable[ResultRecord]:
    for record in load_records(path, ResultRecord):
        comparison_prompt = ""
        record.preferred_option_index = await parse_preference(
            num_options=len(record.comparison),
            comparison_prompt=comparison_prompt,
            comparison_response=record.api_response.answer,
            parsing_api=parsing_api,
        )
        yield record


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reparse LLM responses using current parsing logic",
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()

    settings = Settings(
        input_path=args.input_path,
        output_path=args.output_path,
        model=LLM.MOCK_MODEL,
    )
    parsing_api = instantiate_api(settings, ApiStage.PARSING)
    logging.basicConfig(level=logging.INFO)
    results = process_results(args.input_path, parsing_api)
    await save_results_jsonl(results, args.output_path)


if __name__ == "__main__":
    asyncio.run(main())
