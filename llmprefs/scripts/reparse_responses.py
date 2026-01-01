import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv
from tqdm.asyncio import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from llmprefs.api.base import BaseApi
from llmprefs.api.instantiate import instantiate_api
from llmprefs.api.structs import LLM, AnyApiResponse, ApiStage
from llmprefs.file_io.load_records import load_records
from llmprefs.file_io.save_results import save_results_jsonl
from llmprefs.parsing import parse_preference
from llmprefs.settings import Settings
from llmprefs.task_structs import ResultRecord


class ReparseSettings(Settings):
    model: LLM = LLM.MOCK_MODEL


async def process_results(
    path: Path,
    parsing_api: BaseApi[AnyApiResponse],
) -> AsyncGenerator[ResultRecord, None]:
    for record in load_records(path, ResultRecord):
        record.preferred_option_index = await parse_preference(
            num_options=len(record.comparison),
            comparison_prompt=record.comparison_prompt,
            comparison_response=record.api_response.answer,
            parsing_api=parsing_api,
        )
        yield record


def count_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as fin:
        return sum(1 for _ in fin)


async def main() -> None:
    settings = ReparseSettings()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    if not load_dotenv():
        logger.warning("No .env file found")

    parsing_api = instantiate_api(settings, ApiStage.PARSING)
    line_count = count_lines(settings.input_path)
    results = process_results(settings.input_path, parsing_api)
    progress = tqdm(results, total=line_count)
    with logging_redirect_tqdm():
        await save_results_jsonl(progress, settings.output_path)


if __name__ == "__main__":
    asyncio.run(main())
