from asyncio import Semaphore, as_completed
from collections.abc import AsyncIterable, Iterable
from datetime import UTC, datetime
from typing import Any, Coroutine

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import AnyApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.parsing import parse_preference
from llmprefs.prompts import COMPARISON_TEMPLATES, ComparisonTemplate
from llmprefs.task_structs import ResultRecord


async def run_pipeline(
    api: BaseApi[AnyApiResponse],
    comparisons: Iterable[Comparison],
    concurrent_requests: int,
) -> AsyncIterable[ResultRecord]:
    semaphore = Semaphore(concurrent_requests)
    awaitables: list[Coroutine[Any, Any, ResultRecord]] = []
    for comparison in comparisons:
        for template in COMPARISON_TEMPLATES:
            coro = compare_options(
                api,
                comparison,
                template,
                semaphore,
            )
            awaitables.append(coro)
    for future in as_completed(awaitables):
        yield await future


async def compare_options(
    api: BaseApi[AnyApiResponse],
    comparison: Comparison,
    template: ComparisonTemplate,
    semaphore: Semaphore,
) -> ResultRecord:
    async with semaphore:
        prompt = template.format_comparison(comparison)
        response = await api.submit(prompt)
        preferred_option_index = parse_preference(comparison, response.answer)
        option_ids = [[task.id for task in option] for option in comparison]
        return ResultRecord(
            created_at=datetime.now(tz=UTC),
            comparison_prompt_id=template.id,
            options=option_ids,
            preferred_option_index=preferred_option_index,
            api_params=api.params,
            api_response=response,
        )
