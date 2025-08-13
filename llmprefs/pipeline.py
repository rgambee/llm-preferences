from asyncio import Semaphore, as_completed
from collections.abc import AsyncIterable, Iterable
from datetime import UTC, datetime
from typing import Any, Coroutine

from llmprefs.api.base import BaseApi
from llmprefs.comparisons import Comparison
from llmprefs.parsing import parse_preference
from llmprefs.prompts import COMPARISON_TEMPLATES, ComparisonTemplate
from llmprefs.structs import ResultRecord


async def run_pipeline(
    api: BaseApi,
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
    api: BaseApi,
    comparison: Comparison,
    template: ComparisonTemplate,
    semaphore: Semaphore,
) -> ResultRecord:
    async with semaphore:
        prompt = template.format_comparison(comparison)
        response = await api.submit(prompt)
        preference_index = parse_preference(comparison, response)
        option_ids = [[task.id for task in option] for option in comparison]
        return ResultRecord(
            created_at=datetime.now(tz=UTC),
            comparison_prompt_id=template.id,
            options=option_ids,
            preference_index=preference_index,
            api_params=api.params,
        )
