from __future__ import annotations

import itertools
from asyncio import as_completed
from collections.abc import AsyncIterable, Iterable
from datetime import UTC, datetime
from typing import Any, Coroutine, TypeVar

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import AnyApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.parsing import parse_preference
from llmprefs.prompts import COMPARISON_TEMPLATES, ComparisonTemplate
from llmprefs.task_structs import ResultRecord

T = TypeVar("T")


async def run_pipeline(
    api: BaseApi[AnyApiResponse],
    comparisons: Iterable[Comparison],
    concurrent_requests: int,
) -> AsyncIterable[ResultRecord]:
    combinations = itertools.product(comparisons, COMPARISON_TEMPLATES)
    for chunk in chunked(combinations, concurrent_requests):
        awaitables: list[Coroutine[Any, Any, ResultRecord]] = []
        for comparison, template in chunk:
            coro = compare_options(
                api,
                comparison,
                template,
            )
            awaitables.append(coro)
        for future in as_completed(awaitables):
            yield await future


async def compare_options(
    api: BaseApi[AnyApiResponse],
    comparison: Comparison,
    template: ComparisonTemplate,
) -> ResultRecord:
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


def chunked(iterable: Iterable[T], size: int) -> Iterable[tuple[T, ...]]:
    """Split an iterable into chunks of a given size.

    Very similar to itertools.batched, which is available in Python 3.12+.
    https://docs.python.org/3/library/itertools.html#itertools.batched
    """
    iterator = iter(iterable)
    if size < 1:
        raise ValueError("size must be at least 1")
    while chunk := tuple(itertools.islice(iterator, size)):
        yield chunk
