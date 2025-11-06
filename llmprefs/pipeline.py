from __future__ import annotations

import itertools
from asyncio import as_completed
from collections.abc import AsyncIterable, Iterable
from datetime import UTC, datetime
from typing import Any, Coroutine, TypeVar

from pydantic import BaseModel

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import AnyApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.parsing import parse_preference
from llmprefs.prompts import ComparisonTemplate
from llmprefs.settings import Settings
from llmprefs.task_structs import ResultRecord

T = TypeVar("T")


class Sample(BaseModel):
    index: int
    comparison: Comparison
    template: ComparisonTemplate


async def run_pipeline(
    api: BaseApi[AnyApiResponse],
    comparisons: Iterable[Comparison],
    templates: Iterable[ComparisonTemplate],
    settings: Settings,
) -> AsyncIterable[ResultRecord]:
    samples = generate_samples(
        comparisons=comparisons,
        templates=templates,
        samples_per_comparison=settings.samples_per_comparison,
    )
    for chunk in chunked(samples, settings.concurrent_requests):
        awaitables: list[Coroutine[Any, Any, ResultRecord]] = []
        for sample in chunk:
            coro = compare_options(api, sample)
            awaitables.append(coro)
        for future in as_completed(awaitables):
            yield await future


async def compare_options(
    api: BaseApi[AnyApiResponse],
    sample: Sample,
) -> ResultRecord:
    prompt = sample.template.format_comparison(sample.comparison)
    response = await api.submit(prompt)
    preferred_option_index = parse_preference(sample.comparison, response.answer)
    option_a, option_b = sample.comparison
    option_ids = (
        [task.id for task in option_a],
        [task.id for task in option_b],
    )
    return ResultRecord(
        created_at=datetime.now(tz=UTC),
        comparison_prompt_id=sample.template.id,
        comparison=option_ids,
        sample_index=sample.index,
        preferred_option_index=preferred_option_index,
        api_params=api.params,
        api_response=response,
    )


def generate_samples(
    comparisons: Iterable[Comparison],
    templates: Iterable[ComparisonTemplate],
    samples_per_comparison: int,
) -> Iterable[Sample]:
    for comparison, template in itertools.product(comparisons, templates):
        for sample_index in range(samples_per_comparison):
            yield Sample(
                index=sample_index,
                comparison=comparison,
                template=template,
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
