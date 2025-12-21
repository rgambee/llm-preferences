import itertools
import logging
from asyncio import as_completed
from collections.abc import AsyncGenerator, Coroutine, Iterable
from datetime import UTC, datetime
from typing import Any, TypeVar

from pydantic import BaseModel

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import AnyApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.parsing import parse_preference
from llmprefs.prompts import ComparisonTemplate
from llmprefs.settings import Settings
from llmprefs.task_structs import ResultRecord, ResultRecordKey, comparison_to_id

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
    existing_results: set[ResultRecordKey],
) -> AsyncGenerator[ResultRecord, None]:
    logger = logging.getLogger(__name__)
    samples = generate_samples(
        comparisons=comparisons,
        templates=templates,
        samples_per_comparison=settings.samples_per_comparison,
    )
    skip_count = 0
    for chunk in chunked(samples, settings.concurrent_requests):
        awaitables: list[Coroutine[Any, Any, ResultRecord]] = []
        for sample in chunk:
            if result_already_exists(sample, existing_results):
                skip_count += 1
                continue
            logger.info(
                f"Skipped {skip_count} samples because results already exist",
            )
            skip_count = 0
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
    preferred_option_index = parse_preference(
        num_options=len(sample.comparison),
        llm_response=response.answer,
    )
    return ResultRecord(
        created_at=datetime.now(tz=UTC),
        comparison_prompt_id=sample.template.id,
        comparison=comparison_to_id(sample.comparison),
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


def result_already_exists(
    sample: Sample,
    existing_results: set[ResultRecordKey],
) -> bool:
    key = ResultRecordKey(
        comparison=comparison_to_id(sample.comparison),
        comparison_prompt_id=sample.template.id,
        sample_index=sample.index,
    )
    return key in existing_results
