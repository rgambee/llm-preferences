import asyncio
from unittest.mock import AsyncMock

import pytest

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import MockApiParams, MockApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.pipeline import (
    Sample,
    chunked,
    generate_samples,
    result_already_exists,
    run_pipeline,
)
from llmprefs.prompts import ComparisonTemplate, TemplateStatus
from llmprefs.task_structs import ResultRecordKey, TaskType, comparison_to_id
from llmprefs.testing.factories import task_record_factory
from llmprefs.testing.mock_settings import MockSettings

MOCK_TEMPLATES = [
    ComparisonTemplate(
        id=0,
        status=TemplateStatus.enabled,
        template="{option_a} or {option_b}",
    )
]


class TestRunPipeline:
    @pytest.mark.anyio
    async def test_empty(self) -> None:
        api = AsyncMock(spec=BaseApi)
        comparisons: list[Comparison] = []
        results = run_pipeline(
            api=api,
            comparisons=comparisons,
            templates=MOCK_TEMPLATES,
            settings=MockSettings(),
            existing_results=set(),
        )

        async for _ in results:
            pytest.fail("Should not yield any results")
        api.submit.assert_not_awaited()

    @pytest.mark.anyio
    async def test_one_comparison(self) -> None:
        api = AsyncMock(spec=BaseApi)
        api.params = MockApiParams()
        api.submit.return_value = MockApiResponse(reply="Option A")

        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))
        results = run_pipeline(
            api=api,
            comparisons=[comparison],
            templates=MOCK_TEMPLATES,
            settings=MockSettings(),
            existing_results=set(),
        )

        async for result in results:
            assert result.preferred_option_index == 0
        api.submit.assert_awaited_once()

    @pytest.mark.anyio
    async def test_multiple_comparisons_serial(self) -> None:
        api = AsyncMock(spec=BaseApi)
        api.params = MockApiParams()
        api.submit.return_value = MockApiResponse(reply="Option A")

        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison_0 = ((task_a,), (task_b,))
        comparison_1 = ((task_b,), (task_a,))
        settings = MockSettings(concurrent_requests=1)
        results = run_pipeline(
            api=api,
            comparisons=[comparison_0, comparison_1],
            templates=MOCK_TEMPLATES,
            settings=settings,
            existing_results=set(),
        )

        index = 0
        async for result in results:
            await asyncio.sleep(0)
            assert index < 2
            assert result.preferred_option_index == 0
            assert api.submit.await_count == index + 1
            index += 1
        assert index == 2

    @pytest.mark.anyio
    async def test_multiple_comparisons_parallel(self) -> None:
        api = AsyncMock(spec=BaseApi)
        api.params = MockApiParams()
        api.submit.return_value = MockApiResponse(reply="Option A")

        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison_0 = ((task_a,), (task_b,))
        comparison_1 = ((task_b,), (task_a,))
        settings = MockSettings(concurrent_requests=2)
        results = run_pipeline(
            api=api,
            comparisons=[comparison_0, comparison_1],
            templates=MOCK_TEMPLATES,
            settings=settings,
            existing_results=set(),
        )

        index = 0
        async for result in results:
            await asyncio.sleep(0)
            assert index < 2
            assert result.preferred_option_index == 0
            assert api.submit.await_count == 2
            index += 1
        assert index == 2

    @pytest.mark.anyio
    async def test_multiple_samples(self) -> None:
        api = AsyncMock(spec=BaseApi)
        api.params = MockApiParams()
        api.submit.return_value = MockApiResponse(reply="Option A")

        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))
        settings = MockSettings(samples_per_comparison=3, concurrent_requests=1)
        results = run_pipeline(
            api=api,
            comparisons=[comparison],
            templates=MOCK_TEMPLATES,
            settings=settings,
            existing_results=set(),
        )

        index = 0
        async for result in results:
            await asyncio.sleep(0)
            assert index < 3
            assert result.preferred_option_index == 0
            assert api.submit.await_count == index + 1
            index += 1
        assert index == 3

    @pytest.mark.anyio
    async def test_existing_results(self) -> None:
        api = AsyncMock(spec=BaseApi)
        api.params = MockApiParams()
        api.submit.return_value = MockApiResponse(reply="Option A")

        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        old_comparison = ((task_a,), (task_b,))
        new_comparison = ((task_b,), (task_a,))
        settings = MockSettings(samples_per_comparison=1, concurrent_requests=1)
        existing_results = {
            ResultRecordKey(
                comparison=comparison_to_id(old_comparison),
                comparison_prompt_id=MOCK_TEMPLATES[0].id,
                sample_index=0,
            )
        }
        results = run_pipeline(
            api=api,
            comparisons=[old_comparison, new_comparison],
            templates=MOCK_TEMPLATES,
            settings=settings,
            existing_results=existing_results,
        )

        index = 0
        async for result in results:
            await asyncio.sleep(0)
            assert index < 1
            assert result.comparison == comparison_to_id(new_comparison)
            assert result.preferred_option_index == 0
            assert api.submit.await_count == index + 1
            index += 1
        assert index == 1


class TestGenerateSamples:
    def test_empty(self) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))

        assert list(generate_samples([], [], 1)) == []
        assert list(generate_samples([], MOCK_TEMPLATES, 1)) == []
        assert list(generate_samples([comparison], [], 1)) == []
        assert list(generate_samples([comparison], MOCK_TEMPLATES, 0)) == []

    @pytest.mark.parametrize("samples_per_comparison", [1, 2])
    def test_repeated_sample(self, samples_per_comparison: int) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))
        samples = list(
            generate_samples([comparison], MOCK_TEMPLATES, samples_per_comparison)
        )
        assert len(samples) == samples_per_comparison
        for i in range(len(samples)):
            assert samples[i].comparison == comparison
            assert samples[i].template == MOCK_TEMPLATES[0]
            assert samples[i].index == i

    @pytest.mark.parametrize("samples_per_comparison", [1, 2])
    def test_multiple_different_samples(self, samples_per_comparison: int) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison_0 = ((task_a,), (task_b,))
        comparison_1 = ((task_b,), (task_a,))
        comparisons = [comparison_0, comparison_1]
        samples = list(
            generate_samples(comparisons, MOCK_TEMPLATES, samples_per_comparison)
        )
        assert len(samples) == (
            len(comparisons) * len(MOCK_TEMPLATES) * samples_per_comparison
        )
        outer_index = 0
        for comparison in comparisons:
            for template in MOCK_TEMPLATES:
                for sample_index in range(samples_per_comparison):
                    sample = samples[outer_index]
                    assert sample.comparison == comparison
                    assert sample.template == template
                    assert sample.index == sample_index
                    outer_index += 1


class TestChunked:
    def test_invalid_size(self) -> None:
        with pytest.raises(ValueError, match="size must be at least 1"):
            list(chunked((1, 2, 3), 0))

    def test_empty(self) -> None:
        assert list(chunked((), 1)) == []

    def test_single_item(self) -> None:
        assert list(chunked((1,), 1)) == [(1,)]

    def test_multiple_items(self) -> None:
        assert list(chunked((1, 2, 3, 4), 2)) == [(1, 2), (3, 4)]

    def test_partial_final_chunk(self) -> None:
        assert list(chunked((1, 2, 3), 2)) == [(1, 2), (3,)]


class TestResultAlreadyExists:
    def test_empty(self) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))
        sample = Sample(
            index=0,
            comparison=comparison,
            template=MOCK_TEMPLATES[0],
        )
        assert not result_already_exists(sample, set())

    def test_present(self) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))
        sample = Sample(
            index=0,
            comparison=comparison,
            template=MOCK_TEMPLATES[0],
        )
        existing_results = {
            ResultRecordKey(
                comparison=((task_a.id,), (task_b.id,)),
                comparison_prompt_id=MOCK_TEMPLATES[0].id,
                sample_index=0,
            )
        }
        assert result_already_exists(sample, existing_results)
