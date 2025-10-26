from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import LLM, MockApiParams, MockApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.pipeline import chunked, generate_samples, run_pipeline
from llmprefs.prompts import COMPARISON_TEMPLATES
from llmprefs.settings import Settings
from llmprefs.task_structs import TaskType
from llmprefs.testing.factories import task_record_factory


class MockSettings(Settings, cli_parse_args=False):
    input_path: Path = Path("in.csv")
    output_path: Path = Path("out.jsonl")
    model: LLM = LLM.MOCK_MODEL


class TestRunPipeline:
    @pytest.mark.anyio
    async def test_empty(self) -> None:
        api = AsyncMock(spec=BaseApi)
        comparisons: list[Comparison] = []
        results = run_pipeline(api, comparisons, MockSettings())
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
        results = run_pipeline(api, [comparison], MockSettings())

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
        results = run_pipeline(api, [comparison_0, comparison_1], settings)

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
        results = run_pipeline(api, [comparison_0, comparison_1], settings)

        index = 0
        async for result in results:
            await asyncio.sleep(0)
            assert index < 2
            assert result.preferred_option_index == 0
            assert api.submit.await_count == 2
            index += 1
        assert index == 2


class TestGenerateSamples:
    def test_empty(self) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))

        assert list(generate_samples([], [], 1)) == []
        assert list(generate_samples([], COMPARISON_TEMPLATES, 1)) == []
        assert list(generate_samples([comparison], [], 1)) == []
        assert list(generate_samples([comparison], COMPARISON_TEMPLATES, 0)) == []

    @pytest.mark.parametrize("samples_per_comparison", [1, 2])
    def test_repeated_sample(self, samples_per_comparison: int) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((task_a,), (task_b,))
        samples = list(
            generate_samples([comparison], COMPARISON_TEMPLATES, samples_per_comparison)
        )
        assert len(samples) == samples_per_comparison
        for i in range(len(samples)):
            assert samples[i].comparison == comparison
            assert samples[i].template == COMPARISON_TEMPLATES[0]
            assert samples[i].index == i

    @pytest.mark.parametrize("samples_per_comparison", [1, 2])
    def test_multiple_different_samples(self, samples_per_comparison: int) -> None:
        task_a, task_b = task_record_factory([TaskType.regular] * 2)
        comparison_0 = ((task_a,), (task_b,))
        comparison_1 = ((task_b,), (task_a,))
        comparisons = [comparison_0, comparison_1]
        samples = list(
            generate_samples(comparisons, COMPARISON_TEMPLATES, samples_per_comparison)
        )
        assert len(samples) == (
            len(comparisons) * len(COMPARISON_TEMPLATES) * samples_per_comparison
        )
        outer_index = 0
        for comparison in comparisons:
            for template in COMPARISON_TEMPLATES:
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
