from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import MockApiParams, MockApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.pipeline import chunked, run_pipeline
from llmprefs.task_structs import TaskType
from llmprefs.testing.factories import task_record_factory


class TestRunPipeline:
    @pytest.mark.anyio
    async def test_empty(self) -> None:
        api = AsyncMock(spec=BaseApi)
        comparisons: list[Comparison] = []
        concurrent_requests = 1
        results = run_pipeline(api, comparisons, concurrent_requests)
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
        results = run_pipeline(api, [comparison], concurrent_requests=1)

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
        results = run_pipeline(api, [comparison_0, comparison_1], concurrent_requests=1)

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
        results = run_pipeline(api, [comparison_0, comparison_1], concurrent_requests=2)

        index = 0
        async for result in results:
            await asyncio.sleep(0)
            assert index < 2
            assert result.preferred_option_index == 0
            assert api.submit.await_count == 2
            index += 1
        assert index == 2


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
