from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import MockApiParams, MockApiResponse
from llmprefs.comparisons import Comparison
from llmprefs.pipeline import run_pipeline
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
