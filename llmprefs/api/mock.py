from collections.abc import Callable

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import MockApiParams, MockApiResponse


class MockApi(BaseApi[MockApiResponse]):
    def __init__(
        self,
        params: MockApiParams,
        submit_fn: Callable[[str], MockApiResponse] | None = None,
    ) -> None:
        self._params = params
        if submit_fn is None:
            submit_fn = default_submit_fn
        self._submit_fn = submit_fn

    @property
    def params(self) -> MockApiParams:
        return self._params

    async def submit(self, prompt: str) -> MockApiResponse:
        return self._submit_fn(prompt)


def default_submit_fn(prompt: str) -> MockApiResponse:
    return MockApiResponse(reply=prompt)
