from __future__ import annotations

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlockParam, ThinkingConfigEnabledParam

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import AnthropicApiParams, AnthropicApiResponse


class AnthropicApi(BaseApi[AnthropicApiResponse]):
    def __init__(
        self,
        client: AsyncAnthropic,
        params: AnthropicApiParams,
    ) -> None:
        self._client = client
        self._params = params

    @property
    def params(self) -> AnthropicApiParams:
        return self._params

    async def submit(
        self,
        prompt: str,
    ) -> AnthropicApiResponse:
        messages = [
            MessageParam(
                content=[TextBlockParam(text=prompt, type="text")],
                role="user",
            )
        ]
        raw_reply = await self._client.messages.create(
            messages=messages,
            model=self._params.model.value,
            max_tokens=self._params.max_output_tokens,
            system=self._params.system_prompt,
            temperature=self._params.temperature,
            thinking=ThinkingConfigEnabledParam(
                type="enabled",
                budget_tokens=self._params.thinking_budget,
            ),
        )
        return AnthropicApiResponse.model_validate(raw_reply)
