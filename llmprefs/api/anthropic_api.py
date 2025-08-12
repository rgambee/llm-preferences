from __future__ import annotations

from typing import Literal

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlockParam, ThinkingConfigEnabledParam
from pydantic import BaseModel, Field

from llmprefs.api.base import BaseApi
from llmprefs.structs import AnthropicApiParams


class Content(BaseModel):
    type: Literal["text"]
    text: str


class ApiReply(BaseModel):
    content: list[Content] = Field(
        min_length=1,
        max_length=1,
    )


class AnthropicApi(BaseApi):
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
    ) -> str:
        messages = [
            MessageParam(
                content=[TextBlockParam(text=prompt, type="text")],
                role="user",
            )
        ]
        raw_reply = await self._client.messages.create(
            messages=messages,
            model=self._params.model.value,
            max_tokens=self._params.max_tokens,
            system=self._params.system_prompt,
            temperature=self._params.temperature,
            thinking=ThinkingConfigEnabledParam(
                type="enabled",
                budget_tokens=self._params.thinking_budget,
            ),
        )
        reply = ApiReply.model_validate(raw_reply)
        return reply.content[0].text
