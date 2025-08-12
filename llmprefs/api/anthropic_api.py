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
        self.client = client
        self.params = params

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
        raw_reply = await self.client.messages.create(
            messages=messages,
            model=self.params.model.value,
            max_tokens=self.params.max_tokens,
            system=self.params.system_prompt,
            temperature=self.params.temperature,
            thinking=ThinkingConfigEnabledParam(
                type="enabled",
                budget_tokens=self.params.thinking_budget,
            ),
        )
        reply = ApiReply.model_validate(raw_reply)
        return reply.content[0].text
