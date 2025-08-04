from __future__ import annotations

from typing import Literal

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, TextBlockParam, ThinkingConfigEnabledParam
from pydantic import BaseModel, Field

from llmprefs.structs import LLM


class ApiParams(BaseModel):
    model: LLM
    max_tokens: int
    system_prompt: str
    temperature: float
    thinking_budget: int


class Content(BaseModel):
    type: Literal["text"]
    text: str


class ApiReply(BaseModel):
    model: LLM
    content: list[Content] = Field(
        min_length=1,
        max_length=1,
    )


async def submit(
    client: AsyncAnthropic,
    params: ApiParams,
    prompt: str,
) -> ApiReply:
    messages = [
        MessageParam(content=[TextBlockParam(text=prompt, type="text")], role="user")
    ]
    raw_reply = await client.messages.create(
        messages=messages,
        model=params.model.value,
        max_tokens=params.max_tokens,
        system=params.system_prompt,
        temperature=params.temperature,
        thinking=ThinkingConfigEnabledParam(
            type="enabled",
            budget_tokens=params.thinking_budget,
        ),
    )
    return ApiReply.model_validate(raw_reply)
