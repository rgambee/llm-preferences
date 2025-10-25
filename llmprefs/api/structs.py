from __future__ import annotations

import enum
from typing import Literal

from anthropic.types import Message as AnthropicMessage
from openai.types.responses import Response as _OpenAiResponse
from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import BaseModel


class Provider(enum.StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLM(enum.StrEnum):
    CLAUDE_SONNET_4_0_2025_05_14 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4_0_2025_05_14 = "claude-opus-4-20250514"


class BaseApiParameters(BaseModel):
    model: LLM
    max_tokens: int
    system_prompt: str
    temperature: float


class AnthropicApiParams(BaseApiParameters):
    provider: Literal[Provider.ANTHROPIC] = Provider.ANTHROPIC
    thinking_budget: int


class OpenAiApiParams(BaseApiParameters):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI
    reasoning_effort: ReasoningEffort


AnyApiParameters = AnthropicApiParams | OpenAiApiParams


class AnthropicApiResponse(AnthropicMessage):
    provider: Literal[Provider.ANTHROPIC] = Provider.ANTHROPIC

    @property
    def answer(self) -> str:
        try:
            text_block = next(block for block in self.content if block.type == "text")
        except StopIteration:
            return ""
        return text_block.text


class OpenAiApiResponse(_OpenAiResponse):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI

    @property
    def answer(self) -> str:
        return self.output_text


AnyApiResponse = AnthropicApiResponse | OpenAiApiResponse
