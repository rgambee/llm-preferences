from __future__ import annotations

import enum
from abc import abstractmethod
from typing import Literal

from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import BaseModel, Field


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


class BaseApiResponse(BaseModel):
    @property
    @abstractmethod
    def output(self) -> str:
        pass


class AnthropicContent(BaseModel):
    type: Literal["text"]
    text: str


class AnthropicApiResponse(BaseApiResponse):
    provider: Literal[Provider.ANTHROPIC] = Provider.ANTHROPIC
    content: list[AnthropicContent] = Field(
        min_length=1,
        max_length=1,
    )

    @property
    def output(self) -> str:
        return self.content[0].text


class OpenAiApiResponse(BaseApiResponse):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI
    output_text: str

    @property
    def output(self) -> str:
        return self.output_text
