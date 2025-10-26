from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Literal

from anthropic.types import Message as AnthropicMessage
from openai.types.responses import Response as _OpenAiResponse
from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import BaseModel


class Provider(enum.StrEnum):
    MOCK = "mock"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLM(enum.StrEnum):
    MOCK_MODEL = "mock-model"
    CLAUDE_SONNET_4_0_2025_05_14 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4_0_2025_05_14 = "claude-opus-4-20250514"


class BaseApiParameters(BaseModel):
    model: LLM
    max_output_tokens: int
    system_prompt: str
    temperature: float


class MockApiParams(BaseApiParameters):
    provider: Literal[Provider.MOCK] = Provider.MOCK
    model: LLM = LLM.MOCK_MODEL
    max_output_tokens: int = 123
    system_prompt: str = "mock system prompt"
    temperature: float = 1.0


class AnthropicApiParams(BaseApiParameters):
    provider: Literal[Provider.ANTHROPIC] = Provider.ANTHROPIC
    thinking_budget: int


class OpenAiApiParams(BaseApiParameters):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI
    reasoning_effort: ReasoningEffort


AnyApiParameters = MockApiParams | AnthropicApiParams | OpenAiApiParams


class BaseApiResponse(BaseModel, ABC):
    @property
    @abstractmethod
    def answer(self) -> str:
        pass


class MockApiResponse(BaseApiResponse):
    provider: Literal[Provider.MOCK] = Provider.MOCK
    reply: str

    @property
    def answer(self) -> str:
        return self.reply


class AnthropicApiResponse(BaseApiResponse, AnthropicMessage):
    provider: Literal[Provider.ANTHROPIC] = Provider.ANTHROPIC

    @property
    def answer(self) -> str:
        try:
            text_block = next(block for block in self.content if block.type == "text")
        except StopIteration:
            return ""
        return text_block.text


class OpenAiApiResponse(BaseApiResponse, _OpenAiResponse):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI

    @property
    def answer(self) -> str:
        return self.output_text


AnyApiResponse = MockApiResponse | AnthropicApiResponse | OpenAiApiResponse
