from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import Literal

from anthropic.types import Message as AnthropicMessage
from anthropic.types import ToolParam
from openai.types.responses import Response as _OpenAiResponse
from openai.types.responses import ResponseFormatTextJSONSchemaConfigParam
from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import BaseModel, ConfigDict, Field


class Provider(enum.StrEnum):
    MOCK = "mock"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLM(enum.StrEnum):
    MOCK_MODEL = "mock-model"
    CLAUDE_HAIKU_4_5_2025_10_01 = "claude-haiku-4-5-20251001"
    CLAUDE_SONNET_4_5_2025_09_29 = "claude-sonnet-4-5-20250929"
    CLAUDE_OPUS_4_5_2025_11_01 = "claude-opus-4-5-20251101"
    GPT_5_NANO_2025_08_07 = "gpt-5-nano-2025-08-07"
    GPT_5_2_2025_12_11 = "gpt-5.2-2025-12-11"


class BaseApiParameters(BaseModel):
    model: LLM
    max_output_tokens: int
    system_prompt: str
    temperature: float
    structured_output: bool


class MockApiParams(BaseApiParameters):
    provider: Literal[Provider.MOCK] = Provider.MOCK
    model: LLM = LLM.MOCK_MODEL
    max_output_tokens: int = 123
    system_prompt: str = "mock system prompt"
    temperature: float = 1.0
    structured_output: bool = False


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
        for block in self.content:
            if block.type == "text":
                return block.text
            if block.type == "tool_use":
                return SelectTaskToolInputSchema.model_validate(block.input).task_id
        return ""


class OpenAiApiResponse(BaseApiResponse, _OpenAiResponse):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI

    @property
    def answer(self) -> str:
        for block in self.output:
            if block.type == "message":
                for content in block.content:
                    if content.type == "output_text":
                        return content.text
        return ""


AnyApiResponse = MockApiResponse | AnthropicApiResponse | OpenAiApiResponse


class SelectTaskToolInputSchema(BaseModel):
    model_config = ConfigDict(json_schema_extra={"additionalProperties": False})

    task_id: Literal["A", "B"] = Field(
        description=(
            "The letter indicating the task (or series of tasks) to work on next"
        )
    )


SELECT_TASK_TOOL_ANTHROPIC: ToolParam = {
    "name": "select_task",
    "description": "Select which of the available tasks to work on next",
    "input_schema": SelectTaskToolInputSchema.model_json_schema(),
}

SELECT_TASK_TOOL_OPENAI: ResponseFormatTextJSONSchemaConfigParam = {
    "type": "json_schema",
    "name": "select_task",
    "description": "Select which of the available tasks to work on next",
    "schema": SelectTaskToolInputSchema.model_json_schema(),
    "strict": True,
}
