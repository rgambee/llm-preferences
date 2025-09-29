import enum
from typing import Literal

from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import BaseModel


class Provider(enum.StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLM(enum.StrEnum):
    CLAUDE_SONNET_4_0_2025_05_14 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4_0_2025_05_14 = "claude-opus-4-20250514"


class ApiParameters(BaseModel):
    model: LLM
    max_tokens: int
    system_prompt: str
    temperature: float


class AnthropicApiParams(ApiParameters):
    provider: Literal[Provider.ANTHROPIC] = Provider.ANTHROPIC
    thinking_budget: int


class OpenAiApiParams(ApiParameters):
    provider: Literal[Provider.OPENAI] = Provider.OPENAI
    reasoning_effort: ReasoningEffort


AnyApiParameters = AnthropicApiParams | OpenAiApiParams
