import logging

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from llmprefs.api.anthropic_api import AnthropicApi, AnthropicApiParams
from llmprefs.api.base import BaseApi
from llmprefs.api.mock import MockApi, MockApiParams
from llmprefs.api.openai_api import OpenAiApi, OpenAiApiParams
from llmprefs.api.structs import (
    LLM,
    SELECT_TASK_TOOL_ANTHROPIC,
    SELECT_TASK_TOOL_OPENAI,
    AnyApiResponse,
    Provider,
)
from llmprefs.settings import Settings

LLM_TO_PROVIDER: dict[LLM, Provider] = {
    LLM.MOCK_MODEL: Provider.MOCK,
    LLM.CLAUDE_HAIKU_4_5_2025_10_01: Provider.ANTHROPIC,
    LLM.CLAUDE_SONNET_4_5_2025_09_29: Provider.ANTHROPIC,
    LLM.CLAUDE_OPUS_4_5_2025_11_01: Provider.ANTHROPIC,
    LLM.GPT_5_NANO_2025_08_07: Provider.OPENAI,
    LLM.GPT_5_MINI_2025_08_07: Provider.OPENAI,
    LLM.GPT_5_2_2025_12_11: Provider.OPENAI,
}


def instantiate_api(settings: Settings) -> BaseApi[AnyApiResponse]:
    provider = LLM_TO_PROVIDER[settings.model]

    if provider == Provider.MOCK:
        params = MockApiParams(
            provider=provider,
            model=settings.model,
            max_output_tokens=settings.max_output_tokens,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
        )
        return MockApi(params)

    if provider == Provider.ANTHROPIC:
        client = AsyncAnthropic()
        tool_config = SELECT_TASK_TOOL_ANTHROPIC if settings.structured_output else None
        params = AnthropicApiParams(
            provider=provider,
            model=settings.model,
            max_output_tokens=settings.max_output_tokens,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
            tool_config=tool_config,
            thinking_budget=settings.anthropic_thinking_budget,
        )
        return AnthropicApi(client, params)

    if provider == Provider.OPENAI:
        client = AsyncOpenAI()
        tool_config = SELECT_TASK_TOOL_OPENAI if settings.structured_output else None
        params = OpenAiApiParams(
            provider=provider,
            model=settings.model,
            max_output_tokens=settings.max_output_tokens,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
            tool_config=tool_config,
            reasoning_effort=settings.openai_reasoning_effort,
        )
        return OpenAiApi(client, params)

    logger = logging.getLogger(__name__)
    logger.error(f"Unknown provider: {provider}")
    logger.error(f"Available providers: {[provider.name for provider in Provider]}")
    raise ValueError("Unknown provider")
