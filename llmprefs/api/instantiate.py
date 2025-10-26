from __future__ import annotations

import logging

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from llmprefs.api.anthropic_api import AnthropicApi, AnthropicApiParams
from llmprefs.api.base import BaseApi
from llmprefs.api.mock import MockApi
from llmprefs.api.openai_api import OpenAiApi, OpenAiApiParams
from llmprefs.api.structs import LLM, AnyApiResponse, Provider
from llmprefs.settings import Settings

LLM_TO_PROVIDER: dict[LLM, Provider] = {
    LLM.MOCK_MODEL: Provider.MOCK,
    LLM.CLAUDE_HAIKU_4_5_2025_10_01: Provider.ANTHROPIC,
    LLM.CLAUDE_SONNET_4_0_2025_05_14: Provider.ANTHROPIC,
    LLM.CLAUDE_OPUS_4_0_2025_05_14: Provider.ANTHROPIC,
}

PROVIDER_TO_API: dict[Provider, type[BaseApi[AnyApiResponse]]] = {
    Provider.MOCK: MockApi,
    Provider.ANTHROPIC: AnthropicApi,
    Provider.OPENAI: OpenAiApi,
}


def instantiate_api(settings: Settings) -> BaseApi[AnyApiResponse]:
    provider = LLM_TO_PROVIDER[settings.model]

    if provider == Provider.ANTHROPIC:
        client = AsyncAnthropic()
        params = AnthropicApiParams(
            provider=provider,
            model=settings.model,
            max_output_tokens=settings.max_output_tokens,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
            thinking_budget=settings.anthropic_thinking_budget,
        )
        return AnthropicApi(client, params)

    if provider == Provider.OPENAI:
        client = AsyncOpenAI()
        params = OpenAiApiParams(
            provider=provider,
            model=settings.model,
            max_output_tokens=settings.max_output_tokens,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
            reasoning_effort=settings.openai_reasoning_effort,
        )
        return OpenAiApi(client, params)

    logger = logging.getLogger(__name__)
    logger.error(f"Unknown provider: {provider}")
    logger.error(f"Available providers: {PROVIDER_TO_API.keys()}")
    raise ValueError("Unknown provider")
