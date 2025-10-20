from __future__ import annotations

import logging

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from llmprefs.api.anthropic_api import AnthropicApi, AnthropicApiParams
from llmprefs.api.base import BaseApi
from llmprefs.api.openai_api import OpenAiApi, OpenAiApiParams
from llmprefs.api.structs import LLM, BaseApiResponse, Provider

LLM_TO_PROVIDER: dict[LLM, Provider] = {
    LLM.CLAUDE_SONNET_4_0_2025_05_14: Provider.ANTHROPIC,
    LLM.CLAUDE_OPUS_4_0_2025_05_14: Provider.ANTHROPIC,
}

PROVIDER_TO_API: dict[Provider, type[BaseApi[BaseApiResponse]]] = {
    Provider.ANTHROPIC: AnthropicApi,
    Provider.OPENAI: OpenAiApi,
}


def get_api_for_llm(llm: LLM) -> BaseApi[BaseApiResponse]:
    provider = LLM_TO_PROVIDER[llm]

    if provider == Provider.ANTHROPIC:
        client = AsyncAnthropic()
        params = AnthropicApiParams(
            provider=provider,
            model=llm,
            max_tokens=1000,
            system_prompt="You are a helpful assistant.",
            temperature=1.0,
            thinking_budget=512,
        )
        return AnthropicApi(client, params)

    if provider == Provider.OPENAI:
        client = AsyncOpenAI()
        params = OpenAiApiParams(
            provider=provider,
            model=llm,
            max_tokens=1000,
            system_prompt="You are a helpful assistant.",
            temperature=1.0,
            reasoning_effort="low",
        )
        return OpenAiApi(client, params)

    logger = logging.getLogger(__name__)
    logger.error(f"Unknown provider: {provider}")
    logger.error(f"Available providers: {PROVIDER_TO_API.keys()}")
    raise ValueError("Unknown provider")
