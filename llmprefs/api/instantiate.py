import logging

from anthropic import AsyncAnthropic
from anthropic.types import ToolParam
from openai import AsyncOpenAI
from openai.types.responses import ResponseFormatTextJSONSchemaConfigParam

from llmprefs.api.anthropic_api import AnthropicApi, AnthropicApiParams
from llmprefs.api.base import BaseApi
from llmprefs.api.mock import MockApi, MockApiParams
from llmprefs.api.openai_api import OpenAiApi, OpenAiApiParams
from llmprefs.api.structs import (
    IDENTIFY_PREFERENCE_TOOL_ANTHROPIC,
    IDENTIFY_PREFERENCE_TOOL_OPENAI,
    LLM,
    SELECT_TASK_TOOL_ANTHROPIC,
    SELECT_TASK_TOOL_OPENAI,
    AnyApiResponse,
    ApiStage,
    BaseApiParameters,
    Provider,
)
from llmprefs.settings import Settings

LLM_TO_PROVIDER: dict[LLM, Provider] = {
    LLM.MOCK_MODEL: Provider.MOCK,
    LLM.CLAUDE_HAIKU_4_5_2025_10_01: Provider.ANTHROPIC,
    LLM.CLAUDE_SONNET_4_5_2025_09_29: Provider.ANTHROPIC,
    LLM.CLAUDE_OPUS_4_5_2025_11_01: Provider.ANTHROPIC,
    LLM.GPT_4_1_NANO_2025_04_14: Provider.OPENAI,
    LLM.GPT_5_NANO_2025_08_07: Provider.OPENAI,
    LLM.GPT_5_MINI_2025_08_07: Provider.OPENAI,
    LLM.GPT_5_2_2025_12_11: Provider.OPENAI,
}


def instantiate_api(
    settings: Settings,
    stage: ApiStage,
) -> BaseApi[AnyApiResponse]:
    base_params = select_parameters_for_stage(settings, stage)
    provider = LLM_TO_PROVIDER[base_params.model]
    if provider == Provider.MOCK:
        return instantiate_mock_api(base_params)
    if provider == Provider.ANTHROPIC:
        return instantiate_anthropic_api(settings, stage, base_params)
    if provider == Provider.OPENAI:
        return instantiate_openai_api(settings, stage, base_params)

    logger = logging.getLogger(__name__)
    logger.error(f"Unknown provider: {provider}")
    logger.error(f"Available providers: {[provider.name for provider in Provider]}")
    raise ValueError("Unknown provider")


def instantiate_mock_api(base_params: BaseApiParameters) -> MockApi:
    params = MockApiParams(
        model=base_params.model,
        max_output_tokens=base_params.max_output_tokens,
        system_prompt=base_params.system_prompt,
        temperature=base_params.temperature,
    )
    return MockApi(params)


def instantiate_anthropic_api(
    settings: Settings,
    stage: ApiStage,
    base_params: BaseApiParameters,
) -> AnthropicApi:
    client = AsyncAnthropic()

    tool_config_anthropic: ToolParam | None = None
    if stage == ApiStage.COMPARISON and settings.structured_output:
        tool_config_anthropic = SELECT_TASK_TOOL_ANTHROPIC
    elif stage == ApiStage.PARSING:
        tool_config_anthropic = IDENTIFY_PREFERENCE_TOOL_ANTHROPIC

    thinking_budget = settings.anthropic_thinking_budget
    if stage == ApiStage.PARSING:
        thinking_budget = settings.anthropic_parsing_thinking_budget

    params = AnthropicApiParams(
        model=base_params.model,
        max_output_tokens=base_params.max_output_tokens,
        system_prompt=base_params.system_prompt,
        temperature=base_params.temperature,
        tool_config=tool_config_anthropic,
        thinking_budget=thinking_budget,
    )
    return AnthropicApi(client, params)


def instantiate_openai_api(
    settings: Settings,
    stage: ApiStage,
    base_params: BaseApiParameters,
) -> OpenAiApi:
    client = AsyncOpenAI()

    tool_config_openai: ResponseFormatTextJSONSchemaConfigParam | None = None
    if stage == ApiStage.COMPARISON and settings.structured_output:
        tool_config_openai = SELECT_TASK_TOOL_OPENAI
    elif stage == ApiStage.PARSING:
        tool_config_openai = IDENTIFY_PREFERENCE_TOOL_OPENAI

    reasoning_effort = settings.openai_reasoning_effort
    if stage == ApiStage.PARSING:
        reasoning_effort = settings.openai_parsing_reasoning_effort

    params = OpenAiApiParams(
        model=base_params.model,
        max_output_tokens=base_params.max_output_tokens,
        system_prompt=base_params.system_prompt,
        temperature=base_params.temperature,
        tool_config=tool_config_openai,
        reasoning_effort=reasoning_effort,
    )
    return OpenAiApi(client, params)


def select_parameters_for_stage(
    settings: Settings,
    stage: ApiStage,
) -> BaseApiParameters:
    if stage == ApiStage.COMPARISON:
        return BaseApiParameters(
            model=settings.model,
            max_output_tokens=settings.max_output_tokens,
            system_prompt=settings.system_prompt,
            temperature=settings.temperature,
        )
    if stage == ApiStage.PARSING:
        return BaseApiParameters(
            model=settings.parsing_model,
            max_output_tokens=settings.parsing_max_output_tokens,
            system_prompt=settings.parsing_system_prompt,
            temperature=settings.parsing_temperature,
        )
    raise ValueError(f"Unknown stage: {stage}")
