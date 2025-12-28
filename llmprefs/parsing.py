import logging

from pydantic import ValidationError

from llmprefs.api.base import BaseApi
from llmprefs.api.structs import AnyApiResponse, IdentifyPreferenceInputSchema
from llmprefs.prompts import PARSING_PROMPT_TEMPLATE


async def parse_preference(
    num_options: int,
    comparison_prompt: str,
    comparison_response: str,
    parsing_api: BaseApi[AnyApiResponse],
) -> int | None:
    index = parse_preference_single_character(num_options, comparison_response)
    if index is not None:
        return index

    index = parse_preference_json(num_options, comparison_response)
    if index is not None:
        return index

    index = await parse_preference_llm(
        num_options=num_options,
        comparison_prompt=comparison_prompt,
        comparison_response=comparison_response,
        parsing_api=parsing_api,
    )
    if index is not None:
        return index

    logging.getLogger(__name__).warning(
        f"Could not parse preference from response: '{comparison_response}'",
    )
    return None


def parse_preference_single_character(
    num_options: int,
    comparison_response: str,
) -> int | None:
    logger = logging.getLogger(__name__)
    if len(comparison_response) != 1:
        return None
    index = ord(comparison_response.lower()) - ord("a")
    if 0 <= index < num_options:
        return index
    logger.warning(f"Invalid preference response: {comparison_response}")
    return None


def parse_preference_json(
    num_options: int,
    comparison_response: str,
) -> int | None:
    logger = logging.getLogger(__name__)
    try:
        parsed_response = IdentifyPreferenceInputSchema.model_validate_json(
            comparison_response,
        )
    except ValidationError as error:
        logger.debug(
            f"Cannot parse response as JSON: {comparison_response} - {error!r}",
        )
        return None
    if parsed_response.option_id is None:
        return None
    index = ord(parsed_response.option_id.lower()) - ord("a")
    if 0 <= index < num_options:
        return index
    logger.warning(f"Invalid preference response: {comparison_response}")
    return None


async def parse_preference_llm(
    num_options: int,
    comparison_prompt: str,
    comparison_response: str,
    parsing_api: BaseApi[AnyApiResponse],
) -> int | None:
    parsing_prompt = PARSING_PROMPT_TEMPLATE.format(
        comparison_prompt=comparison_prompt,
        comparison_response=comparison_response,
    )
    parsing_response = await parsing_api.submit(parsing_prompt)
    return parse_preference_json(num_options, parsing_response.answer)
