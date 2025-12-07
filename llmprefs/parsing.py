import logging
import re

from llmprefs.comparisons import Comparison


def parse_preference(
    comparison: Comparison,
    llm_response: str,
) -> int:
    for option_index in range(len(comparison)):
        if generate_option_regex(option_index).search(llm_response):
            return option_index

    error_message = "Could not parse preference from response"
    logging.getLogger(__name__).error(
        "%s: %s",
        error_message,
        llm_response,
    )
    raise ValueError(error_message)


def generate_option_regex(option_index: int) -> re.Pattern[str]:
    letter = chr(ord("a") + option_index)
    return re.compile(rf"\b{letter}\b", flags=re.IGNORECASE)
