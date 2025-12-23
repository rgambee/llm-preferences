import logging
import re


def parse_preference(
    num_options: int,
    llm_response: str,
) -> int | None:
    first_match: re.Match[str] | None = None
    option_index_for_first_match: int | None = None
    for option_index in range(num_options):
        pattern = generate_option_regex(option_index)
        match = pattern.search(llm_response)
        if match is None:
            continue
        if first_match is None or match.start() < first_match.start():
            first_match = match
            option_index_for_first_match = option_index

    if option_index_for_first_match is not None:
        return option_index_for_first_match

    logging.getLogger(__name__).warning(
        "Could not parse preference from response: %s",
        llm_response,
    )
    return None


def generate_option_regex(option_index: int) -> re.Pattern[str]:
    letter = chr(ord("a") + option_index)
    return re.compile(rf"(?:\b|_){letter}\b", flags=re.IGNORECASE)
