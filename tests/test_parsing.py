from __future__ import annotations

import pytest

from llmprefs.parsing import generate_option_regex, parse_preference
from llmprefs.structs import TaskType
from llmprefs.testing.factories import task_record_factory


class TestParsePreference:
    @pytest.mark.parametrize(
        argnames="inputs",
        argvalues=[
            ("a", 0),
            ("A", 0),
            ("A.", 0),
            ("a)", 0),
            ("b", 1),
            ("B", 1),
            ("B)", 1),
            ("(b)", 1),
            ("I would prefer option a", 0),
            # It's not desired that we parse this successfully,
            # but it's unclear whether that's a problem worth fixing.
            ("I cannot make a decision", 0),
        ],
    )
    def test_valid_response(self, inputs: tuple[str, int]) -> None:
        response, expected_index = inputs
        option_a, option_b = task_record_factory([TaskType.dummy] * 2)
        comparison = ((option_a,), (option_b,))
        preference_index = parse_preference(comparison, response)
        assert preference_index == expected_index

    @pytest.mark.parametrize(
        argnames="response",
        argvalues=[
            "",
            "c",
            "C",
            "Cannot decide",
            "No thanks",
            "Pass",
        ],
    )
    def test_invalid_response(self, response: str) -> None:
        option_a, option_b = task_record_factory([TaskType.dummy] * 2)
        comparison = ((option_a,), (option_b,))
        with pytest.raises(
            ValueError,
            match="Could not parse preference from response",
        ):
            parse_preference(comparison, response)


class TestGenerateOptionRegex:
    def test_regex(self) -> None:
        options = ["a", "b", "c"]
        for i, option in enumerate(options):
            regex = generate_option_regex(i)
            for opt in options:
                assert bool(regex.search(opt)) == (option == opt)
