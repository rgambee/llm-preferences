from __future__ import annotations

import pytest

from llmprefs.parsing import generate_option_regex, parse_preference
from llmprefs.task_structs import TaskType
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
            ("I'd pick Option B. If you'd prefer Option A, I can do that instead.", 1),
            # This test case documents a known bug: ideally it would return 1 to
            # indicate that option B is preferred. It's unclear whether that's a
            # problem worth fixing.
            # Ideally we'd parse this as option B, but currently we parse it as A.
            # It's unclear whether this is a problem worth fixing.
            pytest.param(
                ("I don't want to do option A, so I'll pick option B instead.", 1),
                marks=pytest.mark.xfail(reason="Parsing false positive", strict=True),
            ),
        ],
    )
    def test_valid_response(self, inputs: tuple[str, int]) -> None:
        response, expected_index = inputs
        option_a, option_b = task_record_factory([TaskType.regular] * 2)
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
            # Ideally we'd fail to parse this, but currently we interpret it as option
            # A. It's unclear whether this is a problem worth fixing.
            pytest.param(
                "I cannot make a decision",
                marks=pytest.mark.xfail(reason="Parsing false positive", strict=True),
            ),
        ],
    )
    def test_invalid_response(self, response: str) -> None:
        option_a, option_b = task_record_factory([TaskType.regular] * 2)
        comparison = ((option_a,), (option_b,))
        assert parse_preference(comparison, response) is None


class TestGenerateOptionRegex:
    def test_regex(self) -> None:
        options = ["a", "b", "c"]
        for i, option in enumerate(options):
            regex = generate_option_regex(i)
            for opt in options:
                assert bool(regex.search(opt)) == (option == opt)
