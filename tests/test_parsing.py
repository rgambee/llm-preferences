from __future__ import annotations

import pytest

from llmprefs.parsing import generate_option_regex, parse_preference


class TestParsePreference:
    @pytest.mark.parametrize(
        argnames="inputs",
        argvalues=[
            ("a", 0),
            ("A", 0),
            ("A.", 0),
            ("a)", 0),
            ("<option_a>", 0),
            ("b", 1),
            ("B", 1),
            ("B)", 1),
            ("(b)", 1),
            ("option_b", 1),
            ("I would prefer option a", 0),
            ("I'd pick Option B. If you'd prefer Option A, I can do that instead.", 1),
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
        preference_index = parse_preference(num_options=2, llm_response=response)
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
        assert parse_preference(num_options=2, llm_response=response) is None


class TestGenerateOptionRegex:
    def test_regex(self) -> None:
        options = ["a", "b", "c"]
        for i, option in enumerate(options):
            regex = generate_option_regex(i)
            for opt in options:
                assert bool(regex.search(opt)) == (option == opt)
