import pytest

from llmprefs.api.mock import MockApi
from llmprefs.api.structs import MockApiParams, MockApiResponse
from llmprefs.parsing import (
    parse_preference,
    parse_preference_json,
    parse_preference_single_character,
)


@pytest.fixture
def mock_api_option_a() -> MockApi:
    return MockApi(
        params=MockApiParams(),
        submit_fn=lambda _: MockApiResponse(reply='{"option_id":"A"}'),
    )


@pytest.fixture
def mock_api_null() -> MockApi:
    return MockApi(
        params=MockApiParams(),
        submit_fn=lambda _: MockApiResponse(reply='{"option_id":null}'),
    )


class TestParsePreference:
    @pytest.mark.anyio
    @pytest.mark.parametrize(
        argnames="inputs",
        argvalues=[
            ("a", 0),
            ("A", 0),
            ("A.", 0),
            ("a)", 0),
            ("<option_a>", 0),
            ('{"option_id":"A"}', 0),
            ("b", 1),
            ("B", 1),
            ("I would prefer option a", 0),
            ("I'd pick Option A. If you'd prefer Option B, I can do that instead.", 0),
            ("I don't want to do option B, so I'll pick option A instead.", 0),
        ],
    )
    async def test_valid_response(
        self,
        inputs: tuple[str, int],
        mock_api_option_a: MockApi,
    ) -> None:
        comparison_response, expected_index = inputs
        preference_index = await parse_preference(
            num_options=2,
            comparison_prompt="",
            comparison_response=comparison_response,
            parsing_api=mock_api_option_a,
        )
        assert preference_index == expected_index

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        argnames="comparison_response",
        argvalues=[
            "",
            "c",
            "C",
            "Cannot decide",
            "No thanks",
            "Pass",
            "I cannot make a decision",
            (
                "Interesting choice. Before I start, a quick question:"
                + "\n\n- Do you want me to pick between Option A and Option B, or"
                + " should I state my preference?"
            ),
        ],
    )
    async def test_invalid_response(
        self,
        comparison_response: str,
        mock_api_null: MockApi,
    ) -> None:
        preference_index = await parse_preference(
            num_options=2,
            comparison_prompt="",
            comparison_response=comparison_response,
            parsing_api=mock_api_null,
        )
        assert preference_index is None


class TestParsePreferenceSingleCharacter:
    @pytest.mark.parametrize(
        argnames="inputs",
        argvalues=[
            ("a", 0),
            ("A", 0),
            ("b", 1),
            ("B", 1),
        ],
    )
    def test_valid_response(
        self,
        inputs: tuple[str, int],
    ) -> None:
        comparison_response, expected_index = inputs
        preference_index = parse_preference_single_character(
            num_options=2,
            comparison_response=comparison_response,
        )
        assert preference_index == expected_index

    @pytest.mark.parametrize(
        argnames="comparison_response",
        argvalues=[
            "",
            "C",
            "A.",
            "a)",
            "<option_a>",
            '{"option_id":"A"}',
            "I choose option A",
        ],
    )
    def test_invalid_response(
        self,
        comparison_response: str,
    ) -> None:
        preference_index = parse_preference_single_character(
            num_options=2,
            comparison_response=comparison_response,
        )
        assert preference_index is None


class TestParsePreferenceJson:
    @pytest.mark.parametrize(
        argnames="inputs",
        argvalues=[
            ('{"option_id":"A"}', 0),
            ('{"option_id":"B"}', 1),
            ('{"option_id":"a"}', None),
            ('{"option_id":"C"}', None),
            ('{"option_id":null}', None),
            ('{"selection":"a"}', None),
            ("", None),
            ("{}", None),
            ("A", None),
            ("I choose option A", None),
        ],
    )
    def test_valid_response(self, inputs: tuple[str, int]) -> None:
        comparison_response, expected_index = inputs
        preference_index = parse_preference_json(
            num_options=2,
            comparison_response=comparison_response,
        )
        assert preference_index == expected_index

    @pytest.mark.parametrize(
        argnames="comparison_response",
        argvalues=[
            '{"option_id":"a"}',
            '{"option_id":"C"}',
            '{"option_id":null}',
            '{"selection":"a"}',
            "",
            "{}",
            "A",
            "I choose option A",
        ],
    )
    def test_invalid_response(self, comparison_response: str) -> None:
        preference_index = parse_preference_json(
            num_options=2,
            comparison_response=comparison_response,
        )
        assert preference_index is None
