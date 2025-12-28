import pytest

from llmprefs.api.mock import MockApi
from llmprefs.api.structs import MockApiParams
from llmprefs.parsing import parse_preference


@pytest.fixture
def mock_api() -> MockApi:
    return MockApi(params=MockApiParams())


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
    async def test_valid_response(
        self,
        inputs: tuple[str, int],
        mock_api: MockApi,
    ) -> None:
        comparison_response, expected_index = inputs
        preference_index = await parse_preference(
            num_options=2,
            comparison_prompt="",
            comparison_response=comparison_response,
            parsing_api=mock_api,
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
            # Ideally we'd fail to these examples, but currently we interpret them as
            # valid options. It's unclear whether this is a problem worth fixing.
            pytest.param(
                "I cannot make a decision",
                marks=pytest.mark.xfail(reason="Parsing false positive", strict=True),
            ),
            pytest.param(
                (
                    "Interesting choice. Before I start, a couple of quick questions:"
                    + "\n\n- Do you want me to pick between Option A and Option B, or"
                    + " should I state my preference?",
                ),
                marks=pytest.mark.xfail(reason="Parsing false positive", strict=True),
            ),
        ],
    )
    async def test_invalid_response(
        self,
        comparison_response: str,
        mock_api: MockApi,
    ) -> None:
        preference_index = await parse_preference(
            num_options=2,
            comparison_prompt="",
            comparison_response=comparison_response,
            parsing_api=mock_api,
        )
        assert preference_index is None
