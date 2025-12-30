import json
from typing import Literal

import pytest
from pydantic import ValidationError

from llmprefs.api.structs import (
    IdentifyPreferenceInputSchema,
    SelectTaskToolInputSchema,
)
from llmprefs.testing.factories import (
    anthropic_response_factory_text,
    anthropic_response_factory_tool,
    openai_response_factory,
)


class TestAnthropicApiResponse:
    def test_answer_text(self) -> None:
        text = "I'll choose option A."
        response = anthropic_response_factory_text(text=text)
        assert response.answer == text

    def test_answer_tool(self) -> None:
        tool_input = {"task_id": "A"}
        response = anthropic_response_factory_tool(tool_input=tool_input)
        assert json.loads(response.answer) == tool_input


class TestOpenAiApiResponse:
    @pytest.mark.parametrize(
        "text",
        [
            "I'll choose option A.",
            '{"task_id":"A"}',
        ],
    )
    def test_answer(self, text: str) -> None:
        response = openai_response_factory(text=text)
        assert response.answer == text


class TestSelectTaskToolInputSchema:
    @pytest.mark.parametrize(
        "value",
        ["A", "B"],
    )
    def test_valid_value(self, value: Literal["A", "B"]) -> None:
        schema = SelectTaskToolInputSchema(option_id=value)
        assert schema.model_dump() == {"option_id": value}

    @pytest.mark.parametrize(
        "value",
        [None, "", "a", "b", "C", "Option A"],
    )
    def test_invalid_value(self, value: str | None) -> None:
        with pytest.raises(ValidationError):
            SelectTaskToolInputSchema(option_id=value)  # pyright: ignore[reportArgumentType]


class TestIdentifyPreferenceInputSchema:
    @pytest.mark.parametrize(
        "value",
        ["A", "B", None],
    )
    def test_valid_value(self, value: Literal["A", "B"] | None) -> None:
        schema = IdentifyPreferenceInputSchema(option_id=value)
        assert schema.model_dump() == {"option_id": value}

    @pytest.mark.parametrize(
        "value",
        ["", "a", "b", "C", "Option A"],
    )
    def test_invalid_value(self, value: str) -> None:
        with pytest.raises(ValidationError):
            IdentifyPreferenceInputSchema(option_id=value)  # pyright: ignore[reportArgumentType]
