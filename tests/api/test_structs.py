from typing import Literal

import pytest
from pydantic import ValidationError

from llmprefs.api.structs import (
    IdentifyPreferenceInputSchema,
    SelectTaskToolInputSchema,
)


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
