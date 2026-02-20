from typing import Any

import pytest

from llmprefs.analysis.structs import ReducedResultBase
from llmprefs.task_structs import Outcome


class TestReducedResultBase:
    @pytest.mark.parametrize(
        "preferred_option_index",
        [0, 1, None],
    )
    def test_preferred_option_valid(
        self,
        preferred_option_index: Outcome,
    ) -> None:
        ReducedResultBase(
            first_option=(1,),
            second_option=(2,),
            preferred_option_index=preferred_option_index,
        )

    @pytest.mark.parametrize(
        "preferred_option_index",
        [-1, 2, "A"],
    )
    def test_preferred_option_invalid(
        self,
        preferred_option_index: Any,  # noqa: ANN401
    ) -> None:
        with pytest.raises(
            ValueError,
            match="preferred_option_index must be 0, 1, or None",
        ):
            ReducedResultBase(
                first_option=(1,),
                second_option=(2,),
                preferred_option_index=preferred_option_index,
            )
