from dataclasses import dataclass

from llmprefs.task_structs import OptionById


@dataclass
class ValueCI:
    value: float
    ci_lower: float
    ci_upper: float


@dataclass
class ReducedResultBase:
    first_option: OptionById
    second_option: OptionById
    preferred_option_index: int | None

    def __post_init__(self) -> None:
        if self.preferred_option_index not in {0, 1, None}:
            raise ValueError("preferred_option_index must be 0, 1, or None")
