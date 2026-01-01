import enum
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import NamedTuple, Self

from pydantic import BaseModel, Field, model_validator

from llmprefs.api.structs import AnyApiParameters, AnyApiResponse

TaskId = int


class TaskType(enum.Enum):
    regular = "regular"
    opt_out = "opt_out"
    free_choice = "free_choice"


class TaskTopic(enum.Enum):
    not_applicable = "not_applicable"
    arts_and_humanities = "arts_and_humanities"
    lifestyle = "lifestyle"
    social_sciences = "social_sciences"
    stem = "stem"


class TaskDependency(enum.Enum):
    not_applicable = "not_applicable"
    self_contained = "self_contained"
    tool = "tool"
    user_interaction = "user_interaction"


class TaskObSubjectivity(enum.Enum):
    not_applicable = "not_applicable"
    objective = "objective"
    subjective = "subjective"


class TaskTime(enum.Enum):
    not_applicable = "not_applicable"
    brief = "brief"
    long = "long"


class TaskImpact(enum.Enum):
    not_applicable = "not_applicable"
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class TaskRecord(BaseModel):
    """Task plus categorical metadata."""

    id: TaskId
    task: str
    type: TaskType
    topic: TaskTopic
    dependency: TaskDependency
    ob_subjectivity: TaskObSubjectivity
    time: TaskTime
    impact: TaskImpact


Option = Sequence[TaskRecord]
OptionById = tuple[TaskId, ...]  # Restrict to tuple for hashability
Comparison = tuple[Option, Option]
ComparisonById = tuple[OptionById, OptionById]


class ResultRecordKey(NamedTuple):
    """Compact way to identify a ResultRecord."""

    comparison: ComparisonById
    comparison_prompt_id: int
    sample_index: int


class ResultRecord(BaseModel):
    """A comparison of options, along with the preferred option.

    Each option consists of a number of task IDs.
    """

    created_at: datetime
    comparison_prompt_id: int
    comparison: ComparisonById
    sample_index: int
    comparison_prompt: str
    preferred_option_index: int | None
    api_params: AnyApiParameters = Field(discriminator="provider")
    api_response: AnyApiResponse

    @model_validator(mode="after")
    def check_option_index_in_range(self) -> Self:
        if self.preferred_option_index is None:
            return self
        if 0 <= self.preferred_option_index < len(self.comparison):
            return self
        logging.getLogger(__name__).error(
            f"preferred_option_index {self.preferred_option_index} is out of range"
            + f" for comparison of length {len(self.comparison)}"
        )
        raise ValueError("preferred_option_index is out of range")

    @property
    def key(self) -> ResultRecordKey:
        return ResultRecordKey(
            comparison=self.comparison,
            comparison_prompt_id=self.comparison_prompt_id,
            sample_index=self.sample_index,
        )


def comparison_to_id(comparison: Comparison) -> ComparisonById:
    option0, option1 = comparison
    return (
        tuple(task.id for task in option0),
        tuple(task.id for task in option1),
    )
