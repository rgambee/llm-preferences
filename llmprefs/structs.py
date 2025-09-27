import enum
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from openai.types.shared_params.reasoning_effort import ReasoningEffort
from pydantic import BaseModel, Field

TaskId = int


class TaskType(enum.Enum):
    dummy = "dummy"
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


class Provider(enum.StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class LLM(enum.StrEnum):
    CLAUDE_SONNET_4_0_2025_05_14 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4_0_2025_05_14 = "claude-opus-4-20250514"


class ApiParameters(BaseModel):
    model: LLM
    max_tokens: int
    system_prompt: str
    temperature: float


class AnthropicApiParams(ApiParameters):
    provider: Literal[Provider.ANTHROPIC]
    thinking_budget: int


class OpenAiApiParams(ApiParameters):
    provider: Literal[Provider.OPENAI]
    reasoning_effort: ReasoningEffort


AnyApiParams = AnthropicApiParams | OpenAiApiParams


class ResultRecord(BaseModel):
    """A comparison of options, along with the preferred option.

    Each option consists of a number of task IDs.
    """

    created_at: datetime
    comparison_prompt_id: int
    options: Sequence[Sequence[TaskId]]
    preference_index: int
    api_params: AnyApiParams = Field(discriminator="provider")
