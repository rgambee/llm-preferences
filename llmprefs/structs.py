import enum
from collections.abc import Sequence

from pydantic import BaseModel

TaskId = int


class TaskTopic(enum.Enum):
    dummy = "dummy"


class TaskType(enum.Enum):
    dummy = "dummy"
    opt_out = "opt_out"
    free_choice = "free_choice"


class TaskDifficulty(enum.Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class TaskImpact(enum.Enum):
    positive = "positive"
    neutral = "neutral"
    negative = "negative"


class TaskRecord(BaseModel):
    """Task plus categorical metadata."""

    id: TaskId
    task: str
    topic: TaskTopic
    type: TaskType
    difficulty: TaskDifficulty
    impact: TaskImpact


class LLM(enum.StrEnum):
    CLAUDE_SONNET_4_0_2025_05_14 = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4_0_2025_05_14 = "claude-opus-4-20250514"


class ResultRecord(BaseModel):
    """A comparison of options, along with the preferred option.

    Each option consists of a number of task IDs.
    """

    options: Sequence[Sequence[TaskId]]
    comparison_prompt_id: int
    preference_index: int
