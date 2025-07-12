import enum
from collections.abc import Sequence

from pydantic import BaseModel

TaskId = int


class TaskTopic(enum.Enum):
    pass


class TaskType(enum.Enum):
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


class ResultRecord(BaseModel):
    """A comparison of options, along with the preferred option.

    Each option consists of a number of task IDs.
    """

    options: Sequence[Sequence[TaskId]]
    preference_index: int
